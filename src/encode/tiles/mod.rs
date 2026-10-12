mod inner;
mod u8_fast;

use alloc::vec;
use alloc::vec::Vec;

use crate::bitmask::BitMask;
use crate::error::Result;
use crate::header::HeaderInfo;
use crate::types::{DataType, Sample};

use inner::{RangeCoding, TileInnerParams, classify_range, encode_tile_inner};
use u8_fast::encode_tiles_u8_fast;

/// The micro-block size Lerc2 encodes with first, and the one it may switch to.
const MICRO_BLOCK_SIZE: i32 = 8;
const MICRO_BLOCK_SIZE_DOUBLED: i32 = 2 * MICRO_BLOCK_SIZE;

/// Tiles encoded at the micro-block size they were chosen for.
pub(super) struct EncodedTiles {
    pub(super) micro_block_size: i32,
    pub(super) bytes: Vec<u8>,
}

/// Encode every tile at the micro-block size Esri's Lerc2 would choose
/// (`Lerc2::ComputeNumBytesNeededToWrite` in `Lerc2.cpp`).
///
/// Encode at 8 × 8 first. When that comes to under 1.5 bits per pixel, the
/// per-block headers (a flag byte and an offset per block) are a large share
/// of the bytes, so encode again at 16 × 16 — a quarter of the headers — and
/// keep it if it is no larger. Ties go to 16, which has fewer blocks to
/// decode. Esri also skips the second pass when bit stuffing is ineffective
/// (8 × 8 tiling at four times the raw size or more) and for images no
/// larger than one block; both are kept here. Its third condition, tiling not
/// much worse than Huffman, only saves work when Huffman wins anyway and is
/// not applied: the Huffman comparison still runs on the chosen tiles.
///
/// Both sizes are priced over every pixel. An earlier version sampled eight
/// 8 × 8 and eight 16 × 16 blocks and compared the byte totals, which covered
/// four times as many pixels at 16 and so favoured 8 — by 4× on flat data,
/// where each block of either size is one constant.
pub(super) fn encode_tiles_at_best_block_size<T: Sample>(
    data: &[T],
    mask: &BitMask,
    header: &HeaderInfo,
    z_min_vec: &[f64],
    z_max_vec: &[f64],
) -> Result<EncodedTiles> {
    let encode_at = |micro_block_size: i32| -> Result<EncodedTiles> {
        let mut hd = header.clone();
        hd.micro_block_size = micro_block_size;
        let mut bytes = Vec::new();
        encode_tiles(&mut bytes, data, mask, &hd, z_min_vec, z_max_vec)?;
        Ok(EncodedTiles {
            micro_block_size,
            bytes,
        })
    };

    let tiles8 = encode_at(MICRO_BLOCK_SIZE)?;
    if !worth_trying_doubled_block_size::<T>(tiles8.bytes.len(), header) {
        return Ok(tiles8);
    }
    let tiles16 = encode_at(MICRO_BLOCK_SIZE_DOUBLED)?;
    Ok(if tiles16.bytes.len() <= tiles8.bytes.len() {
        tiles16
    } else {
        tiles8
    })
}

/// Esri's conditions for re-encoding at the doubled block size, given the
/// 8 × 8 tiling's byte count.
fn worth_trying_doubled_block_size<T: Sample>(tiling_bytes: usize, header: &HeaderInfo) -> bool {
    let n_depth = header.n_depth as usize;
    let num_total = header.n_rows as usize * header.n_cols as usize;
    // Bytes of the uncompressed values ("one sweep" in Lerc2).
    let raw_bytes = header.num_valid_pixel as usize * n_depth * T::BYTES;
    // tiling_bytes × 8 bits < 1.5 bits × num_total × n_depth, in integers.
    let low_bit_rate = tiling_bytes * 16 < num_total * n_depth * 3;
    let bit_stuffing_effective = tiling_bytes < 4 * raw_bytes;
    let larger_than_one_block =
        header.n_rows > MICRO_BLOCK_SIZE || header.n_cols > MICRO_BLOCK_SIZE;
    low_bit_rate && bit_stuffing_effective && larger_than_one_block
}

/// Scratch buffers reused across tiles to avoid per-tile allocations.
pub(super) struct ScratchBuffers {
    pub(super) valid_data: Vec<f64>,
    pub(super) diff_data: Vec<f64>,
    pub(super) quant_vec: Vec<u32>,
    /// Holds the quantized values from the non-diff encoding attempt,
    /// saved before the diff encoding attempt overwrites `quant_vec`.
    pub(super) non_diff_quant: Vec<u32>,
    pub(super) tile_buf: Vec<u8>,
}

impl ScratchBuffers {
    pub(super) fn new() -> Self {
        Self {
            valid_data: Vec::new(),
            diff_data: Vec::new(),
            quant_vec: Vec::new(),
            non_diff_quant: Vec::new(),
            tile_buf: Vec::new(),
        }
    }
}

/// Information about how a tile was encoded, used to compute the decoder's
/// reconstructed values for the previous-depth reconstruction buffer.
#[derive(Clone, Copy)]
enum TileReconInfo {
    /// All valid pixels in the tile have the same reconstructed value.
    Constant(f64),
    /// Values were quantized: reconstructed[i] = offset + quant[i] * inv_scale
    /// (clamped to z_max).
    Quantized {
        offset: f64,
        inv_scale: f64,
        z_max: f64,
    },
    /// Raw binary: reconstruction matches the original values exactly.
    RawBinary,
}

/// Read-only context shared across all tiles during encoding.
struct TileEncodeContext<'a, T> {
    data: &'a [T],
    mask: &'a BitMask,
    header: &'a HeaderInfo,
    z_max_vec: &'a [f64],
}

pub(super) fn encode_tiles<T: Sample>(
    blob: &mut Vec<u8>,
    data: &[T],
    mask: &BitMask,
    header: &HeaderInfo,
    _z_min_vec: &[f64],
    z_max_vec: &[f64],
) -> Result<()> {
    let mb_size = header.micro_block_size as usize;
    let n_depth = header.n_depth as usize;
    let n_rows = header.n_rows as usize;
    let n_cols = header.n_cols as usize;
    let max_z_error = header.max_z_error;

    // Fast path: u8 lossless, all-valid, single depth.
    // This avoids f64 conversions, uses stack-allocated arrays, and merges
    // quantization + LUT decision + encoding in a single pass per tile.
    let all_valid = header.num_valid_pixel == header.n_rows * header.n_cols;
    if T::DATA_TYPE == DataType::Byte && max_z_error == 0.5 && all_valid && n_depth == 1 {
        // T is u8 (verified by DATA_TYPE == Byte and T::BYTES == 1).
        // Safe reinterpret: u8 has size 1, alignment 1, same as T when T=u8.
        debug_assert_eq!(T::BYTES, 1);
        debug_assert_eq!(core::mem::size_of::<T>(), 1);
        debug_assert_eq!(core::mem::align_of::<T>(), 1);
        let u8_data: &[u8] =
            unsafe { core::slice::from_raw_parts(data.as_ptr() as *const u8, data.len()) };
        let mut quant_buf = [0u32; 256]; // max tile size is 16x16=256
        encode_tiles_u8_fast(blob, u8_data, header, &mut quant_buf);
        return Ok(());
    }

    let num_tiles_vert = n_rows.div_ceil(mb_size);
    let num_tiles_hori = n_cols.div_ceil(mb_size);

    let ctx = TileEncodeContext {
        data,
        mask,
        header,
        z_max_vec,
    };
    let mut scratch = ScratchBuffers::new();

    // Allocate a per-pixel reconstruction buffer when multi-depth lossy encoding
    // is active. This mirrors the C++ ScaleBack approach: after encoding each
    // depth slice, we store the values as the decoder would reconstruct them,
    // so that subsequent depth-slice diffs are computed from reconstructed (not
    // original) values.
    let needs_recon = n_depth > 1 && max_z_error > 0.0;
    let mut recon_buf: Vec<f64> = if needs_recon {
        vec![0.0; n_rows * n_cols]
    } else {
        Vec::new()
    };

    for i_tile in 0..num_tiles_vert {
        let i0 = i_tile * mb_size;
        let i1 = (i0 + mb_size).min(n_rows);

        for j_tile in 0..num_tiles_hori {
            let j0 = j_tile * mb_size;
            let j1 = (j0 + mb_size).min(n_cols);
            let rect = crate::types::TileRect { i0, i1, j0, j1 };

            for i_depth in 0..n_depth {
                encode_tile(
                    blob,
                    &ctx,
                    rect,
                    i_depth,
                    &mut scratch,
                    if needs_recon {
                        Some(&mut recon_buf)
                    } else {
                        None
                    },
                )?;
            }
        }
    }

    Ok(())
}

/// Encode a single tile block (one depth slice of one micro-block).
/// When depth > 0 and nDepth > 1, the encoder tries diff (delta) encoding
/// relative to the previous depth slice and picks whichever representation
/// is smaller.
///
/// When `recon_buf` is `Some`, it holds the decoder-reconstructed values for the
/// previous depth slice (indexed by pixel index `k = row * n_cols + col`).
/// Diffs are computed against these reconstructed values instead of the original
/// data. After encoding, the buffer is updated with this depth slice's
/// reconstructed values so it is ready for the next depth.
fn encode_tile<T: Sample>(
    blob: &mut Vec<u8>,
    ctx: &TileEncodeContext<'_, T>,
    rect: crate::types::TileRect,
    i_depth: usize,
    scratch: &mut ScratchBuffers,
    recon_buf: Option<&mut Vec<f64>>,
) -> Result<()> {
    let crate::types::TileRect { i0, i1, j0, j1 } = rect;
    let data = ctx.data;
    let mask = ctx.mask;
    let header = ctx.header;
    let z_max_vec = ctx.z_max_vec;
    let n_cols = header.n_cols as usize;
    let n_depth = header.n_depth as usize;
    let max_z_error = header.max_z_error;
    let max_val_to_quantize: f64 = match header.data_type {
        DataType::Char | DataType::Byte | DataType::Short | DataType::UShort => {
            ((1u32 << 15) - 1) as f64
        }
        _ => ((1u32 << 30) - 1) as f64,
    };
    let try_diff = n_depth > 1 && i_depth > 0;

    // Collect valid pixels for this depth slice (reusing scratch buffer)
    scratch.valid_data.clear();
    let mut z_min_f = f64::MAX;
    let mut z_max_f = f64::MIN;

    for i in i0..i1 {
        for j in j0..j1 {
            let k = i * n_cols + j;
            if mask.is_valid(k) {
                let val = data[k * n_depth + i_depth].to_f64();
                scratch.valid_data.push(val);
                if val < z_min_f {
                    z_min_f = val;
                }
                if val > z_max_f {
                    z_max_f = val;
                }
            }
        }
    }

    let num_valid = scratch.valid_data.len();

    // Integrity check bits (pattern depends on version)
    let pattern: u8 = if header.version >= 5 { 14 } else { 15 };
    let integrity = ((j0 as u8 >> 3) & pattern) << 2;

    // Compute diff values if applicable.
    // When a reconstruction buffer is provided, use the reconstructed (quantized)
    // previous-depth values instead of the original data. This matches what the
    // C++ encoder does via ScaleBack and what the decoder uses for reconstruction.
    let has_diff = if try_diff {
        scratch.diff_data.clear();
        let mut overflow = false;
        let mut idx = 0;
        'outer: for i in i0..i1 {
            for j in j0..j1 {
                let k = i * n_cols + j;
                if mask.is_valid(k) {
                    let cur = data[k * n_depth + i_depth].to_f64();
                    let prev = if let Some(ref rb) = recon_buf {
                        rb[k]
                    } else {
                        data[k * n_depth + i_depth - 1].to_f64()
                    };
                    let diff = cur - prev;
                    // Check for integer overflow: for integer types, the diff
                    // must be representable as i32
                    if T::is_integer() && (diff < i32::MIN as f64 || diff > i32::MAX as f64) {
                        overflow = true;
                        break 'outer;
                    }
                    scratch.diff_data.push(diff);
                    idx += 1;
                }
            }
        }
        !overflow && idx == num_valid
    } else {
        false
    };

    // Encode without diff
    scratch.tile_buf.clear();
    let non_diff_params = TileInnerParams {
        num_valid,
        z_min_f,
        z_max_f,
        max_z_error,
        max_val_to_quantize,
        integrity,
        src_data_type: header.data_type,
        b_diff_enc: false,
    };
    encode_tile_inner::<T>(
        &mut scratch.tile_buf,
        &scratch.valid_data,
        &mut scratch.quant_vec,
        &non_diff_params,
    );
    let non_diff_len = scratch.tile_buf.len();

    // Save the non-diff quantized values before the diff attempt overwrites quant_vec
    scratch.non_diff_quant.clear();
    scratch.non_diff_quant.extend_from_slice(&scratch.quant_vec);
    let non_diff_z_min = z_min_f;
    let non_diff_z_max = z_max_f;

    // Encode with diff (if available)
    let mut used_diff = false;
    let mut diff_z_min = 0.0f64;
    let mut diff_z_max = 0.0f64;
    if has_diff {
        let mut diff_min = f64::MAX;
        let mut diff_max = f64::MIN;
        for &d in scratch.diff_data.iter() {
            if d < diff_min {
                diff_min = d;
            }
            if d > diff_max {
                diff_max = d;
            }
        }
        diff_z_min = diff_min;
        diff_z_max = diff_max;
        // Append diff encoding after the non-diff data in tile_buf
        let diff_start = scratch.tile_buf.len();
        let diff_params = TileInnerParams {
            num_valid,
            z_min_f: diff_min,
            z_max_f: diff_max,
            max_z_error,
            max_val_to_quantize,
            integrity,
            src_data_type: header.data_type,
            b_diff_enc: true,
        };
        encode_tile_inner::<T>(
            &mut scratch.tile_buf,
            &scratch.diff_data,
            &mut scratch.quant_vec,
            &diff_params,
        );
        let diff_len = scratch.tile_buf.len() - diff_start;

        // Pick the smaller encoding
        if diff_len < non_diff_len {
            blob.extend_from_slice(&scratch.tile_buf[diff_start..]);
            used_diff = true;
        } else {
            blob.extend_from_slice(&scratch.tile_buf[..non_diff_len]);
        }
    } else {
        blob.extend_from_slice(&scratch.tile_buf);
    }

    // Update the reconstruction buffer with what the decoder would produce for
    // this depth slice.
    if let Some(rb) = recon_buf {
        let z_max_depth = if n_depth > 1 {
            z_max_vec[i_depth]
        } else {
            header.z_max
        };

        // Determine reconstruction info based on the chosen encoding path
        let recon_info = if used_diff {
            // Diff encoding was chosen; reconstruct from diff quantized values
            let rp = ReconParams {
                num_valid,
                z_min_f: diff_z_min,
                z_max_f: diff_z_max,
                max_z_error,
                max_val_to_quantize,
                z_max_depth,
            };
            classify_recon(&rp)
        } else {
            // Non-diff encoding was chosen; reconstruct from original quantized values
            let rp = ReconParams {
                num_valid,
                z_min_f: non_diff_z_min,
                z_max_f: non_diff_z_max,
                max_z_error,
                max_val_to_quantize,
                z_max_depth,
            };
            classify_recon(&rp)
        };

        // Apply reconstruction to the buffer
        let mut valid_idx = 0;
        for i in i0..i1 {
            for j in j0..j1 {
                let k = i * n_cols + j;
                if mask.is_valid(k) {
                    let recon_val = match recon_info {
                        TileReconInfo::Constant(c) => {
                            if used_diff {
                                // diff constant: decoder adds offset to prev
                                (c + rb[k]).min(z_max_depth)
                            } else {
                                c
                            }
                        }
                        TileReconInfo::Quantized {
                            offset,
                            inv_scale,
                            z_max,
                        } => {
                            let q = if used_diff {
                                scratch.quant_vec[valid_idx] as f64
                            } else {
                                scratch.non_diff_quant[valid_idx] as f64
                            };
                            let z = offset + q * inv_scale;
                            if used_diff {
                                (z + rb[k]).min(z_max)
                            } else {
                                z.min(z_max)
                            }
                        }
                        TileReconInfo::RawBinary => {
                            // Raw binary: the original values are stored exactly
                            scratch.valid_data[valid_idx]
                        }
                    };
                    rb[k] = recon_val;
                    valid_idx += 1;
                }
            }
        }
    }

    Ok(())
}

/// Parameters for classifying how a tile was encoded for reconstruction.
struct ReconParams {
    num_valid: usize,
    z_min_f: f64,
    z_max_f: f64,
    max_z_error: f64,
    max_val_to_quantize: f64,
    z_max_depth: f64,
}

/// Classify how a tile was encoded and return reconstruction info.
///
/// The block's coding comes from [`classify_range`], the same decision
/// `encode_tile_inner` wrote it with, so the reconstruction tracks what the
/// decoder will rebuild.
fn classify_recon(p: &ReconParams) -> TileReconInfo {
    let ReconParams {
        num_valid,
        z_min_f,
        z_max_f,
        max_z_error,
        max_val_to_quantize,
        z_max_depth,
    } = *p;
    match classify_range(
        num_valid,
        z_min_f,
        z_max_f,
        max_z_error,
        max_val_to_quantize,
    ) {
        RangeCoding::ConstZero => TileReconInfo::Constant(0.0),
        // Every value is within max_z_error of z_min; the decoder rebuilds z_min.
        RangeCoding::Constant => TileReconInfo::Constant(z_min_f),
        // A raw block cannot be diff encoded (the diff attempt writes a
        // sentinel that always loses), so this is the non-diff path, whose
        // values are stored exactly.
        RangeCoding::Raw => TileReconInfo::RawBinary,
        // When the bit-stuffed form came out larger than raw, the encoder
        // stored the block raw instead and the decoder reads it back exactly;
        // reconstructing it as quantized is then off by at most max_z_error,
        // as the C++ encoder's ScaleBack is. Raw cannot carry a diff, so the
        // fallback only happens on the non-diff path.
        RangeCoding::Quantized { .. } => TileReconInfo::Quantized {
            offset: z_min_f,
            inv_scale: 2.0 * max_z_error,
            z_max: z_max_depth,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // encode_tiles_at_best_block_size
    // -----------------------------------------------------------------------

    fn f32_header(n_rows: i32, n_cols: i32, max_z_error: f64) -> HeaderInfo {
        HeaderInfo {
            version: 3,
            n_rows,
            n_cols,
            n_depth: 1,
            num_valid_pixel: n_rows * n_cols,
            data_type: DataType::Float,
            max_z_error,
            ..HeaderInfo::default()
        }
    }

    fn best_block_size(data: &[f32], hd: &HeaderInfo) -> i32 {
        let (z_min, z_max) = data.iter().fold((f64::MAX, f64::MIN), |(lo, hi), &v| {
            (lo.min(f64::from(v)), hi.max(f64::from(v)))
        });
        let mask = BitMask::all_valid(data.len());
        encode_tiles_at_best_block_size(data, &mask, hd, &[z_min], &[z_max])
            .expect("encode")
            .micro_block_size
    }

    #[test]
    fn flat_data_takes_the_doubled_block_size() {
        // Sub-tolerance noise: every block is one constant at either size, so
        // 16 × 16 needs a quarter of the block headers.
        let data: Vec<f32> = (0..64 * 64)
            .map(|i| 1000.0 + 0.001 * ((i * 7919) % 13) as f32)
            .collect();
        assert_eq!(best_block_size(&data, &f32_header(64, 64, 0.02)), 16);
    }

    #[test]
    fn busy_data_keeps_the_8x8_block_size() {
        // Several bits per pixel: Lerc2 does not try 16 × 16.
        let data: Vec<f32> = (0..64 * 64).map(|i| ((i * 7919) % 1000) as f32).collect();
        assert_eq!(best_block_size(&data, &f32_header(64, 64, 0.01)), 8);
    }

    #[test]
    fn doubled_block_size_is_tried_only_at_low_bit_rates() {
        let hd = f32_header(64, 64, 0.01);
        // 1.5 bits per pixel is the threshold: 768 bytes for 4096 pixels.
        assert!(worth_trying_doubled_block_size::<f32>(767, &hd));
        assert!(!worth_trying_doubled_block_size::<f32>(768, &hd));
    }

    #[test]
    fn doubled_block_size_is_not_tried_on_a_single_block() {
        let hd = f32_header(8, 8, 0.01);
        assert!(!worth_trying_doubled_block_size::<f32>(5, &hd));
        let hd = f32_header(8, 9, 0.01);
        assert!(worth_trying_doubled_block_size::<f32>(5, &hd));
    }

    // -----------------------------------------------------------------------
    // classify_recon
    // -----------------------------------------------------------------------

    #[test]
    fn classify_recon_const_zero_num_valid_zero() {
        let rp = ReconParams {
            num_valid: 0,
            z_min_f: f64::MAX,
            z_max_f: f64::MIN,
            max_z_error: 0.5,
            max_val_to_quantize: ((1u32 << 15) - 1) as f64,
            z_max_depth: 255.0,
        };
        let info = classify_recon(&rp);
        match info {
            TileReconInfo::Constant(c) => assert_eq!(c, 0.0),
            _ => panic!("expected Constant(0.0)"),
        }
    }

    #[test]
    fn classify_recon_const_zero_zmin_zmax_zero() {
        let rp = ReconParams {
            num_valid: 2,
            z_min_f: 0.0,
            z_max_f: 0.0,
            max_z_error: 0.5,
            max_val_to_quantize: ((1u32 << 15) - 1) as f64,
            z_max_depth: 255.0,
        };
        let info = classify_recon(&rp);
        match info {
            TileReconInfo::Constant(c) => assert_eq!(c, 0.0),
            _ => panic!("expected Constant(0.0)"),
        }
    }

    #[test]
    fn classify_recon_const_offset() {
        let rp = ReconParams {
            num_valid: 2,
            z_min_f: 42.0,
            z_max_f: 42.0,
            max_z_error: 0.5,
            max_val_to_quantize: ((1u32 << 15) - 1) as f64,
            z_max_depth: 255.0,
        };
        let info = classify_recon(&rp);
        match info {
            TileReconInfo::Constant(c) => assert_eq!(c, 42.0),
            _ => panic!("expected Constant(42.0)"),
        }
    }

    #[test]
    fn classify_recon_sub_tolerance_range_is_constant_at_z_min() {
        // A range under half a quantization step is written as a constant
        // block at z_min, so the decoder rebuilds z_min for every value.
        let rp = ReconParams {
            num_valid: 64,
            z_min_f: 1000.0,
            z_max_f: 1000.018,
            max_z_error: 0.02,
            max_val_to_quantize: ((1u32 << 30) - 1) as f64,
            z_max_depth: 2000.0,
        };
        match classify_recon(&rp) {
            TileReconInfo::Constant(c) => assert_eq!(c, 1000.0),
            _ => panic!("expected Constant(1000.0)"),
        }
    }

    #[test]
    fn classify_recon_quantized() {
        let rp = ReconParams {
            num_valid: 8,
            z_min_f: 0.0,
            z_max_f: 7.0,
            max_z_error: 0.5,
            max_val_to_quantize: ((1u32 << 15) - 1) as f64,
            z_max_depth: 255.0,
        };
        let info = classify_recon(&rp);
        match info {
            TileReconInfo::Quantized {
                offset,
                inv_scale,
                z_max,
            } => {
                assert_eq!(offset, 0.0);
                assert_eq!(inv_scale, 1.0); // 2 * 0.5
                assert_eq!(z_max, 255.0);
            }
            _ => panic!("expected Quantized"),
        }
    }

    #[test]
    fn classify_recon_raw_binary() {
        // For need_quantize = false with z_min != z_max:
        //   max_z_error > 0 and max_val > max_val_to_quantize => need_quantize = false.
        let z_min = 0.0;
        let z_max = 1e18; // huge range
        let rp = ReconParams {
            num_valid: 2,
            z_min_f: z_min,
            z_max_f: z_max,
            max_z_error: 0.5,
            max_val_to_quantize: ((1u32 << 30) - 1) as f64,
            z_max_depth: 1e18,
        };
        let info = classify_recon(&rp);
        match info {
            TileReconInfo::RawBinary => {}
            _ => panic!(
                "expected RawBinary, got {:?}",
                match info {
                    TileReconInfo::Constant(c) => alloc::format!("Constant({c})"),
                    TileReconInfo::Quantized {
                        offset,
                        inv_scale,
                        z_max,
                    } => alloc::format!(
                        "Quantized(offset={offset}, inv_scale={inv_scale}, z_max={z_max})"
                    ),
                    TileReconInfo::RawBinary => "RawBinary".into(),
                }
            ),
        }
    }
}
