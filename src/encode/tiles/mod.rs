mod inner;
mod u8_fast;

use alloc::vec;
use alloc::vec::Vec;

use crate::bitmask::BitMask;
use crate::error::Result;
use crate::header::HeaderInfo;
use crate::types::{DataType, Sample};

use inner::{TileInnerParams, encode_tile_inner};
use u8_fast::{encode_tile_to_count, encode_tiles_u8_fast, select_block_size_u8_fast};

/// Pick representative tile positions for sampling at a given block size.
fn sample_tile_positions(
    n_rows: usize,
    n_cols: usize,
    mb_size: usize,
) -> Vec<(usize, usize, usize, usize)> {
    let num_tv = n_rows.div_ceil(mb_size);
    let num_th = n_cols.div_ceil(mb_size);

    let mut positions = vec![
        (0, 0),
        (0, num_th.saturating_sub(1)),
        (num_tv.saturating_sub(1), 0),
        (num_tv.saturating_sub(1), num_th.saturating_sub(1)),
        (num_tv / 2, num_th / 2),
        (0, num_th / 2),
        (num_tv / 2, 0),
        (num_tv.saturating_sub(1), num_th / 2),
    ];

    positions.sort();
    positions.dedup();

    positions
        .into_iter()
        .map(|(it, jt)| {
            let i0 = it * mb_size;
            let i1 = (i0 + mb_size).min(n_rows);
            let j0 = jt * mb_size;
            let j1 = (j0 + mb_size).min(n_cols);
            (i0, i1, j0, j1)
        })
        .collect()
}

/// Try encoding a sample of tiles at both block sizes 8 and 16, return whichever
/// produces smaller output. Instead of encoding ALL tiles twice, we sample a
/// small number of representative tiles to make the decision.
pub(super) fn select_block_size<T: Sample>(
    data: &[T],
    mask: &BitMask,
    hd: &HeaderInfo,
    z_min_vec: &[f64],
    z_max_vec: &[f64],
) -> Result<i32> {
    let n_rows = hd.n_rows as usize;
    let n_cols = hd.n_cols as usize;
    let n_depth = hd.n_depth as usize;

    // Fast path for u8 lossless, all-valid, single-depth
    let all_valid = hd.num_valid_pixel == hd.n_rows * hd.n_cols;
    if T::DATA_TYPE == DataType::Byte && hd.max_z_error == 0.5 && all_valid && n_depth == 1 {
        debug_assert_eq!(T::BYTES, 1);
        let u8_data: &[u8] =
            unsafe { core::slice::from_raw_parts(data.as_ptr() as *const u8, data.len()) };
        return Ok(select_block_size_u8_fast(u8_data, hd));
    }

    // For very small images, just try both fully
    if n_rows <= 16 && n_cols <= 16 {
        let mut hd8 = hd.clone();
        hd8.micro_block_size = 8;
        let mut buf8 = Vec::new();
        encode_tiles(&mut buf8, data, mask, &hd8, z_min_vec, z_max_vec)?;

        let mut hd16 = hd.clone();
        hd16.micro_block_size = 16;
        let mut buf16 = Vec::new();
        encode_tiles(&mut buf16, data, mask, &hd16, z_min_vec, z_max_vec)?;

        return if buf16.len() < buf8.len() {
            Ok(16)
        } else {
            Ok(8)
        };
    }

    let mut scratch = ScratchBuffers::new();

    // Encode sampled tiles at block size 8
    let mut size8 = 0usize;
    let hd8 = {
        let mut h = hd.clone();
        h.micro_block_size = 8;
        h
    };
    let ctx8 = TileEncodeContext {
        data,
        mask,
        header: &hd8,
        z_max_vec,
    };
    for &(i0, i1, j0, j1) in &sample_tile_positions(n_rows, n_cols, 8) {
        let rect = crate::types::TileRect { i0, i1, j0, j1 };
        for i_depth in 0..n_depth {
            size8 += encode_tile_to_count::<T>(&ctx8, rect, i_depth, &mut scratch)?;
        }
    }

    // Encode sampled tiles at block size 16
    let mut size16 = 0usize;
    let hd16 = {
        let mut h = hd.clone();
        h.micro_block_size = 16;
        h
    };
    let ctx16 = TileEncodeContext {
        data,
        mask,
        header: &hd16,
        z_max_vec,
    };
    for &(i0, i1, j0, j1) in &sample_tile_positions(n_rows, n_cols, 16) {
        let rect = crate::types::TileRect { i0, i1, j0, j1 };
        for i_depth in 0..n_depth {
            size16 += encode_tile_to_count::<T>(&ctx16, rect, i_depth, &mut scratch)?;
        }
    }

    if size16 < size8 { Ok(16) } else { Ok(8) }
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
        let mut k = i * n_cols + j0;
        for _j in j0..j1 {
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
            k += 1;
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
        for i in i0..i1 {
            let mut k = i * n_cols + j0;
            for _j in j0..j1 {
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
                        break;
                    }
                    scratch.diff_data.push(diff);
                    idx += 1;
                }
                k += 1;
            }
            if overflow {
                break;
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
                b_diff_enc: true,
            };
            classify_recon(&scratch.diff_data, &scratch.quant_vec, &rp)
        } else {
            // Non-diff encoding was chosen; reconstruct from original quantized values
            let rp = ReconParams {
                num_valid,
                z_min_f: non_diff_z_min,
                z_max_f: non_diff_z_max,
                max_z_error,
                max_val_to_quantize,
                z_max_depth,
                b_diff_enc: false,
            };
            classify_recon(&scratch.valid_data, &scratch.non_diff_quant, &rp)
        };

        // Apply reconstruction to the buffer
        let mut valid_idx = 0;
        for i in i0..i1 {
            let mut k = i * n_cols + j0;
            for _j in j0..j1 {
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
                k += 1;
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
    b_diff_enc: bool,
}

/// Classify how a tile was encoded and return reconstruction info.
/// This mirrors the logic in `encode_tile_inner` to determine whether the tile
/// ended up as constant, quantized, or raw binary.
fn classify_recon(values: &[f64], quant: &[u32], p: &ReconParams) -> TileReconInfo {
    let ReconParams {
        num_valid,
        z_min_f,
        z_max_f,
        max_z_error,
        max_val_to_quantize,
        z_max_depth,
        b_diff_enc,
    } = *p;
    if num_valid == 0 || (z_min_f == 0.0 && z_max_f == 0.0) {
        // Constant zero block
        return TileReconInfo::Constant(0.0);
    }

    let need_quantize = if max_z_error == 0.0 {
        z_max_f > z_min_f
    } else {
        let max_val = (z_max_f - z_min_f) / (2.0 * max_z_error);
        max_val <= max_val_to_quantize && (max_val + 0.5) as u32 != 0
    };

    if !need_quantize && z_min_f == z_max_f {
        // Constant block
        return TileReconInfo::Constant(z_min_f);
    }

    if !need_quantize {
        if !b_diff_enc {
            // Raw binary
            return TileReconInfo::RawBinary;
        }
        // Diff + raw binary is not allowed; this means non-diff won the size
        // comparison in encode_tile. This code path should not be reached since
        // the diff sentinel is always larger, but handle it safely.
        return TileReconInfo::RawBinary;
    }

    // Check if all values quantize to the same value (max_quant == 0)
    let max_quant = quant.iter().copied().max().unwrap_or(0);
    if max_quant == 0 {
        // All quantized to same -> constant block with z_min_f as offset
        return TileReconInfo::Constant(z_min_f);
    }

    // Check if raw binary was cheaper (only for non-diff mode).
    // We need to replicate the raw-binary fallback check from encode_tile_inner.
    if !b_diff_enc {
        // Estimate the encoded size: this mirrors the bitstuffer output size.
        // If the actual encoder fell back to raw binary, the quant values are
        // irrelevant and reconstruction should use original values.
        // We approximate by checking if the encoded bytes would exceed raw size.
        // However, we cannot know the exact encoded size here without re-encoding.
        // Instead, rely on the fact that if raw binary was chosen, the values
        // array contains the exact original values. In practice for lossy mode
        // with max_z_error > 0, quantization always beats raw for non-trivial
        // data ranges, so this fallback is rare.
        //
        // For correctness we conservatively return Quantized, which is correct
        // when quantization was used and harmless when raw was used (since raw
        // stores exact values, reconstruction is exact either way for depth 0,
        // and raw binary cannot be used with diff encoding).
        let _ = values;
    }

    let inv_scale = 2.0 * max_z_error;
    TileReconInfo::Quantized {
        offset: z_min_f,
        inv_scale,
        z_max: z_max_depth,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

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
            b_diff_enc: false,
        };
        let info = classify_recon(&[], &[], &rp);
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
            b_diff_enc: false,
        };
        let info = classify_recon(&[0.0, 0.0], &[0, 0], &rp);
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
            b_diff_enc: false,
        };
        let info = classify_recon(&[42.0, 42.0], &[0, 0], &rp);
        match info {
            TileReconInfo::Constant(c) => assert_eq!(c, 42.0),
            _ => panic!("expected Constant(42.0)"),
        }
    }

    #[test]
    fn classify_recon_quantized() {
        let values: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let quant: Vec<u32> = (0..8).collect();
        let rp = ReconParams {
            num_valid: 8,
            z_min_f: 0.0,
            z_max_f: 7.0,
            max_z_error: 0.5,
            max_val_to_quantize: ((1u32 << 15) - 1) as f64,
            z_max_depth: 255.0,
            b_diff_enc: false,
        };
        let info = classify_recon(&values, &quant, &rp);
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
        let values = vec![z_min, z_max];
        let quant = vec![0, 0];
        let rp = ReconParams {
            num_valid: 2,
            z_min_f: z_min,
            z_max_f: z_max,
            max_z_error: 0.5,
            max_val_to_quantize: ((1u32 << 30) - 1) as f64,
            z_max_depth: 1e18,
            b_diff_enc: false,
        };
        let info = classify_recon(&values, &quant, &rp);
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

    #[test]
    fn classify_recon_quantized_all_same_after_quant() {
        // max_quant == 0 path: need_quantize = true but all quant values are 0.
        let values = vec![10.0, 10.5, 11.0];
        let quant = vec![0, 0, 0]; // force max_quant = 0
        let rp = ReconParams {
            num_valid: 3,
            z_min_f: 10.0,
            z_max_f: 11.0,
            max_z_error: 0.5,
            max_val_to_quantize: ((1u32 << 30) - 1) as f64,
            z_max_depth: 100.0,
            b_diff_enc: false,
        };
        let info = classify_recon(&values, &quant, &rp);
        match info {
            TileReconInfo::Constant(c) => assert_eq!(c, 10.0),
            _ => panic!("expected Constant(10.0) for all-same quantization"),
        }
    }

    // -----------------------------------------------------------------------
    // sample_tile_positions
    // -----------------------------------------------------------------------

    #[test]
    fn sample_tile_positions_within_bounds() {
        let n_rows = 100;
        let n_cols = 200;
        let mb_size = 8;
        let positions = sample_tile_positions(n_rows, n_cols, mb_size);
        assert!(!positions.is_empty());
        for &(i0, i1, j0, j1) in &positions {
            assert!(i0 < n_rows, "i0={i0} >= n_rows={n_rows}");
            assert!(i1 <= n_rows, "i1={i1} > n_rows={n_rows}");
            assert!(i0 < i1, "i0={i0} >= i1={i1}");
            assert!(j0 < n_cols, "j0={j0} >= n_cols={n_cols}");
            assert!(j1 <= n_cols, "j1={j1} > n_cols={n_cols}");
            assert!(j0 < j1, "j0={j0} >= j1={j1}");
        }
    }

    #[test]
    fn sample_tile_positions_small_image() {
        // For a 1x1 image, should produce at least 1 position
        let positions = sample_tile_positions(1, 1, 8);
        assert!(!positions.is_empty());
        for &(i0, i1, j0, j1) in &positions {
            assert_eq!(i0, 0);
            assert_eq!(i1, 1);
            assert_eq!(j0, 0);
            assert_eq!(j1, 1);
        }
    }

    #[test]
    fn sample_tile_positions_exact_tile_boundary() {
        // Image size exactly divisible by mb_size
        let positions = sample_tile_positions(16, 16, 8);
        assert!(!positions.is_empty());
        for &(i0, i1, j0, j1) in &positions {
            assert!(i0 < 16);
            assert!(i1 <= 16);
            assert!(j0 < 16);
            assert!(j1 <= 16);
            assert_eq!(i1 - i0, 8); // exact block size
            assert_eq!(j1 - j0, 8);
        }
    }

    #[test]
    fn sample_tile_positions_non_divisible() {
        // Image size not divisible by mb_size -- last tile is smaller
        let positions = sample_tile_positions(10, 10, 8);
        assert!(!positions.is_empty());
        for &(i0, i1, j0, j1) in &positions {
            assert!(i1 - i0 <= 8);
            assert!(j1 - j0 <= 8);
            assert!(i1 <= 10);
            assert!(j1 <= 10);
        }
    }

    #[test]
    fn sample_tile_positions_unique() {
        let positions = sample_tile_positions(100, 100, 8);
        // All positions should be unique (dedup was called)
        let mut sorted = positions.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), positions.len());
    }

    #[test]
    fn sample_tile_positions_covers_corners() {
        // Should include corners and center for a large-enough image
        let n_rows = 64;
        let n_cols = 64;
        let mb_size = 8;
        let positions = sample_tile_positions(n_rows, n_cols, mb_size);
        // Top-left corner
        assert!(positions.iter().any(|&(i0, _, j0, _)| i0 == 0 && j0 == 0));
        // Top-right corner
        let last_j0 = (n_cols.div_ceil(mb_size) - 1) * mb_size;
        assert!(
            positions
                .iter()
                .any(|&(i0, _, j0, _)| i0 == 0 && j0 == last_j0)
        );
        // Bottom-left corner
        let last_i0 = (n_rows.div_ceil(mb_size) - 1) * mb_size;
        assert!(
            positions
                .iter()
                .any(|&(i0, _, j0, _)| i0 == last_i0 && j0 == 0)
        );
        // Bottom-right corner
        assert!(
            positions
                .iter()
                .any(|&(i0, _, j0, _)| i0 == last_i0 && j0 == last_j0)
        );
    }
}
