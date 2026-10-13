mod inner;
mod u8_fast;

use alloc::vec::Vec;

use crate::bitmask::BitMask;
use crate::error::Result;
use crate::header::HeaderInfo;
use crate::types::{DataType, Sample};

use inner::{TileInnerParams, encode_tile_inner};
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
) -> Result<EncodedTiles> {
    let encode_at = |micro_block_size: i32| -> Result<EncodedTiles> {
        let mut hd = header.clone();
        hd.micro_block_size = micro_block_size;
        let mut bytes = Vec::new();
        encode_tiles(&mut bytes, data, mask, &hd)?;
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
    pub(super) tile_buf: Vec<u8>,
}

impl ScratchBuffers {
    pub(super) fn new() -> Self {
        Self {
            valid_data: Vec::new(),
            diff_data: Vec::new(),
            quant_vec: Vec::new(),
            tile_buf: Vec::new(),
        }
    }
}

/// Read-only context shared across all tiles during encoding.
struct TileEncodeContext<'a, T> {
    data: &'a [T],
    mask: &'a BitMask,
    header: &'a HeaderInfo,
    /// Whether depth slices after the first may be coded as differences from
    /// the slice before ([`tries_diff_encoding`]).
    try_diff: bool,
}

/// Whether a blob's depth slices after the first may be coded as differences
/// from the slice before, as Lerc2 v5 allows.
///
/// Esri's encoder tries it only for lossless integer data
/// (`Lerc2::WriteTiles`: `bTryDiffEnc = version >= 5 && nDepth > 1 &&
/// bIntLossless`, "to avoid error propagation"), and so does this one. There
/// the previous slice decodes to exactly its source values, every difference
/// is an integer, and a block's offset is stored exactly in an integer type.
///
/// This encoder once tried it for lossy and float data too. The previous
/// slice then had to be tracked as the decoder would rebuild it, and a
/// block's offset became an f64 difference that its storage type could not
/// hold: a float offset of -0.99999997 was judged by its f32 rounding (-1),
/// stored as a 16-bit integer and written truncated to 0, so every value in
/// the block decoded 1.0 too high.
fn tries_diff_encoding<T: Sample>(header: &HeaderInfo) -> bool {
    header.version >= 5 && header.n_depth > 1 && T::is_integer() && header.max_z_error == 0.5
}

pub(super) fn encode_tiles<T: Sample>(
    blob: &mut Vec<u8>,
    data: &[T],
    mask: &BitMask,
    header: &HeaderInfo,
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
        try_diff: tries_diff_encoding::<T>(header),
    };
    let mut scratch = ScratchBuffers::new();

    for i_tile in 0..num_tiles_vert {
        let i0 = i_tile * mb_size;
        let i1 = (i0 + mb_size).min(n_rows);

        for j_tile in 0..num_tiles_hori {
            let j0 = j_tile * mb_size;
            let j1 = (j0 + mb_size).min(n_cols);
            let rect = crate::types::TileRect { i0, i1, j0, j1 };

            for i_depth in 0..n_depth {
                encode_tile(blob, &ctx, rect, i_depth, &mut scratch)?;
            }
        }
    }

    Ok(())
}

/// Encode one tile (micro-block) for a single depth slice.
///
/// Where [`tries_diff_encoding`] allows it and this is not the first depth
/// slice, the block is also coded as differences from the previous slice's
/// source values (exact, since the data is lossless integer), and the smaller
/// of the two codings is kept.
fn encode_tile<T: Sample>(
    blob: &mut Vec<u8>,
    ctx: &TileEncodeContext<'_, T>,
    rect: crate::types::TileRect,
    i_depth: usize,
    scratch: &mut ScratchBuffers,
) -> Result<()> {
    let crate::types::TileRect { i0, i1, j0, j1 } = rect;
    let data = ctx.data;
    let mask = ctx.mask;
    let header = ctx.header;
    let n_cols = header.n_cols as usize;
    let n_depth = header.n_depth as usize;
    let max_z_error = header.max_z_error;
    let max_val_to_quantize: f64 = match header.data_type {
        DataType::Char | DataType::Byte | DataType::Short | DataType::UShort => {
            ((1u32 << 15) - 1) as f64
        }
        _ => ((1u32 << 30) - 1) as f64,
    };

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

    // Differences from the previous slice, when diff coding is tried. The
    // data is lossless integer, so the previous slice decodes to exactly its
    // source values; a difference that does not fit in i32 rules it out.
    let has_diff = if ctx.try_diff && i_depth > 0 {
        scratch.diff_data.clear();
        let mut overflow = false;
        'outer: for i in i0..i1 {
            for j in j0..j1 {
                let k = i * n_cols + j;
                if mask.is_valid(k) {
                    let diff = data[k * n_depth + i_depth].to_f64()
                        - data[k * n_depth + i_depth - 1].to_f64();
                    if diff < i32::MIN as f64 || diff > i32::MAX as f64 {
                        overflow = true;
                        break 'outer;
                    }
                    scratch.diff_data.push(diff);
                }
            }
        }
        !overflow
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

    if !has_diff {
        blob.extend_from_slice(&scratch.tile_buf);
        return Ok(());
    }

    // Encode with diff, appended after the non-diff coding, and keep the
    // smaller of the two.
    let (diff_min, diff_max) = scratch
        .diff_data
        .iter()
        .fold((f64::MAX, f64::MIN), |(lo, hi), &d| (lo.min(d), hi.max(d)));
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
    let diff_len = scratch.tile_buf.len() - non_diff_len;
    if diff_len < non_diff_len {
        blob.extend_from_slice(&scratch.tile_buf[non_diff_len..]);
    } else {
        blob.extend_from_slice(&scratch.tile_buf[..non_diff_len]);
    }

    Ok(())
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
        let mask = BitMask::all_valid(data.len());
        encode_tiles_at_best_block_size(data, &mask, hd)
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
}
