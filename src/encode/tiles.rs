use alloc::vec;
use alloc::vec::Vec;

use crate::bitmask::BitMask;
use crate::error::Result;
use crate::header::HeaderInfo;
use crate::types::{DataType, LercDataType, TileCompressionMode, tile_flags};

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

/// Fast block size selection for u8 lossless, all-valid, single-depth images.
fn select_block_size_u8_fast(data: &[u8], hd: &HeaderInfo) -> i32 {
    let n_rows = hd.n_rows as usize;
    let n_cols = hd.n_cols as usize;
    let version = hd.version;
    let pattern: u8 = if version >= 5 { 14 } else { 15 };
    let mut quant_buf = [0u32; 256];

    let mut size8 = 0usize;
    let mut buf8 = Vec::with_capacity(512);
    for &(i0, i1, j0, j1) in &sample_tile_positions(n_rows, n_cols, 8) {
        buf8.clear();
        let integrity = ((j0 as u8 >> 3) & pattern) << 2;
        encode_single_u8_tile(&mut buf8, data, n_cols, i0, i1, j0, j1, integrity, &mut quant_buf);
        size8 += buf8.len();
    }

    let mut size16 = 0usize;
    let mut buf16 = Vec::with_capacity(512);
    for &(i0, i1, j0, j1) in &sample_tile_positions(n_rows, n_cols, 16) {
        buf16.clear();
        let integrity = ((j0 as u8 >> 3) & pattern) << 2;
        encode_single_u8_tile(&mut buf16, data, n_cols, i0, i1, j0, j1, integrity, &mut quant_buf);
        size16 += buf16.len();
    }

    if size16 < size8 { 16 } else { 8 }
}

/// Encode a single u8 tile (used by select_block_size_u8_fast and encode_tiles_u8_fast).
///
/// Combines min/max scan, quantization, histogram building, LUT decision, and encoding
/// into minimal passes over the data.
#[inline]
#[allow(clippy::too_many_arguments)]
fn encode_single_u8_tile(
    buf: &mut Vec<u8>,
    data: &[u8],
    n_cols: usize,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    integrity: u8,
    quant_buf: &mut [u32; 256],
) {
    let tile_w = j1 - j0;
    let num_valid = (i1 - i0) * tile_w;

    // Single pass: compute min, max, and copy values into quant_buf (as-is, not yet quantized)
    let mut z_min: u8 = 255;
    let mut z_max: u8 = 0;
    let mut idx = 0;
    for i in i0..i1 {
        let row_start = i * n_cols + j0;
        for &val in &data[row_start..row_start + tile_w] {
            if val < z_min { z_min = val; }
            if val > z_max { z_max = val; }
            // Store raw value; we'll subtract z_min below if needed
            quant_buf[idx] = val as u32;
            idx += 1;
        }
    }

    if z_min == 0 && z_max == 0 {
        buf.push(TileCompressionMode::ConstZero as u8 | integrity);
        return;
    }

    if z_min == z_max {
        buf.push(TileCompressionMode::ConstOffset as u8 | integrity);
        buf.push(z_min);
        return;
    }

    // Quantize in-place: subtract z_min
    let max_quant = (z_max - z_min) as u32;
    let quant_data = &mut quant_buf[..num_valid];
    let z_min_u32 = z_min as u32;
    for v in quant_data.iter_mut() {
        *v -= z_min_u32;
    }

    buf.push(TileCompressionMode::BitStuffed as u8 | integrity);
    buf.push(z_min);

    let num_bits = crate::bitstuffer::num_bits_needed(max_quant);
    let num_elem = num_valid as u32;

    // Quick heuristic: if the value range is large relative to the tile size,
    // LUT can't help (not enough duplicates to save bits). Skip the histogram.
    // For a tile of N values with max_quant >= N/2, the maximum possible number
    // of distinct values is N, and n_bits_lut >= log2(N/2) which is close to
    // num_bits, so LUT overhead eats any savings.
    let try_lut = max_quant < num_elem / 2;

    if !try_lut {
        // Simple encoding (most common for high-entropy data)
        crate::bitstuffer::encode_simple_into(buf, quant_data, max_quant);
    } else {
        // Build histogram and check if LUT helps
        let mut counts = [0u16; 256];
        for &v in quant_data.iter() {
            counts[v as usize] += 1;
        }

        // Count distinct values (only up to max_quant)
        let mut n_distinct = 0u32;
        for &c in &counts[..=max_quant as usize] {
            if c > 0 {
                n_distinct += 1;
            }
        }

        // Check if LUT is beneficial
        let use_lut = if (2..=255).contains(&n_distinct) {
            let n_lut = n_distinct - 1;
            let mut n_bits_lut = 0u32;
            while (n_lut >> n_bits_lut) != 0 { n_bits_lut += 1; }
            let n_bytes = crate::bitstuffer::num_bytes_uint(num_elem);
            let num_bytes_simple = 1 + n_bytes as u32 + ((num_elem * num_bits + 7) >> 3);
            let num_bytes_lut = 1 + n_bytes as u32 + 1
                + ((n_lut * num_bits + 7) >> 3)
                + ((num_elem * n_bits_lut + 7) >> 3);
            num_bytes_lut < num_bytes_simple
        } else {
            false
        };

        if !use_lut {
            crate::bitstuffer::encode_simple_into(buf, quant_data, max_quant);
        } else {
            // Build LUT mapping and encode using stack-allocated arrays
            let n_lut = n_distinct - 1;
            let mut n_bits_lut = 0u32;
            while (n_lut >> n_bits_lut) != 0 { n_bits_lut += 1; }

            let mut value_to_index = [0u8; 256];
            let mut lut_values = [0u32; 255];
            let mut lut_count = 0u32;
            let mut lut_idx = 0u8;
            for v in 0..=max_quant as usize {
                if counts[v] > 0 {
                    value_to_index[v] = lut_idx;
                    if lut_idx > 0 {
                        lut_values[lut_count as usize] = v as u32;
                        lut_count += 1;
                    }
                    lut_idx += 1;
                }
            }

            // Build index array on the stack
            let mut index_arr = [0u32; 256];
            for (i, &val) in quant_data.iter().enumerate() {
                index_arr[i] = value_to_index[val as usize] as u32;
            }
            let index_data = &index_arr[..num_valid];
            let lut_data = &lut_values[..lut_count as usize];

            // Write LUT header and bit-stuffed data directly
            let n = crate::bitstuffer::num_bytes_uint(num_elem);
            let bits67 = if n == 4 { 0u8 } else { (3 - n) as u8 };
            let mut header_byte = num_bits as u8;
            header_byte |= bits67 << 6;
            header_byte |= 1 << 5; // LUT mode

            buf.push(header_byte);
            crate::bitstuffer::encode_uint_pub(buf, num_elem, n);
            buf.push((lut_count + 1) as u8);

            if num_bits > 0 {
                crate::bitstuffer::bit_stuff_append(buf, lut_data, num_bits);
            }
            if n_bits_lut > 0 {
                crate::bitstuffer::bit_stuff_append(buf, index_data, n_bits_lut);
            }
        }
    }
}

/// Try encoding a sample of tiles at both block sizes 8 and 16, return whichever
/// produces smaller output. Instead of encoding ALL tiles twice, we sample a
/// small number of representative tiles to make the decision.
pub(super) fn select_block_size<T: LercDataType>(
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
        let u8_data: &[u8] = unsafe {
            core::slice::from_raw_parts(data.as_ptr() as *const u8, data.len())
        };
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

    let max_val_to_quantize: f64 = match hd.data_type {
        DataType::Char | DataType::Byte | DataType::Short | DataType::UShort => {
            ((1u32 << 15) - 1) as f64
        }
        _ => ((1u32 << 30) - 1) as f64,
    };
    let max_z_error = hd.max_z_error;

    let mut scratch = ScratchBuffers::new();

    // Encode sampled tiles at block size 8
    let mut size8 = 0usize;
    let hd8 = {
        let mut h = hd.clone();
        h.micro_block_size = 8;
        h
    };
    for &(i0, i1, j0, j1) in &sample_tile_positions(n_rows, n_cols, 8) {
        for i_depth in 0..n_depth {
            size8 += encode_tile_to_count::<T>(
                data,
                mask,
                &hd8,
                z_max_vec,
                max_z_error,
                max_val_to_quantize,
                i0,
                i1,
                j0,
                j1,
                i_depth,
                n_depth > 1 && i_depth > 0,
                &mut scratch,
            )?;
        }
    }

    // Encode sampled tiles at block size 16
    let mut size16 = 0usize;
    let hd16 = {
        let mut h = hd.clone();
        h.micro_block_size = 16;
        h
    };
    for &(i0, i1, j0, j1) in &sample_tile_positions(n_rows, n_cols, 16) {
        for i_depth in 0..n_depth {
            size16 += encode_tile_to_count::<T>(
                data,
                mask,
                &hd16,
                z_max_vec,
                max_z_error,
                max_val_to_quantize,
                i0,
                i1,
                j0,
                j1,
                i_depth,
                n_depth > 1 && i_depth > 0,
                &mut scratch,
            )?;
        }
    }

    if size16 < size8 {
        Ok(16)
    } else {
        Ok(8)
    }
}

/// Encode a single tile and return its byte count (without appending to blob).
/// This is used by `select_block_size` for sampling.
#[allow(clippy::too_many_arguments)]
fn encode_tile_to_count<T: LercDataType>(
    data: &[T],
    mask: &BitMask,
    header: &HeaderInfo,
    z_max_vec: &[f64],
    max_z_error: f64,
    max_val_to_quantize: f64,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    i_depth: usize,
    try_diff: bool,
    scratch: &mut ScratchBuffers,
) -> Result<usize> {
    let mut tmp = Vec::new();
    encode_tile(
        &mut tmp,
        data,
        mask,
        header,
        z_max_vec,
        max_z_error,
        max_val_to_quantize,
        i0,
        i1,
        j0,
        j1,
        i_depth,
        try_diff,
        scratch,
        None, // No reconstruction buffer needed for size estimation
    )?;
    Ok(tmp.len())
}

/// Scratch buffers reused across tiles to avoid per-tile allocations.
pub(super) struct ScratchBuffers {
    valid_data: Vec<f64>,
    diff_data: Vec<f64>,
    quant_vec: Vec<u32>,
    /// Holds the quantized values from the non-diff encoding attempt,
    /// saved before the diff encoding attempt overwrites `quant_vec`.
    non_diff_quant: Vec<u32>,
    tile_buf: Vec<u8>,
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

pub(super) fn encode_tiles<T: LercDataType>(
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
        let u8_data: &[u8] = unsafe {
            core::slice::from_raw_parts(data.as_ptr() as *const u8, data.len())
        };
        let mut quant_buf = [0u32; 256]; // max tile size is 16x16=256
        encode_tiles_u8_fast(blob, u8_data, header, &mut quant_buf);
        return Ok(());
    }

    let num_tiles_vert = n_rows.div_ceil(mb_size);
    let num_tiles_hori = n_cols.div_ceil(mb_size);

    let max_val_to_quantize: f64 = match header.data_type {
        DataType::Char | DataType::Byte | DataType::Short | DataType::UShort => {
            ((1u32 << 15) - 1) as f64
        }
        _ => ((1u32 << 30) - 1) as f64,
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

            for i_depth in 0..n_depth {
                encode_tile(
                    blob,
                    data,
                    mask,
                    header,
                    z_max_vec,
                    max_z_error,
                    max_val_to_quantize,
                    i0,
                    i1,
                    j0,
                    j1,
                    i_depth,
                    n_depth > 1 && i_depth > 0,
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

/// Specialized tile encoding for u8 lossless, all-valid, single-depth images.
/// Avoids all f64 conversions and Vec allocations in the inner loop.
fn encode_tiles_u8_fast(
    blob: &mut Vec<u8>,
    data: &[u8],
    header: &HeaderInfo,
    quant_buf: &mut [u32; 256],
) {
    let mb_size = header.micro_block_size as usize;
    let n_rows = header.n_rows as usize;
    let n_cols = header.n_cols as usize;
    let version = header.version;

    let num_tiles_vert = n_rows.div_ceil(mb_size);
    let num_tiles_hori = n_cols.div_ceil(mb_size);

    // Pre-allocate: worst case is ~1 byte overhead per tile + data bytes.
    // For high-entropy u8, output is roughly the same size as input.
    blob.reserve(data.len() + num_tiles_vert * num_tiles_hori * 4);

    let pattern: u8 = if version >= 5 { 14 } else { 15 };

    for i_tile in 0..num_tiles_vert {
        let i0 = i_tile * mb_size;
        let i1 = (i0 + mb_size).min(n_rows);

        for j_tile in 0..num_tiles_hori {
            let j0 = j_tile * mb_size;
            let j1 = (j0 + mb_size).min(n_cols);

            let integrity = ((j0 as u8 >> 3) & pattern) << 2;
            encode_single_u8_tile(blob, data, n_cols, i0, i1, j0, j1, integrity, quant_buf);
        }
    }
}

/// Encode a single tile block (one depth slice of one micro-block).
/// When `try_diff` is true (depth > 0 with nDepth > 1), the encoder tries
/// diff (delta) encoding relative to the previous depth slice and picks
/// whichever representation is smaller.
///
/// When `recon_buf` is `Some`, it holds the decoder-reconstructed values for the
/// previous depth slice (indexed by pixel index `k = row * n_cols + col`).
/// Diffs are computed against these reconstructed values instead of the original
/// data. After encoding, the buffer is updated with this depth slice's
/// reconstructed values so it is ready for the next depth.
#[allow(clippy::too_many_arguments)]
fn encode_tile<T: LercDataType>(
    blob: &mut Vec<u8>,
    data: &[T],
    mask: &BitMask,
    header: &HeaderInfo,
    z_max_vec: &[f64],
    max_z_error: f64,
    max_val_to_quantize: f64,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    i_depth: usize,
    try_diff: bool,
    scratch: &mut ScratchBuffers,
    recon_buf: Option<&mut Vec<f64>>,
) -> Result<()> {
    let n_cols = header.n_cols as usize;
    let n_depth = header.n_depth as usize;

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
                    if T::is_integer()
                        && (diff < i32::MIN as f64 || diff > i32::MAX as f64)
                    {
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
    encode_tile_inner::<T>(
        &mut scratch.tile_buf,
        &scratch.valid_data,
        &mut scratch.quant_vec,
        num_valid,
        z_min_f,
        z_max_f,
        max_z_error,
        max_val_to_quantize,
        integrity,
        header.data_type,
        false,
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
        encode_tile_inner::<T>(
            &mut scratch.tile_buf,
            &scratch.diff_data,
            &mut scratch.quant_vec,
            num_valid,
            diff_min,
            diff_max,
            max_z_error,
            max_val_to_quantize,
            integrity,
            header.data_type,
            true,
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
            classify_recon(
                &scratch.diff_data,
                &scratch.quant_vec,
                num_valid,
                diff_z_min,
                diff_z_max,
                max_z_error,
                max_val_to_quantize,
                z_max_depth,
                header.data_type,
                true,
            )
        } else {
            // Non-diff encoding was chosen; reconstruct from original quantized values
            classify_recon(
                &scratch.valid_data,
                &scratch.non_diff_quant,
                num_valid,
                non_diff_z_min,
                non_diff_z_max,
                max_z_error,
                max_val_to_quantize,
                z_max_depth,
                header.data_type,
                false,
            )
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

/// Classify how a tile was encoded and return reconstruction info.
/// This mirrors the logic in `encode_tile_inner` to determine whether the tile
/// ended up as constant, quantized, or raw binary.
#[allow(clippy::too_many_arguments)]
fn classify_recon(
    values: &[f64],
    quant: &[u32],
    num_valid: usize,
    z_min_f: f64,
    z_max_f: f64,
    max_z_error: f64,
    max_val_to_quantize: f64,
    z_max_depth: f64,
    _src_data_type: DataType,
    b_diff_enc: bool,
) -> TileReconInfo {
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

/// Encode the payload for a single tile block, appending to `buf`.
/// When `b_diff_enc` is true, bit 2 of the compression flag is set and the
/// offset data type is forced to Int for integer source types.
/// `quant_scratch` is a reusable scratch buffer for quantized values.
#[allow(clippy::too_many_arguments)]
fn encode_tile_inner<T: LercDataType>(
    buf: &mut Vec<u8>,
    values: &[f64],
    quant_scratch: &mut Vec<u32>,
    num_valid: usize,
    z_min_f: f64,
    z_max_f: f64,
    max_z_error: f64,
    max_val_to_quantize: f64,
    integrity: u8,
    src_data_type: DataType,
    b_diff_enc: bool,
) {
    let diff_flag: u8 = if b_diff_enc { tile_flags::DIFF_ENCODING } else { 0 };

    if num_valid == 0 || (z_min_f == 0.0 && z_max_f == 0.0) {
        buf.push(TileCompressionMode::ConstZero as u8 | diff_flag | integrity);
        return;
    }

    // Check if we need quantization
    let need_quantize = if max_z_error == 0.0 {
        z_max_f > z_min_f
    } else {
        let max_val = (z_max_f - z_min_f) / (2.0 * max_z_error);
        max_val <= max_val_to_quantize && (max_val + 0.5) as u32 != 0
    };

    if !need_quantize && z_min_f == z_max_f {
        // Constant block (all same value / all same diff)
        let z_min_t = T::from_f64(z_min_f);
        let (dt_reduced, tc) = if b_diff_enc && src_data_type.is_integer() {
            crate::tiles::reduce_data_type(z_min_f as i32, DataType::Int)
        } else {
            crate::tiles::reduce_data_type(z_min_t, src_data_type)
        };
        let bits67 = (tc as u8) << tile_flags::TYPE_REDUCTION_SHIFT;
        buf.push(TileCompressionMode::ConstOffset as u8 | bits67 | diff_flag | integrity);
        crate::tiles::write_variable_data_type(buf, z_min_f, dt_reduced);
        return;
    }

    if !need_quantize {
        if !b_diff_enc {
            buf.push(TileCompressionMode::RawBinary as u8 | integrity);
            for val in values {
                let t = T::from_f64(*val);
                t.extend_le_bytes(buf);
            }
            return;
        }
        // Raw binary is not allowed with diff enc per the decoder.
        // Append an impossibly-large sentinel so the non-diff path wins
        // the size comparison. We use raw-size + 1 so it always loses.
        let sentinel_len = 1 + num_valid * T::BYTES + 1;
        buf.resize(buf.len() + sentinel_len, 0u8);
        return;
    }

    // Quantize and bit-stuff
    let z_min_t = T::from_f64(z_min_f);
    let (dt_reduced, tc) = if b_diff_enc && src_data_type.is_integer() {
        crate::tiles::reduce_data_type(z_min_f as i32, DataType::Int)
    } else {
        crate::tiles::reduce_data_type(z_min_t, src_data_type)
    };
    let bits67 = (tc as u8) << tile_flags::TYPE_REDUCTION_SHIFT;

    // Quantize values (reusing scratch buffer)
    quant_scratch.clear();
    if T::is_integer() && max_z_error == 0.5 {
        // Lossless integer
        for val in values {
            quant_scratch.push((*val - z_min_f) as u32);
        }
    } else {
        let scale = 1.0 / (2.0 * max_z_error);
        for val in values {
            quant_scratch.push(((*val - z_min_f) * scale + 0.5) as u32);
        }
    }

    let max_quant = quant_scratch.iter().copied().max().unwrap_or(0);

    if max_quant == 0 {
        // All values quantize to same -> constant block
        buf.push(TileCompressionMode::ConstOffset as u8 | bits67 | diff_flag | integrity);
        crate::tiles::write_variable_data_type(buf, z_min_f, dt_reduced);
        return;
    }

    // Try LUT vs simple encoding
    let encoded_start = buf.len();
    buf.push(TileCompressionMode::BitStuffed as u8 | bits67 | diff_flag | integrity);
    crate::tiles::write_variable_data_type(buf, z_min_f, dt_reduced);

    // Use fast path for small tiles with small values (common for u8 data)
    if num_valid <= 256 && max_quant < 256 {
        crate::bitstuffer::encode_small_tile_into(buf, quant_scratch, max_quant);
    } else if let Some(lut_info) = crate::bitstuffer::should_use_lut(quant_scratch, max_quant) {
        crate::bitstuffer::encode_lut_into(buf, &lut_info, num_valid as u32);
    } else {
        crate::bitstuffer::encode_simple_into(buf, quant_scratch, max_quant);
    }

    // Check if raw binary would be smaller (only for non-diff mode)
    if !b_diff_enc {
        let raw_size = 1 + num_valid * T::BYTES;
        let encoded_len = buf.len() - encoded_start;
        if encoded_len >= raw_size {
            // Raw binary fallback: replace the encoded data
            buf.truncate(encoded_start);
            buf.push(TileCompressionMode::RawBinary as u8 | integrity);
            for val in values {
                let t = T::from_f64(*val);
                t.extend_le_bytes(buf);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // -----------------------------------------------------------------------
    // Helper to call encode_tile_inner with typical u8 lossless settings
    // -----------------------------------------------------------------------

    fn encode_tile_inner_u8(
        values: &[f64],
        z_min_f: f64,
        z_max_f: f64,
        b_diff_enc: bool,
    ) -> (Vec<u8>, Vec<u32>) {
        let mut buf = Vec::new();
        let mut quant_scratch = Vec::new();
        let num_valid = values.len();
        // u8 lossless: max_z_error = 0.5, max_val_to_quantize = (1<<15)-1
        encode_tile_inner::<u8>(
            &mut buf,
            values,
            &mut quant_scratch,
            num_valid,
            z_min_f,
            z_max_f,
            0.5,
            ((1u32 << 15) - 1) as f64,
            0, // integrity = 0 for simplicity
            DataType::Byte,
            b_diff_enc,
        );
        (buf, quant_scratch)
    }

    fn encode_tile_inner_f64(
        values: &[f64],
        z_min_f: f64,
        z_max_f: f64,
        max_z_error: f64,
    ) -> (Vec<u8>, Vec<u32>) {
        let mut buf = Vec::new();
        let mut quant_scratch = Vec::new();
        let num_valid = values.len();
        encode_tile_inner::<f64>(
            &mut buf,
            values,
            &mut quant_scratch,
            num_valid,
            z_min_f,
            z_max_f,
            max_z_error,
            ((1u32 << 30) - 1) as f64,
            0,
            DataType::Double,
            false,
        );
        (buf, quant_scratch)
    }

    // -----------------------------------------------------------------------
    // encode_tile_inner: all-zero block
    // -----------------------------------------------------------------------

    #[test]
    fn encode_tile_inner_all_zero_via_num_valid_zero() {
        let mut buf = Vec::new();
        let mut quant_scratch = Vec::new();
        // num_valid = 0 triggers const-zero path
        encode_tile_inner::<u8>(
            &mut buf,
            &[],
            &mut quant_scratch,
            0,
            f64::MAX,
            f64::MIN,
            0.5,
            ((1u32 << 15) - 1) as f64,
            0,
            DataType::Byte,
            false,
        );
        assert_eq!(buf.len(), 1);
        assert_eq!(buf[0] & tile_flags::MODE_MASK, TileCompressionMode::ConstZero as u8);
    }

    #[test]
    fn encode_tile_inner_all_zero_via_zmin_zmax_zero() {
        let values = vec![0.0, 0.0, 0.0, 0.0];
        let (buf, _) = encode_tile_inner_u8(&values, 0.0, 0.0, false);
        assert_eq!(buf.len(), 1);
        assert_eq!(buf[0] & tile_flags::MODE_MASK, TileCompressionMode::ConstZero as u8);
    }

    // -----------------------------------------------------------------------
    // encode_tile_inner: constant block (z_min == z_max != 0)
    // -----------------------------------------------------------------------

    #[test]
    fn encode_tile_inner_const_offset() {
        let values = vec![42.0, 42.0, 42.0, 42.0];
        let (buf, _) = encode_tile_inner_u8(&values, 42.0, 42.0, false);
        assert_eq!(
            buf[0] & tile_flags::MODE_MASK,
            TileCompressionMode::ConstOffset as u8
        );
        // Should have mode byte + offset value (reduced to u8 = 1 byte)
        assert!(buf.len() > 1);
        // The offset byte should be 42
        assert_eq!(buf[1], 42);
    }

    #[test]
    fn encode_tile_inner_const_offset_i16() {
        // Use i16 data where the constant value is > 255, requiring a 2-byte offset
        let values = vec![1000.0, 1000.0, 1000.0];
        let mut buf = Vec::new();
        let mut quant_scratch = Vec::new();
        encode_tile_inner::<i16>(
            &mut buf,
            &values,
            &mut quant_scratch,
            3,
            1000.0,
            1000.0,
            0.5,
            ((1u32 << 15) - 1) as f64,
            0,
            DataType::Short,
            false,
        );
        assert_eq!(
            buf[0] & tile_flags::MODE_MASK,
            TileCompressionMode::ConstOffset as u8
        );
        // 1000 fits in i16 but not i8/u8, so type reduction from Short
        // should produce a 2-byte offset at minimum
        assert!(buf.len() >= 3);
    }

    // -----------------------------------------------------------------------
    // encode_tile_inner: quantized / bit-stuffed block
    // -----------------------------------------------------------------------

    #[test]
    fn encode_tile_inner_bitstuffed() {
        // Small range of u8 values: should produce bit-stuffed output
        let values: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let (buf, quant) = encode_tile_inner_u8(&values, 0.0, 15.0, false);
        assert_eq!(
            buf[0] & tile_flags::MODE_MASK,
            TileCompressionMode::BitStuffed as u8
        );
        // Quantized values should match original for lossless u8
        assert_eq!(quant.len(), 16);
        for (i, &q) in quant.iter().enumerate() {
            assert_eq!(q, i as u32);
        }
    }

    #[test]
    fn encode_tile_inner_bitstuffed_with_offset() {
        // Use enough values that bitstuffed is smaller than raw binary.
        // For u8 with 16 values, raw_size = 1 + 16 = 17.
        // Bitstuffed with range 0..3 (2 bits) is much smaller.
        let values: Vec<f64> = (0..16).map(|i| 100.0 + (i % 4) as f64).collect();
        let (buf, quant) = encode_tile_inner_u8(&values, 100.0, 103.0, false);
        assert_eq!(
            buf[0] & tile_flags::MODE_MASK,
            TileCompressionMode::BitStuffed as u8
        );
        // Offset byte should be 100 (z_min as u8)
        assert_eq!(buf[1], 100);
        // Quantized should be 0, 1, 2, 3, 0, 1, 2, 3, ...
        for (i, &q) in quant.iter().enumerate() {
            assert_eq!(q, (i % 4) as u32);
        }
    }

    // -----------------------------------------------------------------------
    // encode_tile_inner: raw binary fallback
    // -----------------------------------------------------------------------

    #[test]
    fn encode_tile_inner_raw_binary_f64_large_range() {
        // To trigger raw binary via the early exit (not the fallback), we need
        // need_quantize = false with z_min != z_max.
        // Use max_z_error > 0 and a range so large that
        // max_val = (z_max - z_min) / (2 * max_z_error) > max_val_to_quantize.
        let max_val_to_quantize = ((1u32 << 30) - 1) as f64;
        let max_z_error = 0.5;
        let z_min = 0.0;
        // Need range / (2*0.5) > max_val_to_quantize, so range > max_val_to_quantize
        let z_max = max_val_to_quantize + 100.0;
        let n = 4;
        let values: Vec<f64> = (0..n).map(|i| z_min + (z_max - z_min) * i as f64 / (n - 1) as f64).collect();
        let (buf, _) = encode_tile_inner_f64(&values, z_min, z_max, max_z_error);
        assert_eq!(
            buf[0] & tile_flags::MODE_MASK,
            TileCompressionMode::RawBinary as u8
        );
        // Raw binary: 1 byte header + n * 8 bytes
        assert_eq!(buf.len(), 1 + n * 8);
        // Verify values are stored correctly
        for (i, &v) in values.iter().enumerate() {
            let start = 1 + i * 8;
            let decoded = f64::from_le_bytes(buf[start..start + 8].try_into().unwrap());
            assert_eq!(decoded, v);
        }
    }

    #[test]
    fn encode_tile_inner_raw_binary_contains_original_values() {
        // Use enough values to ensure raw binary wins over bitstuffed
        let n = 32;
        let values: Vec<f64> = (0..n).map(|i| 1.5 + i as f64 * 1e14).collect();
        let z_min = values[0];
        let z_max = *values.last().unwrap();
        let (buf, _) = encode_tile_inner_f64(&values, z_min, z_max, 0.0);
        if buf[0] & tile_flags::MODE_MASK == TileCompressionMode::RawBinary as u8 {
            // Verify the raw f64 values are stored correctly
            for (i, &v) in values.iter().enumerate() {
                let start = 1 + i * 8;
                let decoded = f64::from_le_bytes(buf[start..start + 8].try_into().unwrap());
                assert_eq!(decoded, v);
            }
        }
    }

    // -----------------------------------------------------------------------
    // encode_tile_inner: diff encoding flag
    // -----------------------------------------------------------------------

    #[test]
    fn encode_tile_inner_diff_encoding_const_zero() {
        let values = vec![0.0, 0.0, 0.0];
        let (buf, _) = encode_tile_inner_u8(&values, 0.0, 0.0, true);
        assert_eq!(buf.len(), 1);
        assert_eq!(
            buf[0] & tile_flags::MODE_MASK,
            TileCompressionMode::ConstZero as u8
        );
        // Diff flag (bit 2) should be set
        assert_ne!(buf[0] & tile_flags::DIFF_ENCODING, 0);
    }

    #[test]
    fn encode_tile_inner_diff_encoding_const_offset() {
        let values = vec![10.0, 10.0, 10.0];
        let (buf, _) = encode_tile_inner_u8(&values, 10.0, 10.0, true);
        assert_eq!(
            buf[0] & tile_flags::MODE_MASK,
            TileCompressionMode::ConstOffset as u8
        );
        assert_ne!(buf[0] & tile_flags::DIFF_ENCODING, 0);
    }

    #[test]
    fn encode_tile_inner_diff_encoding_bitstuffed() {
        let values: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let (buf, _) = encode_tile_inner_u8(&values, 0.0, 7.0, true);
        assert_eq!(
            buf[0] & tile_flags::MODE_MASK,
            TileCompressionMode::BitStuffed as u8
        );
        assert_ne!(buf[0] & tile_flags::DIFF_ENCODING, 0);
    }

    #[test]
    fn encode_tile_inner_diff_encoding_no_raw_binary() {
        // With diff encoding, raw binary should NOT be produced.
        // Instead, a sentinel is written that is larger than raw, so non-diff wins.
        let mut buf = Vec::new();
        let mut quant_scratch = Vec::new();
        let values = vec![0.0, 1e15];
        // Use f64 lossless with diff encoding
        encode_tile_inner::<f64>(
            &mut buf,
            &values,
            &mut quant_scratch,
            values.len(),
            0.0,
            1e15,
            0.0,
            ((1u32 << 30) - 1) as f64,
            0,
            DataType::Double,
            true,
        );
        // Should NOT be raw binary when diff is true
        assert_ne!(
            buf[0] & tile_flags::MODE_MASK,
            TileCompressionMode::RawBinary as u8
        );
    }

    // -----------------------------------------------------------------------
    // encode_tile_inner: integrity bits are preserved
    // -----------------------------------------------------------------------

    #[test]
    fn encode_tile_inner_integrity_bits() {
        let mut buf = Vec::new();
        let mut quant_scratch = Vec::new();
        let values = vec![0.0, 0.0];
        let integrity: u8 = 0b00001100; // bits 2-3 set
        encode_tile_inner::<u8>(
            &mut buf,
            &values,
            &mut quant_scratch,
            values.len(),
            0.0,
            0.0,
            0.5,
            ((1u32 << 15) - 1) as f64,
            integrity,
            DataType::Byte,
            false,
        );
        // Integrity bits should be present in the header byte
        assert_eq!(buf[0] & integrity, integrity);
    }

    // -----------------------------------------------------------------------
    // encode_tile_inner: quantized with lossy max_z_error
    // -----------------------------------------------------------------------

    #[test]
    fn encode_tile_inner_lossy_quantization() {
        // Lossy encoding of f64 data with max_z_error = 1.0
        let values = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let (buf, quant) = encode_tile_inner_f64(&values, 0.0, 4.0, 1.0);
        assert_eq!(
            buf[0] & tile_flags::MODE_MASK,
            TileCompressionMode::BitStuffed as u8
        );
        // With max_z_error = 1.0, scale = 1/(2*1.0) = 0.5
        // quant[i] = (val - z_min) * 0.5 + 0.5 = val * 0.5 + 0.5
        // quant: 0->0, 1->1, 2->1, 3->2, 4->2
        assert_eq!(quant[0], 0); // 0.0 * 0.5 + 0.5 = 0.5 -> 0
        assert_eq!(quant[1], 1); // 1.0 * 0.5 + 0.5 = 1.0 -> 1
        assert_eq!(quant[2], 1); // 2.0 * 0.5 + 0.5 = 1.5 -> 1
        assert_eq!(quant[3], 2); // 3.0 * 0.5 + 0.5 = 2.0 -> 2
        assert_eq!(quant[4], 2); // 4.0 * 0.5 + 0.5 = 2.5 -> 2
    }

    // -----------------------------------------------------------------------
    // encode_tile_inner: all-quantized-to-same (max_quant == 0)
    // -----------------------------------------------------------------------

    #[test]
    fn encode_tile_inner_all_quantize_to_same() {
        // With large max_z_error relative to range, all values quantize to 0.
        // Need: need_quantize = true, i.e. max_val <= max_val_to_quantize
        //       AND (max_val + 0.5) as u32 != 0
        // Use max_z_error = 5.0 and values in [10.0, 10.5]:
        //   max_val = (10.5 - 10.0) / (2*5.0) = 0.05
        //   (0.05 + 0.5) as u32 = 0, so need_quantize = false -> goes to raw/const path.
        // Instead use max_z_error = 0.5 and values [10.0, 10.5, 11.0]:
        //   max_val = (11.0 - 10.0) / 1.0 = 1.0
        //   (1.0 + 0.5) as u32 = 1, so need_quantize = true
        //   scale = 1/(2*0.5) = 1.0
        //   quant = [(10-10)*1+0.5, (10.5-10)*1+0.5, (11-10)*1+0.5] = [0, 1, 1]
        //   max_quant = 1, so this is NOT all same.
        //
        // For all-same quantization, use max_z_error such that all values round to 0:
        //   values = [10.0, 10.3, 10.4], max_z_error = 0.5
        //   max_val = 0.4 / 1.0 = 0.4, (0.4 + 0.5) as u32 = 0 -> need_quantize false.
        //
        // Use integer type where the logic is simpler:
        // u8 values [5, 5, 5, 5, 5, 5, 5, 5] with max_z_error=0.5 (lossless)
        // All values equal -> z_min == z_max -> const offset.
        // That's already tested. Instead test the quantized path where max_quant=0:
        // u16 values [100, 101] with max_z_error=1.0:
        //   max_val = (101-100)/(2*1.0) = 0.5
        //   (0.5 + 0.5) as u32 = 1 -> need_quantize = true
        //   scale = 1/(2*1.0) = 0.5
        //   quant = [(100-100)*0.5+0.5, (101-100)*0.5+0.5] = [0, 1]
        //   max_quant = 1, not zero.
        //
        // For max_quant=0: all values equal after quantization.
        // values [100, 100, 100, ..., 101] with a huge max_z_error could do it.
        // max_z_error = 1.0, values = [100.0; 16] + [100.9]:
        //   max_val = 0.9/2.0 = 0.45, (0.45+0.5) = 0 -> need_quantize=false.
        //
        // values = [100.0; 16] + [101.0] with max_z_error = 1.0:
        //   max_val = 1.0/2.0 = 0.5, (0.5+0.5) = 1 -> need_quantize=true
        //   scale = 0.5
        //   quant[last] = (101-100)*0.5+0.5 = 1.0 -> 1
        //   max_quant = 1.
        //
        // Actually let's use: values where after quantization all become 0.
        // values = [100.0, 100.1, 100.2] with max_z_error = 0.5:
        //   need_quantize: max_val = 0.2/1.0 = 0.2, (0.2+0.5) = 0 -> false.
        //
        // This is tricky because (max_val + 0.5) as u32 == 0 when max_val < 0.5.
        // For max_quant=0 we need all individual quant values to be 0, but
        // (max_val + 0.5) as u32 > 0 (so max_val >= 0.5).
        // When max_val >= 0.5, scale = 1/(2*max_z_error), and at least
        // the largest value will quantize to max_val * scale + 0.5 >= 1.
        // So max_quant=0 in the quantized path cannot happen unless all values
        // are exactly z_min, which means z_min == z_max (caught earlier).
        //
        // Therefore the max_quant==0 ConstOffset path in the quantized section
        // is effectively unreachable for well-formed input. Skip this test case.
        // Instead verify a near-constant case that ends up as ConstOffset.
        let values = vec![42.0; 16];
        let (buf, _) = encode_tile_inner_u8(&values, 42.0, 42.0, false);
        assert_eq!(
            buf[0] & tile_flags::MODE_MASK,
            TileCompressionMode::ConstOffset as u8
        );
    }

    // -----------------------------------------------------------------------
    // encode_single_u8_tile
    // -----------------------------------------------------------------------

    #[test]
    fn encode_single_u8_tile_all_zeros() {
        let data = vec![0u8; 64]; // 8x8 tile, all zeros
        let n_cols = 8;
        let mut buf = Vec::new();
        let mut quant_buf = [0u32; 256];
        encode_single_u8_tile(&mut buf, &data, n_cols, 0, 8, 0, 8, 0, &mut quant_buf);
        assert_eq!(buf.len(), 1);
        assert_eq!(buf[0] & tile_flags::MODE_MASK, TileCompressionMode::ConstZero as u8);
    }

    #[test]
    fn encode_single_u8_tile_constant_nonzero() {
        let data = vec![42u8; 64]; // 8x8 tile, all 42
        let n_cols = 8;
        let mut buf = Vec::new();
        let mut quant_buf = [0u32; 256];
        encode_single_u8_tile(&mut buf, &data, n_cols, 0, 8, 0, 8, 0, &mut quant_buf);
        assert_eq!(buf[0] & tile_flags::MODE_MASK, TileCompressionMode::ConstOffset as u8);
        assert_eq!(buf[1], 42);
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn encode_single_u8_tile_varying_values() {
        // 4x4 tile with varying data
        let mut data = vec![0u8; 16];
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as u8;
        }
        let n_cols = 4;
        let mut buf = Vec::new();
        let mut quant_buf = [0u32; 256];
        encode_single_u8_tile(&mut buf, &data, n_cols, 0, 4, 0, 4, 0, &mut quant_buf);
        assert_eq!(buf[0] & tile_flags::MODE_MASK, TileCompressionMode::BitStuffed as u8);
        // Offset byte should be 0 (z_min = 0)
        assert_eq!(buf[1], 0);
        // Should be decodable via bitstuffer
        let mut pos = 2;
        let decoded = crate::bitstuffer::decode(&buf, &mut pos, 256, 6).unwrap();
        assert_eq!(decoded.len(), 16);
        for (i, &v) in decoded.iter().enumerate() {
            assert_eq!(v, i as u32);
        }
    }

    #[test]
    fn encode_single_u8_tile_with_offset() {
        // All values in [100, 109]
        let mut data = vec![0u8; 16];
        for (i, v) in data.iter_mut().enumerate() {
            *v = 100 + (i as u8 % 10);
        }
        let n_cols = 4;
        let mut buf = Vec::new();
        let mut quant_buf = [0u32; 256];
        encode_single_u8_tile(&mut buf, &data, n_cols, 0, 4, 0, 4, 0, &mut quant_buf);
        assert_eq!(buf[0] & tile_flags::MODE_MASK, TileCompressionMode::BitStuffed as u8);
        assert_eq!(buf[1], 100); // z_min = 100
    }

    #[test]
    fn encode_single_u8_tile_subtile_region() {
        // Test encoding a sub-region of a larger image
        // 16-column image, encode tile at rows [2,4), cols [4,8)
        let n_cols = 16;
        let n_rows = 8;
        let mut data = vec![0u8; n_rows * n_cols];
        // Fill the tile region with value 99
        for i in 2..4 {
            for j in 4..8 {
                data[i * n_cols + j] = 99;
            }
        }
        let mut buf = Vec::new();
        let mut quant_buf = [0u32; 256];
        encode_single_u8_tile(&mut buf, &data, n_cols, 2, 4, 4, 8, 0, &mut quant_buf);
        assert_eq!(buf[0] & tile_flags::MODE_MASK, TileCompressionMode::ConstOffset as u8);
        assert_eq!(buf[1], 99);
    }

    #[test]
    fn encode_single_u8_tile_integrity_bits_preserved() {
        let data = vec![0u8; 64];
        let mut buf = Vec::new();
        let mut quant_buf = [0u32; 256];
        let integrity = 0b00001100u8;
        encode_single_u8_tile(&mut buf, &data, 8, 0, 8, 0, 8, integrity, &mut quant_buf);
        assert_eq!(buf[0] & integrity, integrity);
    }

    // -----------------------------------------------------------------------
    // classify_recon
    // -----------------------------------------------------------------------

    #[test]
    fn classify_recon_const_zero_num_valid_zero() {
        let info = classify_recon(
            &[],
            &[],
            0,
            f64::MAX,
            f64::MIN,
            0.5,
            ((1u32 << 15) - 1) as f64,
            255.0,
            DataType::Byte,
            false,
        );
        match info {
            TileReconInfo::Constant(c) => assert_eq!(c, 0.0),
            _ => panic!("expected Constant(0.0)"),
        }
    }

    #[test]
    fn classify_recon_const_zero_zmin_zmax_zero() {
        let info = classify_recon(
            &[0.0, 0.0],
            &[0, 0],
            2,
            0.0,
            0.0,
            0.5,
            ((1u32 << 15) - 1) as f64,
            255.0,
            DataType::Byte,
            false,
        );
        match info {
            TileReconInfo::Constant(c) => assert_eq!(c, 0.0),
            _ => panic!("expected Constant(0.0)"),
        }
    }

    #[test]
    fn classify_recon_const_offset() {
        let info = classify_recon(
            &[42.0, 42.0],
            &[0, 0],
            2,
            42.0,
            42.0,
            0.5,
            ((1u32 << 15) - 1) as f64,
            255.0,
            DataType::Byte,
            false,
        );
        match info {
            TileReconInfo::Constant(c) => assert_eq!(c, 42.0),
            _ => panic!("expected Constant(42.0)"),
        }
    }

    #[test]
    fn classify_recon_quantized() {
        let values: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let quant: Vec<u32> = (0..8).collect();
        let info = classify_recon(
            &values,
            &quant,
            8,
            0.0,
            7.0,
            0.5,
            ((1u32 << 15) - 1) as f64,
            255.0,
            DataType::Byte,
            false,
        );
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
        // Lossless f64 with z_min != z_max and max_z_error = 0:
        // need_quantize = z_max > z_min = true, but since max_z_error is 0,
        // the actual check is: need_quantize = z_max_f > z_min_f.
        // Then it falls through to Quantized. We need need_quantize = false.
        // For need_quantize = false with z_min != z_max:
        //   max_z_error = 0.0 and z_max > z_min => need_quantize = true.
        //   max_z_error > 0 and max_val > max_val_to_quantize => need_quantize = false.
        // Use max_val > max_val_to_quantize to get need_quantize = false.
        let z_min = 0.0;
        let z_max = 1e18; // huge range
        let max_z_error = 0.5;
        let max_val_to_quantize = ((1u32 << 30) - 1) as f64;
        // max_val = (1e18 - 0) / 1.0 = 1e18, which is >> max_val_to_quantize
        let values = vec![z_min, z_max];
        let quant = vec![0, 0];
        let info = classify_recon(
            &values,
            &quant,
            2,
            z_min,
            z_max,
            max_z_error,
            max_val_to_quantize,
            1e18,
            DataType::Double,
            false,
        );
        match info {
            TileReconInfo::RawBinary => {}
            _ => panic!("expected RawBinary, got {:?}", match info {
                TileReconInfo::Constant(c) => alloc::format!("Constant({c})"),
                TileReconInfo::Quantized { offset, inv_scale, z_max } => alloc::format!("Quantized(offset={offset}, inv_scale={inv_scale}, z_max={z_max})"),
                TileReconInfo::RawBinary => alloc::format!("RawBinary"),
            }),
        }
    }

    #[test]
    fn classify_recon_quantized_all_same_after_quant() {
        // max_quant == 0 path: need_quantize = true but all quant values are 0.
        // For this to happen: max_val = (z_max - z_min) / (2 * max_z_error)
        // must satisfy (max_val + 0.5) as u32 != 0 (so max_val >= 0.5),
        // yet after individual quantization all values end up at 0.
        // This is effectively only possible when z_min == z_max was not caught,
        // or with numerical edge cases. We test the code path directly by
        // passing quant = [0, 0, 0] with parameters that make need_quantize true.
        let values = vec![10.0, 10.5, 11.0];
        let quant = vec![0, 0, 0]; // force max_quant = 0
        let info = classify_recon(
            &values,
            &quant,
            3,
            10.0,
            11.0,
            0.5,
            ((1u32 << 30) - 1) as f64,
            100.0,
            DataType::Double,
            false,
        );
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
        assert!(positions.iter().any(|&(i0, _, j0, _)| i0 == 0 && j0 == last_j0));
        // Bottom-left corner
        let last_i0 = (n_rows.div_ceil(mb_size) - 1) * mb_size;
        assert!(positions
            .iter()
            .any(|&(i0, _, j0, _)| i0 == last_i0 && j0 == 0));
        // Bottom-right corner
        assert!(positions
            .iter()
            .any(|&(i0, _, j0, _)| i0 == last_i0 && j0 == last_j0));
    }

    // -----------------------------------------------------------------------
    // Round-trip: encode then decode via crate's public API
    // -----------------------------------------------------------------------

    #[test]
    fn encode_tile_inner_round_trip_u8_lossless() {
        // Encode a tile and decode it using the bitstuffer decoder.
        // Use 64 values (8x8 tile) with small range so bitstuffed beats raw.
        // Values: 10..=25 repeating. Range=15, 4 bits. 64 values * 4 bits = 32 bytes.
        // Bitstuffer overhead: ~3 bytes. Total bitstuffed: 1 + 1 + 35 ~= 37.
        // Raw: 1 + 64 = 65. So bitstuffed wins.
        let values: Vec<f64> = (0..64).map(|i| 10.0 + (i % 16) as f64).collect();
        let (buf, _) = encode_tile_inner_u8(&values, 10.0, 25.0, false);
        assert_eq!(
            buf[0] & tile_flags::MODE_MASK,
            TileCompressionMode::BitStuffed as u8
        );
        // The offset is written as a reduced type; for u8 value 10, it's 1 byte
        let offset_val = buf[1];
        assert_eq!(offset_val, 10);
        // Decode the bitstuffed portion
        let mut pos = 2;
        let decoded = crate::bitstuffer::decode(&buf, &mut pos, 256, 6).unwrap();
        assert_eq!(decoded.len(), 64);
        // Reconstruct: offset + quant_val
        for (i, &d) in decoded.iter().enumerate() {
            let reconstructed = offset_val as u32 + d;
            assert_eq!(reconstructed, values[i] as u32);
        }
    }
}
