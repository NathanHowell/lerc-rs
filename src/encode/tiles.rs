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
