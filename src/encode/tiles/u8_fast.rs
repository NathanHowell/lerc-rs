use alloc::vec::Vec;

use crate::error::Result;
use crate::header::HeaderInfo;
use crate::types::{LercDataType, TileCompressionMode, TileRect};

use super::ScratchBuffers;

/// Fast block size selection for u8 lossless, all-valid, single-depth images.
pub(super) fn select_block_size_u8_fast(data: &[u8], hd: &HeaderInfo) -> i32 {
    let n_rows = hd.n_rows as usize;
    let n_cols = hd.n_cols as usize;
    let version = hd.version;
    let pattern: u8 = if version >= 5 { 14 } else { 15 };
    let mut quant_buf = [0u32; 256];

    let mut size8 = 0usize;
    let mut buf8 = Vec::with_capacity(512);
    for &(i0, i1, j0, j1) in &super::sample_tile_positions(n_rows, n_cols, 8) {
        buf8.clear();
        let rect = TileRect { i0, i1, j0, j1 };
        let integrity = ((j0 as u8 >> 3) & pattern) << 2;
        encode_single_u8_tile(&mut buf8, data, n_cols, rect, integrity, &mut quant_buf);
        size8 += buf8.len();
    }

    let mut size16 = 0usize;
    let mut buf16 = Vec::with_capacity(512);
    for &(i0, i1, j0, j1) in &super::sample_tile_positions(n_rows, n_cols, 16) {
        buf16.clear();
        let rect = TileRect { i0, i1, j0, j1 };
        let integrity = ((j0 as u8 >> 3) & pattern) << 2;
        encode_single_u8_tile(&mut buf16, data, n_cols, rect, integrity, &mut quant_buf);
        size16 += buf16.len();
    }

    if size16 < size8 { 16 } else { 8 }
}

/// Encode a single u8 tile (used by select_block_size_u8_fast and encode_tiles_u8_fast).
///
/// Combines min/max scan, quantization, histogram building, LUT decision, and encoding
/// into minimal passes over the data.
#[inline]
pub(super) fn encode_single_u8_tile(
    buf: &mut Vec<u8>,
    data: &[u8],
    n_cols: usize,
    rect: TileRect,
    integrity: u8,
    quant_buf: &mut [u32; 256],
) {
    let TileRect { i0, i1, j0, j1 } = rect;
    let tile_w = j1 - j0;
    let num_valid = (i1 - i0) * tile_w;

    // Single pass: compute min, max, and copy values into quant_buf (as-is, not yet quantized)
    let mut z_min: u8 = 255;
    let mut z_max: u8 = 0;
    let mut idx = 0;
    for i in i0..i1 {
        let row_start = i * n_cols + j0;
        for &val in &data[row_start..row_start + tile_w] {
            if val < z_min {
                z_min = val;
            }
            if val > z_max {
                z_max = val;
            }
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
            while (n_lut >> n_bits_lut) != 0 {
                n_bits_lut += 1;
            }
            let n_bytes = crate::bitstuffer::num_bytes_uint(num_elem);
            let num_bytes_simple = 1 + n_bytes as u32 + ((num_elem * num_bits + 7) >> 3);
            let num_bytes_lut = 1
                + n_bytes as u32
                + 1
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
            while (n_lut >> n_bits_lut) != 0 {
                n_bits_lut += 1;
            }

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

/// Specialized tile encoding for u8 lossless, all-valid, single-depth images.
/// Avoids all f64 conversions and Vec allocations in the inner loop.
pub(super) fn encode_tiles_u8_fast(
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

            let rect = TileRect { i0, i1, j0, j1 };
            let integrity = ((j0 as u8 >> 3) & pattern) << 2;
            encode_single_u8_tile(blob, data, n_cols, rect, integrity, quant_buf);
        }
    }
}

/// Encode a single tile and return its byte count (without appending to blob).
/// This is used by `select_block_size` for sampling.
pub(super) fn encode_tile_to_count<T: LercDataType>(
    ctx: &super::TileEncodeContext<'_, T>,
    rect: crate::types::TileRect,
    i_depth: usize,
    scratch: &mut ScratchBuffers,
) -> Result<usize> {
    let mut tmp = Vec::new();
    super::encode_tile(
        &mut tmp, ctx, rect, i_depth, scratch,
        None, // No reconstruction buffer needed for size estimation
    )?;
    Ok(tmp.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::tile_flags;
    use alloc::vec;

    // -----------------------------------------------------------------------
    // encode_single_u8_tile
    // -----------------------------------------------------------------------

    #[test]
    fn encode_single_u8_tile_all_zeros() {
        let data = vec![0u8; 64]; // 8x8 tile, all zeros
        let n_cols = 8;
        let mut buf = Vec::new();
        let mut quant_buf = [0u32; 256];
        encode_single_u8_tile(
            &mut buf,
            &data,
            n_cols,
            TileRect {
                i0: 0,
                i1: 8,
                j0: 0,
                j1: 8,
            },
            0,
            &mut quant_buf,
        );
        assert_eq!(buf.len(), 1);
        assert_eq!(
            buf[0] & tile_flags::MODE_MASK,
            TileCompressionMode::ConstZero as u8
        );
    }

    #[test]
    fn encode_single_u8_tile_constant_nonzero() {
        let data = vec![42u8; 64]; // 8x8 tile, all 42
        let n_cols = 8;
        let mut buf = Vec::new();
        let mut quant_buf = [0u32; 256];
        encode_single_u8_tile(
            &mut buf,
            &data,
            n_cols,
            TileRect {
                i0: 0,
                i1: 8,
                j0: 0,
                j1: 8,
            },
            0,
            &mut quant_buf,
        );
        assert_eq!(
            buf[0] & tile_flags::MODE_MASK,
            TileCompressionMode::ConstOffset as u8
        );
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
        encode_single_u8_tile(
            &mut buf,
            &data,
            n_cols,
            TileRect {
                i0: 0,
                i1: 4,
                j0: 0,
                j1: 4,
            },
            0,
            &mut quant_buf,
        );
        assert_eq!(
            buf[0] & tile_flags::MODE_MASK,
            TileCompressionMode::BitStuffed as u8
        );
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
        encode_single_u8_tile(
            &mut buf,
            &data,
            n_cols,
            TileRect {
                i0: 0,
                i1: 4,
                j0: 0,
                j1: 4,
            },
            0,
            &mut quant_buf,
        );
        assert_eq!(
            buf[0] & tile_flags::MODE_MASK,
            TileCompressionMode::BitStuffed as u8
        );
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
        encode_single_u8_tile(
            &mut buf,
            &data,
            n_cols,
            TileRect {
                i0: 2,
                i1: 4,
                j0: 4,
                j1: 8,
            },
            0,
            &mut quant_buf,
        );
        assert_eq!(
            buf[0] & tile_flags::MODE_MASK,
            TileCompressionMode::ConstOffset as u8
        );
        assert_eq!(buf[1], 99);
    }

    #[test]
    fn encode_single_u8_tile_integrity_bits_preserved() {
        let data = vec![0u8; 64];
        let mut buf = Vec::new();
        let mut quant_buf = [0u32; 256];
        let integrity = 0b00001100u8;
        encode_single_u8_tile(
            &mut buf,
            &data,
            8,
            TileRect {
                i0: 0,
                i1: 8,
                j0: 0,
                j1: 8,
            },
            integrity,
            &mut quant_buf,
        );
        assert_eq!(buf[0] & integrity, integrity);
    }
}
