use alloc::vec;
use alloc::vec::Vec;

use crate::error::{LercError, Result};
use crate::types::LercDataType;

mod compression;
mod predictor;

/// Decode float-point lossless (FPL) Huffman-encoded data.
pub(crate) fn decode_huffman_flt<T: LercDataType>(
    data: &[u8],
    pos: &mut usize,
    is_double: bool,
    width: usize,
    height: usize,
    n_depth: usize,
    output: &mut [T],
) -> Result<()> {
    let unit_size = if is_double { 8 } else { 4 };

    for i_depth in 0..n_depth {
        decode_huffman_flt_slice(
            data, pos, is_double, width, height, i_depth, n_depth, unit_size, output,
        )?;
    }

    Ok(())
}

fn decode_huffman_flt_slice<T: LercDataType>(
    data: &[u8],
    pos: &mut usize,
    is_double: bool,
    width: usize,
    height: usize,
    i_depth: usize,
    n_depth: usize,
    unit_size: usize,
    output: &mut [T],
) -> Result<()> {
    if *pos >= data.len() {
        return Err(LercError::BufferTooSmall {
            needed: *pos + 1,
            available: data.len(),
        });
    }

    // Read predictor code
    let predictor_code = data[*pos];
    *pos += 1;

    if predictor_code > 2 {
        return Err(LercError::InvalidData("invalid FPL predictor code".into()));
    }

    let num_pixels = width * height;
    let mut byte_planes: Vec<Vec<u8>> = vec![Vec::new(); unit_size];
    let mut byte_plane_indices: Vec<usize> = vec![0; unit_size];
    let mut byte_plane_levels: Vec<u8> = vec![0; unit_size];

    // Decode each byte plane
    for _ in 0..unit_size {
        if *pos + 6 > data.len() {
            return Err(LercError::BufferTooSmall {
                needed: *pos + 6,
                available: data.len(),
            });
        }

        let byte_index = data[*pos] as usize;
        let best_level = data[*pos + 1];
        let compressed_size =
            u32::from_le_bytes(data[*pos + 2..*pos + 6].try_into().unwrap()) as usize;
        *pos += 6;

        if byte_index >= unit_size || best_level > 5 {
            return Err(LercError::InvalidData("invalid FPL byte plane header".into()));
        }

        if *pos + compressed_size > data.len() {
            return Err(LercError::BufferTooSmall {
                needed: *pos + compressed_size,
                available: data.len(),
            });
        }

        // Decompress byte plane
        let plane = compression::extract_buffer(&data[*pos..*pos + compressed_size], num_pixels)?;
        *pos += compressed_size;

        // Restore derivatives
        let mut restored = plane;
        predictor::restore_sequence(&mut restored, width, best_level);

        byte_planes[byte_index] = restored;
        byte_plane_indices[byte_index] = byte_index;
        byte_plane_levels[byte_index] = best_level;
    }

    // Reconstruct float/double values by interleaving byte planes
    let mut unit_buffer = vec![0u8; num_pixels * unit_size];
    for pixel in 0..num_pixels {
        for b in 0..unit_size {
            unit_buffer[pixel * unit_size + b] = byte_planes[b][pixel];
        }
    }

    // Apply predictor restoration on the interleaved unit data
    match predictor_code {
        2 => {
            // ROWS_COLS: restore columns then rows
            predictor::restore_cross_bytes(&mut unit_buffer, width, height, unit_size);
        }
        1 => {
            // DELTA1: restore rows
            predictor::restore_byte_order(&mut unit_buffer, width, height, unit_size);
        }
        0 => {
            // NONE: nothing to do
        }
        _ => unreachable!(),
    }

    // Reverse float transform and write to output
    if is_double {
        for pixel in 0..num_pixels {
            let offset = pixel * unit_size;
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&unit_buffer[offset..offset + 8]);
            let bits = u64::from_le_bytes(bytes);
            let restored = undo_double_transform(bits);
            let val = f64::from_bits(restored);
            let m = pixel * n_depth + i_depth;
            output[m] = T::from_f64(val);
        }
    } else {
        for pixel in 0..num_pixels {
            let offset = pixel * unit_size;
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&unit_buffer[offset..offset + 4]);
            let bits = u32::from_le_bytes(bytes);
            let restored = undo_float_transform(bits);
            let val = f32::from_bits(restored);
            let m = pixel * n_depth + i_depth;
            output[m] = T::from_f64(val as f64);
        }
    }

    Ok(())
}

/// Reverse the float bit reorganization: [mantissa(23) | exponent(8) | sign(1)] -> IEEE 754
fn undo_float_transform(a: u32) -> u32 {
    let mantissa = a >> 9; // upper 23 bits
    let exponent = (a >> 1) & 0xFF; // next 8 bits
    let sign = a & 1; // lowest bit
    (sign << 31) | (exponent << 23) | mantissa
}

/// Reverse the double bit reorganization
fn undo_double_transform(a: u64) -> u64 {
    // The C++ moveBits2Front puts:
    //   [mantissa(52) | exponent(11) | sign(1)] in a 64-bit word
    // Undo: mantissa = a >> 12, exponent = (a >> 1) & 0x7FF, sign = a & 1
    let mantissa_d = a >> 12;
    let exponent_d = (a >> 1) & 0x7FF;
    let sign_d = a & 1;
    (sign_d << 63) | (exponent_d << 52) | mantissa_d
}

/// Apply the float bit reorganization for encoding
pub(crate) fn float_transform(a: u32) -> u32 {
    let mantissa = a & 0x007FFFFF;
    let exponent = (a >> 23) & 0xFF;
    let sign = (a >> 31) & 1;
    (mantissa << 9) | (exponent << 1) | sign
}

/// Apply the double bit reorganization for encoding
pub(crate) fn double_transform(a: u64) -> u64 {
    let mantissa = a & 0x000FFFFFFFFFFFFF;
    let exponent = (a >> 52) & 0x7FF;
    let sign = (a >> 63) & 1;
    (mantissa << 12) | (exponent << 1) | sign
}

/// Encode float-point lossless data.
pub(crate) fn encode_huffman_flt<T: LercDataType>(
    _input: &[T],
    _is_double: bool,
    _width: usize,
    _height: usize,
    _n_depth: usize,
) -> Result<Vec<u8>> {
    // TODO: implement FPL encoding
    Err(LercError::InvalidData("FPL encoding not yet implemented".into()))
}
