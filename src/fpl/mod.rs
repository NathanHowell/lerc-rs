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

#[allow(clippy::too_many_arguments)]
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
    input: &[T],
    is_double: bool,
    width: usize,
    height: usize,
    n_depth: usize,
) -> Result<Vec<u8>> {
    let unit_size = if is_double { 8 } else { 4 };
    let mut result = Vec::new();

    for i_depth in 0..n_depth {
        let slice_data =
            encode_huffman_flt_slice(input, is_double, width, height, i_depth, n_depth, unit_size)?;
        result.extend_from_slice(&slice_data);
    }

    Ok(result)
}

fn encode_huffman_flt_slice<T: LercDataType>(
    input: &[T],
    is_double: bool,
    width: usize,
    height: usize,
    i_depth: usize,
    n_depth: usize,
    unit_size: usize,
) -> Result<Vec<u8>> {
    let num_pixels = width * height;

    // Step 1: Apply float bit reorganization and build interleaved unit buffer
    let mut unit_buffer = vec![0u8; num_pixels * unit_size];

    if is_double {
        for pixel in 0..num_pixels {
            let m = pixel * n_depth + i_depth;
            let val = input[m].to_f64();
            let bits = f64::to_bits(val);
            let transformed = double_transform(bits);
            let bytes = transformed.to_le_bytes();
            unit_buffer[pixel * unit_size..pixel * unit_size + unit_size]
                .copy_from_slice(&bytes);
        }
    } else {
        for pixel in 0..num_pixels {
            let m = pixel * n_depth + i_depth;
            let val = input[m].to_f64() as f32;
            let bits = f32::to_bits(val);
            let transformed = float_transform(bits);
            let bytes = transformed.to_le_bytes();
            unit_buffer[pixel * unit_size..pixel * unit_size + unit_size]
                .copy_from_slice(&bytes);
        }
    }

    // Step 2: Select and apply predictor
    let predictor_code = select_predictor(&unit_buffer, width, height, unit_size);
    let mut predicted_buffer = unit_buffer;

    match predictor_code {
        2 => {
            predictor::apply_cross_bytes(&mut predicted_buffer, width, height, unit_size);
        }
        1 => {
            predictor::apply_byte_order(&mut predicted_buffer, width, height, unit_size);
        }
        0 => {
            // No prediction
        }
        _ => unreachable!(),
    }

    // Step 3: Decompose into byte planes
    let mut byte_planes: Vec<Vec<u8>> = vec![vec![0u8; num_pixels]; unit_size];
    for pixel in 0..num_pixels {
        for b in 0..unit_size {
            byte_planes[b][pixel] = predicted_buffer[pixel * unit_size + b];
        }
    }

    // Step 4: For each byte plane, find best delta level and compress
    let mut result = Vec::new();

    // Write predictor code
    result.push(predictor_code);

    for (byte_index, plane) in byte_planes[..unit_size].iter().enumerate() {
        // Try different delta levels and pick the best
        let (best_level, compressed) = compress_byte_plane(plane, width);

        // Write byte plane header
        result.push(byte_index as u8);
        result.push(best_level);
        result.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
        result.extend_from_slice(&compressed);
    }

    Ok(result)
}

/// Select the best predictor by trying all three and measuring entropy.
fn select_predictor(
    unit_buffer: &[u8],
    width: usize,
    height: usize,
    unit_size: usize,
) -> u8 {
    // For very small images, skip prediction
    if width <= 1 && height <= 1 {
        return 0;
    }

    let mut best_code = 0u8;
    let mut best_entropy = compute_interleaved_entropy(unit_buffer, unit_size);

    // Try predictor 1 (DELTA1)
    {
        let mut buf = unit_buffer.to_vec();
        predictor::apply_byte_order(&mut buf, width, height, unit_size);
        let entropy = compute_interleaved_entropy(&buf, unit_size);
        if entropy < best_entropy {
            best_entropy = entropy;
            best_code = 1;
        }
    }

    // Try predictor 2 (ROWS_COLS)
    {
        let mut buf = unit_buffer.to_vec();
        predictor::apply_cross_bytes(&mut buf, width, height, unit_size);
        let entropy = compute_interleaved_entropy(&buf, unit_size);
        if entropy < best_entropy {
            best_code = 2;
        }
    }

    best_code
}

/// Compute the total entropy of all byte planes from an interleaved buffer.
fn compute_interleaved_entropy(data: &[u8], unit_size: usize) -> f64 {
    let num_pixels = data.len() / unit_size;
    if num_pixels == 0 {
        return 0.0;
    }

    let mut total = 0.0;
    for b in 0..unit_size {
        let mut histo = [0u32; 256];
        for pixel in 0..num_pixels {
            histo[data[pixel * unit_size + b] as usize] += 1;
        }
        total += compute_entropy_from_histo(&histo, num_pixels);
    }
    total
}

/// Compute entropy in bits from a histogram.
fn compute_entropy_from_histo(histo: &[u32; 256], total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let n = total as f64;
    let mut entropy = 0.0;
    for &count in histo {
        if count > 0 {
            let p = count as f64 / n;
            entropy -= p * p.log2();
        }
    }
    entropy * n // total bits for this plane
}

/// Compress a single byte plane with the best delta level.
/// Returns (best_level, compressed_data).
#[allow(clippy::needless_range_loop)]
fn compress_byte_plane(plane: &[u8], width: usize) -> (u8, Vec<u8>) {
    let mut best_level = 0u8;
    let mut best_compressed = compression::compress_buffer(plane);

    // Try delta levels 1 through 5
    let mut delta_plane = plane.to_vec();
    for level in 1..=5u8 {
        // Apply one more level of delta
        predictor::apply_sequence(&mut delta_plane, width, 1);

        let compressed = compression::compress_buffer(&delta_plane);
        if compressed.len() < best_compressed.len() {
            best_compressed = compressed;
            best_level = level;
        }
    }

    (best_level, best_compressed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn float_transform_round_trip() {
        for bits in [0u32, 1, 0x3F800000, 0x40000000, 0xBF800000, 0x7F7FFFFF, 0xFF7FFFFF] {
            let transformed = float_transform(bits);
            let restored = undo_float_transform(transformed);
            assert_eq!(restored, bits, "float transform round trip failed for {bits:#010x}");
        }
    }

    #[test]
    fn double_transform_round_trip() {
        for bits in [0u64, 1, 0x3FF0000000000000, 0x4000000000000000, 0xBFF0000000000000] {
            let transformed = double_transform(bits);
            let restored = undo_double_transform(transformed);
            assert_eq!(restored, bits, "double transform round trip failed for {bits:#018x}");
        }
    }

    #[test]
    fn fpl_encode_decode_f32() {
        let width = 8;
        let height = 8;
        let n_depth = 1;
        let input: Vec<f32> = (0..width * height)
            .map(|i| (i as f32) * 0.1 + 100.0)
            .collect();

        let is_double = false;
        let encoded = encode_huffman_flt(&input, is_double, width, height, n_depth).unwrap();

        let mut output = vec![0.0f32; width * height * n_depth];
        let mut pos = 0;
        decode_huffman_flt(&encoded, &mut pos, is_double, width, height, n_depth, &mut output).unwrap();

        for (i, (&orig, &dec)) in input.iter().zip(output.iter()).enumerate() {
            assert_eq!(
                orig.to_bits(),
                dec.to_bits(),
                "pixel {i}: orig={orig} (bits={:#010x}), decoded={dec} (bits={:#010x})",
                orig.to_bits(),
                dec.to_bits()
            );
        }
    }

    #[test]
    fn fpl_encode_decode_f64() {
        let width = 8;
        let height = 8;
        let n_depth = 1;
        let input: Vec<f64> = (0..width * height)
            .map(|i| (i as f64) * 0.123456789 + 1000.0)
            .collect();

        let is_double = true;
        let encoded = encode_huffman_flt(&input, is_double, width, height, n_depth).unwrap();

        let mut output = vec![0.0f64; width * height * n_depth];
        let mut pos = 0;
        decode_huffman_flt(&encoded, &mut pos, is_double, width, height, n_depth, &mut output).unwrap();

        for (i, (&orig, &dec)) in input.iter().zip(output.iter()).enumerate() {
            assert_eq!(
                orig.to_bits(),
                dec.to_bits(),
                "pixel {i}: orig={orig}, decoded={dec}"
            );
        }
    }
}
