use alloc::vec;
use alloc::vec::Vec;

use crate::error::{LercError, Result};
use crate::types::LercDataType;

mod compression;
mod predictor;

/// Parameters for decoding a single FPL (float-point lossless) depth slice.
struct FplSliceParams {
    is_double: bool,
    width: usize,
    height: usize,
    n_depth: usize,
}

impl FplSliceParams {
    fn unit_size(&self) -> usize {
        if self.is_double { 8 } else { 4 }
    }
}

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
    let params = FplSliceParams {
        is_double,
        width,
        height,
        n_depth,
    };

    for i_depth in 0..n_depth {
        decode_huffman_flt_slice(data, pos, &params, i_depth, output)?;
    }

    Ok(())
}

fn decode_huffman_flt_slice<T: LercDataType>(
    data: &[u8],
    pos: &mut usize,
    params: &FplSliceParams,
    i_depth: usize,
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

    let width = params.width;
    let height = params.height;
    let n_depth = params.n_depth;
    let unit_size = params.unit_size();
    let is_double = params.is_double;

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
            return Err(LercError::InvalidData(
                "invalid FPL byte plane header".into(),
            ));
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
        // C++ does NOT apply bit reorganization for doubles (no undoFloatTransform).
        // Just reinterpret the raw LE bytes as f64.
        for pixel in 0..num_pixels {
            let offset = pixel * unit_size;
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&unit_buffer[offset..offset + 8]);
            let val = f64::from_le_bytes(bytes);
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

/// Reverse the float bit reorganization: [mantissa(23) | sign(1) | exponent(8)] -> IEEE 754
/// Layout: mantissa in bits 0-22, sign in bit 23, exponent in bits 24-31.
fn undo_float_transform(a: u32) -> u32 {
    let mantissa = a & 0x007FFFFF;
    let sign = (a >> 23) & 1;
    let exponent = (a >> 24) & 0xFF;
    (sign << 31) | (exponent << 23) | mantissa
}

/// Reverse the double bit reorganization: [mantissa(52) | sign(1) | exponent(11)] -> IEEE 754
/// Layout: mantissa in bits 0-51, sign in bit 52, exponent in bits 53-63.
/// Note: not used in production since C++ does NOT apply this transform for doubles.
#[cfg(test)]
fn undo_double_transform(a: u64) -> u64 {
    let mantissa = a & 0x000FFFFFFFFFFFFF;
    let sign = (a >> 52) & 1;
    let exponent = (a >> 53) & 0x7FF;
    (sign << 63) | (exponent << 52) | mantissa
}

/// Apply the float bit reorganization for encoding.
/// Rearranges IEEE 754 bits to [mantissa(23) | sign(1) | exponent(8)] for better
/// byte-plane compression. This matches the C++ moveBits2Front layout.
pub(crate) fn float_transform(a: u32) -> u32 {
    let mantissa = a & 0x007FFFFF;
    let exponent = (a >> 23) & 0xFF;
    let sign = (a >> 31) & 1;
    mantissa | (sign << 23) | (exponent << 24)
}

/// Apply the double bit reorganization for encoding.
/// Rearranges IEEE 754 bits to [mantissa(52) | sign(1) | exponent(11)].
/// Note: the C++ reference does NOT apply this transform for doubles, so it is
/// not used in the FPL encode/decode paths. Kept for testing.
#[cfg(test)]
fn double_transform(a: u64) -> u64 {
    let mantissa = a & 0x000FFFFFFFFFFFFF;
    let exponent = (a >> 52) & 0x7FF;
    let sign = (a >> 63) & 1;
    mantissa | (sign << 52) | (exponent << 53)
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
        // C++ does NOT apply bit reorganization for doubles (no doFloatTransform).
        // Just copy the raw LE bytes.
        for pixel in 0..num_pixels {
            let m = pixel * n_depth + i_depth;
            let val = input[m].to_f64();
            let bytes = val.to_le_bytes();
            unit_buffer[pixel * unit_size..pixel * unit_size + unit_size].copy_from_slice(&bytes);
        }
    } else {
        for pixel in 0..num_pixels {
            let m = pixel * n_depth + i_depth;
            // For f32, read bits directly without f64 round-trip
            let bits = input[m].to_bits_u64() as u32;
            let transformed = float_transform(bits);
            let bytes = transformed.to_le_bytes();
            unit_buffer[pixel * unit_size..pixel * unit_size + unit_size].copy_from_slice(&bytes);
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

    // Compute max byte delta based on predictor, matching C++
    // `Predictor::getMaxByteDelta(p) = MAX_DELTA - getIntDelta(p)`
    const MAX_DELTA: u8 = 5;
    let predictor_int_delta = match predictor_code {
        0 => 0u8, // NONE
        1 => 1,   // DELTA1
        2 => 2,   // ROWS_COLS
        _ => unreachable!(),
    };
    let max_byte_delta = MAX_DELTA - predictor_int_delta;

    // Step 3 & 4: Decompose into byte planes and compress each one.
    // Process one byte plane at a time to reuse the extraction buffer and
    // reduce peak memory usage (avoids allocating all byte planes at once).
    // Pre-allocate result with a conservative estimate (header + compressed planes).
    let mut result = Vec::with_capacity(1 + unit_size * (6 + num_pixels));

    // Write predictor code
    result.push(predictor_code);

    let mut plane_buf = vec![0u8; num_pixels];
    for byte_index in 0..unit_size {
        // Extract this byte plane from the interleaved buffer
        for pixel in 0..num_pixels {
            plane_buf[pixel] = predicted_buffer[pixel * unit_size + byte_index];
        }

        let (best_level, compressed) = compress_byte_plane(&plane_buf, max_byte_delta);

        // Write byte plane header
        result.push(byte_index as u8);
        result.push(best_level);
        result.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
        result.extend_from_slice(&compressed);
    }

    Ok(result)
}

/// A test block used for sampling-based predictor selection and level search.
/// Matches the C++ `TestBlock` struct.
struct TestBlock {
    top: usize,
    height: usize,
}

/// Generate evenly-spaced test blocks for predictor selection.
/// Matches the C++ `generateListOfTestBlocks`.
fn generate_test_blocks(width: usize, height: usize) -> Vec<TestBlock> {
    let size = width * height;
    if size == 0 {
        return Vec::new();
    }

    const BLOCK_TARGET_SIZE: usize = 8 * 1024;

    let t = (size as f64 / BLOCK_TARGET_SIZE as f64).round();
    let mut count = (t + 1.0).sqrt().round() as usize;
    // count is always >= 1

    let mut block_height = BLOCK_TARGET_SIZE / width;
    if block_height < 4 {
        block_height = 4;
    }

    while count * block_height > height && count > 1 {
        count -= 1;
    }

    let top_margin = (height as f64 - (count * block_height) as f64) / (2.0 * count as f64);
    let delta = 2.0 * top_margin + block_height as f64;

    let mut blocks = Vec::with_capacity(count);
    for i in 0..count {
        let mut top = (top_margin + delta * i as f64) as isize;
        let mut bh = block_height;
        if top < 0 {
            top = 0;
        }
        let top = top as usize;
        if top + bh > height {
            bh = height - top;
        }
        if bh > 0 {
            blocks.push(TestBlock { top, height: bh });
        }
    }

    blocks
}

/// Apply the prime-stride delta (stride=7) to a buffer, matching C++
/// `setDerivativePrime`.
fn set_derivative_prime(data: &mut [u8]) {
    const PRIME_MULT: usize = 7;
    if data.is_empty() {
        return;
    }
    let mut off = data.len() - 1;
    off = PRIME_MULT * (off / PRIME_MULT);

    while off >= 1 {
        data[off] = data[off].wrapping_sub(data[off - 1]);
        if off < PRIME_MULT {
            break;
        }
        off -= PRIME_MULT;
    }
}

/// Compute the estimated compressed size for test blocks extracted from
/// interleaved unit data. Matches the C++ `testBlocksSize`.
fn test_blocks_size(
    blocks: &[TestBlock],
    data: &[u8],
    raster_width: usize,
    unit_size: usize,
    test_first_byte_delta: bool,
) -> usize {
    let mut ret = 0usize;

    // Pre-allocate buffers sized to the largest block
    let max_length = blocks
        .iter()
        .map(|tb| tb.height * raster_width)
        .max()
        .unwrap_or(0);
    let mut plane_buffer = vec![0u8; max_length];
    let mut delta_buf = if test_first_byte_delta {
        vec![0u8; max_length]
    } else {
        Vec::new()
    };

    for tb in blocks {
        let start = unit_size * tb.top * raster_width;
        let length = tb.height * raster_width;

        for byte in 0..unit_size {
            // Extract byte plane from interleaved data
            let mut ptr_offset = start + byte;
            for dest in &mut plane_buffer[..length] {
                *dest = data[ptr_offset];
                ptr_offset += unit_size;
            }

            let plane_encoded = compression::estimate_compressed_size(&plane_buffer[..length]);

            if test_first_byte_delta {
                delta_buf[..length].copy_from_slice(&plane_buffer[..length]);
                set_derivative_prime(&mut delta_buf[..length]);
                let plane_encoded2 = compression::estimate_compressed_size(&delta_buf[..length]);
                ret += plane_encoded.min(plane_encoded2);
            } else {
                ret += plane_encoded;
            }
        }
    }

    ret
}

/// Select the best predictor by generating test blocks and measuring estimated
/// compressed sizes, matching C++ `selectInitialLinearOrCrossDelta`.
fn select_predictor(unit_buffer: &[u8], width: usize, height: usize, unit_size: usize) -> u8 {
    // For very small images, skip prediction
    if width <= 1 && height <= 1 {
        return 0;
    }

    let blocks = generate_test_blocks(width, height);
    if blocks.is_empty() {
        return 0;
    }

    let test_first_byte_delta = true;
    let mut stats = [0usize; 3];

    // Predictor 0 (NONE): test on a copy of the original data
    let mut copy = unit_buffer.to_vec();
    stats[0] = test_blocks_size(&blocks, &copy, width, unit_size, test_first_byte_delta);

    // Predictor 1 (DELTA1): apply row-wise delta (phase 1 of setBlockDerivative)
    predictor::apply_byte_order(&mut copy, width, height, unit_size);
    stats[1] = test_blocks_size(&blocks, &copy, width, unit_size, test_first_byte_delta);

    // Predictor 2 (ROWS_COLS): apply column delta on top of existing row delta
    // (phase 2 of setCrossDerivative)
    predictor::apply_cross_bytes_phase2(&mut copy, width, height, unit_size);
    stats[2] = test_blocks_size(&blocks, &copy, width, unit_size, test_first_byte_delta);

    // Pick the predictor with the smallest total
    let min_index = if stats[0] <= stats[1] && stats[0] <= stats[2] {
        0
    } else if stats[1] <= stats[2] {
        1
    } else {
        2
    };

    min_index as u8
}

/// Find the best delta level for a byte plane using snippet-based sampling,
/// then compress the full plane once at that level.
/// Matches the C++ `getBestLevel2` + single full compress.
fn compress_byte_plane(plane: &[u8], max_delta: u8) -> (u8, Vec<u8>) {
    let best_level = get_best_level(plane, max_delta);

    // Apply the winning delta level to a copy and compress once
    let mut final_plane = plane.to_vec();
    predictor::apply_sequence(&mut final_plane, 0, best_level);

    let compressed = compression::compress_buffer(&final_plane);
    (best_level, compressed)
}

/// Determine the best delta level using snippet-based sampling.
/// Matches the C++ `getBestLevel2` exactly.
fn get_best_level(plane: &[u8], max_delta: u8) -> u8 {
    if max_delta == 0 {
        return 0;
    }

    let size = plane.len();
    const TARGET_SAMPLE_SIZE: usize = 1024 * 8;

    let t = (size as f64 / TARGET_SAMPLE_SIZE as f64).round();
    let mut count = (t + 1.0).sqrt().round() as isize;

    while count as usize * TARGET_SAMPLE_SIZE > size && count > 0 {
        count -= 1;
    }

    let count = count as usize;
    if count == 0 {
        // Plane too small for snippet sampling; fall back to testing full plane
        return get_best_level_full(plane, max_delta);
    }

    let top_margin = (size as f64 - (count * TARGET_SAMPLE_SIZE) as f64) / (2.0 * count as f64);
    let delta = 2.0 * top_margin + TARGET_SAMPLE_SIZE as f64;

    let mut snippets: Vec<(usize, usize)> = Vec::with_capacity(count);
    for i in 0..count {
        let mut start = (top_margin + delta * i as f64) as isize;
        let mut len = TARGET_SAMPLE_SIZE;
        if start < 0 {
            start = 0;
        }
        let start = start as usize;
        if start + len > size {
            len = size - start;
        }
        if len > 0 {
            snippets.push((start, len));
        }
    }

    // Work on a copy of the full plane (needed so snippet deltas accumulate correctly)
    let mut copy = plane.to_vec();

    let mut best_comp = 0usize;
    let mut ret = 0u8;

    for l in 0..=max_delta {
        if l > 0 {
            // Apply one additional delta level to only the snippet regions
            for &(start, len) in &snippets {
                let end = start + len;
                for i in (start + l as usize..end).rev() {
                    copy[i] = copy[i].wrapping_sub(copy[i - 1]);
                }
            }
        }

        // Estimate compressed size of all snippets
        let mut comp = 0usize;
        for &(start, len) in &snippets {
            comp += compression::estimate_compressed_size(&copy[start..start + len]);
        }

        if comp < best_comp || l == 0 {
            best_comp = comp;
            ret = l;
        } else {
            break; // if deteriorating, stop
        }
    }

    ret
}

/// Fallback for planes too small for snippet sampling: test each delta level
/// with entropy estimation on the full plane.
fn get_best_level_full(plane: &[u8], max_delta: u8) -> u8 {
    let mut copy = plane.to_vec();
    let mut best_comp = compression::estimate_compressed_size(&copy);
    let mut ret = 0u8;

    for l in 1..=max_delta {
        // Apply one delta level
        for i in (l as usize..copy.len()).rev() {
            copy[i] = copy[i].wrapping_sub(copy[i - 1]);
        }
        let comp = compression::estimate_compressed_size(&copy);
        if comp < best_comp {
            best_comp = comp;
            ret = l;
        } else {
            break;
        }
    }

    ret
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn float_transform_round_trip() {
        for bits in [
            0u32, 1, 0x3F800000, 0x40000000, 0xBF800000, 0x7F7FFFFF, 0xFF7FFFFF,
        ] {
            let transformed = float_transform(bits);
            let restored = undo_float_transform(transformed);
            assert_eq!(
                restored, bits,
                "float transform round trip failed for {bits:#010x}"
            );
        }
    }

    #[test]
    fn double_transform_round_trip() {
        for bits in [
            0u64,
            1,
            0x3FF0000000000000,
            0x4000000000000000,
            0xBFF0000000000000,
        ] {
            let transformed = double_transform(bits);
            let restored = undo_double_transform(transformed);
            assert_eq!(
                restored, bits,
                "double transform round trip failed for {bits:#018x}"
            );
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
        decode_huffman_flt(
            &encoded,
            &mut pos,
            is_double,
            width,
            height,
            n_depth,
            &mut output,
        )
        .unwrap();

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
    fn fpl_encode_decode_f32_ramp_multi_depth() {
        // Directly test the depth=1 slice that fails
        let width = 16;
        let height = 16;
        // Depth 1 slice: values = (pixel*3+1)*0.01 + 1.0
        let input: Vec<f32> = (0..width * height)
            .map(|pixel| ((pixel * 3 + 1) as f32) * 0.01 + 1.0)
            .collect();

        let is_double = false;
        let encoded = encode_huffman_flt(&input, is_double, width, height, 1).unwrap();
        let mut output = vec![0.0f32; width * height];
        let mut pos = 0;
        decode_huffman_flt(&encoded, &mut pos, is_double, width, height, 1, &mut output).unwrap();

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
    fn test_apply_restore_sequence_level2() {
        // Test that apply_sequence and restore_sequence are inverses for level 2
        let original: Vec<u8> = (0..256).map(|i| (i * 7 + 13) as u8).collect();
        let mut data = original.clone();
        predictor::apply_sequence(&mut data, 16, 2);
        predictor::restore_sequence(&mut data, 16, 2);
        assert_eq!(
            data, original,
            "apply_sequence/restore_sequence round-trip failed for level 2"
        );
    }

    #[test]
    fn test_compress_byte_plane_round_trip() {
        // Test that compress_byte_plane produces data that can be restored
        let width = 16;
        let original: Vec<u8> = (0..256).map(|i| (i * 7 + 13) as u8).collect();
        let (level, compressed) = compress_byte_plane(&original, 5);

        // Decompress
        let decompressed = compression::extract_buffer(&compressed, original.len()).unwrap();
        // Restore
        let mut restored = decompressed;
        predictor::restore_sequence(&mut restored, width, level);
        assert_eq!(restored, original, "compress_byte_plane round-trip failed");
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
        decode_huffman_flt(
            &encoded,
            &mut pos,
            is_double,
            width,
            height,
            n_depth,
            &mut output,
        )
        .unwrap();

        for (i, (&orig, &dec)) in input.iter().zip(output.iter()).enumerate() {
            assert_eq!(
                orig.to_bits(),
                dec.to_bits(),
                "pixel {i}: orig={orig}, decoded={dec}"
            );
        }
    }
}
