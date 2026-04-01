use alloc::vec::Vec;

use crate::bitmask::BitMask;
use crate::header::HeaderInfo;
use crate::huffman::HuffmanCodec;
use crate::types::{DataType, ImageEncodeMode, Sample};

/// Quick check whether 8-bit integer data has high entropy (nearly all 256
/// byte values present with roughly uniform distribution). When this is true,
/// Huffman encoding cannot beat tiling, so we skip the expensive Huffman attempt.
pub(super) fn is_high_entropy_u8<T: Sample>(
    data: &[T],
    mask: &BitMask,
    header: &HeaderInfo,
) -> bool {
    let width = header.n_cols as usize;
    let height = header.n_rows as usize;
    let n_depth = header.n_depth as usize;
    let offset: i32 = if header.data_type == DataType::Char {
        128
    } else {
        0
    };
    let all_valid = header.num_valid_pixel == header.n_rows * header.n_cols;

    // Build a quick histogram
    let mut histo = [0u32; 256];
    let mut total = 0u32;

    // Fast path for u8, all-valid, single-depth: avoid to_f64 and mask checks
    if T::DATA_TYPE == DataType::Byte && all_valid && n_depth == 1 {
        debug_assert_eq!(offset, 0);
        let u8_data: &[u8] =
            unsafe { core::slice::from_raw_parts(data.as_ptr() as *const u8, data.len()) };
        for &val in u8_data {
            histo[val as usize] += 1;
        }
        total = u8_data.len() as u32;
    } else {
        for i in 0..height {
            for j in 0..width {
                let k = i * width + j;
                if all_valid || mask.is_valid(k) {
                    for m in 0..n_depth {
                        let val = data[k * n_depth + m].to_f64() as i32;
                        let bin = (val + offset) as usize;
                        histo[bin] += 1;
                        total += 1;
                    }
                }
            }
        }
    }

    if total == 0 {
        return false;
    }

    let n_distinct = histo.iter().filter(|&&c| c > 0).count();

    // If fewer than 248 distinct values, Huffman may still help.
    if n_distinct < 248 {
        return false;
    }

    let avg = total as f64 / 256.0;
    let max_count = *histo.iter().max().unwrap_or(&0) as f64;

    // If the distribution isn't roughly uniform, Huffman may help.
    if max_count >= avg * 2.0 {
        return false;
    }

    // Direct histogram is uniform, but delta-encoding might still compress well.
    // Quick check: compute a delta histogram on a sample of rows.
    let mut delta_histo = [0u32; 256];
    let sample_rows = 8.min(height);
    let row_step = height / sample_rows;
    let mut delta_total = 0u32;

    if T::DATA_TYPE == DataType::Byte && all_valid && n_depth == 1 {
        let u8_data: &[u8] =
            unsafe { core::slice::from_raw_parts(data.as_ptr() as *const u8, data.len()) };
        for row_idx in 0..sample_rows {
            let i = row_idx * row_step;
            let row_start = i * width;
            let mut prev_val: u8 = 0;
            for j in 0..width {
                let val = u8_data[row_start + j];
                let delta = if j > 0 {
                    val.wrapping_sub(prev_val)
                } else {
                    val
                };
                delta_histo[delta as usize] += 1;
                delta_total += 1;
                prev_val = val;
            }
        }
    } else {
        for row_idx in 0..sample_rows {
            let i = row_idx * row_step;
            let mut prev_val: i32 = 0;
            for j in 0..width {
                let k = i * width + j;
                if all_valid || mask.is_valid(k) {
                    let val = data[k * n_depth].to_f64() as i32;
                    let delta = if j > 0 {
                        val.wrapping_sub(prev_val)
                    } else {
                        val
                    };
                    delta_histo[(delta + offset) as u8 as usize] += 1;
                    delta_total += 1;
                    prev_val = val;
                }
            }
        }
    }

    if delta_total == 0 {
        return true;
    }

    // If delta histogram has few distinct values, delta-Huffman will compress well.
    let delta_distinct = delta_histo.iter().filter(|&&c| c > 0).count();
    if delta_distinct < 64 {
        return false; // Don't skip — delta-Huffman will likely win
    }

    // Both direct and delta are high entropy — safe to skip Huffman
    true
}

/// Compute histograms for Huffman encoding of 8-bit data.
/// Returns (direct_histogram, delta_histogram) each of size 256.
/// The offset is 128 for i8 (DT_Char) and 0 for u8 (DT_Byte).
fn compute_histo_for_huffman<T: Sample>(
    data: &[T],
    mask: &BitMask,
    header: &HeaderInfo,
) -> ([i32; 256], [i32; 256]) {
    let width = header.n_cols as usize;
    let height = header.n_rows as usize;
    let n_depth = header.n_depth as usize;
    let offset: i32 = if header.data_type == DataType::Char {
        128
    } else {
        0
    };
    let all_valid = header.num_valid_pixel == header.n_rows * header.n_cols;

    let mut histo = [0i32; 256];
    let mut delta_histo = [0i32; 256];

    // Direct histogram
    for i in 0..height {
        for j in 0..width {
            let k = i * width + j;
            if all_valid || mask.is_valid(k) {
                for m in 0..n_depth {
                    let val = data[k * n_depth + m].to_f64() as i32;
                    let bin = (val + offset) as usize;
                    histo[bin] += 1;
                }
            }
        }
    }

    // Delta histogram
    for i_depth in 0..n_depth {
        let mut prev_val: i32 = 0;
        for i in 0..height {
            for j in 0..width {
                let k = i * width + j;
                if all_valid || mask.is_valid(k) {
                    let val = data[k * n_depth + i_depth].to_f64() as i32;
                    let predictor = if j > 0 && (all_valid || mask.is_valid(k - 1)) {
                        prev_val
                    } else if i > 0 && (all_valid || mask.is_valid(k - width)) {
                        data[(k - width) * n_depth + i_depth].to_f64() as i32
                    } else {
                        prev_val
                    };
                    let delta = val.wrapping_sub(predictor);
                    let bin = (delta + offset) as u8 as usize;
                    delta_histo[bin] += 1;
                    prev_val = val;
                }
            }
        }
    }

    (histo, delta_histo)
}

/// Try Huffman encoding for 8-bit integer data (u8/i8).
/// Returns Some(bytes) with the mode flag + Huffman data if beneficial, or None.
pub(super) fn try_encode_huffman_int<T: Sample>(
    data: &[T],
    mask: &BitMask,
    header: &HeaderInfo,
) -> Option<Vec<u8>> {
    let width = header.n_cols as usize;
    let height = header.n_rows as usize;
    let n_depth = header.n_depth as usize;
    let offset: i32 = if header.data_type == DataType::Char {
        128
    } else {
        0
    };
    let all_valid = header.num_valid_pixel == header.n_rows * header.n_cols;

    let (histo, delta_histo) = compute_histo_for_huffman(data, mask, header);

    // Try delta Huffman
    let mut delta_codec = HuffmanCodec::new();
    let delta_ok = delta_codec.compute_codes(&delta_histo);
    let delta_size = if delta_ok {
        delta_codec
            .compute_compressed_size(&delta_histo)
            .map(|(bytes, _)| bytes)
    } else {
        None
    };

    // Try direct Huffman (IEM_Huffman, requires v4+)
    let mut direct_codec = HuffmanCodec::new();
    let direct_ok = header.version >= 4 && direct_codec.compute_codes(&histo);
    let direct_size = if direct_ok {
        direct_codec
            .compute_compressed_size(&histo)
            .map(|(bytes, _)| bytes)
    } else {
        None
    };

    // Pick the best: delta Huffman or direct Huffman
    enum HuffMode {
        Delta,
        Direct,
    }

    let (mode, codec) = match (delta_size, direct_size) {
        (Some(ds), Some(hs)) => {
            if ds <= hs {
                (HuffMode::Delta, delta_codec)
            } else {
                (HuffMode::Direct, direct_codec)
            }
        }
        (Some(_), None) => (HuffMode::Delta, delta_codec),
        (None, Some(_)) => (HuffMode::Direct, direct_codec),
        (None, None) => return None,
    };

    // Write the Huffman-encoded blob
    let mut buf = Vec::new();

    // Write mode flag
    match mode {
        HuffMode::Delta => buf.push(ImageEncodeMode::DeltaHuffman as u8),
        HuffMode::Direct => buf.push(ImageEncodeMode::Huffman as u8),
    }

    // Write code table
    let code_table_bytes = codec.write_code_table(header.version).ok()?;
    buf.extend_from_slice(&code_table_bytes);

    // Compute total bits needed for the data
    let code_table = codec.code_table();

    let mut total_bits = 0u64;
    match mode {
        HuffMode::Delta => {
            for i_depth in 0..n_depth {
                let mut prev_val: i32 = 0;
                for i in 0..height {
                    for j in 0..width {
                        let k = i * width + j;
                        if all_valid || mask.is_valid(k) {
                            let val = data[k * n_depth + i_depth].to_f64() as i32;
                            let predictor = if j > 0 && (all_valid || mask.is_valid(k - 1)) {
                                prev_val
                            } else if i > 0 && (all_valid || mask.is_valid(k - width)) {
                                data[(k - width) * n_depth + i_depth].to_f64() as i32
                            } else {
                                prev_val
                            };
                            let delta = val.wrapping_sub(predictor);
                            let bin = (delta + offset) as u8 as usize;
                            total_bits += code_table[bin].0 as u64;
                            prev_val = val;
                        }
                    }
                }
            }
        }
        HuffMode::Direct => {
            for i in 0..height {
                for j in 0..width {
                    let k = i * width + j;
                    if all_valid || mask.is_valid(k) {
                        for m in 0..n_depth {
                            let val = data[k * n_depth + m].to_f64() as i32;
                            let bin = (val + offset) as usize;
                            total_bits += code_table[bin].0 as u64;
                        }
                    }
                }
            }
        }
    }

    // Allocate buffer for Huffman-encoded data
    // numUInts = (bitPos > 0 ? 1 : 0) + 1 for padding (decode read-ahead)
    let num_uints_data = total_bits.div_ceil(32) as usize;
    let num_uints_total = num_uints_data + 1; // +1 for decode read-ahead padding
    let mut encoded = vec![0u8; num_uints_total * 4];

    // Use a u64 accumulator (MSB-aligned) instead of per-symbol push_value
    // calls. This avoids repeated u32 reads/writes from the output buffer and
    // is the same pattern used in the FPL Huffman encoder.
    let mut accum: u64 = 0;
    let mut accum_bits: u32 = 0;
    let mut out_idx: usize = 0;

    // Encode data
    match mode {
        HuffMode::Delta => {
            for i_depth in 0..n_depth {
                let mut prev_val: i32 = 0;
                for i in 0..height {
                    for j in 0..width {
                        let k = i * width + j;
                        if all_valid || mask.is_valid(k) {
                            let val = data[k * n_depth + i_depth].to_f64() as i32;
                            let predictor = if j > 0 && (all_valid || mask.is_valid(k - 1)) {
                                prev_val
                            } else if i > 0 && (all_valid || mask.is_valid(k - width)) {
                                data[(k - width) * n_depth + i_depth].to_f64() as i32
                            } else {
                                prev_val
                            };
                            let delta = val.wrapping_sub(predictor);
                            let bin = (delta + offset) as u8 as usize;
                            let (len, code) = code_table[bin];
                            let len = len as u32;
                            accum |= (code as u64) << (64 - accum_bits - len);
                            accum_bits += len;
                            if accum_bits >= 32 {
                                let word = (accum >> 32) as u32;
                                encoded[out_idx..out_idx + 4].copy_from_slice(&word.to_le_bytes());
                                out_idx += 4;
                                accum <<= 32;
                                accum_bits -= 32;
                            }
                            prev_val = val;
                        }
                    }
                }
            }
        }
        HuffMode::Direct => {
            for i in 0..height {
                for j in 0..width {
                    let k = i * width + j;
                    if all_valid || mask.is_valid(k) {
                        for m in 0..n_depth {
                            let val = data[k * n_depth + m].to_f64() as i32;
                            let bin = (val + offset) as usize;
                            let (len, code) = code_table[bin];
                            let len = len as u32;
                            accum |= (code as u64) << (64 - accum_bits - len);
                            accum_bits += len;
                            if accum_bits >= 32 {
                                let word = (accum >> 32) as u32;
                                encoded[out_idx..out_idx + 4].copy_from_slice(&word.to_le_bytes());
                                out_idx += 4;
                                accum <<= 32;
                                accum_bits -= 32;
                            }
                        }
                    }
                }
            }
        }
    }

    // Flush remaining bits
    if accum_bits > 0 {
        let word = (accum >> 32) as u32;
        encoded[out_idx..out_idx + 4].copy_from_slice(&word.to_le_bytes());
        out_idx += 4;
    }

    // Compute final size with padding: round up to next u32 boundary + 1 extra
    // u32 for decode read-ahead
    let num_uints_final = (out_idx / 4) + 1; // +1 for decode read-ahead padding
    encoded.truncate(num_uints_final * 4);

    buf.extend_from_slice(&encoded);
    Some(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build a HeaderInfo for u8 data (DataType::Byte).
    fn make_header_byte(
        width: usize,
        height: usize,
        n_depth: usize,
        num_valid: usize,
    ) -> HeaderInfo {
        HeaderInfo {
            version: 6,
            n_rows: height as i32,
            n_cols: width as i32,
            n_depth: n_depth as i32,
            num_valid_pixel: num_valid as i32,
            data_type: DataType::Byte,
            ..HeaderInfo::default()
        }
    }

    // ---- is_high_entropy_u8 ----

    #[test]
    fn high_entropy_uniform_data() {
        // Create pseudo-random data where all 256 byte values are roughly uniform
        // and deltas are also high entropy (many distinct delta values).
        // Use a simple LCG to generate pseudo-random bytes.
        let width = 256;
        let height = 256;
        let n = width * height;
        let mut data = Vec::with_capacity(n);
        let mut state: u32 = 12345;
        for _ in 0..n {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            data.push((state >> 16) as u8);
        }
        let mask = BitMask::all_valid(n);
        let header = make_header_byte(width, height, 1, n);
        assert!(is_high_entropy_u8(&data, &mask, &header));
    }

    #[test]
    fn low_entropy_constant_data() {
        // All same value => 1 distinct value, not high entropy
        let width = 32;
        let height = 32;
        let n = width * height;
        let data = vec![42u8; n];
        let mask = BitMask::all_valid(n);
        let header = make_header_byte(width, height, 1, n);
        assert!(!is_high_entropy_u8(&data, &mask, &header));
    }

    #[test]
    fn low_entropy_constant_stride() {
        // Data with a constant stride: (i*7)%256 — direct histogram is uniform
        // but delta is highly compressible (constant delta of 7).
        // This should return false because delta-Huffman can compress it.
        let width = 128;
        let height = 128;
        let n = width * height;
        let data: Vec<u8> = (0..n).map(|i| ((i * 7) % 256) as u8).collect();
        let mask = BitMask::all_valid(n);
        let header = make_header_byte(width, height, 1, n);
        // The delta histogram should be concentrated (mostly value 7),
        // so is_high_entropy_u8 should return false.
        assert!(!is_high_entropy_u8(&data, &mask, &header));
    }

    // ---- compute_histo_for_huffman ----

    #[test]
    fn direct_histo_known_data() {
        // 4x1 image with known byte values
        let width = 4;
        let height = 1;
        let data: Vec<u8> = vec![0, 1, 1, 2];
        let mask = BitMask::all_valid(width * height);
        let header = make_header_byte(width, height, 1, width * height);
        let (histo, _delta_histo) = compute_histo_for_huffman(&data, &mask, &header);
        assert_eq!(histo[0], 1); // one 0
        assert_eq!(histo[1], 2); // two 1s
        assert_eq!(histo[2], 1); // one 2
        // All other bins should be 0
        let sum: i32 = histo.iter().sum();
        assert_eq!(sum, 4);
    }

    #[test]
    fn delta_histo_constant_stride() {
        // Constant-stride data: 0, 3, 6, 9, 12, ...
        // Delta should be mostly 3 (except the first pixel of each row).
        let width = 8;
        let height = 4;
        let n = width * height;
        let data: Vec<u8> = (0..n).map(|i| ((i * 3) % 256) as u8).collect();
        let mask = BitMask::all_valid(n);
        let header = make_header_byte(width, height, 1, n);
        let (_histo, delta_histo) = compute_histo_for_huffman(&data, &mask, &header);
        // The delta value 3 should be the most common entry in delta_histo.
        let max_bin = delta_histo
            .iter()
            .enumerate()
            .max_by_key(|(_i, c)| *c)
            .unwrap()
            .0;
        assert_eq!(max_bin, 3, "delta of 3 should be most frequent");
    }

    #[test]
    fn histo_with_mask() {
        // 4x1 image, only first 2 pixels valid
        let width = 4;
        let height = 1;
        let data: Vec<u8> = vec![10, 20, 30, 40];
        let mut mask = BitMask::new(width * height);
        mask.set_valid(0);
        mask.set_valid(1);
        let header = make_header_byte(width, height, 1, 2);
        let (histo, _) = compute_histo_for_huffman(&data, &mask, &header);
        let sum: i32 = histo.iter().sum();
        assert_eq!(sum, 2, "only 2 valid pixels should be counted");
        assert_eq!(histo[10], 1);
        assert_eq!(histo[20], 1);
    }
}
