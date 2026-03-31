use alloc::vec;
use alloc::vec::Vec;

use crate::error::{LercError, Result};

const HUFFMAN_RLE: u8 = 0x01;
const HUFFMAN_NO_ENCODING: u8 = 0x02;
const HUFFMAN_PACKBITS: u8 = 0x03;

/// Extract (decompress) a byte plane from the FPL compressed format.
pub(super) fn extract_buffer(compressed: &[u8], expected_size: usize) -> Result<Vec<u8>> {
    if compressed.is_empty() {
        return Err(LercError::InvalidData("empty FPL compressed data".into()));
    }

    let mode = compressed[0];
    let data = &compressed[1..];

    match mode {
        HUFFMAN_RLE => decode_rle(data, expected_size),
        HUFFMAN_NO_ENCODING => {
            if data.len() < expected_size {
                return Err(LercError::BufferTooSmall {
                    needed: expected_size,
                    available: data.len(),
                });
            }
            Ok(data[..expected_size].to_vec())
        }
        HUFFMAN_PACKBITS => decode_packbits(data, expected_size),
        0x00 => decode_fpl_huffman(data, expected_size),
        _ => Err(LercError::InvalidData(
            alloc::format!("unknown FPL compression mode: {mode:#x}"),
        )),
    }
}

fn decode_rle(data: &[u8], expected_size: usize) -> Result<Vec<u8>> {
    if data.len() < 5 {
        return Err(LercError::InvalidData("FPL RLE too short".into()));
    }

    let count = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let byte_val = data[4];

    if count != expected_size {
        return Err(LercError::InvalidData("FPL RLE count mismatch".into()));
    }

    Ok(vec![byte_val; count])
}

fn decode_packbits(data: &[u8], expected_size: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(expected_size);
    let mut pos = 0;

    while pos < data.len() && out.len() < expected_size {
        let header = data[pos] as i8;
        pos += 1;

        if header >= 0 {
            // Literal: copy next (header + 1) bytes
            let count = (header as usize) + 1;
            if pos + count > data.len() {
                return Err(LercError::InvalidData("FPL PackBits overflow".into()));
            }
            out.extend_from_slice(&data[pos..pos + count]);
            pos += count;
        } else if header != -128 {
            // Run: repeat next byte (-header + 1) times
            let count = (-header as usize) + 1;
            if pos >= data.len() {
                return Err(LercError::InvalidData("FPL PackBits missing byte".into()));
            }
            let byte_val = data[pos];
            pos += 1;
            out.resize(out.len() + count, byte_val);
        }
        // header == -128 is a no-op
    }

    if out.len() != expected_size {
        return Err(LercError::InvalidData("FPL PackBits size mismatch".into()));
    }

    Ok(out)
}

fn decode_fpl_huffman(data: &[u8], expected_size: usize) -> Result<Vec<u8>> {
    // FPL Huffman: simplified Huffman for byte values
    // Read code table, then decode
    use crate::huffman::HuffmanCodec;

    let mut codec = HuffmanCodec::new();
    let mut pos = 0;
    codec.read_code_table(data, &mut pos, 6)?;
    let num_bits_lut = codec.build_tree_from_codes()?;

    let mut result = Vec::with_capacity(expected_size);
    let mut byte_pos = pos;
    let mut bit_pos = 0i32;

    for _ in 0..expected_size {
        let val = codec.decode_one_value(data, &mut byte_pos, &mut bit_pos, num_bits_lut)?;
        result.push(val as u8);
    }

    Ok(result)
}

// === Encoding functions ===

const COMPRESS_HUFFMAN: u8 = 0x00;
const COMPRESS_RLE: u8 = 0x01;
const COMPRESS_NO_ENCODING: u8 = 0x02;
const COMPRESS_PACKBITS: u8 = 0x03;

/// Compress a byte plane using the best available method.
/// Returns the compressed data (including the mode byte prefix).
pub(super) fn compress_buffer(data: &[u8]) -> Vec<u8> {
    compress_buffer_with_histo(data, None)
}

/// Compress a byte plane using the best available method, optionally reusing a
/// pre-computed histogram. Returns the compressed data (including the mode byte
/// prefix).
pub(super) fn compress_buffer_with_histo(data: &[u8], precomputed_histo: Option<&[i32; 256]>) -> Vec<u8> {
    if data.is_empty() {
        return encode_raw(data);
    }

    // Build histogram (or reuse the caller's).
    let mut local_histo = [0i32; 256];
    let histo: &[i32; 256] = if let Some(h) = precomputed_histo {
        h
    } else {
        for &b in data {
            local_histo[b as usize] += 1;
        }
        &local_histo
    };

    // Count distinct values and find max frequency
    let mut distinct = 0u32;
    let mut sole_value = 0u8;
    let mut max_count = 0i32;
    for (i, &count) in histo.iter().enumerate() {
        if count > 0 {
            distinct += 1;
            sole_value = i as u8;
            if count > max_count {
                max_count = count;
            }
        }
    }

    // RLE: all same byte
    if distinct == 1 {
        let mut buf = Vec::with_capacity(6);
        buf.push(COMPRESS_RLE);
        buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
        buf.push(sole_value);
        return buf;
    }

    // Raw size is 1 + data.len(). Track best size without allocating raw eagerly.
    let raw_size = 1 + data.len();
    let mut best: Option<Vec<u8>> = None;
    let mut best_len = raw_size;

    // Compute entropy once, used for both PackBits and Huffman skip decisions.
    let entropy_bits_per_byte = compute_entropy_bits_per_byte(histo, data.len());

    // Try PackBits -- skip if data looks uniformly distributed (no runs expected).
    // PackBits helps when there are many runs of identical bytes. A rough heuristic:
    // if the most frequent value covers < 1% of the data and there are many distinct
    // values, or if entropy is high, PackBits is unlikely to help.
    let skip_packbits = (distinct > 128 && (max_count as usize) < data.len() / 100)
        || entropy_bits_per_byte > 6.0;
    if !skip_packbits {
        let packbits = encode_packbits(data);
        if packbits.len() < best_len {
            best_len = packbits.len();
            best = Some(packbits);
        }
    }

    // Skip Huffman if entropy is too high (near 8 bits/byte). The overhead of
    // the Huffman code table makes it a net loss when entropy > ~7.5 bits/byte.
    let skip_huffman = entropy_bits_per_byte > 7.5;

    // Try Huffman if we have enough data and at least 2 distinct values.
    if !skip_huffman
        && data.len() >= 4
        && distinct >= 2
        && let Some(huffman) = encode_fpl_huffman_with_histo_bounded(data, histo, best_len)
        && huffman.len() < best_len
    {
        best = Some(huffman);
    }

    best.unwrap_or_else(|| encode_raw(data))
}

/// Compute entropy in bits per byte from a histogram.
fn compute_entropy_bits_per_byte(histo: &[i32; 256], total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let n = total as f64;
    let mut entropy = 0.0;
    for &count in histo.iter() {
        if count > 0 {
            let p = count as f64 / n;
            entropy -= p * p.log2();
        }
    }
    entropy
}


fn encode_raw(data: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(1 + data.len());
    buf.push(COMPRESS_NO_ENCODING);
    buf.extend_from_slice(data);
    buf
}

fn encode_rle(data: &[u8]) -> Option<Vec<u8>> {
    if data.is_empty() {
        return None;
    }

    let first = data[0];
    if data.iter().all(|&b| b == first) {
        let mut buf = Vec::with_capacity(6);
        buf.push(COMPRESS_RLE);
        buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
        buf.push(first);
        Some(buf)
    } else {
        None
    }
}

fn encode_packbits(data: &[u8]) -> Vec<u8> {
    // Worst case: every byte is a literal, needing 1 header byte per 128 bytes
    // plus the data itself. Pre-allocate to avoid repeated growing.
    let mut buf = Vec::with_capacity(1 + data.len() + data.len() / 128 + 2);
    buf.push(COMPRESS_PACKBITS);

    let len = data.len();
    let mut i = 0;

    while i < len {
        // Check for a run of at least 3 identical bytes
        let mut run_len = 1;
        while i + run_len < len && data[i + run_len] == data[i] && run_len < 128 {
            run_len += 1;
        }

        if run_len >= 3 {
            // Write run: header = -(run_len - 1), then one byte
            buf.push((-(run_len as isize - 1)) as i8 as u8);
            buf.push(data[i]);
            i += run_len;
        } else {
            // Literal sequence
            let start = i;
            let mut lit_len = 0;

            while i + lit_len < len && lit_len < 128 {
                // Check if a run of 3+ starts here
                if i + lit_len + 2 < len
                    && data[i + lit_len] == data[i + lit_len + 1]
                    && data[i + lit_len] == data[i + lit_len + 2]
                {
                    break;
                }
                lit_len += 1;
            }

            if lit_len == 0 {
                // If we couldn't form a literal (edge case: run of exactly 2),
                // just emit them as literals
                lit_len = run_len.min(128);
            }

            buf.push((lit_len as u8).wrapping_sub(1));
            buf.extend_from_slice(&data[start..start + lit_len]);
            i = start + lit_len;
        }
    }

    buf
}

/// Huffman encode with a pre-built histogram (avoids rebuilding it).
/// `max_size` is an upper bound: if the estimated Huffman output
/// would exceed this, skip the expensive full encoding entirely.
fn encode_fpl_huffman_with_histo_bounded(
    data: &[u8],
    histo: &[i32; 256],
    max_size: usize,
) -> Option<Vec<u8>> {
    if data.len() < 4 {
        return None;
    }

    // Build Huffman codes.
    use crate::huffman::{HuffmanCodec, encode_huffman_with_codec_histo};
    let mut codec = HuffmanCodec::new();
    if !codec.compute_codes(histo) {
        return None;
    }

    // Estimate compressed size before doing the expensive full encode.
    // If the estimate already exceeds the current best, skip.
    if let Some((est_bytes, _)) = codec.compute_compressed_size(histo)
        && (est_bytes as usize + 1) >= max_size
    {
        return None;
    }

    // Encode using the codec we already built (avoids recomputing codes).
    // Pass the histogram so total_bits can be computed from it (O(256))
    // instead of scanning all data bytes (O(n)).
    let encoded = encode_huffman_with_codec_histo(&codec, data, Some(histo))?;

    let mut buf = Vec::with_capacity(1 + encoded.len());
    buf.push(COMPRESS_HUFFMAN);
    buf.extend_from_slice(&encoded);
    Some(buf)
}

fn encode_fpl_huffman(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 4 {
        return None;
    }

    let mut histo = [0i32; 256];
    for &b in data {
        histo[b as usize] += 1;
    }

    // Use usize::MAX as bound so we never skip
    encode_fpl_huffman_with_histo_bounded(data, &histo, usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_raw() {
        let data: Vec<u8> = (0..100).map(|i| (i * 7 % 256) as u8).collect();
        let compressed = encode_raw(&data);
        let decompressed = extract_buffer(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn round_trip_rle() {
        let data = vec![42u8; 200];
        let compressed = encode_rle(&data).unwrap();
        let decompressed = extract_buffer(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn round_trip_packbits() {
        let data: Vec<u8> = (0..100).map(|i| (i * 3 % 256) as u8).collect();
        let compressed = encode_packbits(&data);
        let decompressed = extract_buffer(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn round_trip_packbits_runs() {
        let mut data = Vec::new();
        data.extend_from_slice(&[42u8; 50]);
        data.extend_from_slice(&[99u8; 50]);
        data.extend_from_slice(&(0u8..50).collect::<Vec<_>>());
        let compressed = encode_packbits(&data);
        let decompressed = extract_buffer(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn round_trip_huffman() {
        let data: Vec<u8> = (0..500).map(|i| (i * 13 % 256) as u8).collect();
        if let Some(compressed) = encode_fpl_huffman(&data) {
            let decompressed = extract_buffer(&compressed, data.len()).unwrap();
            assert_eq!(decompressed, data);
        }
    }

    #[test]
    fn round_trip_compress_buffer() {
        let data: Vec<u8> = (0..500).map(|i| (i * 7 % 256) as u8).collect();
        let compressed = compress_buffer(&data);
        let decompressed = extract_buffer(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }
}
