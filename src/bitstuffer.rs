use alloc::vec;
use alloc::vec::Vec;

use crate::error::{LercError, Result};

fn num_bytes_uint(k: u32) -> usize {
    if k < 256 {
        1
    } else if k < (1 << 16) {
        2
    } else {
        4
    }
}

fn num_tail_bytes_not_needed(num_elem: u32, num_bits: u32) -> u32 {
    let num_bits_tail = ((num_elem as u64 * num_bits as u64) & 31) as u32;
    let num_bytes_tail = (num_bits_tail + 7) >> 3;
    if num_bytes_tail > 0 { 4 - num_bytes_tail } else { 0 }
}

fn encode_uint(buf: &mut Vec<u8>, k: u32, num_bytes: usize) {
    match num_bytes {
        1 => buf.push(k as u8),
        2 => buf.extend_from_slice(&(k as u16).to_le_bytes()),
        4 => buf.extend_from_slice(&k.to_le_bytes()),
        _ => unreachable!(),
    }
}

fn decode_uint(data: &[u8], pos: &mut usize, num_bytes: usize) -> Result<u32> {
    if *pos + num_bytes > data.len() {
        return Err(LercError::BufferTooSmall {
            needed: *pos + num_bytes,
            available: data.len(),
        });
    }
    let val = match num_bytes {
        1 => data[*pos] as u32,
        2 => u16::from_le_bytes(data[*pos..*pos + 2].try_into().unwrap()) as u32,
        4 => u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap()),
        _ => return Err(LercError::InvalidData("invalid uint size".into())),
    };
    *pos += num_bytes;
    Ok(val)
}

/// Bit-stuff values into a buffer (v3+ format: LSB-first packing into u32 words).
fn bit_stuff(data: &[u32], num_bits: u32) -> Vec<u8> {
    if num_bits == 0 || data.is_empty() {
        return Vec::new();
    }

    let num_elements = data.len() as u32;
    let num_uints = (num_elements as u64 * num_bits as u64).div_ceil(32) as usize;
    let num_bytes_used =
        (num_uints * 4) - num_tail_bytes_not_needed(num_elements, num_bits) as usize;

    // Allocate output bytes directly (little-endian) and pack using a 64-bit accumulator
    let mut out = vec![0u8; num_bytes_used];
    bit_stuff_into(&mut out, data, num_bits);
    out
}

/// Bit-stuff values into an already-allocated byte slice using a 64-bit accumulator.
/// The output slice must be at least as large as the bit-stuffed data.
fn bit_stuff_into(out: &mut [u8], data: &[u32], num_bits: u32) {
    if num_bits == 0 || data.is_empty() {
        return;
    }

    let mut accum: u64 = 0;
    let mut bits_in_accum: u32 = 0;
    let mut dst = 0usize;

    for &val in data {
        accum |= (val as u64) << bits_in_accum;
        bits_in_accum += num_bits;

        if bits_in_accum >= 32 {
            let word = accum as u32;
            // Write the u32 word as little-endian bytes
            out[dst..dst + 4].copy_from_slice(&word.to_le_bytes());
            dst += 4;
            accum >>= 32;
            bits_in_accum -= 32;
        }
    }

    // Flush remaining bits
    if bits_in_accum > 0 {
        let remaining_bytes = bits_in_accum.div_ceil(8) as usize;
        let word_bytes = (accum as u32).to_le_bytes();
        let n = remaining_bytes.min(out.len() - dst);
        out[dst..dst + n].copy_from_slice(&word_bytes[..n]);
    }
}

/// Bit-unstuff values from a buffer (v3+ format: LSB-first packing).
fn bit_unstuff(
    data: &[u8],
    pos: &mut usize,
    num_elements: u32,
    num_bits: u32,
) -> Result<Vec<u32>> {
    if num_elements == 0 || num_bits >= 32 {
        return Err(LercError::InvalidData("bitstuffer: invalid params".into()));
    }

    let num_uints = (num_elements as u64 * num_bits as u64).div_ceil(32) as usize;
    let num_bytes_used =
        (num_uints * 4) - num_tail_bytes_not_needed(num_elements, num_bits) as usize;

    if *pos + num_bytes_used > data.len() {
        return Err(LercError::BufferTooSmall {
            needed: *pos + num_bytes_used,
            available: data.len(),
        });
    }

    // Copy into aligned u32 buffer
    let mut src = vec![0u32; num_uints];
    let bytes = &data[*pos..*pos + num_bytes_used];
    // Copy in chunks of 4, handling the last partial chunk
    for (i, word) in src.iter_mut().enumerate() {
        let start = i * 4;
        if start + 4 <= bytes.len() {
            *word = u32::from_le_bytes(bytes[start..start + 4].try_into().unwrap());
        } else {
            let mut tmp = [0u8; 4];
            tmp[..bytes.len() - start].copy_from_slice(&bytes[start..]);
            *word = u32::from_le_bytes(tmp);
        }
    }

    let mut result = vec![0u32; num_elements as usize];
    let mut bit_pos: i32 = 0;
    let mut src_idx = 0;
    let nb = 32 - num_bits as i32;

    for dst in result.iter_mut() {
        if nb - bit_pos >= 0 {
            *dst = (src[src_idx] << (nb - bit_pos)) >> nb;
            bit_pos += num_bits as i32;
            if bit_pos == 32 {
                src_idx += 1;
                bit_pos = 0;
            }
        } else {
            *dst = src[src_idx] >> bit_pos as u32;
            src_idx += 1;
            *dst |= (src[src_idx] << (64 - num_bits as i32 - bit_pos)) >> nb;
            bit_pos -= nb;
        }
    }

    *pos += num_bytes_used;
    Ok(result)
}

/// Bit-unstuff values from a buffer (pre-v3 format: MSB-first packing).
fn bit_unstuff_before_v3(
    data: &[u8],
    pos: &mut usize,
    num_elements: u32,
    num_bits: u32,
) -> Result<Vec<u32>> {
    if num_elements == 0 || num_bits >= 32 {
        return Err(LercError::InvalidData("bitstuffer: invalid params".into()));
    }

    let num_uints = (num_elements as u64 * num_bits as u64).div_ceil(32) as usize;
    let ntbnn = num_tail_bytes_not_needed(num_elements, num_bits) as usize;

    let num_bytes_to_copy = (num_elements as u64 * num_bits as u64).div_ceil(8) as usize;

    if *pos + num_bytes_to_copy > data.len() {
        return Err(LercError::BufferTooSmall {
            needed: *pos + num_bytes_to_copy,
            available: data.len(),
        });
    }

    // Copy into aligned u32 buffer
    let mut src = vec![0u32; num_uints];
    let bytes = &data[*pos..*pos + num_bytes_to_copy];
    for (i, word) in src.iter_mut().enumerate() {
        let start = i * 4;
        if start + 4 <= bytes.len() {
            *word = u32::from_le_bytes(bytes[start..start + 4].try_into().unwrap());
        } else {
            let mut tmp = [0u8; 4];
            tmp[..bytes.len() - start].copy_from_slice(&bytes[start..]);
            *word = u32::from_le_bytes(tmp);
        }
    }

    // Shift the last uint left to compensate for unused tail bytes
    if let Some(last) = src.last_mut() {
        for _ in 0..ntbnn {
            *last <<= 8;
        }
    }

    let mut result = vec![0u32; num_elements as usize];
    let mut bit_pos: i32 = 0;
    let mut src_idx = 0;

    for dst in result.iter_mut() {
        if 32 - bit_pos >= num_bits as i32 {
            let val = src[src_idx];
            let n = val << bit_pos as u32;
            *dst = n >> (32 - num_bits);
            bit_pos += num_bits as i32;
            if bit_pos == 32 {
                bit_pos = 0;
                src_idx += 1;
            }
        } else {
            let val1 = src[src_idx];
            src_idx += 1;
            let n = val1 << bit_pos as u32;
            *dst = n >> (32 - num_bits);
            bit_pos -= 32 - num_bits as i32;
            let val2 = src[src_idx];
            *dst |= val2 >> (32 - bit_pos);
        }
    }

    *pos += num_bytes_to_copy;
    Ok(result)
}

/// Encode an array of unsigned ints using simple mode (no LUT).
pub fn encode_simple(data: &[u32]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }

    let max_elem = data.iter().copied().max().unwrap_or(0);
    let mut num_bits: u32 = 0;
    while num_bits < 32 && (max_elem >> num_bits) != 0 {
        num_bits += 1;
    }

    let num_elements = data.len() as u32;
    let n = num_bytes_uint(num_elements);
    let bits67 = if n == 4 { 0u8 } else { (3 - n) as u8 };

    let mut header_byte = num_bits as u8;
    header_byte |= bits67 << 6;
    // bit 5 = 0 for simple mode

    let mut buf = Vec::new();
    buf.push(header_byte);
    encode_uint(&mut buf, num_elements, n);

    if num_bits > 0 {
        let stuffed = bit_stuff(data, num_bits);
        buf.extend_from_slice(&stuffed);
    }

    buf
}

/// Pre-computed LUT encoding info from `should_use_lut`, avoiding recomputation in `encode_lut`.
pub struct LutInfo {
    /// Distinct values excluding the minimum (which maps to index 0)
    pub lut_vec: Vec<u32>,
    /// Per-element LUT index
    pub index_vec: Vec<u32>,
    /// Number of bits needed for the max value in lut_vec
    pub num_bits: u32,
    /// Number of bits needed for lut indices
    pub n_bits_lut: u32,
}

/// Encode an array of unsigned ints using LUT mode from pre-computed info.
pub fn encode_lut(info: &LutInfo, num_elem: u32) -> Vec<u8> {
    if num_elem == 0 {
        return Vec::new();
    }

    let n = num_bytes_uint(num_elem);
    let bits67 = if n == 4 { 0u8 } else { (3 - n) as u8 };

    let mut header_byte = info.num_bits as u8;
    header_byte |= bits67 << 6;
    header_byte |= 1 << 5; // bit 5 = 1 for LUT mode

    // Pre-compute output size for a single allocation
    let n_lut = info.lut_vec.len() as u32;
    let lut_bytes = if info.num_bits > 0 {
        let num_uints = (n_lut as u64 * info.num_bits as u64).div_ceil(32) as usize;
        (num_uints * 4) - num_tail_bytes_not_needed(n_lut, info.num_bits) as usize
    } else {
        0
    };
    let idx_bytes = if info.n_bits_lut > 0 {
        let num_uints = (num_elem as u64 * info.n_bits_lut as u64).div_ceil(32) as usize;
        (num_uints * 4) - num_tail_bytes_not_needed(num_elem, info.n_bits_lut) as usize
    } else {
        0
    };
    let total_size = 1 + n + 1 + lut_bytes + idx_bytes;

    let mut buf = Vec::with_capacity(total_size);
    buf.push(header_byte);
    encode_uint(&mut buf, num_elem, n);
    buf.push((n_lut + 1) as u8); // size including the 0

    if info.num_bits > 0 {
        let stuffed_lut = bit_stuff(&info.lut_vec, info.num_bits);
        buf.extend_from_slice(&stuffed_lut);
    }

    if info.n_bits_lut > 0 {
        let stuffed_idx = bit_stuff(&info.index_vec, info.n_bits_lut);
        buf.extend_from_slice(&stuffed_idx);
    }

    buf
}

/// Decode a bit-stuffed array (auto-detects simple vs LUT mode).
pub fn decode(
    data: &[u8],
    pos: &mut usize,
    max_element_count: usize,
    lerc2_version: i32,
) -> Result<Vec<u32>> {
    if *pos >= data.len() {
        return Err(LercError::BufferTooSmall {
            needed: *pos + 1,
            available: data.len(),
        });
    }

    let header_byte = data[*pos];
    *pos += 1;

    let bits67 = header_byte >> 6;
    let nb = if bits67 == 0 { 4 } else { (3 - bits67) as usize };
    let do_lut = (header_byte & (1 << 5)) != 0;
    let num_bits = (header_byte & 31) as u32;

    let num_elements = decode_uint(data, pos, nb)?;
    if num_elements as usize > max_element_count {
        return Err(LercError::InvalidData("too many elements".into()));
    }

    if !do_lut {
        if num_bits > 0 {
            if lerc2_version >= 3 {
                bit_unstuff(data, pos, num_elements, num_bits)
            } else {
                bit_unstuff_before_v3(data, pos, num_elements, num_bits)
            }
        } else {
            // numBits == 0 means all values are 0
            Ok(vec![0u32; num_elements as usize])
        }
    } else {
        if num_bits == 0 {
            return Err(LercError::InvalidData("LUT mode with 0 bits".into()));
        }

        if *pos >= data.len() {
            return Err(LercError::BufferTooSmall {
                needed: *pos + 1,
                available: data.len(),
            });
        }

        let n_lut_byte = data[*pos];
        *pos += 1;
        let n_lut = (n_lut_byte as u32).wrapping_sub(1);

        // Unstuff LUT values (without the 0)
        let mut lut_vec = if lerc2_version >= 3 {
            bit_unstuff(data, pos, n_lut, num_bits)?
        } else {
            bit_unstuff_before_v3(data, pos, n_lut, num_bits)?
        };

        let mut n_bits_lut: u32 = 0;
        while (n_lut >> n_bits_lut) != 0 {
            n_bits_lut += 1;
        }
        if n_bits_lut == 0 {
            return Err(LercError::InvalidData("LUT with 0 index bits".into()));
        }

        // Unstuff indexes
        let mut result = if lerc2_version >= 3 {
            bit_unstuff(data, pos, num_elements, n_bits_lut)?
        } else {
            bit_unstuff_before_v3(data, pos, num_elements, n_bits_lut)?
        };

        // Replace indexes by values
        lut_vec.insert(0, 0); // put back the 0
        for val in result.iter_mut() {
            let idx = *val as usize;
            if idx >= lut_vec.len() {
                return Err(LercError::InvalidData("LUT index out of bounds".into()));
            }
            *val = lut_vec[idx];
        }

        Ok(result)
    }
}

/// Compute the number of bits needed to represent `max_elem`.
pub fn num_bits_needed(max_elem: u32) -> u32 {
    let mut num_bits: u32 = 0;
    while num_bits < 32 && (max_elem >> num_bits) != 0 {
        num_bits += 1;
    }
    num_bits
}

/// Decide whether LUT encoding is beneficial, and if so build the LUT info directly.
///
/// For small domains (max_elem < 256, common for u8 data), uses a histogram-based
/// approach that avoids sorting entirely. For larger domains, uses a bitset to count
/// distinct values first, then builds the LUT via counting sort when possible.
///
/// Returns `Some(LutInfo)` with pre-computed LUT data ready for `encode_lut`,
/// or `None` if simple encoding would be smaller.
pub fn should_use_lut(data: &[u32], max_elem: u32) -> Option<LutInfo> {
    if data.len() < 2 {
        return None;
    }

    let num_elem = data.len() as u32;
    let num_bits = num_bits_needed(max_elem);

    if max_elem < 256 {
        // Fast path for u8-range data: histogram-based, no sorting needed
        should_use_lut_small(data, num_elem, num_bits, max_elem)
    } else if max_elem < 65536 {
        // Medium domain: bitset count then counting sort
        should_use_lut_medium(data, num_elem, num_bits, max_elem)
    } else {
        // Large domain: sample-based estimate
        should_use_lut_large(data, num_elem, num_bits)
    }
}

/// Fast LUT construction for u8-range data (max_elem < 256).
/// Uses a 256-entry histogram to count distinct values AND build the value-to-index
/// mapping in a single pass, completely avoiding any sorting.
fn should_use_lut_small(data: &[u32], num_elem: u32, num_bits: u32, max_elem: u32) -> Option<LutInfo> {
    // Build histogram
    let mut counts = [0u32; 256];
    for &val in data {
        counts[val as usize] += 1;
    }

    // Count distinct values
    let n_distinct: u32 = counts.iter().filter(|&&c| c > 0).count() as u32;

    if !(2..=255).contains(&n_distinct) {
        return None;
    }

    let n_lut = n_distinct - 1;

    // Estimate size before building the full LUT
    let num_bytes_simple =
        1 + num_bytes_uint(num_elem) as u32 + ((num_elem * num_bits + 7) >> 3);

    let mut n_bits_lut = 0u32;
    while (n_lut >> n_bits_lut) != 0 {
        n_bits_lut += 1;
    }

    let num_bytes_lut =
        1 + num_bytes_uint(num_elem) as u32 + 1 + ((n_lut * num_bits + 7) >> 3)
            + ((num_elem * n_bits_lut + 7) >> 3);

    if num_bytes_lut >= num_bytes_simple {
        return None;
    }

    // Build value-to-lut-index mapping: iterate values in order, assign indices.
    // Index 0 = minimum value, then 1, 2, ... for subsequent distinct values.
    let mut value_to_index = [0u32; 256];
    let mut lut_vec = Vec::with_capacity(n_lut as usize);
    let mut idx = 0u32;
    for v in 0..=max_elem as usize {
        if counts[v] > 0 {
            value_to_index[v] = idx;
            if idx > 0 {
                lut_vec.push(v as u32);
            }
            idx += 1;
        }
    }

    // Build index_vec by looking up each data value
    let mut index_vec = vec![0u32; num_elem as usize];
    for (i, &val) in data.iter().enumerate() {
        index_vec[i] = value_to_index[val as usize];
    }

    Some(LutInfo {
        lut_vec,
        index_vec,
        num_bits,
        n_bits_lut,
    })
}

/// LUT construction for medium domains (256..65536).
fn should_use_lut_medium(data: &[u32], num_elem: u32, num_bits: u32, max_elem: u32) -> Option<LutInfo> {
    // Use a bitset to count distinct values
    let num_words = (max_elem as usize / 64) + 1;
    let mut bitset = vec![0u64; num_words];
    for &val in data {
        let word = val as usize / 64;
        let bit = val as usize % 64;
        bitset[word] |= 1u64 << bit;
    }
    let n_distinct: u32 = bitset.iter().map(|w| w.count_ones()).sum();

    if !(2..=255).contains(&n_distinct) {
        return None;
    }

    let n_lut = n_distinct - 1;

    let num_bytes_simple =
        1 + num_bytes_uint(num_elem) as u32 + ((num_elem * num_bits + 7) >> 3);

    let mut n_bits_lut = 0u32;
    while (n_lut >> n_bits_lut) != 0 {
        n_bits_lut += 1;
    }

    let num_bytes_lut =
        1 + num_bytes_uint(num_elem) as u32 + 1 + ((n_lut * num_bits + 7) >> 3)
            + ((num_elem * n_bits_lut + 7) >> 3);

    if num_bytes_lut >= num_bytes_simple {
        return None;
    }

    // Build value-to-index map using the bitset
    // We need a map from value -> lut index. Use a Vec since domain is < 65536.
    let mut value_to_index = vec![0u32; max_elem as usize + 1];
    let mut lut_vec = Vec::with_capacity(n_lut as usize);
    let mut idx = 0u32;
    for (v, slot) in value_to_index.iter_mut().enumerate() {
        let word = v / 64;
        let bit = v % 64;
        if (bitset[word] >> bit) & 1 != 0 {
            *slot = idx;
            if idx > 0 {
                lut_vec.push(v as u32);
            }
            idx += 1;
        }
    }

    let mut index_vec = vec![0u32; num_elem as usize];
    for (i, &val) in data.iter().enumerate() {
        index_vec[i] = value_to_index[val as usize];
    }

    Some(LutInfo {
        lut_vec,
        index_vec,
        num_bits,
        n_bits_lut,
    })
}

/// LUT construction for large domains (>= 65536).
fn should_use_lut_large(data: &[u32], num_elem: u32, num_bits: u32) -> Option<LutInfo> {
    // Sample to estimate distinct count
    let step = (data.len() / 512).max(1);
    let mut seen = Vec::with_capacity(256);
    for (i, &val) in data.iter().enumerate() {
        if i % step != 0 {
            continue;
        }
        if !seen.contains(&val) {
            seen.push(val);
            if seen.len() > 255 {
                return None; // Exceeds LUT threshold
            }
        }
    }
    let n_distinct_est = seen.len() as u32;

    if !(2..=255).contains(&n_distinct_est) {
        return None;
    }

    // Need to do the full sort for large domains
    let mut sorted: Vec<(u32, u32)> = data.iter().copied().zip(0u32..).collect();
    sorted.sort_unstable_by_key(|&(val, _)| val);

    // Build LUT from sorted data
    let mut lut_vec: Vec<u32> = Vec::new();
    let mut index_vec = vec![0u32; num_elem as usize];
    let mut index_lut = 0u32;

    for i in 1..sorted.len() {
        let prev = sorted[i - 1].0;
        index_vec[sorted[i - 1].1 as usize] = index_lut;
        if sorted[i].0 != prev {
            lut_vec.push(sorted[i].0);
            index_lut += 1;
        }
    }
    index_vec[sorted[num_elem as usize - 1].1 as usize] = index_lut;

    let n_lut = lut_vec.len() as u32;

    // Check if LUT is beneficial
    let num_bytes_simple =
        1 + num_bytes_uint(num_elem) as u32 + ((num_elem * num_bits + 7) >> 3);

    let mut n_bits_lut = 0u32;
    while (n_lut >> n_bits_lut) != 0 {
        n_bits_lut += 1;
    }

    let num_bytes_lut =
        1 + num_bytes_uint(num_elem) as u32 + 1 + ((n_lut * num_bits + 7) >> 3)
            + ((num_elem * n_bits_lut + 7) >> 3);

    if num_bytes_lut >= num_bytes_simple {
        return None;
    }

    let max_lut_elem = *lut_vec.last().unwrap_or(&0);
    let actual_num_bits = num_bits_needed(max_lut_elem);
    // Use the original num_bits (based on max_elem of the full data) for consistency
    let _ = actual_num_bits;

    Some(LutInfo {
        lut_vec,
        index_vec,
        num_bits,
        n_bits_lut,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_simple() {
        let data = vec![0, 1, 2, 3, 100, 255, 42, 7];
        let encoded = encode_simple(&data);
        let mut pos = 0;
        let decoded = decode(&encoded, &mut pos, 100, 6).unwrap();
        assert_eq!(decoded, data);
        assert_eq!(pos, encoded.len());
    }

    #[test]
    fn round_trip_simple_zeros() {
        let data = vec![0, 0, 0, 0, 0];
        let encoded = encode_simple(&data);
        let mut pos = 0;
        let decoded = decode(&encoded, &mut pos, 100, 6).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn round_trip_simple_large() {
        let data: Vec<u32> = (0..1000).collect();
        let encoded = encode_simple(&data);
        let mut pos = 0;
        let decoded = decode(&encoded, &mut pos, 2000, 6).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn round_trip_lut() {
        // Data with few distinct values — good for LUT
        let data = vec![0, 100, 200, 0, 100, 200, 0, 100, 200, 0];
        let max_elem = *data.iter().max().unwrap();
        let info = should_use_lut(&data, max_elem).expect("should use LUT");

        let encoded = encode_lut(&info, data.len() as u32);
        let mut pos = 0;
        let decoded = decode(&encoded, &mut pos, 100, 6).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn num_bits() {
        assert_eq!(num_bits_needed(0), 0);
        assert_eq!(num_bits_needed(1), 1);
        assert_eq!(num_bits_needed(2), 2);
        assert_eq!(num_bits_needed(3), 2);
        assert_eq!(num_bits_needed(255), 8);
        assert_eq!(num_bits_needed(256), 9);
    }

    #[test]
    fn bit_stuff_round_trip_various_widths() {
        for num_bits in 1..32 {
            let max_val = (1u32 << num_bits) - 1;
            let data: Vec<u32> = (0..50).map(|i| i % (max_val + 1)).collect();
            let stuffed = bit_stuff(&data, num_bits);
            let mut pos = 0;
            // Wrap in a full decode context
            let unstuffed =
                bit_unstuff(&stuffed, &mut pos, data.len() as u32, num_bits).unwrap();
            assert_eq!(unstuffed, data, "failed for num_bits={num_bits}");
        }
    }
}
