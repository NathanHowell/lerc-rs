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
    let mut buf = vec![0u32; num_uints];

    let mut bit_pos: u32 = 0;
    let mut dst_idx = 0;

    for &val in data {
        if 32 - bit_pos >= num_bits {
            buf[dst_idx] |= val << bit_pos;
            bit_pos += num_bits;
            if bit_pos == 32 {
                dst_idx += 1;
                bit_pos = 0;
            }
        } else {
            buf[dst_idx] |= val << bit_pos;
            dst_idx += 1;
            buf[dst_idx] |= val >> (32 - bit_pos);
            bit_pos = bit_pos + num_bits - 32;
        }
    }

    let num_bytes_used =
        (num_uints * 4) - num_tail_bytes_not_needed(num_elements, num_bits) as usize;

    let mut out = Vec::with_capacity(num_bytes_used);
    for &word in &buf {
        out.extend_from_slice(&word.to_le_bytes());
    }
    out.truncate(num_bytes_used);
    out
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

/// Encode an array of unsigned ints using LUT mode.
///
/// `sorted_data` is a list of (value, original_index) pairs sorted by value.
pub fn encode_lut(sorted_data: &[(u32, u32)]) -> Vec<u8> {
    if sorted_data.is_empty() {
        return Vec::new();
    }

    let num_elem = sorted_data.len() as u32;

    // Build LUT (omitting the 0 which corresponds to min)
    let mut lut_vec: Vec<u32> = Vec::new();
    let mut index_vec = vec![0u32; num_elem as usize];
    let mut index_lut = 0u32;

    for i in 1..sorted_data.len() {
        let prev = sorted_data[i - 1].0;
        index_vec[sorted_data[i - 1].1 as usize] = index_lut;
        if sorted_data[i].0 != prev {
            lut_vec.push(sorted_data[i].0);
            index_lut += 1;
        }
    }
    index_vec[sorted_data[num_elem as usize - 1].1 as usize] = index_lut;

    let max_elem = *lut_vec.last().unwrap();
    let mut num_bits: u32 = 0;
    while num_bits < 32 && (max_elem >> num_bits) != 0 {
        num_bits += 1;
    }

    let n = num_bytes_uint(num_elem);
    let bits67 = if n == 4 { 0u8 } else { (3 - n) as u8 };

    let mut header_byte = num_bits as u8;
    header_byte |= bits67 << 6;
    header_byte |= 1 << 5; // bit 5 = 1 for LUT mode

    let mut buf = Vec::new();
    buf.push(header_byte);
    encode_uint(&mut buf, num_elem, n);

    let n_lut = lut_vec.len() as u32;
    buf.push((n_lut + 1) as u8); // size including the 0

    let stuffed_lut = bit_stuff(&lut_vec, num_bits);
    buf.extend_from_slice(&stuffed_lut);

    let mut n_bits_lut: u32 = 0;
    while (n_lut >> n_bits_lut) != 0 {
        n_bits_lut += 1;
    }

    let stuffed_idx = bit_stuff(&index_vec, n_bits_lut);
    buf.extend_from_slice(&stuffed_idx);

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

/// Decide whether LUT encoding is beneficial, and return the sorted data if so.
pub fn should_use_lut(data: &[u32]) -> Option<Vec<(u32, u32)>> {
    if data.len() < 2 {
        return None;
    }

    let mut sorted: Vec<(u32, u32)> = data.iter().copied().zip(0u32..).collect();
    sorted.sort_by_key(|&(val, _)| val);

    let max_elem = sorted.last().unwrap().0;
    let num_elem = sorted.len() as u32;

    let num_bits = num_bits_needed(max_elem);
    let num_bytes_simple =
        1 + num_bytes_uint(num_elem) as u32 + ((num_elem * num_bits + 7) >> 3);

    // Count distinct values (excluding the min which maps to 0)
    let mut n_lut = 0u32;
    for i in 1..sorted.len() {
        if sorted[i].0 != sorted[i - 1].0 {
            n_lut += 1;
        }
    }

    if !(1..255).contains(&n_lut) {
        return None;
    }

    let mut n_bits_lut = 0u32;
    while (n_lut >> n_bits_lut) != 0 {
        n_bits_lut += 1;
    }

    let num_bytes_lut =
        1 + num_bytes_uint(num_elem) as u32 + 1 + ((n_lut * num_bits + 7) >> 3)
            + ((num_elem * n_bits_lut + 7) >> 3);

    if num_bytes_lut < num_bytes_simple {
        Some(sorted)
    } else {
        None
    }
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
        let mut sorted: Vec<(u32, u32)> = data.iter().copied().zip(0u32..).collect();
        sorted.sort_by_key(|&(val, _)| val);

        let encoded = encode_lut(&sorted);
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
