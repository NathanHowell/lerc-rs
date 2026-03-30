use alloc::collections::BinaryHeap;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use crate::bitstuffer;
use crate::error::{LercError, Result};

const MAX_HISTO_SIZE: usize = 1 << 15;
const MAX_NUM_BITS_LUT: i32 = 12;

/// A Huffman code entry: (code_length, code_bits).
type CodeEntry = (u16, u32);

/// Decode LUT entry: (code_length, symbol). -1 means not in LUT.
type DecodeLutEntry = (i16, i16);

pub struct HuffmanCodec {
    code_table: Vec<CodeEntry>,
    decode_lut: Vec<DecodeLutEntry>,
    num_bits_to_skip_in_tree: i32,
    tree_root: Option<Box<TreeNode>>,
}

struct TreeNode {
    value: i16,
    child0: Option<Box<TreeNode>>,
    child1: Option<Box<TreeNode>>,
}

impl TreeNode {
    fn leaf(value: i16) -> Self {
        Self {
            value,
            child0: None,
            child1: None,
        }
    }

    fn internal(c0: Box<TreeNode>, c1: Box<TreeNode>) -> Self {
        Self {
            value: -1,
            child0: Some(c0),
            child1: Some(c1),
        }
    }

    fn tree_to_lut(&self, num_bits: u16, bits: u32, table: &mut [CodeEntry]) -> bool {
        if let (Some(c0), Some(c1)) = (&self.child0, &self.child1) {
            if num_bits == 32 {
                return false;
            }
            if !c0.tree_to_lut(num_bits + 1, bits << 1, table) {
                return false;
            }
            if !c1.tree_to_lut(num_bits + 1, (bits << 1) + 1, table) {
                return false;
            }
        } else {
            table[self.value as usize] = (num_bits, bits);
        }
        true
    }
}

fn get_index_wrap_around(i: i32, size: i32) -> i32 {
    i - if i < size { 0 } else { size }
}

impl HuffmanCodec {
    pub fn new() -> Self {
        Self {
            code_table: Vec::new(),
            decode_lut: Vec::new(),
            num_bits_to_skip_in_tree: 0,
            tree_root: None,
        }
    }

    /// Build Huffman codes from a histogram.
    pub fn compute_codes(&mut self, histo: &[i32]) -> bool {
        if histo.is_empty() || histo.len() >= MAX_HISTO_SIZE {
            return false;
        }

        // Use a BinaryHeap (min-heap via Reverse) of (weight, index)
        // Weight is negative count so min-heap gives us smallest counts first
        // Collect leaf nodes: (negative_weight, node)
        let mut nodes: Vec<Option<Box<TreeNode>>> = Vec::new();
        let mut heap: BinaryHeap<Reverse<(i32, usize)>> = BinaryHeap::new();

        let size = histo.len();
        for (i, &count) in histo.iter().enumerate() {
            if count > 0 {
                let idx = nodes.len();
                nodes.push(Some(Box::new(TreeNode::leaf(i as i16))));
                heap.push(Reverse((count, idx)));
            }
        }

        if heap.len() < 2 {
            return false;
        }

        // Build tree
        while heap.len() > 1 {
            let Reverse((w0, i0)) = heap.pop().unwrap();
            let Reverse((w1, i1)) = heap.pop().unwrap();
            let c0 = nodes[i0].take().unwrap();
            let c1 = nodes[i1].take().unwrap();
            let idx = nodes.len();
            nodes.push(Some(Box::new(TreeNode::internal(c0, c1))));
            heap.push(Reverse((w0 + w1, idx)));
        }

        let Reverse((_, root_idx)) = heap.pop().unwrap();
        let root = nodes[root_idx].take().unwrap();

        self.code_table = vec![(0u16, 0u32); size];
        if !root.tree_to_lut(0, 0, &mut self.code_table) {
            return false;
        }

        self.convert_codes_to_canonical()
    }

    /// Convert tree-derived codes to canonical form.
    fn convert_codes_to_canonical(&mut self) -> bool {
        let table_size = self.code_table.len() as u32;

        // Sort by (codeLength * tableSize - index) descending
        let mut sort_vec: Vec<(i32, u32)> = Vec::with_capacity(table_size as usize);
        for (i, entry) in self.code_table.iter().enumerate() {
            if entry.0 > 0 {
                sort_vec.push((
                    entry.0 as i32 * table_size as i32 - i as i32,
                    i as u32,
                ));
            } else {
                sort_vec.push((0, i as u32));
            }
        }
        sort_vec.sort_by(|a, b| b.0.cmp(&a.0));

        // Assign canonical codes
        let mut i = 0usize;
        if sort_vec[0].0 <= 0 {
            return false;
        }

        let index = sort_vec[0].1 as usize;
        let mut code_len = self.code_table[index].0;
        let mut code_canonical: u32 = 0;

        while i < sort_vec.len() && sort_vec[i].0 > 0 {
            let idx = sort_vec[i].1 as usize;
            let delta = code_len - self.code_table[idx].0;
            code_canonical >>= delta;
            code_len -= delta;
            self.code_table[idx].1 = code_canonical;
            code_canonical += 1;
            i += 1;
        }

        true
    }

    /// Get the range [i0, i1) and max code length, with possible wrap-around.
    fn get_range(&self) -> Option<(i32, i32, i32)> {
        if self.code_table.is_empty() || self.code_table.len() >= MAX_HISTO_SIZE {
            return None;
        }

        let size = self.code_table.len() as i32;

        // Find first and last non-zero entries
        let mut first = 0;
        while first < size && self.code_table[first as usize].0 == 0 {
            first += 1;
        }
        let mut last = size - 1;
        while last >= 0 && self.code_table[last as usize].0 == 0 {
            last -= 1;
        }
        let i0_simple = first;
        let i1_simple = last + 1;

        if i1_simple <= i0_simple {
            return None;
        }

        // Find largest stretch of zeros for possible wrap-around optimization
        let mut best_start = 0i32;
        let mut best_len = 0i32;
        let mut j = 0;
        while j < size {
            while j < size && self.code_table[j as usize].0 > 0 {
                j += 1;
            }
            let k0 = j;
            while j < size && self.code_table[j as usize].0 == 0 {
                j += 1;
            }
            let gap = j - k0;
            if gap > best_len {
                best_start = k0;
                best_len = gap;
            }
        }

        let (i0, i1) = if size - best_len < i1_simple - i0_simple {
            (best_start + best_len, best_start + size)
        } else {
            (i0_simple, i1_simple)
        };

        if i1 <= i0 {
            return None;
        }

        let mut max_len = 0i32;
        for i in i0..i1 {
            let k = get_index_wrap_around(i, size) as usize;
            max_len = max_len.max(self.code_table[k].0 as i32);
        }

        if max_len <= 0 || max_len > 32 {
            return None;
        }

        Some((i0, i1, max_len))
    }

    /// Write the code table to a byte buffer.
    pub fn write_code_table(&self, lerc2_version: i32) -> Result<Vec<u8>> {
        let (i0, i1, _max_len) =
            self.get_range().ok_or(LercError::InvalidData("empty code table".into()))?;

        let size = self.code_table.len() as i32;
        let mut data_vec: Vec<u32> = Vec::with_capacity((i1 - i0) as usize);
        for i in i0..i1 {
            let k = get_index_wrap_around(i, size) as usize;
            data_vec.push(self.code_table[k].0 as u32);
        }

        let mut buf = Vec::new();

        // Header: 4 ints
        buf.extend_from_slice(&4i32.to_le_bytes()); // huffman version
        buf.extend_from_slice(&size.to_le_bytes());
        buf.extend_from_slice(&i0.to_le_bytes());
        buf.extend_from_slice(&i1.to_le_bytes());

        // Bit-stuffed code lengths
        let _ = lerc2_version; // always use v3+ encoding for writing
        let encoded_lengths = bitstuffer::encode_simple(&data_vec);
        buf.extend_from_slice(&encoded_lengths);

        // Bit-stuffed codes (MSB-first via PushValue)
        self.bit_stuff_codes(&mut buf, i0, i1);

        Ok(buf)
    }

    /// Read the code table from a byte buffer.
    pub fn read_code_table(
        &mut self,
        data: &[u8],
        pos: &mut usize,
        lerc2_version: i32,
    ) -> Result<()> {
        if *pos + 16 > data.len() {
            return Err(LercError::BufferTooSmall {
                needed: *pos + 16,
                available: data.len(),
            });
        }

        let version = i32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
        let size = i32::from_le_bytes(data[*pos + 4..*pos + 8].try_into().unwrap());
        let i0 = i32::from_le_bytes(data[*pos + 8..*pos + 12].try_into().unwrap());
        let i1 = i32::from_le_bytes(data[*pos + 12..*pos + 16].try_into().unwrap());
        *pos += 16;

        if version < 2 {
            return Err(LercError::InvalidData("huffman version too old".into()));
        }
        if i0 >= i1 || i0 < 0 || size < 0 || size as usize > MAX_HISTO_SIZE {
            return Err(LercError::InvalidData("invalid code table range".into()));
        }

        // Decode code lengths
        let data_vec =
            bitstuffer::decode(data, pos, (i1 - i0) as usize, lerc2_version)?;

        self.code_table = vec![(0u16, 0u32); size as usize];
        for i in i0..i1 {
            let k = get_index_wrap_around(i, size) as usize;
            self.code_table[k].0 = data_vec[(i - i0) as usize] as u16;
        }

        // Decode codes (MSB-first bit-unstuffing)
        self.bit_unstuff_codes(data, pos, i0, i1)?;

        Ok(())
    }

    /// Bit-stuff the Huffman codes using MSB-first PushValue encoding.
    fn bit_stuff_codes(&self, buf: &mut Vec<u8>, i0: i32, i1: i32) {
        let size = self.code_table.len() as i32;
        let mut bit_pos: i32 = 0;

        // Calculate total bits needed
        let mut total_bits = 0u64;
        for i in i0..i1 {
            let k = get_index_wrap_around(i, size) as usize;
            total_bits += self.code_table[k].0 as u64;
        }

        let num_uints = total_bits.div_ceil(32) as usize;
        let start = buf.len();
        buf.resize(start + num_uints * 4 + 4, 0); // +4 for safety

        for i in i0..i1 {
            let k = get_index_wrap_around(i, size) as usize;
            let len = self.code_table[k].0 as i32;
            if len > 0 {
                let val = self.code_table[k].1;
                push_value(&mut buf[start..], &mut bit_pos, val, len);
            }
        }

        let bytes_used = if bit_pos > 0 {
            // Round up to next uint boundary
            let current_uint_offset = ((bit_pos - 1) / 32) as usize;
            (current_uint_offset + 1) * 4
        } else if total_bits > 0 {
            num_uints * 4
        } else {
            0
        };

        buf.truncate(start + bytes_used);
    }

    /// Bit-unstuff codes using MSB-first format.
    fn bit_unstuff_codes(
        &mut self,
        data: &[u8],
        pos: &mut usize,
        i0: i32,
        i1: i32,
    ) -> Result<()> {
        let size = self.code_table.len() as i32;
        let start_pos = *pos;
        let mut bit_pos: i32 = 0;

        for i in i0..i1 {
            let k = get_index_wrap_around(i, size) as usize;
            let len = self.code_table[k].0 as i32;
            if len > 0 {
                if len > 32 {
                    return Err(LercError::InvalidData("code length > 32".into()));
                }

                let byte_offset = start_pos + (bit_pos / 32) as usize * 4;
                if byte_offset + 4 > data.len() {
                    return Err(LercError::BufferTooSmall {
                        needed: byte_offset + 4,
                        available: data.len(),
                    });
                }

                let temp = u32::from_le_bytes(
                    data[byte_offset..byte_offset + 4].try_into().unwrap(),
                );
                let local_bit = bit_pos & 31;
                let mut code = (temp << local_bit) >> (32 - len);

                if 32 - local_bit < len {
                    // Spans two u32s
                    let new_byte_offset = byte_offset + 4;
                    if new_byte_offset + 4 > data.len() {
                        return Err(LercError::BufferTooSmall {
                            needed: new_byte_offset + 4,
                            available: data.len(),
                        });
                    }
                    let temp2 = u32::from_le_bytes(
                        data[new_byte_offset..new_byte_offset + 4]
                            .try_into()
                            .unwrap(),
                    );
                    let new_local_bit = (local_bit + len - 32) as u32;
                    code |= temp2 >> (32 - new_local_bit);
                    bit_pos = (((byte_offset - start_pos) / 4 + 1) * 32 + new_local_bit as usize) as i32;
                } else {
                    bit_pos += len;
                    if bit_pos & 31 == 0 && local_bit + len == 32 {
                        // Already at boundary
                    }
                }

                self.code_table[k].1 = code;
            }
        }

        // Advance pos past the consumed data
        let total_uint_boundary = if (bit_pos & 31) > 0 {
            (bit_pos / 32 + 1) as usize * 4
        } else {
            (bit_pos / 32) as usize * 4
        };
        *pos = start_pos + total_uint_boundary;

        Ok(())
    }

    /// Build the decode LUT and optional tree from the code table.
    pub fn build_tree_from_codes(&mut self) -> Result<i32> {
        let (i0, i1, max_len) =
            self.get_range().ok_or(LercError::InvalidData("empty code table".into()))?;

        let size = self.code_table.len() as i32;
        let need_tree = max_len > MAX_NUM_BITS_LUT;
        let num_bits_lut = max_len.min(MAX_NUM_BITS_LUT);
        let size_lut = 1usize << num_bits_lut;

        self.decode_lut = vec![(-1i16, -1i16); size_lut];

        let mut min_num_zero_bits: i32 = 32;

        for i in i0..i1 {
            let k = get_index_wrap_around(i, size) as usize;
            let len = self.code_table[k].0 as i32;
            if len == 0 {
                continue;
            }
            let code = self.code_table[k].1;

            if len <= num_bits_lut {
                let shifted = code << (num_bits_lut - len);
                let num_entries = 1u32 << (num_bits_lut - len);
                let entry = (len as i16, k as i16);
                for j in 0..num_entries {
                    self.decode_lut[(shifted | j) as usize] = entry;
                }
            } else {
                // Count leading zeros for tree skip optimization
                let mut shift = 1;
                let mut c = code;
                while c > 1 {
                    c >>= 1;
                    shift += 1;
                }
                // leading zeros = len - shift (where shift is the number of significant bits)
                min_num_zero_bits = min_num_zero_bits.min(len - shift);
            }
        }

        self.num_bits_to_skip_in_tree = if need_tree { min_num_zero_bits } else { 0 };

        if !need_tree {
            self.tree_root = None;
            return Ok(num_bits_lut);
        }

        // Build tree for codes longer than LUT
        let mut root = Box::new(TreeNode::leaf(-1));

        for i in i0..i1 {
            let k = get_index_wrap_around(i, size) as usize;
            let len = self.code_table[k].0 as i32;

            if len > 0 && len > num_bits_lut {
                let code = self.code_table[k].1;
                let mut node = &mut *root;
                let effective_len = len - self.num_bits_to_skip_in_tree;

                for j in (0..effective_len).rev() {
                    let bit = (code >> j) & 1;
                    if bit == 1 {
                        if node.child1.is_none() {
                            node.child1 = Some(Box::new(TreeNode::leaf(-1)));
                        }
                        node = node.child1.as_mut().unwrap();
                    } else {
                        if node.child0.is_none() {
                            node.child0 = Some(Box::new(TreeNode::leaf(-1)));
                        }
                        node = node.child0.as_mut().unwrap();
                    }

                    if j == 0 {
                        node.value = k as i16;
                    }
                }
            }
        }

        self.tree_root = Some(root);
        Ok(num_bits_lut)
    }

    /// Decode one Huffman-coded value from the bitstream.
    ///
    /// `src` is the data starting at the current byte position.
    /// `bit_pos` is the bit offset within the current u32 word (0..31).
    /// Returns the decoded symbol value.
    pub fn decode_one_value(
        &self,
        data: &[u8],
        byte_pos: &mut usize,
        bit_pos: &mut i32,
        num_bits_lut: i32,
    ) -> Result<i32> {
        if *byte_pos + 4 > data.len() {
            return Err(LercError::BufferTooSmall {
                needed: *byte_pos + 4,
                available: data.len(),
            });
        }

        let temp = u32::from_le_bytes(
            data[*byte_pos..*byte_pos + 4].try_into().unwrap(),
        );
        let mut val_tmp = (temp << *bit_pos) >> (32 - num_bits_lut);

        if 32 - *bit_pos < num_bits_lut {
            if *byte_pos + 8 > data.len() {
                return Err(LercError::BufferTooSmall {
                    needed: *byte_pos + 8,
                    available: data.len(),
                });
            }
            let temp2 = u32::from_le_bytes(
                data[*byte_pos + 4..*byte_pos + 8].try_into().unwrap(),
            );
            val_tmp |= temp2 >> (64 - *bit_pos - num_bits_lut);
        }

        let entry = self.decode_lut[val_tmp as usize];
        if entry.0 >= 0 {
            let value = entry.1 as i32;
            *bit_pos += entry.0 as i32;
            if *bit_pos >= 32 {
                *bit_pos -= 32;
                *byte_pos += 4;
            }
            return Ok(value);
        }

        // Tree traversal for long codes
        let root = self
            .tree_root
            .as_ref()
            .ok_or(LercError::InvalidData("no huffman tree".into()))?;

        // Skip leading zero bits
        *bit_pos += self.num_bits_to_skip_in_tree;
        if *bit_pos >= 32 {
            *bit_pos -= 32;
            *byte_pos += 4;
        }

        let mut node = &**root;
        loop {
            if *byte_pos + 4 > data.len() {
                return Err(LercError::BufferTooSmall {
                    needed: *byte_pos + 4,
                    available: data.len(),
                });
            }

            let temp = u32::from_le_bytes(
                data[*byte_pos..*byte_pos + 4].try_into().unwrap(),
            );
            let bit = (temp << *bit_pos) >> 31;
            *bit_pos += 1;
            if *bit_pos == 32 {
                *bit_pos = 0;
                *byte_pos += 4;
            }

            node = if bit != 0 {
                node.child1
                    .as_ref()
                    .ok_or(LercError::InvalidData("corrupt huffman tree".into()))?
            } else {
                node.child0
                    .as_ref()
                    .ok_or(LercError::InvalidData("corrupt huffman tree".into()))?
            };

            if node.value >= 0 {
                return Ok(node.value as i32);
            }
        }
    }

    pub fn code_table(&self) -> &[CodeEntry] {
        &self.code_table
    }

    /// Compute the compressed size in bytes for the given histogram.
    pub fn compute_compressed_size(&self, histo: &[i32]) -> Option<(i32, f64)> {
        let (i0, i1, max_len) = self.get_range()?;

        let size = self.code_table.len() as i32;

        // Code table size
        let mut num_bytes = 4 * 4i32; // header ints
        // BitStuffer2 for code lengths
        num_bytes +=
            compute_num_bytes_needed_simple((i1 - i0) as u32, max_len as u32) as i32;
        // Bit-stuffed codes
        let mut sum_code_bits = 0i32;
        for i in i0..i1 {
            let k = get_index_wrap_around(i, size) as usize;
            sum_code_bits += self.code_table[k].0 as i32;
        }
        let num_uints_codes = (((sum_code_bits + 7) >> 3) + 3) >> 2;
        num_bytes += 4 * num_uints_codes;

        // Data bits
        let mut num_bits = 0i64;
        let mut num_elem = 0i64;
        for (i, &count) in histo.iter().enumerate() {
            if count > 0 {
                num_bits += count as i64 * self.code_table[i].0 as i64;
                num_elem += count as i64;
            }
        }

        if num_elem == 0 {
            return None;
        }

        let num_uints = ((((num_bits + 7) >> 3) + 3) >> 2) + 1; // +1 for decode read-ahead
        num_bytes += 4 * num_uints as i32;

        let avg_bpp = 8.0 * num_bytes as f64 / num_elem as f64;
        Some((num_bytes, avg_bpp))
    }
}

/// MSB-first bit push (for Huffman code table serialization).
fn push_value(buf: &mut [u8], bit_pos: &mut i32, value: u32, len: i32) {
    let uint_idx = (*bit_pos / 32) as usize;
    let local_bit = *bit_pos & 31;

    if 32 - local_bit >= len {
        if local_bit == 0 {
            // Clear the uint
            buf[uint_idx * 4..uint_idx * 4 + 4].fill(0);
        }
        let mut temp = u32::from_le_bytes(
            buf[uint_idx * 4..uint_idx * 4 + 4].try_into().unwrap(),
        );
        temp |= value << (32 - local_bit - len);
        buf[uint_idx * 4..uint_idx * 4 + 4].copy_from_slice(&temp.to_le_bytes());
        *bit_pos += len;
        // No need to advance; uint_idx will naturally update from bit_pos
    } else {
        let overflow = local_bit + len - 32;
        let mut temp = u32::from_le_bytes(
            buf[uint_idx * 4..uint_idx * 4 + 4].try_into().unwrap(),
        );
        temp |= value >> overflow;
        buf[uint_idx * 4..uint_idx * 4 + 4].copy_from_slice(&temp.to_le_bytes());

        let next_idx = uint_idx + 1;
        let temp2 = value << (32 - overflow);
        buf[next_idx * 4..next_idx * 4 + 4].copy_from_slice(&temp2.to_le_bytes());

        *bit_pos += len;
    }
}

fn compute_num_bytes_needed_simple(num_elem: u32, max_elem: u32) -> u32 {
    let mut num_bits = 0u32;
    while num_bits < 32 && (max_elem >> num_bits) != 0 {
        num_bits += 1;
    }
    let num_bytes_uint: u32 = if num_elem < 256 {
        1
    } else if num_elem < (1 << 16) {
        2
    } else {
        4
    };
    1 + num_bytes_uint + ((num_elem * num_bits + 7) >> 3)
}

/// Encode data using Huffman coding. Returns the encoded bytes.
pub fn encode_huffman(
    data: &[u8],
    histo: &[i32],
    histo_size: usize,
) -> Option<(HuffmanCodec, Vec<u8>)> {
    let mut codec = HuffmanCodec::new();
    if !codec.compute_codes(&histo[..histo_size]) {
        return None;
    }

    let code_table_bytes = codec.write_code_table(6).ok()?;

    // Encode data using MSB-first PushValue
    let mut total_bits = 0u64;
    for &b in data {
        total_bits += codec.code_table[b as usize].0 as u64;
    }

    let num_uints = (total_bits.div_ceil(8).div_ceil(4) + 1) as usize;
    let mut encoded = vec![0u8; num_uints * 4];
    let mut bit_pos = 0i32;

    for &b in data {
        let (len, code) = codec.code_table[b as usize];
        if len > 0 {
            push_value(&mut encoded, &mut bit_pos, code, len as i32);
        }
    }

    let mut result = code_table_bytes;
    result.extend_from_slice(&encoded);

    Some((codec, result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_and_canonical() {
        // Simple histogram: two symbols with different frequencies
        let histo = vec![10, 5, 0, 0];
        let mut codec = HuffmanCodec::new();
        assert!(codec.compute_codes(&histo));

        // Symbol 0 (more frequent) should have shorter code
        assert!(codec.code_table[0].0 <= codec.code_table[1].0);
        // Both should have non-zero lengths
        assert!(codec.code_table[0].0 > 0);
        assert!(codec.code_table[1].0 > 0);
        // Unused symbols should have 0 length
        assert_eq!(codec.code_table[2].0, 0);
        assert_eq!(codec.code_table[3].0, 0);
    }

    #[test]
    fn code_table_round_trip() {
        let histo = vec![100, 50, 30, 10, 5, 1, 0, 0];
        let mut codec = HuffmanCodec::new();
        assert!(codec.compute_codes(&histo));

        let buf = codec.write_code_table(6).unwrap();

        let mut codec2 = HuffmanCodec::new();
        let mut pos = 0;
        codec2.read_code_table(&buf, &mut pos, 6).unwrap();

        assert_eq!(codec.code_table.len(), codec2.code_table.len());
        for i in 0..codec.code_table.len() {
            assert_eq!(
                codec.code_table[i], codec2.code_table[i],
                "mismatch at index {i}"
            );
        }
    }

    #[test]
    fn encode_decode_round_trip() {
        let data: Vec<u8> = vec![0, 1, 0, 1, 0, 2, 0, 1, 2, 3, 0, 0, 1, 1, 2];
        let mut histo = vec![0i32; 256];
        for &b in &data {
            histo[b as usize] += 1;
        }

        let (codec, encoded) = encode_huffman(&data, &histo, 256).unwrap();

        // Decode
        let mut codec2 = HuffmanCodec::new();
        let mut pos = 0;
        codec2.read_code_table(&encoded, &mut pos, 6).unwrap();
        let num_bits_lut = codec2.build_tree_from_codes().unwrap();

        let mut byte_pos = pos;
        let mut bit_pos = 0i32;
        let mut decoded = Vec::new();
        for _ in 0..data.len() {
            let val = codec2
                .decode_one_value(&encoded, &mut byte_pos, &mut bit_pos, num_bits_lut)
                .unwrap();
            decoded.push(val as u8);
        }

        assert_eq!(decoded, data);
    }
}
