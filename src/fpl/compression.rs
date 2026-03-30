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
