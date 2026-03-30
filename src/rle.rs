use alloc::vec::Vec;

use crate::error::{LercError, Result};

const END_MARKER: i16 = -32768;
const MIN_RUN_EVEN: usize = 5;

/// RLE-compress a byte slice using the LERC mask RLE format.
///
/// Format: sequences of `[i16 count][data]`:
/// - count > 0: `count` literal (distinct) bytes follow
/// - count < 0 (and != -32768): next byte is repeated `-count` times
/// - count == -32768: end of stream
pub fn compress(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let len = input.len();
    let mut i = 0;

    while i < len {
        // Look for a run of identical bytes
        let mut run_len = 1usize;
        while i + run_len < len && input[i + run_len] == input[i] && run_len < 32767 {
            run_len += 1;
        }

        if run_len >= MIN_RUN_EVEN {
            // Even mode: repeated byte
            let count = -(run_len as i16);
            out.extend_from_slice(&count.to_le_bytes());
            out.push(input[i]);
            i += run_len;
        } else {
            // Odd mode: literal bytes. Collect until we hit a run >= MIN_RUN_EVEN
            let start = i;
            while i < len {
                let mut next_run = 1usize;
                while i + next_run < len
                    && input[i + next_run] == input[i]
                    && next_run < 32767
                {
                    next_run += 1;
                }
                if next_run >= MIN_RUN_EVEN {
                    break;
                }
                i += 1;
                if (i - start) >= 32767 {
                    break;
                }
            }
            let count = (i - start) as i16;
            out.extend_from_slice(&count.to_le_bytes());
            out.extend_from_slice(&input[start..i]);
        }
    }

    // End marker
    out.extend_from_slice(&END_MARKER.to_le_bytes());
    out
}

/// RLE-decompress data in the LERC mask RLE format.
pub fn decompress(input: &[u8], expected_size: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(expected_size);
    let mut pos = 0;

    loop {
        if pos + 2 > input.len() {
            return Err(LercError::InvalidData("RLE: unexpected end of data".into()));
        }
        let count = i16::from_le_bytes(input[pos..pos + 2].try_into().unwrap());
        pos += 2;

        if count == END_MARKER {
            break;
        }

        if count > 0 {
            // Literal bytes
            let n = count as usize;
            if pos + n > input.len() {
                return Err(LercError::InvalidData("RLE: literal run overflow".into()));
            }
            out.extend_from_slice(&input[pos..pos + n]);
            pos += n;
        } else {
            // Repeated byte
            let n = (-count) as usize;
            if pos >= input.len() {
                return Err(LercError::InvalidData("RLE: missing repeat byte".into()));
            }
            let byte = input[pos];
            pos += 1;
            out.resize(out.len() + n, byte);
        }
    }

    if out.len() != expected_size {
        return Err(LercError::InvalidData(
            alloc::format!(
                "RLE: expected {expected_size} bytes, got {}",
                out.len()
            ),
        ));
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_empty() {
        let compressed = compress(&[]);
        let decompressed = decompress(&compressed, 0).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn round_trip_literals() {
        let data: Vec<u8> = (0..100).collect();
        let compressed = compress(&data);
        let decompressed = decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn round_trip_runs() {
        let mut data = Vec::new();
        data.resize(50, 0xAA);
        data.resize(100, 0xBB);
        data.resize(150, 0xCC);
        let compressed = compress(&data);
        let decompressed = decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn round_trip_mixed() {
        let mut data = Vec::new();
        data.extend_from_slice(&[1, 2, 3, 4]);
        data.resize(data.len() + 20, 0xFF);
        data.extend_from_slice(&[10, 20, 30]);
        data.resize(data.len() + 10, 0x00);
        let compressed = compress(&data);
        let decompressed = decompress(&compressed, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn compression_effective_for_runs() {
        let data = vec![0xAA; 1000];
        let compressed = compress(&data);
        // Should be much smaller: 2 bytes count + 1 byte value + 2 bytes end marker = 5
        assert!(compressed.len() < 10);
    }
}
