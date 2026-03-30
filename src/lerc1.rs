//! Lerc1 (legacy) format decoder.
//!
//! Lerc1 uses a completely different codec from Lerc2. The format is:
//! - Magic: "CntZImage " (10 bytes)
//! - Header: version(i32)=11, type(i32)=8, height(i32), width(i32), maxZError(f64)
//! - Two sections: count part (validity mask) then z part (elevation values)
//! - Each section: numTilesVert(i32), numTilesHori(i32), numBytes(i32), maxValInImg(f32)
//! - Tiles with encoding types: 0=uncompressed, 1=bit-stuffed, 2=constant-zero, 3=constant
//!
//! The bit stuffer uses big-endian / MSB-first packing within u32 words,
//! which is different from Lerc2's BitStuffer2.

use alloc::vec;
use alloc::vec::Vec;

use crate::bitmask::BitMask;
use crate::error::{LercError, Result};
use crate::rle;
use crate::{DataType, LercData, LercImage, LercInfo};

const LERC1_MAGIC: &[u8; 10] = b"CntZImage ";
const LERC1_VERSION: i32 = 11;
const LERC1_TYPE: i32 = 8; // CNT_Z

/// Check if data starts with the Lerc1 magic bytes.
pub fn is_lerc1(data: &[u8]) -> bool {
    data.len() >= 10 && &data[..10] == LERC1_MAGIC
}

/// Lerc1 header information.
struct Lerc1Header {
    width: i32,
    height: i32,
    max_z_error: f64,
}

/// Section header (count part or z part).
struct SectionHeader {
    num_tiles_vert: i32,
    num_tiles_hori: i32,
    num_bytes: i32,
    max_val_in_img: f32,
}

/// A cursor for reading Lerc1 data.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn check(&self, n: usize) -> Result<()> {
        if self.remaining() < n {
            Err(LercError::BufferTooSmall {
                needed: self.pos + n,
                available: self.data.len(),
            })
        } else {
            Ok(())
        }
    }

    fn read_u8(&mut self) -> Result<u8> {
        self.check(1)?;
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_i8(&mut self) -> Result<i8> {
        self.check(1)?;
        let v = self.data[self.pos] as i8;
        self.pos += 1;
        Ok(v)
    }

    fn read_i16_le(&mut self) -> Result<i16> {
        self.check(2)?;
        let v = i16::from_le_bytes(self.data[self.pos..self.pos + 2].try_into().unwrap());
        self.pos += 2;
        Ok(v)
    }

    fn read_i32_le(&mut self) -> Result<i32> {
        self.check(4)?;
        let v = i32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn read_u32_le(&mut self) -> Result<u32> {
        self.check(4)?;
        let v = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn read_f32_le(&mut self) -> Result<f32> {
        self.check(4)?;
        let v = f32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn read_f64_le(&mut self) -> Result<f64> {
        self.check(8)?;
        let v = f64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        self.check(n)?;
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }
}

/// Read a float value stored in 1, 2, or 4 bytes.
///
/// Matches C++ `CntZImage::readFlt`:
/// - 1 byte: signed i8 cast to f32
/// - 2 bytes: signed i16 (LE) cast to f32
/// - 4 bytes: f32 (LE)
fn read_flt(cursor: &mut Cursor, num_bytes: usize) -> Result<f32> {
    match num_bytes {
        1 => Ok(cursor.read_i8()? as f32),
        2 => Ok(cursor.read_i16_le()? as f32),
        4 => cursor.read_f32_le(),
        _ => Err(LercError::InvalidData(
            "lerc1: invalid float byte count".into(),
        )),
    }
}

/// Read a uint value stored in 1, 2, or 4 bytes (Lerc1 BitStuffer format).
///
/// Matches C++ `BitStuffer::readUInt`.
fn read_uint(cursor: &mut Cursor, num_bytes: usize) -> Result<u32> {
    match num_bytes {
        1 => Ok(cursor.read_u8()? as u32),
        2 => {
            cursor.check(2)?;
            let v =
                u16::from_le_bytes(cursor.data[cursor.pos..cursor.pos + 2].try_into().unwrap());
            cursor.pos += 2;
            Ok(v as u32)
        }
        4 => cursor.read_u32_le(),
        _ => Err(LercError::InvalidData(
            "lerc1: invalid uint byte count".into(),
        )),
    }
}

/// Lerc1 bit-unstuffing.
///
/// This is the OLD BitStuffer format (different from Lerc2's BitStuffer2).
/// Values are packed MSB-first within u32 words. The on-disk format stores
/// u32 words in little-endian byte order (the C++ reference uses `memcpy`
/// into native u32 followed by a no-op `SWAP_4` on LE).
///
/// Matches C++ `BitStuffer::read`.
fn bit_unstuff(cursor: &mut Cursor) -> Result<Vec<u32>> {
    // First byte: bits 0-5 = numBits, bits 6-7 encode the byte width of numElements
    let first_byte = cursor.read_u8()?;
    let bits67 = (first_byte >> 6) as usize;
    let n = if bits67 == 0 { 4 } else { 3 - bits67 };
    let num_bits = (first_byte & 63) as u32;

    let num_elements = read_uint(cursor, n)? as usize;

    if num_bits >= 32 {
        return Err(LercError::InvalidData(
            "lerc1: numBits >= 32 in bit stuffer".into(),
        ));
    }

    let mut data_vec = vec![0u32; num_elements];

    if num_bits > 0 {
        let num_uints = (num_elements as u64 * num_bits as u64 + 31) / 32;
        let n_bytes_to_copy = ((num_elements as u64 * num_bits as u64 + 7) / 8) as usize;

        cursor.check(n_bytes_to_copy)?;

        // Copy raw bytes into u32 array using LE byte order (matching C++ memcpy on LE).
        // The last u32 is pre-initialized to 0 so partial fills work correctly.
        let mut stuffed = vec![0u32; num_uints as usize];
        for (i, word) in stuffed.iter_mut().enumerate() {
            let base = cursor.pos + i * 4;
            let mut bytes = [0u8; 4];
            let end = (cursor.pos + n_bytes_to_copy).min(base + 4);
            let count = end.saturating_sub(base);
            bytes[..count].copy_from_slice(&cursor.data[base..base + count]);
            *word = u32::from_le_bytes(bytes);
        }

        // Fix up the last u32: shift left by the number of tail bytes not needed.
        // This aligns the packed bits to the MSB of the u32 for extraction.
        let num_bits_tail = (num_elements * num_bits as usize) & 31;
        let num_bytes_tail = (num_bits_tail + 7) >> 3;
        let tail_shift = if num_bytes_tail > 0 {
            4 - num_bytes_tail
        } else {
            0
        };
        if let Some(last) = stuffed.last_mut() {
            for _ in 0..tail_shift {
                *last <<= 8;
            }
        }

        // Un-stuff: extract num_bits-wide values from the MSB-first packed stream.
        let mut src_idx = 0usize;
        let mut bit_pos = 0u32;

        for dst in data_vec.iter_mut() {
            if 32 - bit_pos >= num_bits {
                let val = stuffed[src_idx];
                let shifted = val << bit_pos;
                *dst = shifted >> (32 - num_bits);
                bit_pos += num_bits;
                if bit_pos == 32 {
                    bit_pos = 0;
                    src_idx += 1;
                }
            } else {
                let val = stuffed[src_idx];
                src_idx += 1;
                let shifted = val << bit_pos;
                *dst = shifted >> (32 - num_bits);
                bit_pos -= 32 - num_bits;
                let val2 = stuffed[src_idx];
                *dst |= val2 >> (32 - bit_pos);
            }
        }

        cursor.pos += n_bytes_to_copy;
    }

    Ok(data_vec)
}

/// Read the Lerc1 file header.
fn read_lerc1_header(cursor: &mut Cursor) -> Result<Lerc1Header> {
    let magic = cursor.read_bytes(10)?;
    if magic != LERC1_MAGIC {
        return Err(LercError::InvalidMagic);
    }

    let version = cursor.read_i32_le()?;
    let type_id = cursor.read_i32_le()?;
    let height = cursor.read_i32_le()?;
    let width = cursor.read_i32_le()?;
    let max_z_error = cursor.read_f64_le()?;

    if version != LERC1_VERSION {
        return Err(LercError::UnsupportedVersion(version));
    }
    if type_id != LERC1_TYPE {
        return Err(LercError::InvalidData(
            alloc::format!("lerc1: unexpected type {type_id}, expected {LERC1_TYPE}"),
        ));
    }
    if width <= 0 || height <= 0 || width > 20000 || height > 20000 {
        return Err(LercError::InvalidData(
            alloc::format!("lerc1: invalid dimensions {width}x{height}"),
        ));
    }

    Ok(Lerc1Header {
        width,
        height,
        max_z_error,
    })
}

/// Read a section header (count or z part).
fn read_section_header(cursor: &mut Cursor) -> Result<SectionHeader> {
    let num_tiles_vert = cursor.read_i32_le()?;
    let num_tiles_hori = cursor.read_i32_le()?;
    let num_bytes = cursor.read_i32_le()?;
    let max_val_in_img = cursor.read_f32_le()?;

    Ok(SectionHeader {
        num_tiles_vert,
        num_tiles_hori,
        num_bytes,
        max_val_in_img,
    })
}

/// Read a count tile (validity mask tile).
///
/// Matches C++ `CntZImage::readCntTile`.
fn read_cnt_tile(
    cursor: &mut Cursor,
    cnt: &mut [f32],
    width: usize,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
) -> Result<()> {
    let compr_flag = cursor.read_u8()?;

    if compr_flag == 2 {
        // Entire tile is constant 0 (invalid) — already zeroed
        return Ok(());
    }

    if compr_flag == 3 || compr_flag == 4 {
        // Entire tile is constant: 3 = -1 (invalid), 4 = 1 (valid)
        let val = if compr_flag == 3 { -1.0f32 } else { 1.0f32 };
        for i in i0..i1 {
            for j in j0..j1 {
                cnt[i * width + j] = val;
            }
        }
        return Ok(());
    }

    if (compr_flag & 63) > 4 {
        return Err(LercError::InvalidData(
            "lerc1: invalid cnt tile compression flag".into(),
        ));
    }

    if compr_flag == 0 {
        // Uncompressed: read f32 values directly
        for i in i0..i1 {
            for j in j0..j1 {
                cnt[i * width + j] = cursor.read_f32_le()?;
            }
        }
    } else {
        // Bit-stuffed integer array
        let bits67 = (compr_flag >> 6) as usize;
        let n = if bits67 == 0 { 4 } else { 3 - bits67 };
        let offset = read_flt(cursor, n)?;

        let data_vec = bit_unstuff(cursor)?;

        let num_pixel = (i1 - i0) * (j1 - j0);
        if data_vec.len() < num_pixel {
            return Err(LercError::InvalidData(
                "lerc1: bit stuffer returned too few elements for cnt tile".into(),
            ));
        }

        let mut src_idx = 0;
        for i in i0..i1 {
            for j in j0..j1 {
                cnt[i * width + j] = offset + data_vec[src_idx] as f32;
                src_idx += 1;
            }
        }
    }

    Ok(())
}

/// Read a z tile (elevation values tile).
///
/// Matches C++ `CntZImage::readZTile`.
fn read_z_tile(
    cursor: &mut Cursor,
    cnt: &[f32],
    z: &mut [f32],
    width: usize,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    max_z_error_in_file: f64,
    max_z_in_img: f32,
    decoder_can_ignore_mask: bool,
) -> Result<()> {
    let byte0 = cursor.read_u8()?;
    let bits67 = (byte0 >> 6) as usize;
    let compr_flag = byte0 & 63;

    if compr_flag == 2 {
        // Entire tile constant 0
        for i in i0..i1 {
            for j in j0..j1 {
                let idx = i * width + j;
                if cnt[idx] > 0.0 {
                    z[idx] = 0.0;
                }
            }
        }
        return Ok(());
    }

    if compr_flag > 3 {
        return Err(LercError::InvalidData(
            "lerc1: invalid z tile compression flag".into(),
        ));
    }

    if compr_flag == 0 {
        // Uncompressed: read f32 for each valid pixel
        for i in i0..i1 {
            for j in j0..j1 {
                let idx = i * width + j;
                if cnt[idx] > 0.0 {
                    z[idx] = cursor.read_f32_le()?;
                }
            }
        }
    } else {
        // Bit-stuffed with offset
        let n = if bits67 == 0 { 4 } else { 3 - bits67 };
        let offset = read_flt(cursor, n)?;

        if compr_flag == 3 {
            // Constant tile
            for i in i0..i1 {
                for j in j0..j1 {
                    let idx = i * width + j;
                    if cnt[idx] > 0.0 {
                        z[idx] = offset;
                    }
                }
            }
        } else {
            // compr_flag == 1: bit-stuffed
            let data_vec = bit_unstuff(cursor)?;
            let inv_scale = 2.0 * max_z_error_in_file;
            let mut src_idx = 0;

            if decoder_can_ignore_mask {
                for i in i0..i1 {
                    for j in j0..j1 {
                        let idx = i * width + j;
                        let val =
                            (offset as f64 + data_vec[src_idx] as f64 * inv_scale) as f32;
                        z[idx] = val.min(max_z_in_img);
                        src_idx += 1;
                    }
                }
            } else {
                for i in i0..i1 {
                    for j in j0..j1 {
                        let idx = i * width + j;
                        if cnt[idx] > 0.0 {
                            let val = (offset as f64 + data_vec[src_idx] as f64 * inv_scale)
                                as f32;
                            z[idx] = val.min(max_z_in_img);
                            src_idx += 1;
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Read tiles for a section (count or z part).
///
/// Matches C++ `CntZImage::readTiles`. The tile loop iterates
/// `0..=numTiles{Vert,Hori}`, with tiles `< numTiles` getting
/// `dim / numTiles` size and tile `== numTiles` getting `dim % numTiles`.
fn read_tiles_cnt(
    cursor: &mut Cursor,
    cnt: &mut [f32],
    width: usize,
    height: usize,
    num_tiles_vert: i32,
    num_tiles_hori: i32,
) -> Result<()> {
    if num_tiles_vert <= 0 || num_tiles_hori <= 0 {
        return Err(LercError::InvalidData(
            "lerc1: invalid tile counts".into(),
        ));
    }

    for i_tile in 0..=num_tiles_vert {
        let tile_h = if i_tile == num_tiles_vert {
            height % num_tiles_vert as usize
        } else {
            height / num_tiles_vert as usize
        };
        if tile_h == 0 {
            continue;
        }
        let i0 = i_tile as usize * (height / num_tiles_vert as usize);

        for j_tile in 0..=num_tiles_hori {
            let tile_w = if j_tile == num_tiles_hori {
                width % num_tiles_hori as usize
            } else {
                width / num_tiles_hori as usize
            };
            if tile_w == 0 {
                continue;
            }
            let j0 = j_tile as usize * (width / num_tiles_hori as usize);

            read_cnt_tile(cursor, cnt, width, i0, i0 + tile_h, j0, j0 + tile_w)?;
        }
    }

    Ok(())
}

fn read_tiles_z(
    cursor: &mut Cursor,
    cnt: &[f32],
    z: &mut [f32],
    width: usize,
    height: usize,
    num_tiles_vert: i32,
    num_tiles_hori: i32,
    max_z_error_in_file: f64,
    max_z_in_img: f32,
    decoder_can_ignore_mask: bool,
) -> Result<()> {
    if num_tiles_vert <= 0 || num_tiles_hori <= 0 {
        return Err(LercError::InvalidData(
            "lerc1: invalid tile counts".into(),
        ));
    }

    for i_tile in 0..=num_tiles_vert {
        let tile_h = if i_tile == num_tiles_vert {
            height % num_tiles_vert as usize
        } else {
            height / num_tiles_vert as usize
        };
        if tile_h == 0 {
            continue;
        }
        let i0 = i_tile as usize * (height / num_tiles_vert as usize);

        for j_tile in 0..=num_tiles_hori {
            let tile_w = if j_tile == num_tiles_hori {
                width % num_tiles_hori as usize
            } else {
                width / num_tiles_hori as usize
            };
            if tile_w == 0 {
                continue;
            }
            let j0 = j_tile as usize * (width / num_tiles_hori as usize);

            read_z_tile(
                cursor,
                cnt,
                z,
                width,
                i0,
                i0 + tile_h,
                j0,
                j0 + tile_w,
                max_z_error_in_file,
                max_z_in_img,
                decoder_can_ignore_mask,
            )?;
        }
    }

    Ok(())
}

/// Extract decode info from a Lerc1 blob without decoding pixel data.
pub fn decode_info(data: &[u8]) -> Result<LercInfo> {
    let mut cursor = Cursor::new(data);
    let hd = read_lerc1_header(&mut cursor)?;

    // Read count section header to get past it
    let cnt_sec = read_section_header(&mut cursor)?;

    // Skip the count section data
    if cnt_sec.num_bytes < 0 || cursor.pos + cnt_sec.num_bytes as usize > data.len() {
        return Err(LercError::BufferTooSmall {
            needed: cursor.pos + cnt_sec.num_bytes.max(0) as usize,
            available: data.len(),
        });
    }
    cursor.pos += cnt_sec.num_bytes as usize;

    // Read z section header
    let z_sec = read_section_header(&mut cursor)?;

    // Compute total blob size
    // We can't know the exact valid pixel count without decoding the mask,
    // but we provide an estimate. For decode_info we report the z range.
    // The blob size is everything up to end of z section data.
    let blob_size = cursor.pos + z_sec.num_bytes.max(0) as usize;

    Ok(LercInfo {
        version: LERC1_VERSION,
        width: hd.width as u32,
        height: hd.height as u32,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Float,
        // We can't know exact valid count without decoding; use total pixels as upper bound
        num_valid_pixels: (hd.width as u32) * (hd.height as u32),
        max_z_error: hd.max_z_error,
        z_min: f64::NAN, // Not available from header alone
        z_max: z_sec.max_val_in_img as f64,
        blob_size: blob_size as u32,
        ..Default::default()
    })
}

/// Fully decode a Lerc1 blob into an `LercImage`.
pub fn decode(data: &[u8]) -> Result<LercImage> {
    let mut cursor = Cursor::new(data);
    let hd = read_lerc1_header(&mut cursor)?;

    let width = hd.width as usize;
    let height = hd.height as usize;
    let num_pixels = width * height;

    // Allocate cnt and z arrays (both f32, initialized to 0)
    let mut cnt = vec![0.0f32; num_pixels];
    let mut z = vec![0.0f32; num_pixels];

    let mut decoder_can_ignore_mask = false;

    // --- Count part (iPart=0) ---
    let cnt_sec = read_section_header(&mut cursor)?;
    if cnt_sec.num_bytes < 0 {
        return Err(LercError::InvalidData(
            "lerc1: negative numBytes in count section".into(),
        ));
    }

    let cnt_data_start = cursor.pos;

    if cnt_sec.num_tiles_vert == 0 && cnt_sec.num_tiles_hori == 0 {
        // No tiling for count part
        if cnt_sec.num_bytes == 0 {
            // Count part is constant: all pixels have the same count value
            for c in cnt.iter_mut() {
                *c = cnt_sec.max_val_in_img;
            }
            if cnt_sec.max_val_in_img > 0.0 {
                decoder_can_ignore_mask = true;
            }
        } else {
            // Count part is binary mask, RLE compressed
            let mask_byte_count = (num_pixels + 7) / 8;
            let rle_data = &data[cursor.pos..cursor.pos + cnt_sec.num_bytes as usize];
            let mask_bytes = rle::decompress(rle_data, mask_byte_count)?;
            let bitmask = BitMask::from_bytes(mask_bytes, num_pixels);

            for k in 0..num_pixels {
                cnt[k] = if bitmask.is_valid(k) { 1.0 } else { 0.0 };
            }
        }
    } else {
        read_tiles_cnt(
            &mut cursor,
            &mut cnt,
            width,
            height,
            cnt_sec.num_tiles_vert,
            cnt_sec.num_tiles_hori,
        )?;
    }

    // Advance past count section data
    cursor.pos = cnt_data_start + cnt_sec.num_bytes as usize;

    // --- Z part (iPart=1) ---
    let z_sec = read_section_header(&mut cursor)?;
    if z_sec.num_bytes < 0 {
        return Err(LercError::InvalidData(
            "lerc1: negative numBytes in z section".into(),
        ));
    }

    read_tiles_z(
        &mut cursor,
        &cnt,
        &mut z,
        width,
        height,
        z_sec.num_tiles_vert,
        z_sec.num_tiles_hori,
        hd.max_z_error,
        z_sec.max_val_in_img,
        decoder_can_ignore_mask,
    )?;

    // Build validity mask from count array
    let mut mask = BitMask::new(num_pixels);
    for k in 0..num_pixels {
        if cnt[k] > 0.0 {
            mask.set_valid(k);
        }
    }

    Ok(LercImage {
        width: width as u32,
        height: height as u32,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![mask],
        data: LercData::F32(z),
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_unstuff_zero_bits() {
        // numBits=0, numElements=5 => all zeros
        // First byte: bits67=1 (n=2), numBits=0 => byte = 0b01_000000 = 0x40
        // numElements as u16 LE: 5, 0
        let data = [0x40, 0x05, 0x00];
        let mut cursor = Cursor::new(&data);
        let result = bit_unstuff(&mut cursor).unwrap();
        assert_eq!(result, vec![0u32; 5]);
    }

    #[test]
    fn bit_unstuff_1bit_values() {
        // 4 elements, 1 bit each: [1, 0, 1, 1]
        // Packed MSB-first in a u32: bits are 1,0,1,1 in positions 31,30,29,28
        // That's 0b1011_0000... = 0xB0000000 as big-endian
        // But we only copy ceil(4*1/8) = 1 byte.
        // After byte swap fixing: the single byte 0xB0 gets read.
        //
        // First byte: bits67=1 (n=2), numBits=1 => byte = 0b01_000001 = 0x41
        // numElements as u16 LE: 4, 0
        // Data: 1 byte = 0b1011_0000 = 0xB0
        let data = [0x41, 0x04, 0x00, 0xB0];
        let mut cursor = Cursor::new(&data);
        let result = bit_unstuff(&mut cursor).unwrap();
        assert_eq!(result, vec![1, 0, 1, 1]);
    }

    #[test]
    fn read_flt_1byte() {
        let data = [0xFE]; // -2 as i8
        let mut cursor = Cursor::new(&data);
        let val = read_flt(&mut cursor, 1).unwrap();
        assert_eq!(val, -2.0);
    }

    #[test]
    fn read_flt_2bytes() {
        let data = 300i16.to_le_bytes();
        let mut cursor = Cursor::new(&data);
        let val = read_flt(&mut cursor, 2).unwrap();
        assert_eq!(val, 300.0);
    }

    #[test]
    fn read_flt_4bytes() {
        let data = 1.5f32.to_le_bytes();
        let mut cursor = Cursor::new(&data);
        let val = read_flt(&mut cursor, 4).unwrap();
        assert_eq!(val, 1.5);
    }

    #[test]
    fn is_lerc1_detection() {
        assert!(is_lerc1(b"CntZImage extra data"));
        assert!(!is_lerc1(b"Lerc2 data"));
        assert!(!is_lerc1(b"short"));
    }
}
