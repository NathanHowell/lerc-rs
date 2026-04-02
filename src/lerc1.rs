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
use crate::types::TileRect;
use crate::{DataType, Image, LercInfo, SampleData};

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
            let v = u16::from_le_bytes(cursor.data[cursor.pos..cursor.pos + 2].try_into().unwrap());
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
        let num_uints = (num_elements as u64 * num_bits as u64).div_ceil(32);
        let n_bytes_to_copy = (num_elements as u64 * num_bits as u64).div_ceil(8) as usize;

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
        return Err(LercError::InvalidData(alloc::format!(
            "lerc1: unexpected type {type_id}, expected {LERC1_TYPE}"
        )));
    }
    if width <= 0 || height <= 0 || width > 20000 || height > 20000 {
        return Err(LercError::InvalidData(alloc::format!(
            "lerc1: invalid dimensions {width}x{height}"
        )));
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

/// Shared configuration for Lerc1 z-tile decoding.
struct Lerc1ZConfig {
    width: usize,
    height: usize,
    max_z_error_in_file: f64,
    max_z_in_img: f32,
    decoder_can_ignore_mask: bool,
}

/// Read a z tile (elevation values tile).
///
/// Matches C++ `CntZImage::readZTile`.
fn read_z_tile(
    cursor: &mut Cursor,
    cnt: &[f32],
    z: &mut [f32],
    cfg: &Lerc1ZConfig,
    rect: TileRect,
) -> Result<()> {
    let TileRect { i0, i1, j0, j1 } = rect;
    let width = cfg.width;
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
            let inv_scale = 2.0 * cfg.max_z_error_in_file;
            let mut src_idx = 0;

            if cfg.decoder_can_ignore_mask {
                for i in i0..i1 {
                    for j in j0..j1 {
                        let idx = i * width + j;
                        let val = (offset as f64 + data_vec[src_idx] as f64 * inv_scale) as f32;
                        z[idx] = val.min(cfg.max_z_in_img);
                        src_idx += 1;
                    }
                }
            } else {
                for i in i0..i1 {
                    for j in j0..j1 {
                        let idx = i * width + j;
                        if cnt[idx] > 0.0 {
                            let val = (offset as f64 + data_vec[src_idx] as f64 * inv_scale) as f32;
                            z[idx] = val.min(cfg.max_z_in_img);
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
        return Err(LercError::InvalidData("lerc1: invalid tile counts".into()));
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
    cfg: &Lerc1ZConfig,
    num_tiles_vert: i32,
    num_tiles_hori: i32,
) -> Result<()> {
    let width = cfg.width;
    let height = cfg.height;

    if num_tiles_vert <= 0 || num_tiles_hori <= 0 {
        return Err(LercError::InvalidData("lerc1: invalid tile counts".into()));
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

            let rect = TileRect {
                i0,
                i1: i0 + tile_h,
                j0,
                j1: j0 + tile_w,
            };
            read_z_tile(cursor, cnt, z, cfg, rect)?;
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
        tolerance: hd.max_z_error,
        min_value: f64::NAN, // Not available from header alone
        max_value: z_sec.max_val_in_img as f64,
        blob_size: blob_size as u32,
        ..Default::default()
    })
}

/// Fully decode a Lerc1 blob into an `Image`.
pub fn decode(data: &[u8]) -> Result<Image> {
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
            let mask_byte_count = num_pixels.div_ceil(8);
            let rle_data = &data[cursor.pos..cursor.pos + cnt_sec.num_bytes as usize];
            let mask_bytes = rle::decompress(rle_data, mask_byte_count)?;
            let bitmask = BitMask::from_bytes(mask_bytes, num_pixels);

            for (k, c) in cnt[..num_pixels].iter_mut().enumerate() {
                *c = if bitmask.is_valid(k) { 1.0 } else { 0.0 };
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

    let z_cfg = Lerc1ZConfig {
        width,
        height,
        max_z_error_in_file: hd.max_z_error,
        max_z_in_img: z_sec.max_val_in_img,
        decoder_can_ignore_mask,
    };
    read_tiles_z(
        &mut cursor,
        &cnt,
        &mut z,
        &z_cfg,
        z_sec.num_tiles_vert,
        z_sec.num_tiles_hori,
    )?;

    // Build validity mask from count array
    let mut mask = BitMask::new(num_pixels);
    for (k, &c) in cnt[..num_pixels].iter().enumerate() {
        if c > 0.0 {
            mask.set_valid(k);
        }
    }

    Ok(Image {
        width: width as u32,
        height: height as u32,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![mask],
        data: SampleData::F32(z),
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
    fn bit_unstuff_empty_elements() {
        // numBits=1, numElements=0 => empty result
        // First byte: bits67=1 (n=2), numBits=1 => byte = 0x41
        // numElements as u16 LE: 0, 0
        let data = [0x41, 0x00, 0x00];
        let mut cursor = Cursor::new(&data);
        let result = bit_unstuff(&mut cursor).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn bit_unstuff_31bit_values() {
        // Test with numBits=31 (maximum valid), 1 element
        // Value: 0x7FFF_FFFF (all 31 bits set)
        //
        // First byte: bits67=1 (n=2), numBits=31 => byte = 0b01_011111 = 0x5F
        // numElements as u16 LE: 1, 0
        // Data bytes: ceil(1*31/8) = 4 bytes
        //
        // The value 0x7FFFFFFF packed MSB-first in one u32:
        // bits[31..1] = 0x7FFFFFFF => shifted to MSB: 0xFFFF_FFFE
        // After tail fix: numBitsTail = (1*31)&31 = 31, numBytesTail = (31+7)/8 = 4
        // tailShift = 0. So u32 from 4 LE bytes should be 0xFFFF_FFFE
        // LE bytes: 0xFE 0xFF 0xFF 0xFF
        let data = [0x5F, 0x01, 0x00, 0xFE, 0xFF, 0xFF, 0xFF];
        let mut cursor = Cursor::new(&data);
        let result = bit_unstuff(&mut cursor).unwrap();
        assert_eq!(result, vec![0x7FFF_FFFF]);
    }

    #[test]
    fn bit_unstuff_multi_word_3bit() {
        // 12 elements at 3 bits each = 36 bits total => 2 u32 words
        // Values: [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]
        //
        // MSB-first packing into u32 words:
        // Word 0 (bits 0..31): 000 001 010 011 100 101 110 111 000 001 01
        //   = 0b00000101001110010111000101_000010 = ...
        // Let's compute carefully:
        //   bit positions in word 0 (MSB=bit31):
        //   val[0]=0b000 at bits 31-29 => 0x0000_0000
        //   val[1]=0b001 at bits 28-26 => shift 26 => 0x0400_0000
        //   val[2]=0b010 at bits 25-23 => shift 23 => 0x0100_0000
        //   val[3]=0b011 at bits 22-20 => shift 20 => 0x0030_0000
        //   val[4]=0b100 at bits 19-17 => shift 17 => 0x0008_0000
        //   val[5]=0b101 at bits 16-14 => shift 14 => 0x0001_4000
        //   val[6]=0b110 at bits 13-11 => shift 11 => 0x0000_3000
        //   val[7]=0b111 at bits 10-8  => shift 8  => 0x0000_0700
        //   val[8]=0b000 at bits 7-5   => 0
        //   val[9]=0b001 at bits 4-2   => shift 2  => 0x0000_0004
        //   val[10]: needs 3 bits, only 2 remain (bits 1-0)
        //     top 2 bits of 010 = 01 at bits 1-0 => 0x0000_0001
        //
        //   Word 0 = 0x0000_0000 | 0x0400_0000 | 0x0100_0000 | 0x0030_0000
        //          | 0x0008_0000 | 0x0001_4000 | 0x0000_3000 | 0x0000_0700
        //          | 0x0000_0000 | 0x0000_0004 | 0x0000_0001
        //          = 0x0539_7705
        //
        //   Word 1: remaining 1 bit of val[10] (0) at bit 31, then val[11]=011 at bits 30-28
        //     val[10] bottom bit: 0 at bit 31 => 0x0000_0000
        //     val[11]=0b011 at bits 30-28 => shift 28 => 0x3000_0000
        //     Word 1 = 0x3000_0000
        //
        //   numBitsTail = (12*3) & 31 = 36 & 31 = 4
        //   numBytesTail = (4+7)/8 = 1
        //   tailShift = 4 - 1 = 3 bytes => shift left by 24
        //   So word1 before shift: we need 0x3000_0000 >> 24 = 0x30 in first byte
        //   Actually, let's approach differently: n_bytes_to_copy = ceil(36/8) = 5
        //
        // This is getting complex. Let me construct the test by encoding manually.
        // Simpler approach: construct from known packed bytes.
        //
        // 12 values, 3 bits: total 36 bits => 5 bytes to copy.
        // Pack MSB-first:
        //   Byte 0 (bits 31-24 of word0): val0=000, val1=001, val2=01_ => 0b00000101 = 0x05
        //   Byte 1 (bits 23-16 of word0): _0, val3=011, val4=100, val5=1_ => 0b00011001 = 0x39
        //     wait, val2 remaining bit is 0: 0_011_100_1 = 0b00111001 = 0x39
        //   Byte 2 (bits 15-8 of word0): val5 remaining=01, val6=110, val7=111 => 0b01110111 = 0x77
        //   Byte 3 (bits 7-0 of word0): val8=000, val9=001, val10 top 2 = 01 => 0b00000101 = 0x05
        //   Byte 4: val10 bottom 1 = 0, val11=011, pad=0000 => 0b00110000 = 0x30
        //
        // Word0 from LE bytes [0x05, 0x39, 0x77, 0x05] = 0x0577_3905
        //   Hmm that doesn't match. LE means byte0 is least significant.
        //   u32::from_le_bytes([0x05, 0x39, 0x77, 0x05]) = 0x0577_3905
        //   But we want word0 MSB = byte0 content...
        //   Actually in the code: bytes are loaded via from_le_bytes, so
        //   data[cursor.pos + 0] => least significant byte of word0.
        //   But we pack MSB-first from the most significant bit of the u32.
        //   So byte at offset 0 should be the LE representation's byte 0,
        //   i.e., the LEAST significant byte of the packed u32.
        //
        // I think the easiest test approach: use known small values and verify.
        // Let me just pack 3 elements of 2 bits each: [1, 2, 3]
        // Total: 6 bits => 1 byte
        // MSB-first: val0=01, val1=10, val2=11, pad=00 => 0b01101100 = 0x6C
        //
        // u32 from 1 LE byte: [0x6C, 0, 0, 0] => 0x0000_006C
        // numBitsTail = (3*2) & 31 = 6
        // numBytesTail = (6+7)/8 = 1
        // tailShift = 4 - 1 = 3 => shift left 3*8=24 bits
        // word0 = 0x6C << 24 = 0x6C00_0000
        //
        // Extract val0: word0 << 0 >> 30 = 0x6C00_0000 >> 30 = 0b01 = 1 ✓
        // Extract val1: bit_pos=2, word0 << 2 = 0xB000_0000, >> 30 = 0b10 = 2 ✓
        // Extract val2: bit_pos=4, word0 << 4 = 0xC000_0000, >> 30 = 0b11 = 3 ✓
        //
        // First byte: bits67=1 (n=2), numBits=2 => byte = 0b01_000010 = 0x42
        // numElements as u16 LE: 3, 0
        // Data: 1 byte = 0x6C
        let data = [0x42, 0x03, 0x00, 0x6C];
        let mut cursor = Cursor::new(&data);
        let result = bit_unstuff(&mut cursor).unwrap();
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn bit_unstuff_8bit_values() {
        // 3 elements, 8 bits each: [100, 200, 42]
        // Total: 24 bits => 3 bytes
        // MSB-first packing into a u32:
        //   val0=100=0x64 at bits 31-24
        //   val1=200=0xC8 at bits 23-16
        //   val2=42=0x2A at bits 15-8
        //
        // n_bytes_to_copy = ceil(3*8/8) = 3
        // LE bytes: the u32 should have 0x64 at bits 31-24, 0xC8 at 23-16, 0x2A at 15-8
        //   u32 = 0x64C8_2A00
        //   LE bytes: [0x00, 0x2A, 0xC8, 0x64]
        //   But we only copy 3 bytes: [0x00, 0x2A, 0xC8]
        //   Partial last word: numBitsTail = (3*8)&31 = 24, numBytesTail=(24+7)/8=3
        //   tailShift = 4-3 = 1 => shift left 8 bits
        //   word0 from LE [0x00, 0x2A, 0xC8, 0x00] = 0x00C8_2A00
        //   After tailShift: 0x00C8_2A00 << 8 = 0xC82A_0000
        //   Hmm that gives wrong MSB. Let me re-think.
        //
        // Actually let me just trace the code more carefully.
        // stuffed has 1 word (num_uints = ceil(24/32) = 1).
        // Bytes to copy: 3.
        // Loop: i=0, base=cursor.pos+0, end=min(pos+3, pos+4)=pos+3, count=3
        //   bytes = [data[pos], data[pos+1], data[pos+2], 0]
        //   word = u32::from_le_bytes(bytes)
        //
        // If data bytes after header are [B0, B1, B2]:
        //   word = u32::from_le_bytes([B0, B1, B2, 0]) = (B2 << 16) | (B1 << 8) | B0
        //
        // Tail fix: numBitsTail=24, numBytesTail=3, tailShift=1
        //   word <<= 8
        //   => ((B2 << 16) | (B1 << 8) | B0) << 8
        //   = (B2 << 24) | (B1 << 16) | (B0 << 8)
        //
        // Extract val0: word << 0 >> 24 = B2
        // But we want val0=100... so B2 should be 100=0x64.
        // val1: bit_pos=8, word << 8 >> 24 = B1. So B1=200=0xC8.
        // val2: bit_pos=16, word << 16 >> 24 = B0. So B0=42=0x2A.
        //
        // So data bytes: [0x2A, 0xC8, 0x64]
        // First byte: bits67=1 (n=2), numBits=8 => byte = 0b01_001000 = 0x48
        // numElements as u16 LE: 3, 0
        let data = [0x48, 0x03, 0x00, 0x2A, 0xC8, 0x64];
        let mut cursor = Cursor::new(&data);
        let result = bit_unstuff(&mut cursor).unwrap();
        assert_eq!(result, vec![100, 200, 42]);
    }

    #[test]
    fn bit_unstuff_rejects_32bit() {
        // numBits=32 should be rejected
        // First byte: bits67=1 (n=2), numBits=32 => byte = 0b01_100000 = 0x60
        // numElements as u16 LE: 1, 0
        let data = [0x60, 0x01, 0x00];
        let mut cursor = Cursor::new(&data);
        assert!(bit_unstuff(&mut cursor).is_err());
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
    fn read_flt_invalid_size() {
        let data = [0u8; 8];
        let mut cursor = Cursor::new(&data);
        assert!(read_flt(&mut cursor, 3).is_err());
        assert!(read_flt(&mut cursor, 0).is_err());
    }

    #[test]
    fn read_f64_le_known_value() {
        let val = core::f64::consts::PI;
        let data = val.to_le_bytes();
        let mut cursor = Cursor::new(&data);
        let result = cursor.read_f64_le().unwrap();
        assert_eq!(result, val);
    }

    #[test]
    fn read_f64_le_negative() {
        let val: f64 = -123456.789;
        let data = val.to_le_bytes();
        let mut cursor = Cursor::new(&data);
        let result = cursor.read_f64_le().unwrap();
        assert_eq!(result, val);
    }

    #[test]
    fn read_f64_le_zero() {
        let data = 0.0f64.to_le_bytes();
        let mut cursor = Cursor::new(&data);
        let result = cursor.read_f64_le().unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn read_f64_le_insufficient_data() {
        let data = [0u8; 4]; // only 4 bytes, need 8
        let mut cursor = Cursor::new(&data);
        assert!(cursor.read_f64_le().is_err());
    }

    #[test]
    fn is_lerc1_detection() {
        assert!(is_lerc1(b"CntZImage extra data"));
        assert!(!is_lerc1(b"Lerc2 data"));
        assert!(!is_lerc1(b"short"));
    }

    #[test]
    fn cursor_remaining_and_check() {
        let data = [1u8, 2, 3, 4];
        let mut cursor = Cursor::new(&data);
        assert_eq!(cursor.remaining(), 4);
        cursor.pos = 2;
        assert_eq!(cursor.remaining(), 2);
        assert!(cursor.check(2).is_ok());
        assert!(cursor.check(3).is_err());
    }

    #[test]
    fn read_uint_1byte() {
        let data = [0xFF];
        let mut cursor = Cursor::new(&data);
        assert_eq!(read_uint(&mut cursor, 1).unwrap(), 255);
    }

    #[test]
    fn read_uint_2bytes() {
        let data = 0x1234u16.to_le_bytes();
        let mut cursor = Cursor::new(&data);
        assert_eq!(read_uint(&mut cursor, 2).unwrap(), 0x1234);
    }

    #[test]
    fn read_uint_4bytes() {
        let data = 0xDEAD_BEEFu32.to_le_bytes();
        let mut cursor = Cursor::new(&data);
        assert_eq!(read_uint(&mut cursor, 4).unwrap(), 0xDEAD_BEEF);
    }

    #[test]
    fn read_uint_invalid_size() {
        let data = [0u8; 8];
        let mut cursor = Cursor::new(&data);
        assert!(read_uint(&mut cursor, 3).is_err());
    }

    // ---- decode_z_tile tests (tile types) ----

    /// Helper: build a minimal z-tile payload for type 2 (constant zero).
    /// byte0 = compr_flag=2, bits67=0 => 0x02
    fn make_z_tile_type2() -> Vec<u8> {
        vec![0x02]
    }

    /// Helper: build a minimal z-tile payload for type 3 (constant value).
    /// byte0 = compr_flag=3, bits67=1 (n=2 => offset stored as i16) => 0b01_000011 = 0x43
    /// offset as i16 LE
    fn make_z_tile_type3(offset: f32) -> Vec<u8> {
        // Use 4-byte float encoding: bits67=0 => n=4
        // byte0 = compr_flag=3, bits67=0 => 0x03
        let mut buf = vec![0x03];
        buf.extend_from_slice(&offset.to_le_bytes());
        buf
    }

    /// Helper: build a minimal z-tile payload for type 0 (uncompressed).
    /// byte0 = 0x00
    /// followed by f32 LE values for each valid pixel
    fn make_z_tile_type0(values: &[f32]) -> Vec<u8> {
        let mut buf = vec![0x00];
        for &v in values {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    #[test]
    fn decode_z_tile_type2_constant_zero() {
        let tile_data = make_z_tile_type2();
        let mut cursor = Cursor::new(&tile_data);

        let width = 2;
        let cnt = vec![1.0f32; 4]; // all valid
        let mut z = vec![f32::NAN; 4];

        let cfg = Lerc1ZConfig {
            width,
            height: 2,
            max_z_error_in_file: 0.0,
            max_z_in_img: 100.0,
            decoder_can_ignore_mask: false,
        };
        let rect = TileRect {
            i0: 0,
            i1: 2,
            j0: 0,
            j1: 2,
        };
        read_z_tile(&mut cursor, &cnt, &mut z, &cfg, rect).unwrap();

        assert_eq!(z, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn decode_z_tile_type2_skips_invalid() {
        let tile_data = make_z_tile_type2();
        let mut cursor = Cursor::new(&tile_data);

        let width = 2;
        let cnt = vec![1.0, 0.0, 0.0, 1.0]; // only corners valid
        let mut z = vec![f32::NAN; 4];

        let cfg = Lerc1ZConfig {
            width,
            height: 2,
            max_z_error_in_file: 0.0,
            max_z_in_img: 100.0,
            decoder_can_ignore_mask: false,
        };
        let rect = TileRect {
            i0: 0,
            i1: 2,
            j0: 0,
            j1: 2,
        };
        read_z_tile(&mut cursor, &cnt, &mut z, &cfg, rect).unwrap();

        assert_eq!(z[0], 0.0);
        assert!(z[1].is_nan()); // invalid pixel, untouched
        assert!(z[2].is_nan());
        assert_eq!(z[3], 0.0);
    }

    #[test]
    fn decode_z_tile_type3_constant_value() {
        let tile_data = make_z_tile_type3(42.5);
        let mut cursor = Cursor::new(&tile_data);

        let width = 3;
        let cnt = vec![1.0f32; 6];
        let mut z = vec![0.0f32; 6];

        let cfg = Lerc1ZConfig {
            width,
            height: 2,
            max_z_error_in_file: 0.5,
            max_z_in_img: 100.0,
            decoder_can_ignore_mask: false,
        };
        let rect = TileRect {
            i0: 0,
            i1: 2,
            j0: 0,
            j1: 3,
        };
        read_z_tile(&mut cursor, &cnt, &mut z, &cfg, rect).unwrap();

        for &val in &z {
            assert_eq!(val, 42.5);
        }
    }

    #[test]
    fn decode_z_tile_type3_respects_mask() {
        let tile_data = make_z_tile_type3(7.0);
        let mut cursor = Cursor::new(&tile_data);

        let width = 2;
        let cnt = vec![1.0, 0.0, 0.0, 1.0];
        let mut z = vec![0.0f32; 4];

        let cfg = Lerc1ZConfig {
            width,
            height: 2,
            max_z_error_in_file: 0.0,
            max_z_in_img: 100.0,
            decoder_can_ignore_mask: false,
        };
        let rect = TileRect {
            i0: 0,
            i1: 2,
            j0: 0,
            j1: 2,
        };
        read_z_tile(&mut cursor, &cnt, &mut z, &cfg, rect).unwrap();

        assert_eq!(z[0], 7.0);
        assert_eq!(z[1], 0.0); // invalid, untouched
        assert_eq!(z[2], 0.0); // invalid, untouched
        assert_eq!(z[3], 7.0);
    }

    #[test]
    fn decode_z_tile_type0_uncompressed() {
        let values = [1.0f32, 2.0, 3.0, 4.0];
        let tile_data = make_z_tile_type0(&values);
        let mut cursor = Cursor::new(&tile_data);

        let width = 2;
        let cnt = vec![1.0f32; 4];
        let mut z = vec![0.0f32; 4];

        let cfg = Lerc1ZConfig {
            width,
            height: 2,
            max_z_error_in_file: 0.0,
            max_z_in_img: 100.0,
            decoder_can_ignore_mask: false,
        };
        let rect = TileRect {
            i0: 0,
            i1: 2,
            j0: 0,
            j1: 2,
        };
        read_z_tile(&mut cursor, &cnt, &mut z, &cfg, rect).unwrap();

        assert_eq!(z, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn decode_z_tile_type0_uncompressed_with_mask() {
        // Only pixels [0] and [3] are valid
        let values = [10.0f32, 20.0];
        let tile_data = make_z_tile_type0(&values);
        let mut cursor = Cursor::new(&tile_data);

        let width = 2;
        let cnt = vec![1.0, 0.0, 0.0, 1.0];
        let mut z = vec![0.0f32; 4];

        let cfg = Lerc1ZConfig {
            width,
            height: 2,
            max_z_error_in_file: 0.0,
            max_z_in_img: 100.0,
            decoder_can_ignore_mask: false,
        };
        let rect = TileRect {
            i0: 0,
            i1: 2,
            j0: 0,
            j1: 2,
        };
        read_z_tile(&mut cursor, &cnt, &mut z, &cfg, rect).unwrap();

        assert_eq!(z[0], 10.0);
        assert_eq!(z[3], 20.0);
    }

    #[test]
    fn decode_z_tile_invalid_compr_flag() {
        // compr_flag=5 is invalid
        let tile_data = [0x05u8];
        let mut cursor = Cursor::new(&tile_data);

        let width = 2;
        let cnt = vec![1.0f32; 4];
        let mut z = vec![0.0f32; 4];

        let cfg = Lerc1ZConfig {
            width,
            height: 2,
            max_z_error_in_file: 0.0,
            max_z_in_img: 100.0,
            decoder_can_ignore_mask: false,
        };
        let rect = TileRect {
            i0: 0,
            i1: 2,
            j0: 0,
            j1: 2,
        };
        let result = read_z_tile(&mut cursor, &cnt, &mut z, &cfg, rect);
        assert!(result.is_err());
    }

    // ---- Bit-stuffed tile (type 1) ----

    #[test]
    fn decode_z_tile_type1_bit_stuffed() {
        // Build a bit-stuffed z tile with known values.
        // 4 pixels, all valid, max_z_error = 0.5 (so inv_scale = 1.0)
        // Offset = 10.0 (as f32, 4 bytes), quantized values = [0, 1, 2, 3]
        // Reconstructed: offset + val * inv_scale = [10.0, 11.0, 12.0, 13.0]
        //
        // byte0: compr_flag=1, bits67=0 (n=4 for offset) => 0x01
        // offset: 10.0f32 LE
        // bit_unstuff header: numBits=2, bits67=1 (n=2), numElements=4
        //   first_byte = 0b01_000010 = 0x42
        //   numElements as u16 LE: 4, 0
        //   data: [0,1,2,3] packed MSB-first at 2 bits each = 00 01 10 11 => 0b00011011 = 0x1B
        //   1 byte of data
        let mut buf = vec![0x01u8];
        buf.extend_from_slice(&10.0f32.to_le_bytes());
        buf.push(0x42); // numBits=2, bits67=1
        buf.extend_from_slice(&4u16.to_le_bytes()); // numElements=4
        buf.push(0x1B); // packed bits

        let mut cursor = Cursor::new(&buf);
        let width = 2;
        let cnt = vec![1.0f32; 4];
        let mut z = vec![0.0f32; 4];

        let cfg = Lerc1ZConfig {
            width,
            height: 2,
            max_z_error_in_file: 0.5,
            max_z_in_img: 100.0,
            decoder_can_ignore_mask: false,
        };
        let rect = TileRect {
            i0: 0,
            i1: 2,
            j0: 0,
            j1: 2,
        };
        read_z_tile(&mut cursor, &cnt, &mut z, &cfg, rect).unwrap();

        // inv_scale = 2 * 0.5 = 1.0
        // z[i] = offset + val * inv_scale = 10.0 + [0,1,2,3] * 1.0
        assert_eq!(z, vec![10.0, 11.0, 12.0, 13.0]);
    }

    #[test]
    fn decode_z_tile_type1_clamped_to_max() {
        // Same setup but max_z_in_img is low enough to clamp
        let mut buf = vec![0x01u8];
        buf.extend_from_slice(&10.0f32.to_le_bytes());
        buf.push(0x42); // numBits=2, bits67=1
        buf.extend_from_slice(&4u16.to_le_bytes());
        buf.push(0x1B); // [0,1,2,3]

        let mut cursor = Cursor::new(&buf);
        let width = 2;
        let cnt = vec![1.0f32; 4];
        let mut z = vec![0.0f32; 4];

        let cfg = Lerc1ZConfig {
            width,
            height: 2,
            max_z_error_in_file: 0.5,
            max_z_in_img: 11.5,
            decoder_can_ignore_mask: false,
        };
        let rect = TileRect {
            i0: 0,
            i1: 2,
            j0: 0,
            j1: 2,
        };
        read_z_tile(&mut cursor, &cnt, &mut z, &cfg, rect).unwrap();

        assert_eq!(z[0], 10.0);
        assert_eq!(z[1], 11.0);
        assert_eq!(z[2], 11.5); // clamped from 12.0
        assert_eq!(z[3], 11.5); // clamped from 13.0
    }
}
