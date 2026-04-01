use alloc::vec::Vec;

use crate::bitmask::BitMask;
use crate::bitstuffer;
use crate::error::{LercError, Result};
use crate::header::HeaderInfo;
use crate::types::{DataType, Sample, TileCompressionMode, TileRect, tile_flags};

/// A depth-slice identifier: the depth index plus the z_max for that depth.
struct DepthSlice {
    index: usize,
    z_max: f64,
}

/// Read a variable-width value from the byte stream.
/// DataType determines the wire format (may be reduced from the original type).
fn read_variable_data_type(data: &[u8], pos: &mut usize, dt: DataType) -> Result<f64> {
    let size = dt.size();
    if *pos + size > data.len() {
        return Err(LercError::BufferTooSmall {
            needed: *pos + size,
            available: data.len(),
        });
    }
    let val = match dt {
        DataType::Char => {
            let v = data[*pos] as i8;
            *pos += 1;
            v as f64
        }
        DataType::Byte => {
            let v = data[*pos];
            *pos += 1;
            v as f64
        }
        DataType::Short => {
            let v = i16::from_le_bytes(data[*pos..*pos + 2].try_into().unwrap());
            *pos += 2;
            v as f64
        }
        DataType::UShort => {
            let v = u16::from_le_bytes(data[*pos..*pos + 2].try_into().unwrap());
            *pos += 2;
            v as f64
        }
        DataType::Int => {
            let v = i32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
            *pos += 4;
            v as f64
        }
        DataType::UInt => {
            let v = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
            *pos += 4;
            v as f64
        }
        DataType::Float => {
            let v = f32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
            *pos += 4;
            v as f64
        }
        DataType::Double => {
            let v = f64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
            *pos += 8;
            v
        }
    };
    Ok(val)
}

/// Write a variable-width value to the byte buffer.
pub(crate) fn write_variable_data_type(buf: &mut Vec<u8>, z: f64, dt: DataType) {
    match dt {
        DataType::Char => buf.push(z as i8 as u8),
        DataType::Byte => buf.push(z as u8),
        DataType::Short => buf.extend_from_slice(&(z as i16).to_le_bytes()),
        DataType::UShort => buf.extend_from_slice(&(z as u16).to_le_bytes()),
        DataType::Int => buf.extend_from_slice(&(z as i32).to_le_bytes()),
        DataType::UInt => buf.extend_from_slice(&(z as u32).to_le_bytes()),
        DataType::Float => buf.extend_from_slice(&(z as f32).to_le_bytes()),
        DataType::Double => buf.extend_from_slice(&z.to_le_bytes()),
    }
}

/// Get the reduced data type from the type code in the block header.
fn get_data_type_used(dt: DataType, tc: i32) -> Option<DataType> {
    let raw = match dt {
        DataType::Short | DataType::Int => (dt as i32) - tc,
        DataType::UShort | DataType::UInt => (dt as i32) - 2 * tc,
        DataType::Float => {
            return match tc {
                0 => Some(DataType::Float),
                1 => Some(DataType::Short),
                2 => Some(DataType::Byte),
                _ => None,
            };
        }
        DataType::Double => {
            if tc == 0 {
                return Some(DataType::Double);
            }
            (dt as i32) - 2 * tc + 1
        }
        _ => return Some(dt),
    };
    DataType::from_i32(raw)
}

/// Reduce a data type for storing the offset value more compactly.
pub(crate) fn reduce_data_type<T: Sample>(z: T, dt: DataType) -> (DataType, i32) {
    let zf = z.to_f64();
    match dt {
        DataType::Short => {
            let c = if (-128.0..=127.0).contains(&zf) {
                zf as i8
            } else {
                0
            };
            let b = if (0.0..=255.0).contains(&zf) {
                zf as u8
            } else {
                0
            };
            if T::from_f64(c as f64).to_f64() == zf {
                (DataType::Char, 2)
            } else if T::from_f64(b as f64).to_f64() == zf {
                (DataType::Byte, 1)
            } else {
                (dt, 0)
            }
        }
        DataType::UShort => {
            let b = if (0.0..=255.0).contains(&zf) {
                zf as u8
            } else {
                0
            };
            if T::from_f64(b as f64).to_f64() == zf {
                (DataType::Byte, 1)
            } else {
                (dt, 0)
            }
        }
        DataType::Int => {
            let b = if (0.0..=255.0).contains(&zf) {
                zf as u8
            } else {
                0
            };
            let s = if (-32768.0..=32767.0).contains(&zf) {
                zf as i16
            } else {
                0
            };
            let us = if (0.0..=65535.0).contains(&zf) {
                zf as u16
            } else {
                0
            };
            if T::from_f64(b as f64).to_f64() == zf {
                (DataType::Byte, 3)
            } else if T::from_f64(s as f64).to_f64() == zf {
                (DataType::Short, 2)
            } else if T::from_f64(us as f64).to_f64() == zf {
                (DataType::UShort, 1)
            } else {
                (dt, 0)
            }
        }
        DataType::UInt => {
            let b = if (0.0..=255.0).contains(&zf) {
                zf as u8
            } else {
                0
            };
            let us = if (0.0..=65535.0).contains(&zf) {
                zf as u16
            } else {
                0
            };
            if T::from_f64(b as f64).to_f64() == zf {
                (DataType::Byte, 2)
            } else if T::from_f64(us as f64).to_f64() == zf {
                (DataType::UShort, 1)
            } else {
                (dt, 0)
            }
        }
        DataType::Float => {
            let b = if (0.0..=255.0).contains(&zf) {
                zf as u8
            } else {
                0
            };
            let s = if (-32768.0..=32767.0).contains(&zf) {
                zf as i16
            } else {
                0
            };
            if T::from_f64(b as f64).to_f64() == zf {
                (DataType::Byte, 2)
            } else if T::from_f64(s as f64).to_f64() == zf {
                (DataType::Short, 1)
            } else {
                (dt, 0)
            }
        }
        DataType::Double => {
            let s = if (-32768.0..=32767.0).contains(&zf) {
                zf as i16
            } else {
                0
            };
            let l = if (i32::MIN as f64..=i32::MAX as f64).contains(&zf) {
                zf as i32
            } else {
                0
            };
            let f = if (f32::MIN as f64..=f32::MAX as f64).contains(&zf) {
                zf as f32
            } else {
                0.0
            };
            if T::from_f64(s as f64).to_f64() == zf {
                (DataType::Short, 3)
            } else if T::from_f64(l as f64).to_f64() == zf {
                (DataType::Int, 2)
            } else if T::from_f64(f as f64).to_f64() == zf {
                (DataType::Float, 1)
            } else {
                (dt, 0)
            }
        }
        _ => (dt, 0),
    }
}

/// Decode all tiles for a typed data array.
pub(crate) fn read_tiles<T: Sample>(
    data_buf: &[u8],
    pos: &mut usize,
    header: &HeaderInfo,
    mask: &BitMask,
    _z_min_vec: &[f64],
    z_max_vec: &[f64],
    output: &mut [T],
) -> Result<()> {
    let mb_size = header.micro_block_size as usize;
    let n_depth = header.n_depth as usize;
    let n_rows = header.n_rows as usize;
    let n_cols = header.n_cols as usize;

    let num_tiles_vert = n_rows.div_ceil(mb_size);
    let num_tiles_hori = n_cols.div_ceil(mb_size);

    for i_tile in 0..num_tiles_vert {
        let i0 = i_tile * mb_size;
        let i1 = (i0 + mb_size).min(n_rows);

        for j_tile in 0..num_tiles_hori {
            let j0 = j_tile * mb_size;
            let j1 = (j0 + mb_size).min(n_cols);
            let rect = TileRect { i0, i1, j0, j1 };

            for (i_depth, &z_max_d) in z_max_vec.iter().enumerate().take(n_depth) {
                let z_max = if header.version >= 4 && n_depth > 1 {
                    z_max_d
                } else {
                    header.z_max
                };
                let depth = DepthSlice {
                    index: i_depth,
                    z_max,
                };
                read_tile(data_buf, pos, header, mask, output, rect, &depth)?;
            }
        }
    }

    Ok(())
}

fn read_tile<T: Sample>(
    data: &[u8],
    pos: &mut usize,
    header: &HeaderInfo,
    mask: &BitMask,
    output: &mut [T],
    rect: TileRect,
    depth: &DepthSlice,
) -> Result<()> {
    let TileRect { i0, i1, j0, j1 } = rect;
    let i_depth = depth.index;
    if *pos >= data.len() {
        return Err(LercError::BufferTooSmall {
            needed: *pos + 1,
            available: data.len(),
        });
    }

    let n_cols = header.n_cols as usize;
    let n_depth = header.n_depth as usize;

    let compr_flag = data[*pos];
    *pos += 1;

    let b_diff_enc = if header.version >= 5 {
        (compr_flag & tile_flags::DIFF_ENCODING) != 0
    } else {
        false
    };

    // Integrity check: bits 2-5 encode (j0 >> 3) & pattern
    let pattern: u8 = if header.version >= 5 { 14 } else { 15 };
    if ((compr_flag >> 2) & pattern) != ((j0 as u8 >> 3) & pattern) {
        return Err(LercError::IntegrityCheckFailed);
    }

    if b_diff_enc && i_depth == 0 {
        return Err(LercError::InvalidData("diff encoding on depth 0".into()));
    }

    let bits67 = (compr_flag >> tile_flags::TYPE_REDUCTION_SHIFT) as i32;
    let compr_mode = compr_flag & tile_flags::MODE_MASK;

    if compr_mode == TileCompressionMode::ConstZero as u8 {
        for i in i0..i1 {
            let mut k = i * n_cols + j0;
            let mut m = k * n_depth + i_depth;
            for _j in j0..j1 {
                if mask.is_valid(k) {
                    output[m] = if b_diff_enc {
                        output[m - 1]
                    } else {
                        T::default()
                    };
                }
                k += 1;
                m += n_depth;
            }
        }
        return Ok(());
    }

    if compr_mode == TileCompressionMode::RawBinary as u8 {
        if b_diff_enc {
            return Err(LercError::InvalidData("raw binary with diff enc".into()));
        }

        let type_size = T::BYTES;
        for i in i0..i1 {
            let mut k = i * n_cols + j0;
            let mut m = k * n_depth + i_depth;
            for _j in j0..j1 {
                if mask.is_valid(k) {
                    if *pos + type_size > data.len() {
                        return Err(LercError::BufferTooSmall {
                            needed: *pos + type_size,
                            available: data.len(),
                        });
                    }
                    output[m] = read_typed_value::<T>(data, pos);
                }
                k += 1;
                m += n_depth;
            }
        }
        return Ok(());
    }

    // compr_mode == 1 or 3: bit-stuffed with offset
    let dt_for_reduction = if b_diff_enc && header.data_type.is_integer() {
        DataType::Int
    } else {
        header.data_type
    };
    let dt_used = get_data_type_used(dt_for_reduction, bits67)
        .ok_or(LercError::InvalidData("invalid reduced data type".into()))?;

    let offset = read_variable_data_type(data, pos, dt_used)?;

    let z_max = depth.z_max;

    if compr_mode == TileCompressionMode::ConstOffset as u8 {
        for i in i0..i1 {
            let mut k = i * n_cols + j0;
            let mut m = k * n_depth + i_depth;
            for _j in j0..j1 {
                if mask.is_valid(k) {
                    if !b_diff_enc {
                        output[m] = T::from_f64(offset);
                    } else {
                        let z = offset + output[m - 1].to_f64();
                        output[m] = T::from_f64(z.min(z_max));
                    }
                }
                k += 1;
                m += n_depth;
            }
        }
        return Ok(());
    }

    // compr_mode == 1: bit-stuffed quantized data
    let max_element_count = (i1 - i0) * (j1 - j0);
    let buffer_vec = bitstuffer::decode(data, pos, max_element_count, header.version)?;

    let inv_scale = 2.0 * header.max_z_error;
    let mut src_idx = 0;

    let all_valid = buffer_vec.len() == max_element_count;

    // Check once per tile whether clamping is needed: if the maximum possible
    // dequantized value cannot exceed z_max, we can skip the min() entirely.
    // max_quant is the largest quantized value in the buffer.
    let needs_clamp = if !b_diff_enc {
        // For non-diff: z = offset + quant * inv_scale, max when quant is largest
        let max_quant = buffer_vec.iter().copied().max().unwrap_or(0) as f64;
        offset + max_quant * inv_scale > z_max
    } else {
        // For diff encoding, always clamp (conservative; diff encoding is rare)
        true
    };

    if all_valid && !b_diff_enc && n_depth == 1 {
        // Fast path: all pixels valid, no diff encoding, contiguous output (nDepth=1).
        // This allows the compiler to vectorize the inner loop.
        if needs_clamp {
            for i in i0..i1 {
                let row_start = i * n_cols + j0;
                let row_len = j1 - j0;
                let out_slice = &mut output[row_start..row_start + row_len];
                let src_slice = &buffer_vec[src_idx..src_idx + row_len];
                for (dst, &q) in out_slice.iter_mut().zip(src_slice.iter()) {
                    let z = offset + q as f64 * inv_scale;
                    *dst = T::from_f64(z.min(z_max));
                }
                src_idx += row_len;
            }
        } else {
            for i in i0..i1 {
                let row_start = i * n_cols + j0;
                let row_len = j1 - j0;
                let out_slice = &mut output[row_start..row_start + row_len];
                let src_slice = &buffer_vec[src_idx..src_idx + row_len];
                for (dst, &q) in out_slice.iter_mut().zip(src_slice.iter()) {
                    *dst = T::from_f64(offset + q as f64 * inv_scale);
                }
                src_idx += row_len;
            }
        }
    } else if all_valid && !b_diff_enc {
        // All valid, no diff, but n_depth > 1 (strided output)
        if needs_clamp {
            for i in i0..i1 {
                let mut m = (i * n_cols + j0) * n_depth + i_depth;
                for _ in j0..j1 {
                    let z = offset + buffer_vec[src_idx] as f64 * inv_scale;
                    output[m] = T::from_f64(z.min(z_max));
                    src_idx += 1;
                    m += n_depth;
                }
            }
        } else {
            for i in i0..i1 {
                let mut m = (i * n_cols + j0) * n_depth + i_depth;
                for _ in j0..j1 {
                    output[m] = T::from_f64(offset + buffer_vec[src_idx] as f64 * inv_scale);
                    src_idx += 1;
                    m += n_depth;
                }
            }
        }
    } else {
        // General path: mask checks and/or diff encoding
        for i in i0..i1 {
            let mut k = i * n_cols + j0;
            let mut m = k * n_depth + i_depth;
            for _j in j0..j1 {
                let valid = if all_valid { true } else { mask.is_valid(k) };
                if valid {
                    let quant = buffer_vec[src_idx] as f64;
                    src_idx += 1;

                    if !b_diff_enc {
                        let z = offset + quant * inv_scale;
                        output[m] = T::from_f64(z.min(z_max));
                    } else {
                        let z = offset + quant * inv_scale + output[m - 1].to_f64();
                        output[m] = T::from_f64(z.min(z_max));
                    }
                }
                k += 1;
                m += n_depth;
            }
        }
    }

    Ok(())
}

/// Read a typed value from LE bytes, returning as f64.
pub(crate) fn read_typed_as_f64<T: Sample>(data: &[u8], pos: &mut usize) -> f64 {
    let v = T::from_le_slice(&data[*pos..]);
    *pos += T::BYTES;
    v.to_f64()
}

/// Read a typed value from LE bytes, returning as T.
pub(crate) fn read_typed_value<T: Sample>(data: &[u8], pos: &mut usize) -> T {
    let v = T::from_le_slice(&data[*pos..]);
    *pos += T::BYTES;
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // read_variable_data_type
    // -----------------------------------------------------------------------

    #[test]
    fn read_variable_data_type_i8() {
        // i8 -1 → 0xFF as byte
        let data = [0xFF_u8];
        let mut pos = 0;
        let val = read_variable_data_type(&data, &mut pos, DataType::Char).unwrap();
        assert_eq!(val, -1.0);
        assert_eq!(pos, 1);
    }

    #[test]
    fn read_variable_data_type_u8() {
        let data = [255_u8];
        let mut pos = 0;
        let val = read_variable_data_type(&data, &mut pos, DataType::Byte).unwrap();
        assert_eq!(val, 255.0);
        assert_eq!(pos, 1);
    }

    #[test]
    fn read_variable_data_type_i16() {
        let data = (-1000_i16).to_le_bytes();
        let mut pos = 0;
        let val = read_variable_data_type(&data, &mut pos, DataType::Short).unwrap();
        assert_eq!(val, -1000.0);
        assert_eq!(pos, 2);
    }

    #[test]
    fn read_variable_data_type_u16() {
        let data = 60000_u16.to_le_bytes();
        let mut pos = 0;
        let val = read_variable_data_type(&data, &mut pos, DataType::UShort).unwrap();
        assert_eq!(val, 60000.0);
        assert_eq!(pos, 2);
    }

    #[test]
    fn read_variable_data_type_i32() {
        let data = (-123456_i32).to_le_bytes();
        let mut pos = 0;
        let val = read_variable_data_type(&data, &mut pos, DataType::Int).unwrap();
        assert_eq!(val, -123456.0);
        assert_eq!(pos, 4);
    }

    #[test]
    fn read_variable_data_type_u32() {
        let data = 4_000_000_000_u32.to_le_bytes();
        let mut pos = 0;
        let val = read_variable_data_type(&data, &mut pos, DataType::UInt).unwrap();
        assert_eq!(val, 4_000_000_000.0);
        assert_eq!(pos, 4);
    }

    #[test]
    fn read_variable_data_type_f32() {
        let test_val = 3.25_f32;
        let data = test_val.to_le_bytes();
        let mut pos = 0;
        let val = read_variable_data_type(&data, &mut pos, DataType::Float).unwrap();
        // f32 -> f64 conversion: compare with the actual f32 value promoted to f64
        assert_eq!(val, test_val as f64);
        assert_eq!(pos, 4);
    }

    #[test]
    fn read_variable_data_type_f64() {
        let data = core::f64::consts::PI.to_le_bytes();
        let mut pos = 0;
        let val = read_variable_data_type(&data, &mut pos, DataType::Double).unwrap();
        assert_eq!(val, core::f64::consts::PI);
        assert_eq!(pos, 8);
    }

    #[test]
    fn read_variable_data_type_buffer_too_small() {
        let data = [0_u8; 1]; // only 1 byte, but i16 needs 2
        let mut pos = 0;
        let result = read_variable_data_type(&data, &mut pos, DataType::Short);
        assert!(result.is_err());
    }

    #[test]
    fn read_variable_data_type_with_offset() {
        // Place i16 at offset 3
        let mut data = vec![0_u8; 5];
        let val_bytes = 42_i16.to_le_bytes();
        data[3] = val_bytes[0];
        data[4] = val_bytes[1];
        let mut pos = 3;
        let val = read_variable_data_type(&data, &mut pos, DataType::Short).unwrap();
        assert_eq!(val, 42.0);
        assert_eq!(pos, 5);
    }

    // -----------------------------------------------------------------------
    // reduce_data_type
    // -----------------------------------------------------------------------

    #[test]
    fn reduce_i32_value_100_to_byte() {
        let (dt, tc) = reduce_data_type(100_i32, DataType::Int);
        assert_eq!(dt, DataType::Byte);
        assert_eq!(tc, 3);
    }

    #[test]
    fn reduce_i32_value_1000_to_short() {
        let (dt, tc) = reduce_data_type(1000_i32, DataType::Int);
        assert_eq!(dt, DataType::Short);
        assert_eq!(tc, 2);
    }

    #[test]
    fn reduce_u16_value_200_to_byte() {
        let (dt, tc) = reduce_data_type(200_u16, DataType::UShort);
        assert_eq!(dt, DataType::Byte);
        assert_eq!(tc, 1);
    }

    #[test]
    fn reduce_f32_value_42_to_byte() {
        let (dt, tc) = reduce_data_type(42.0_f32, DataType::Float);
        assert_eq!(dt, DataType::Byte);
        assert_eq!(tc, 2);
    }

    #[test]
    fn reduce_i32_large_value_no_reduction() {
        // 100_000 exceeds u16 range (0..65535), so no reduction possible
        let (dt, tc) = reduce_data_type(100_000_i32, DataType::Int);
        assert_eq!(dt, DataType::Int);
        assert_eq!(tc, 0);
    }

    #[test]
    fn reduce_i32_negative_no_byte_reduction() {
        // -50 fits in i8/i16 but not u8 for Int type; should reduce to Short
        let (dt, tc) = reduce_data_type(-50_i32, DataType::Int);
        assert_eq!(dt, DataType::Short);
        assert_eq!(tc, 2);
    }

    #[test]
    fn reduce_u16_value_300_no_reduction() {
        let (dt, tc) = reduce_data_type(300_u16, DataType::UShort);
        assert_eq!(dt, DataType::UShort);
        assert_eq!(tc, 0);
    }

    #[test]
    fn reduce_f32_fractional_no_reduction() {
        let (dt, tc) = reduce_data_type(3.25_f32, DataType::Float);
        assert_eq!(dt, DataType::Float);
        assert_eq!(tc, 0);
    }

    #[test]
    fn reduce_f64_value_100_to_short() {
        let (dt, tc) = reduce_data_type(100.0_f64, DataType::Double);
        assert_eq!(dt, DataType::Short);
        assert_eq!(tc, 3);
    }

    #[test]
    fn reduce_u32_value_50_to_byte() {
        let (dt, tc) = reduce_data_type(50_u32, DataType::UInt);
        assert_eq!(dt, DataType::Byte);
        assert_eq!(tc, 2);
    }

    #[test]
    fn reduce_i16_value_100_to_char() {
        // i16 value 100 fits in i8 (-128..127) so should reduce to Char
        let (dt, tc) = reduce_data_type(100_i16, DataType::Short);
        assert_eq!(dt, DataType::Char);
        assert_eq!(tc, 2);
    }

    // -----------------------------------------------------------------------
    // read_typed_as_f64
    // -----------------------------------------------------------------------

    #[test]
    fn read_typed_as_f64_i8() {
        let data = [0xFE_u8]; // -2 as i8
        let mut pos = 0;
        let val = read_typed_as_f64::<i8>(&data, &mut pos);
        assert_eq!(val, -2.0);
        assert_eq!(pos, 1);
    }

    #[test]
    fn read_typed_as_f64_u8() {
        let data = [200_u8];
        let mut pos = 0;
        let val = read_typed_as_f64::<u8>(&data, &mut pos);
        assert_eq!(val, 200.0);
        assert_eq!(pos, 1);
    }

    #[test]
    fn read_typed_as_f64_i16() {
        let data = (-500_i16).to_le_bytes();
        let mut pos = 0;
        let val = read_typed_as_f64::<i16>(&data, &mut pos);
        assert_eq!(val, -500.0);
        assert_eq!(pos, 2);
    }

    #[test]
    fn read_typed_as_f64_u16() {
        let data = 50000_u16.to_le_bytes();
        let mut pos = 0;
        let val = read_typed_as_f64::<u16>(&data, &mut pos);
        assert_eq!(val, 50000.0);
        assert_eq!(pos, 2);
    }

    #[test]
    fn read_typed_as_f64_i32() {
        let data = (-999999_i32).to_le_bytes();
        let mut pos = 0;
        let val = read_typed_as_f64::<i32>(&data, &mut pos);
        assert_eq!(val, -999999.0);
        assert_eq!(pos, 4);
    }

    #[test]
    fn read_typed_as_f64_u32() {
        let data = 3_000_000_000_u32.to_le_bytes();
        let mut pos = 0;
        let val = read_typed_as_f64::<u32>(&data, &mut pos);
        assert_eq!(val, 3_000_000_000.0);
        assert_eq!(pos, 4);
    }

    #[test]
    fn read_typed_as_f64_f32() {
        let test_val = 2.75_f32;
        let data = test_val.to_le_bytes();
        let mut pos = 0;
        let val = read_typed_as_f64::<f32>(&data, &mut pos);
        assert_eq!(val, test_val as f64);
        assert_eq!(pos, 4);
    }

    #[test]
    fn read_typed_as_f64_f64() {
        let data = 1.23456789012345_f64.to_le_bytes();
        let mut pos = 0;
        let val = read_typed_as_f64::<f64>(&data, &mut pos);
        assert_eq!(val, 1.23456789012345);
        assert_eq!(pos, 8);
    }

    // -----------------------------------------------------------------------
    // read_typed_value
    // -----------------------------------------------------------------------

    #[test]
    fn read_typed_value_i8() {
        let data = [0x80_u8]; // -128 as i8
        let mut pos = 0;
        let val: i8 = read_typed_value(&data, &mut pos);
        assert_eq!(val, -128);
        assert_eq!(pos, 1);
    }

    #[test]
    fn read_typed_value_u8() {
        let data = [42_u8];
        let mut pos = 0;
        let val: u8 = read_typed_value(&data, &mut pos);
        assert_eq!(val, 42);
        assert_eq!(pos, 1);
    }

    #[test]
    fn read_typed_value_i16() {
        let data = 12345_i16.to_le_bytes();
        let mut pos = 0;
        let val: i16 = read_typed_value(&data, &mut pos);
        assert_eq!(val, 12345);
        assert_eq!(pos, 2);
    }

    #[test]
    fn read_typed_value_u16() {
        let data = 65535_u16.to_le_bytes();
        let mut pos = 0;
        let val: u16 = read_typed_value(&data, &mut pos);
        assert_eq!(val, 65535);
        assert_eq!(pos, 2);
    }

    #[test]
    fn read_typed_value_i32() {
        let data = (-2_000_000_000_i32).to_le_bytes();
        let mut pos = 0;
        let val: i32 = read_typed_value(&data, &mut pos);
        assert_eq!(val, -2_000_000_000);
        assert_eq!(pos, 4);
    }

    #[test]
    fn read_typed_value_u32() {
        let data = 4_294_967_295_u32.to_le_bytes();
        let mut pos = 0;
        let val: u32 = read_typed_value(&data, &mut pos);
        assert_eq!(val, u32::MAX);
        assert_eq!(pos, 4);
    }

    #[test]
    fn read_typed_value_f32() {
        let expected = 1.5_f32;
        let data = expected.to_le_bytes();
        let mut pos = 0;
        let val: f32 = read_typed_value(&data, &mut pos);
        assert_eq!(val, expected);
        assert_eq!(pos, 4);
    }

    #[test]
    fn read_typed_value_f64() {
        let expected = -999.999_f64;
        let data = expected.to_le_bytes();
        let mut pos = 0;
        let val: f64 = read_typed_value(&data, &mut pos);
        assert_eq!(val, expected);
        assert_eq!(pos, 8);
    }

    // -----------------------------------------------------------------------
    // get_data_type_used (the inverse of reduce_data_type's tc values)
    // -----------------------------------------------------------------------

    #[test]
    fn get_data_type_used_identity() {
        // tc=0 should always return the same type
        assert_eq!(get_data_type_used(DataType::Char, 0), Some(DataType::Char));
        assert_eq!(get_data_type_used(DataType::Byte, 0), Some(DataType::Byte));
        assert_eq!(
            get_data_type_used(DataType::Short, 0),
            Some(DataType::Short)
        );
        assert_eq!(
            get_data_type_used(DataType::UShort, 0),
            Some(DataType::UShort)
        );
        assert_eq!(get_data_type_used(DataType::Int, 0), Some(DataType::Int));
        assert_eq!(get_data_type_used(DataType::UInt, 0), Some(DataType::UInt));
        assert_eq!(
            get_data_type_used(DataType::Float, 0),
            Some(DataType::Float)
        );
        assert_eq!(
            get_data_type_used(DataType::Double, 0),
            Some(DataType::Double)
        );
    }

    #[test]
    fn get_data_type_used_float_reduction() {
        assert_eq!(
            get_data_type_used(DataType::Float, 1),
            Some(DataType::Short)
        );
        assert_eq!(get_data_type_used(DataType::Float, 2), Some(DataType::Byte));
        assert_eq!(get_data_type_used(DataType::Float, 3), None);
    }

    #[test]
    fn get_data_type_used_int_reduction() {
        // Int(4) - tc: tc=1 -> UShort(3), tc=2 -> Short(2), tc=3 -> Byte(1)
        assert_eq!(get_data_type_used(DataType::Int, 1), Some(DataType::UShort));
        assert_eq!(get_data_type_used(DataType::Int, 2), Some(DataType::Short));
        assert_eq!(get_data_type_used(DataType::Int, 3), Some(DataType::Byte));
    }

    // -----------------------------------------------------------------------
    // write_variable_data_type round-trips
    // -----------------------------------------------------------------------

    #[test]
    fn write_then_read_variable_data_type() {
        let f32_test_val = 3.25_f32;
        let test_cases: &[(f64, DataType)] = &[
            (-1.0, DataType::Char),
            (255.0, DataType::Byte),
            (-1000.0, DataType::Short),
            (60000.0, DataType::UShort),
            (-123456.0, DataType::Int),
            (4_000_000_000.0, DataType::UInt),
            (f32_test_val as f64, DataType::Float),
            (core::f64::consts::PI, DataType::Double),
        ];

        for &(value, dt) in test_cases {
            let mut buf = Vec::new();
            write_variable_data_type(&mut buf, value, dt);
            let mut pos = 0;
            let result = read_variable_data_type(&buf, &mut pos, dt).unwrap();
            assert_eq!(result, value, "round-trip failed for {dt:?} value={value}");
            assert_eq!(pos, buf.len());
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn reduce_data_type_preserves_value(val in -128i32..127) {
                let (_dt_reduced, tc) = reduce_data_type(val, DataType::Int);
                // The reduced type must be able to represent the value
                let recovered_dt = get_data_type_used(DataType::Int, tc);
                prop_assert!(recovered_dt.is_some());
            }
        }
    }
}
