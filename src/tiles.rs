use alloc::vec::Vec;

use crate::bitstuffer;
use crate::bitmask::BitMask;
use crate::error::{LercError, Result};
use crate::header::HeaderInfo;
use crate::types::{DataType, LercDataType, TileCompressionMode, tile_flags};

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
pub(crate) fn reduce_data_type<T: LercDataType>(z: T, dt: DataType) -> (DataType, i32) {
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
pub(crate) fn read_tiles<T: LercDataType>(
    data_buf: &[u8],
    pos: &mut usize,
    header: &HeaderInfo,
    mask: &BitMask,
    z_min_vec: &[f64],
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

            for i_depth in 0..n_depth {
                read_tile(
                    data_buf, pos, header, mask, z_min_vec, z_max_vec, output, i0, i1, j0,
                    j1, i_depth,
                )?;
            }
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn read_tile<T: LercDataType>(
    data: &[u8],
    pos: &mut usize,
    header: &HeaderInfo,
    mask: &BitMask,
    _z_min_vec: &[f64],
    z_max_vec: &[f64],
    output: &mut [T],
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    i_depth: usize,
) -> Result<()> {
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
                    output[m] = if b_diff_enc { output[m - 1] } else { T::default() };
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

    let z_max = if header.version >= 4 && n_depth > 1 {
        z_max_vec[i_depth]
    } else {
        header.z_max
    };

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
pub(crate) fn read_typed_as_f64<T: LercDataType>(data: &[u8], pos: &mut usize) -> f64 {
    let size = T::BYTES;
    let mut bytes = [0u8; 8];
    bytes[..size].copy_from_slice(&data[*pos..*pos + size]);
    *pos += size;

    match T::DATA_TYPE {
        DataType::Char => i8::from_le_bytes([bytes[0]]) as f64,
        DataType::Byte => bytes[0] as f64,
        DataType::Short => i16::from_le_bytes([bytes[0], bytes[1]]) as f64,
        DataType::UShort => u16::from_le_bytes([bytes[0], bytes[1]]) as f64,
        DataType::Int => {
            i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64
        }
        DataType::UInt => {
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64
        }
        DataType::Float => {
            f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64
        }
        DataType::Double => f64::from_le_bytes(bytes),
    }
}

/// Read a typed value from LE bytes, returning as T.
pub(crate) fn read_typed_value<T: LercDataType>(data: &[u8], pos: &mut usize) -> T {
    T::from_f64(read_typed_as_f64::<T>(data, pos))
}
