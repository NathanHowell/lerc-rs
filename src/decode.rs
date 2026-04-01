use alloc::vec;
use alloc::vec::Vec;

use crate::bitmask::BitMask;
use crate::error::{LercError, Result};
use crate::fpl;
use crate::header::{self, HeaderInfo};
use crate::huffman::HuffmanCodec;
use crate::rle;
use crate::tiles;
use crate::types::{DataType, LercDataType};
use crate::{DecodeResult, LercData, LercImage, LercInfo};

use crate::types::ImageEncodeMode;

pub fn decode_info(data: &[u8]) -> Result<LercInfo> {
    if crate::lerc1::is_lerc1(data) {
        return crate::lerc1::decode_info(data);
    }

    let (hd, _) = header::read_header(data)?;

    // Count total bands by walking concatenated blobs
    let n_bands = if hd.version >= 6 && hd.n_blobs_more > 0 {
        1 + hd.n_blobs_more as u32
    } else {
        // For older versions, scan for more blobs
        let mut count = 1u32;
        let mut offset = hd.blob_size as usize;
        while offset < data.len() {
            if let Ok((next_hd, _)) = header::read_header(&data[offset..]) {
                count += 1;
                offset += next_hd.blob_size as usize;
            } else {
                break;
            }
        }
        count
    };

    let no_data_value = if hd.pass_no_data_values {
        Some(hd.no_data_val_orig)
    } else {
        None
    };

    Ok(LercInfo {
        version: hd.version,
        width: hd.n_cols as u32,
        height: hd.n_rows as u32,
        n_depth: hd.n_depth as u32,
        n_bands,
        data_type: hd.data_type,
        num_valid_pixels: hd.num_valid_pixel as u32,
        max_z_error: hd.max_z_error,
        z_min: hd.z_min,
        z_max: hd.z_max,
        blob_size: hd.blob_size as u32,
        no_data_value,
    })
}

pub fn decode(data: &[u8]) -> Result<LercImage> {
    if crate::lerc1::is_lerc1(data) {
        return crate::lerc1::decode(data);
    }

    let info = decode_info(data)?;
    let n_bands = info.n_bands;
    let width = info.width;
    let height = info.height;
    let n_depth = info.n_depth;
    let dt = info.data_type;
    let no_data_value = info.no_data_value;

    let total_pixels = width as usize * height as usize * n_depth as usize * n_bands as usize;

    macro_rules! decode_bands {
        ($default:expr, $variant:ident) => {{
            let mut output = vec![$default; total_pixels];
            let result = decode_bands_into(data, &info, &mut output)?;
            Ok(LercImage {
                width,
                height,
                n_depth,
                n_bands,
                data_type: dt,
                valid_masks: result.valid_masks,
                data: LercData::$variant(output),
                no_data_value,
            })
        }};
    }

    match dt {
        DataType::Char => decode_bands!(0i8, I8),
        DataType::Byte => decode_bands!(0u8, U8),
        DataType::Short => decode_bands!(0i16, I16),
        DataType::UShort => decode_bands!(0u16, U16),
        DataType::Int => decode_bands!(0i32, I32),
        DataType::UInt => decode_bands!(0u32, U32),
        DataType::Float => decode_bands!(0.0f32, F32),
        DataType::Double => decode_bands!(0.0f64, F64),
    }
}

/// Decode LERC2 data into a pre-allocated typed buffer, returning metadata.
/// The buffer must have at least `width * height * n_depth * n_bands` elements.
/// The data type `T` must match the blob's data type.
pub fn decode_into<T: LercDataType>(data: &[u8], output: &mut [T]) -> Result<DecodeResult> {
    if crate::lerc1::is_lerc1(data) {
        return Err(LercError::InvalidData(
            "decode_into does not support Lerc1 format".into(),
        ));
    }

    let info = decode_info(data)?;

    // Verify type match
    if T::DATA_TYPE != info.data_type {
        return Err(LercError::TypeMismatch {
            expected: info.data_type,
            actual: T::DATA_TYPE,
        });
    }

    let total_pixels =
        info.width as usize * info.height as usize * info.n_depth as usize * info.n_bands as usize;

    // Verify buffer size
    if output.len() < total_pixels {
        return Err(LercError::OutputBufferTooSmall {
            needed: total_pixels,
            available: output.len(),
        });
    }

    decode_bands_into(data, &info, output)
}

/// Shared implementation: decode all bands from LERC2 data into a pre-allocated buffer.
fn decode_bands_into<T: LercDataType>(
    data: &[u8],
    info: &LercInfo,
    output: &mut [T],
) -> Result<DecodeResult> {
    let n_bands = info.n_bands;
    let band_pixels = info.width as usize * info.height as usize * info.n_depth as usize;

    let mut masks = Vec::with_capacity(n_bands as usize);
    let mut offset = 0usize;

    for band in 0..n_bands as usize {
        let prev_mask = masks.last();
        let (mask, hd, consumed) = decode_one_band(
            &data[offset..],
            &mut output[band * band_pixels..(band + 1) * band_pixels],
            prev_mask,
        )?;

        // Remap noData sentinel values if needed (v6+, nDepth > 1)
        if hd.pass_no_data_values && hd.no_data_val != hd.no_data_val_orig {
            remap_no_data(
                &mut output[band * band_pixels..(band + 1) * band_pixels],
                &mask,
                &hd,
            );
        }

        masks.push(mask);
        offset += consumed;
    }

    Ok(DecodeResult {
        width: info.width,
        height: info.height,
        n_depth: info.n_depth,
        n_bands: info.n_bands,
        data_type: info.data_type,
        valid_masks: masks,
        no_data_value: info.no_data_value,
    })
}

/// Remap noData sentinel values from the internal encoding value back to the
/// original user-specified noData value. This mirrors the C++ `RemapNoData`.
fn remap_no_data<T: LercDataType>(
    output: &mut [T],
    mask: &BitMask,
    hd: &HeaderInfo,
) {
    let n_cols = hd.n_cols as usize;
    let n_rows = hd.n_rows as usize;
    let n_depth = hd.n_depth as usize;

    let no_data_old = T::from_f64(hd.no_data_val);
    let no_data_new = T::from_f64(hd.no_data_val_orig);

    for i in 0..n_rows {
        let row_start = i * n_cols * n_depth;
        for j in 0..n_cols {
            let k = i * n_cols + j;
            if mask.is_valid(k) {
                let base = row_start + j * n_depth;
                for m in 0..n_depth {
                    if output[base + m].to_f64() == no_data_old.to_f64() {
                        output[base + m] = no_data_new;
                    }
                }
            }
        }
    }
}

/// Decode one band blob, returning the mask, header info, and number of bytes consumed.
fn decode_one_band<T: LercDataType>(
    blob: &[u8],
    output: &mut [T],
    prev_mask: Option<&BitMask>,
) -> Result<(BitMask, HeaderInfo, usize)> {
    let (hd, header_size) = header::read_header(blob)?;
    let blob_size = hd.blob_size as usize;

    // Verify checksum
    header::verify_checksum(blob, &hd)?;

    let mut pos = header_size;

    // Read mask
    let mask = read_mask(blob, &mut pos, &hd, prev_mask)?;

    // Zero the output for images with invalid pixels. For fully valid images,
    // every element will be overwritten by the decode path, so skip zeroing.
    let all_valid = hd.num_valid_pixel == hd.n_rows * hd.n_cols;
    if !all_valid {
        for v in output.iter_mut() {
            *v = T::default();
        }
    }

    if hd.num_valid_pixel == 0 {
        return Ok((mask, hd, blob_size));
    }

    // Const image
    if hd.z_min == hd.z_max {
        fill_const_image(&hd, &mask, &[], &[], output)?;
        return Ok((mask, hd, blob_size));
    }

    // Read per-depth min/max ranges (v4+)
    let (z_min_vec, z_max_vec) = if hd.version >= 4 {
        read_min_max_ranges::<T>(blob, &mut pos, &hd)?
    } else {
        (vec![hd.z_min], vec![hd.z_max])
    };

    // Check if all bands constant
    if z_min_vec == z_max_vec {
        fill_const_image(&hd, &mask, &z_min_vec, &z_max_vec, output)?;
        return Ok((mask, hd, blob_size));
    }

    // Read "one sweep" flag
    if pos >= blob.len() {
        return Err(LercError::BufferTooSmall {
            needed: pos + 1,
            available: blob.len(),
        });
    }
    let one_sweep = blob[pos];
    pos += 1;

    if one_sweep != 0 {
        // Read data one sweep (raw binary for all valid pixels)
        read_data_one_sweep(blob, &mut pos, &hd, &mask, output)?;
    } else {
        // Check for Huffman modes
        let try_huffman_int = hd.version >= 2
            && matches!(hd.data_type, DataType::Char | DataType::Byte)
            && hd.max_z_error == 0.5;
        let try_huffman_flt = hd.version >= 6
            && matches!(hd.data_type, DataType::Float | DataType::Double)
            && hd.max_z_error == 0.0;

        if try_huffman_int || try_huffman_flt {
            if pos >= blob.len() {
                return Err(LercError::BufferTooSmall {
                    needed: pos + 1,
                    available: blob.len(),
                });
            }
            let flag = blob[pos];
            pos += 1;

            if flag > 3
                || (flag > 2 && hd.version < 6)
                || (flag > 1 && hd.version < 4)
            {
                return Err(LercError::InvalidData("invalid image encode mode".into()));
            }

            let mode = ImageEncodeMode::try_from(flag)?;

            if mode != ImageEncodeMode::Tiling {
                if try_huffman_int {
                    if mode == ImageEncodeMode::DeltaHuffman
                        || (hd.version >= 4 && mode == ImageEncodeMode::Huffman)
                    {
                        decode_huffman(blob, &mut pos, &hd, &mask, mode, output)?;
                        return Ok((mask, hd, blob_size));
                    } else {
                        return Err(LercError::InvalidData(
                            "invalid huffman mode for int".into(),
                        ));
                    }
                } else if try_huffman_flt
                    && mode == ImageEncodeMode::DeltaDeltaHuffman
                {
                    fpl::decode_huffman_flt(
                        blob,
                        &mut pos,
                        hd.data_type == DataType::Double,
                        hd.n_cols as usize,
                        hd.n_rows as usize,
                        hd.n_depth as usize,
                        output,
                    )?;
                    return Ok((mask, hd, blob_size));
                } else {
                    return Err(LercError::InvalidData(
                        "invalid huffman mode for float".into(),
                    ));
                }
            }
        }

        // Tiling mode
        tiles::read_tiles(blob, &mut pos, &hd, &mask, &z_min_vec, &z_max_vec, output)?;
    }

    Ok((mask, hd, blob_size))
}

fn read_mask(
    data: &[u8],
    pos: &mut usize,
    header: &HeaderInfo,
    prev_mask: Option<&BitMask>,
) -> Result<BitMask> {
    if *pos + 4 > data.len() {
        return Err(LercError::BufferTooSmall {
            needed: *pos + 4,
            available: data.len(),
        });
    }

    let num_bytes_mask = i32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;

    let num_pixels = header.n_rows as usize * header.n_cols as usize;

    if header.num_valid_pixel == 0 {
        return Ok(BitMask::new(num_pixels));
    }

    if header.num_valid_pixel == header.n_rows * header.n_cols {
        return Ok(BitMask::all_valid(num_pixels));
    }

    if num_bytes_mask <= 0 {
        // Reuse previous mask if available
        if let Some(prev) = prev_mask {
            return Ok(prev.clone());
        }
        return Err(LercError::InvalidData("expected mask data".into()));
    }

    let mask_size = (num_pixels + 7) >> 3;
    if *pos + num_bytes_mask as usize > data.len() {
        return Err(LercError::BufferTooSmall {
            needed: *pos + num_bytes_mask as usize,
            available: data.len(),
        });
    }

    let mask_bytes =
        rle::decompress(&data[*pos..*pos + num_bytes_mask as usize], mask_size)?;
    *pos += num_bytes_mask as usize;

    Ok(BitMask::from_bytes(mask_bytes, num_pixels))
}

fn read_min_max_ranges<T: LercDataType>(
    data: &[u8],
    pos: &mut usize,
    header: &HeaderInfo,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let n_depth = header.n_depth as usize;
    let type_size = T::BYTES;
    let len = n_depth * type_size;

    if *pos + 2 * len > data.len() {
        return Err(LercError::BufferTooSmall {
            needed: *pos + 2 * len,
            available: data.len(),
        });
    }

    let mut z_min_vec = Vec::with_capacity(n_depth);
    for _ in 0..n_depth {
        let val = read_typed_as_f64::<T>(data, pos);
        z_min_vec.push(val);
    }

    let mut z_max_vec = Vec::with_capacity(n_depth);
    for _ in 0..n_depth {
        let val = read_typed_as_f64::<T>(data, pos);
        z_max_vec.push(val);
    }

    Ok((z_min_vec, z_max_vec))
}

fn read_typed_as_f64<T: LercDataType>(data: &[u8], pos: &mut usize) -> f64 {
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

fn fill_const_image<T: LercDataType>(
    header: &HeaderInfo,
    mask: &BitMask,
    z_min_vec: &[f64],
    _z_max_vec: &[f64],
    output: &mut [T],
) -> Result<()> {
    let n_cols = header.n_cols as usize;
    let n_rows = header.n_rows as usize;
    let n_depth = header.n_depth as usize;

    if n_depth == 1 {
        let z0 = T::from_f64(header.z_min);
        for (k, out) in output[..n_rows * n_cols].iter_mut().enumerate() {
            if mask.is_valid(k) {
                *out = z0;
            }
        }
    } else {
        let mut z_buf: Vec<T> = Vec::with_capacity(n_depth);
        if header.z_min == header.z_max || z_min_vec.is_empty() {
            z_buf.resize(n_depth, T::from_f64(header.z_min));
        } else {
            for val in &z_min_vec[..n_depth] {
                z_buf.push(T::from_f64(*val));
            }
        }

        for k in 0..n_rows * n_cols {
            if mask.is_valid(k) {
                let m0 = k * n_depth;
                output[m0..m0 + n_depth].copy_from_slice(&z_buf[..n_depth]);
            }
        }
    }

    Ok(())
}

fn read_data_one_sweep<T: LercDataType>(
    data: &[u8],
    pos: &mut usize,
    header: &HeaderInfo,
    mask: &BitMask,
    output: &mut [T],
) -> Result<()> {
    let n_depth = header.n_depth as usize;
    let type_size = T::BYTES;
    let n_cols = header.n_cols as usize;
    let n_rows = header.n_rows as usize;

    for i in 0..n_rows {
        for j in 0..n_cols {
            let k = i * n_cols + j;
            if mask.is_valid(k) {
                let m0 = k * n_depth;
                for m in 0..n_depth {
                    if *pos + type_size > data.len() {
                        return Err(LercError::BufferTooSmall {
                            needed: *pos + type_size,
                            available: data.len(),
                        });
                    }
                    output[m0 + m] = read_typed_value::<T>(data, pos);
                }
            }
        }
    }

    Ok(())
}

fn read_typed_value<T: LercDataType>(data: &[u8], pos: &mut usize) -> T {
    let val = read_typed_as_f64::<T>(data, pos);
    T::from_f64(val)
}

fn decode_huffman<T: LercDataType>(
    data: &[u8],
    pos: &mut usize,
    header: &HeaderInfo,
    mask: &BitMask,
    mode: ImageEncodeMode,
    output: &mut [T],
) -> Result<()> {
    let mut codec = HuffmanCodec::new();
    codec.read_code_table(data, pos, header.version)?;
    let num_bits_lut = codec.build_tree_from_codes()?;

    let offset = if header.data_type == DataType::Char {
        128
    } else {
        0
    };

    let height = header.n_rows as usize;
    let width = header.n_cols as usize;
    let n_depth = header.n_depth as usize;

    let mut byte_pos = *pos;
    let mut bit_pos = 0i32;
    let start_byte_pos = byte_pos;

    let all_valid = header.num_valid_pixel == header.n_rows * header.n_cols;

    // For 8-bit types, use wrapping integer arithmetic for reconstruction
    let is_byte_type = matches!(
        header.data_type,
        DataType::Char | DataType::Byte
    );

    if mode == ImageEncodeMode::DeltaHuffman {
        for i_depth in 0..n_depth {
            let mut prev_val = T::default();
            for i in 0..height {
                for j in 0..width {
                    let k = i * width + j;
                    let m = k * n_depth + i_depth;

                    if all_valid || mask.is_valid(k) {
                        let val =
                            codec.decode_one_value(data, &mut byte_pos, &mut bit_pos, num_bits_lut)?;
                        let delta = val - offset;

                        let predicted = if j > 0 && (all_valid || mask.is_valid(k - 1))
                        {
                            prev_val
                        } else if i > 0 && (all_valid || mask.is_valid(k - width)) {
                            output[m - width * n_depth]
                        } else {
                            prev_val
                        };

                        let result = if is_byte_type {
                            // Use wrapping arithmetic for 8-bit types to match C++ behavior
                            let predicted_i = predicted.to_f64() as i32;
                            let reconstructed = (delta + predicted_i) as u8;
                            if header.data_type == DataType::Char {
                                T::from_f64(reconstructed as i8 as f64)
                            } else {
                                T::from_f64(reconstructed as f64)
                            }
                        } else {
                            T::from_f64(delta as f64 + predicted.to_f64())
                        };
                        output[m] = result;
                        prev_val = result;
                    }
                }
            }
        }
    } else if mode == ImageEncodeMode::Huffman {
        for i in 0..height {
            for j in 0..width {
                let k = i * width + j;
                if all_valid || mask.is_valid(k) {
                    let m0 = k * n_depth;
                    for m in 0..n_depth {
                        let val = codec.decode_one_value(
                            data,
                            &mut byte_pos,
                            &mut bit_pos,
                            num_bits_lut,
                        )?;
                        output[m0 + m] = T::from_f64((val - offset) as f64);
                    }
                }
            }
        }
    } else {
        return Err(LercError::InvalidData("unexpected huffman mode".into()));
    }

    // Advance past the consumed data (including read-ahead padding)
    let num_uints = if bit_pos > 0 { 1usize } else { 0 } + 1; // +1 for decode LUT read-ahead
    let consumed = (byte_pos - start_byte_pos) + num_uints * 4;
    *pos = start_byte_pos + consumed;

    Ok(())
}

