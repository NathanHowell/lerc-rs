use alloc::vec;
use alloc::vec::Vec;

use crate::bitmask::BitMask;
use crate::error::Result;
use crate::header::{self, HeaderInfo};
use crate::rle;
use crate::tiles;
use crate::types::{DataType, LercDataType};
use crate::{LercData, LercImage};

pub fn encode(image: &LercImage, max_z_error: f64) -> Result<Vec<u8>> {
    match &image.data {
        LercData::I8(d) => encode_typed(image, d, max_z_error),
        LercData::U8(d) => encode_typed(image, d, max_z_error),
        LercData::I16(d) => encode_typed(image, d, max_z_error),
        LercData::U16(d) => encode_typed(image, d, max_z_error),
        LercData::I32(d) => encode_typed(image, d, max_z_error),
        LercData::U32(d) => encode_typed(image, d, max_z_error),
        LercData::F32(d) => encode_typed(image, d, max_z_error),
        LercData::F64(d) => encode_typed(image, d, max_z_error),
    }
}

fn encode_typed<T: LercDataType>(
    image: &LercImage,
    data: &[T],
    max_z_error: f64,
) -> Result<Vec<u8>> {
    let width = image.width as usize;
    let height = image.height as usize;
    let n_depth = image.n_depth as usize;
    let n_bands = image.n_bands as usize;
    let band_size = width * height * n_depth;

    let max_z_error = if T::is_integer() {
        max_z_error.floor().max(0.5)
    } else {
        max_z_error.max(0.0)
    };

    let mut result = Vec::new();

    for band in 0..n_bands {
        let band_data = &data[band * band_size..(band + 1) * band_size];
        let mask = if band < image.valid_masks.len() {
            &image.valid_masks[band]
        } else {
            &image.valid_masks[0]
        };

        let blobs_more = (n_bands - 1 - band) as i32;
        let blob = encode_one_band(band_data, mask, image, max_z_error, blobs_more)?;
        result.extend_from_slice(&blob);
    }

    Ok(result)
}

fn encode_one_band<T: LercDataType>(
    data: &[T],
    mask: &BitMask,
    image: &LercImage,
    max_z_error: f64,
    n_blobs_more: i32,
) -> Result<Vec<u8>> {
    let width = image.width as usize;
    let height = image.height as usize;
    let n_depth = image.n_depth as usize;
    let num_valid = mask.count_valid().min(width * height);

    // Compute per-depth min/max
    let mut z_min_vec = vec![f64::MAX; n_depth];
    let mut z_max_vec = vec![f64::MIN; n_depth];
    let mut overall_min = f64::MAX;
    let mut overall_max = f64::MIN;

    for i in 0..height {
        for j in 0..width {
            let k = i * width + j;
            if mask.is_valid(k) {
                for m in 0..n_depth {
                    let val = data[k * n_depth + m].to_f64();
                    if val < z_min_vec[m] {
                        z_min_vec[m] = val;
                    }
                    if val > z_max_vec[m] {
                        z_max_vec[m] = val;
                    }
                    if val < overall_min {
                        overall_min = val;
                    }
                    if val > overall_max {
                        overall_max = val;
                    }
                }
            }
        }
    }

    if num_valid == 0 {
        overall_min = 0.0;
        overall_max = 0.0;
    }

    let hd = HeaderInfo {
        version: 6,
        checksum: 0,
        n_rows: height as i32,
        n_cols: width as i32,
        n_depth: n_depth as i32,
        num_valid_pixel: num_valid as i32,
        micro_block_size: 8,
        blob_size: 0, // patched later
        data_type: T::DATA_TYPE,
        n_blobs_more,
        pass_no_data_values: false,
        is_int: false,
        max_z_error,
        z_min: overall_min,
        z_max: overall_max,
        no_data_val: 0.0,
        no_data_val_orig: 0.0,
    };

    let mut blob = header::write_header(&hd);

    // Write mask
    let num_pixels = width * height;
    if num_valid == 0 || num_valid == num_pixels {
        // No mask data needed
        blob.extend_from_slice(&0i32.to_le_bytes());
    } else {
        let mask_compressed = rle::compress(mask.as_bytes());
        let mask_size = mask_compressed.len() as i32;
        blob.extend_from_slice(&mask_size.to_le_bytes());
        blob.extend_from_slice(&mask_compressed);
    }

    if num_valid == 0 {
        header::finalize_blob(&mut blob);
        return Ok(blob);
    }

    // Const image
    if overall_min == overall_max {
        header::finalize_blob(&mut blob);
        return Ok(blob);
    }

    // Write per-depth min/max ranges (always for v6)
    for m in 0..n_depth {
        write_typed_value::<T>(&mut blob, z_min_vec[m]);
    }
    for m in 0..n_depth {
        write_typed_value::<T>(&mut blob, z_max_vec[m]);
    }

    // Check if all depths are constant
    if z_min_vec == z_max_vec {
        header::finalize_blob(&mut blob);
        return Ok(blob);
    }

    // One sweep flag = 0 (we use tiling)
    blob.push(0u8);

    // For now, always use tiling mode (no Huffman)
    // TODO: implement Huffman for 8-bit types and FPL for float types

    // Check if Huffman-eligible types
    let try_huffman_int = matches!(T::DATA_TYPE, DataType::Char | DataType::Byte)
        && max_z_error == 0.5;
    let try_huffman_flt =
        matches!(T::DATA_TYPE, DataType::Float | DataType::Double) && max_z_error == 0.0;

    if try_huffman_int || try_huffman_flt {
        // Write tiling mode flag (IEM_Tiling = 0)
        blob.push(0u8);
    }

    // Write tiles
    encode_tiles(&mut blob, data, mask, &hd, &z_min_vec, &z_max_vec)?;

    header::finalize_blob(&mut blob);
    Ok(blob)
}

fn encode_tiles<T: LercDataType>(
    blob: &mut Vec<u8>,
    data: &[T],
    mask: &BitMask,
    header: &HeaderInfo,
    _z_min_vec: &[f64],
    z_max_vec: &[f64],
) -> Result<()> {
    let mb_size = header.micro_block_size as usize;
    let n_depth = header.n_depth as usize;
    let n_rows = header.n_rows as usize;
    let n_cols = header.n_cols as usize;
    let max_z_error = header.max_z_error;

    let num_tiles_vert = (n_rows + mb_size - 1) / mb_size;
    let num_tiles_hori = (n_cols + mb_size - 1) / mb_size;

    let max_val_to_quantize: f64 = match header.data_type {
        DataType::Char | DataType::Byte | DataType::Short | DataType::UShort => {
            ((1u32 << 15) - 1) as f64
        }
        _ => ((1u32 << 30) - 1) as f64,
    };

    for i_tile in 0..num_tiles_vert {
        let i0 = i_tile * mb_size;
        let i1 = (i0 + mb_size).min(n_rows);

        for j_tile in 0..num_tiles_hori {
            let j0 = j_tile * mb_size;
            let j1 = (j0 + mb_size).min(n_cols);

            for i_depth in 0..n_depth {
                encode_tile(
                    blob,
                    data,
                    mask,
                    header,
                    z_max_vec,
                    max_z_error,
                    max_val_to_quantize,
                    i0,
                    i1,
                    j0,
                    j1,
                    i_depth,
                )?;
            }
        }
    }

    Ok(())
}

fn encode_tile<T: LercDataType>(
    blob: &mut Vec<u8>,
    data: &[T],
    mask: &BitMask,
    header: &HeaderInfo,
    _z_max_vec: &[f64],
    max_z_error: f64,
    max_val_to_quantize: f64,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    i_depth: usize,
) -> Result<()> {
    let n_cols = header.n_cols as usize;
    let n_depth = header.n_depth as usize;

    // Collect valid pixels
    let mut valid_data: Vec<T> = Vec::new();
    let mut z_min = T::default();
    let mut z_max = T::default();
    let mut first = true;

    for i in i0..i1 {
        let mut k = i * n_cols + j0;
        for _j in j0..j1 {
            if mask.is_valid(k) {
                let val = data[k * n_depth + i_depth];
                valid_data.push(val);
                if first {
                    z_min = val;
                    z_max = val;
                    first = false;
                } else {
                    if val.to_f64() < z_min.to_f64() {
                        z_min = val;
                    }
                    if val.to_f64() > z_max.to_f64() {
                        z_max = val;
                    }
                }
            }
            k += 1;
        }
    }

    let num_valid = valid_data.len();

    // Integrity check bits
    let pattern: u8 = 14; // v5+
    let integrity = ((j0 as u8 >> 3) & pattern) << 2;

    if num_valid == 0 || (z_min.to_f64() == 0.0 && z_max.to_f64() == 0.0) {
        // Constant 0 block
        blob.push(2 | integrity);
        return Ok(());
    }

    let z_min_f = z_min.to_f64();
    let z_max_f = z_max.to_f64();

    // Check if we need quantization
    let need_quantize = if max_z_error == 0.0 {
        z_max_f > z_min_f
    } else {
        let max_val = (z_max_f - z_min_f) / (2.0 * max_z_error);
        max_val <= max_val_to_quantize && (max_val + 0.5) as u32 != 0
    };

    if !need_quantize && z_min_f == z_max_f {
        // Constant block (all same value)
        let (dt_reduced, tc) = tiles::reduce_data_type(z_min, header.data_type);
        let bits67 = (tc as u8) << 6;
        blob.push(3 | bits67 | integrity);
        tiles::write_variable_data_type(blob, z_min_f, dt_reduced);
        return Ok(());
    }

    if !need_quantize {
        // Raw binary
        blob.push(0 | integrity);
        for val in &valid_data {
            write_typed_value_raw::<T>(blob, *val);
        }
        return Ok(());
    }

    // Quantize and bit-stuff
    let (dt_reduced, tc) = tiles::reduce_data_type(z_min, header.data_type);
    let bits67 = (tc as u8) << 6;

    // Quantize values
    let mut quant_vec: Vec<u32> = Vec::with_capacity(num_valid);
    if T::is_integer() && max_z_error == 0.5 {
        // Lossless integer
        for val in &valid_data {
            quant_vec.push((val.to_f64() - z_min_f) as u32);
        }
    } else {
        let scale = 1.0 / (2.0 * max_z_error);
        for val in &valid_data {
            quant_vec.push(((val.to_f64() - z_min_f) * scale + 0.5) as u32);
        }
    }

    let max_quant = quant_vec.iter().copied().max().unwrap_or(0);

    if max_quant == 0 {
        // All values quantize to same -> constant block
        blob.push(3 | bits67 | integrity);
        tiles::write_variable_data_type(blob, z_min_f, dt_reduced);
        return Ok(());
    }

    // Try LUT vs simple encoding
    let encoded = if let Some(sorted) = crate::bitstuffer::should_use_lut(&quant_vec) {
        let lut_bits67 = bits67;
        let mut tile_buf = vec![1 | lut_bits67 | integrity];
        tiles::write_variable_data_type(&mut tile_buf, z_min_f, dt_reduced);
        let stuffed = crate::bitstuffer::encode_lut(&sorted);
        tile_buf.extend_from_slice(&stuffed);
        tile_buf
    } else {
        let mut tile_buf = vec![1 | bits67 | integrity];
        tiles::write_variable_data_type(&mut tile_buf, z_min_f, dt_reduced);
        let stuffed = crate::bitstuffer::encode_simple(&quant_vec);
        tile_buf.extend_from_slice(&stuffed);
        tile_buf
    };

    // Check if raw binary would be smaller
    let raw_size = 1 + num_valid * T::BYTES;
    if encoded.len() < raw_size {
        blob.extend_from_slice(&encoded);
    } else {
        // Raw binary fallback
        blob.push(0 | integrity);
        for val in &valid_data {
            write_typed_value_raw::<T>(blob, *val);
        }
    }

    Ok(())
}

fn write_typed_value<T: LercDataType>(blob: &mut Vec<u8>, val: f64) {
    let t = T::from_f64(val);
    write_typed_value_raw::<T>(blob, t);
}

fn write_typed_value_raw<T: LercDataType>(blob: &mut Vec<u8>, val: T) {
    match T::DATA_TYPE {
        DataType::Char => blob.push(T::from_f64(val.to_f64()).to_f64() as i8 as u8),
        DataType::Byte => blob.push(val.to_f64() as u8),
        DataType::Short => {
            blob.extend_from_slice(&(val.to_f64() as i16).to_le_bytes())
        }
        DataType::UShort => {
            blob.extend_from_slice(&(val.to_f64() as u16).to_le_bytes())
        }
        DataType::Int => blob.extend_from_slice(&(val.to_f64() as i32).to_le_bytes()),
        DataType::UInt => {
            blob.extend_from_slice(&(val.to_f64() as u32).to_le_bytes())
        }
        DataType::Float => {
            blob.extend_from_slice(&(val.to_f64() as f32).to_le_bytes())
        }
        DataType::Double => blob.extend_from_slice(&val.to_f64().to_le_bytes()),
    }
}
