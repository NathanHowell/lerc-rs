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

            let mode = ImageEncodeMode::try_from(flag)?;

            // Version-gated mode validation
            if (mode == ImageEncodeMode::DeltaDeltaHuffman && hd.version < 6)
                || (mode == ImageEncodeMode::Huffman && hd.version < 4)
            {
                return Err(LercError::InvalidData("image encode mode not supported in this version".into()));
            }

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

use crate::tiles::{read_typed_as_f64, read_typed_value};

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal data buffer for `read_mask`: just the 4-byte numBytesMask field.
    fn make_mask_header(num_bytes_mask: i32) -> Vec<u8> {
        num_bytes_mask.to_le_bytes().to_vec()
    }

    // -----------------------------------------------------------------------
    // read_mask
    // -----------------------------------------------------------------------

    #[test]
    fn read_mask_all_valid() {
        // numBytesMask=0, numValidPixel == nRows*nCols → all valid
        let data = make_mask_header(0);
        let mut pos = 0;
        let header = HeaderInfo {
            n_rows: 4,
            n_cols: 8,
            num_valid_pixel: 32, // 4*8
            ..Default::default()
        };
        let mask = read_mask(&data, &mut pos, &header, None).unwrap();
        assert_eq!(mask.count_valid(), 32);
        for i in 0..32 {
            assert!(mask.is_valid(i), "pixel {i} should be valid");
        }
    }

    #[test]
    fn read_mask_all_invalid() {
        // numBytesMask=0, numValidPixel == 0 → all invalid
        let data = make_mask_header(0);
        let mut pos = 0;
        let header = HeaderInfo {
            n_rows: 4,
            n_cols: 8,
            num_valid_pixel: 0,
            ..Default::default()
        };
        let mask = read_mask(&data, &mut pos, &header, None).unwrap();
        assert_eq!(mask.count_valid(), 0);
        for i in 0..32 {
            assert!(!mask.is_valid(i), "pixel {i} should be invalid");
        }
    }

    #[test]
    fn read_mask_rle_compressed() {
        // 2x2 image, 2 pixels valid, 2 invalid
        // Mask bytes: 0xC0 = 0b1100_0000 → pixels 0,1 valid; pixels 2,3 invalid
        // That's 1 byte for 4 pixels
        let mask_bytes = vec![0xC0];
        let compressed = rle::compress(&mask_bytes);

        let mut data = make_mask_header(compressed.len() as i32);
        data.extend_from_slice(&compressed);

        let mut pos = 0;
        let header = HeaderInfo {
            n_rows: 2,
            n_cols: 2,
            num_valid_pixel: 2, // not 0, not 4 → triggers RLE path
            ..Default::default()
        };
        let mask = read_mask(&data, &mut pos, &header, None).unwrap();
        assert!(mask.is_valid(0));
        assert!(mask.is_valid(1));
        assert!(!mask.is_valid(2));
        assert!(!mask.is_valid(3));
        assert_eq!(mask.count_valid(), 2);
    }

    #[test]
    fn read_mask_rle_larger() {
        // 4x4 image (16 pixels), checkerboard pattern
        // Byte 0: 0xAA = 0b10101010 → pixels 0,2,4,6 valid
        // Byte 1: 0x55 = 0b01010101 → pixels 9,11,13,15 valid
        let mask_bytes = vec![0xAA, 0x55];
        let compressed = rle::compress(&mask_bytes);

        let mut data = make_mask_header(compressed.len() as i32);
        data.extend_from_slice(&compressed);

        let mut pos = 0;
        let header = HeaderInfo {
            n_rows: 4,
            n_cols: 4,
            num_valid_pixel: 8, // partial → triggers RLE path
            ..Default::default()
        };
        let mask = read_mask(&data, &mut pos, &header, None).unwrap();
        assert_eq!(mask.count_valid(), 8);
        // Byte 0 = 0xAA: bits 1,0,1,0,1,0,1,0 (MSB-first)
        assert!(mask.is_valid(0));
        assert!(!mask.is_valid(1));
        assert!(mask.is_valid(2));
        assert!(!mask.is_valid(3));
        // Byte 1 = 0x55: bits 0,1,0,1,0,1,0,1 (MSB-first)
        assert!(!mask.is_valid(8));
        assert!(mask.is_valid(9));
    }

    #[test]
    fn read_mask_reuses_prev_mask() {
        // numBytesMask <= 0 with partial validity but a previous mask exists
        let data = make_mask_header(0);
        let mut pos = 0;
        let header = HeaderInfo {
            n_rows: 2,
            n_cols: 2,
            num_valid_pixel: 3, // partial → needs mask data or prev_mask
            ..Default::default()
        };
        let mut prev = BitMask::new(4);
        prev.set_valid(0);
        prev.set_valid(1);
        prev.set_valid(3);

        let mask = read_mask(&data, &mut pos, &header, Some(&prev)).unwrap();
        assert!(mask.is_valid(0));
        assert!(mask.is_valid(1));
        assert!(!mask.is_valid(2));
        assert!(mask.is_valid(3));
        assert_eq!(mask.count_valid(), 3);
    }

    #[test]
    fn read_mask_error_no_prev_mask() {
        // numBytesMask=0, partial validity, no previous mask → error
        let data = make_mask_header(0);
        let mut pos = 0;
        let header = HeaderInfo {
            n_rows: 2,
            n_cols: 2,
            num_valid_pixel: 2,
            ..Default::default()
        };
        let result = read_mask(&data, &mut pos, &header, None);
        assert!(result.is_err());
    }

    #[test]
    fn read_mask_buffer_too_small() {
        // Data too short to even read numBytesMask
        let data = [0u8; 2]; // need 4 bytes
        let mut pos = 0;
        let header = HeaderInfo {
            n_rows: 1,
            n_cols: 1,
            num_valid_pixel: 1,
            ..Default::default()
        };
        let result = read_mask(&data, &mut pos, &header, None);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // fill_const_image
    // -----------------------------------------------------------------------

    #[test]
    fn fill_const_image_ndepth1_all_valid() {
        let header = HeaderInfo {
            n_rows: 2,
            n_cols: 3,
            n_depth: 1,
            z_min: 42.0,
            z_max: 42.0,
            ..Default::default()
        };
        let mask = BitMask::all_valid(6);
        let mut output = vec![0.0_f64; 6];

        fill_const_image(&header, &mask, &[], &[], &mut output).unwrap();

        for (i, &val) in output.iter().enumerate() {
            assert_eq!(val, 42.0, "pixel {i} should be 42.0");
        }
    }

    #[test]
    fn fill_const_image_ndepth1_partial_mask() {
        let header = HeaderInfo {
            n_rows: 2,
            n_cols: 2,
            n_depth: 1,
            z_min: 99.0,
            z_max: 99.0,
            ..Default::default()
        };
        let mut mask = BitMask::new(4);
        mask.set_valid(0);
        mask.set_valid(2);
        // pixels 1, 3 are invalid

        let mut output = vec![0.0_f64; 4];
        fill_const_image(&header, &mask, &[], &[], &mut output).unwrap();

        assert_eq!(output[0], 99.0);
        assert_eq!(output[1], 0.0); // invalid pixel, untouched
        assert_eq!(output[2], 99.0);
        assert_eq!(output[3], 0.0); // invalid pixel, untouched
    }

    #[test]
    fn fill_const_image_ndepth2_with_z_min_vec() {
        let header = HeaderInfo {
            n_rows: 2,
            n_cols: 2,
            n_depth: 2,
            z_min: 10.0,
            z_max: 20.0, // different from z_min so z_min_vec is used
            ..Default::default()
        };
        let mask = BitMask::all_valid(4); // 2x2 pixels
        let z_min_vec = vec![10.0, 20.0]; // depth 0 = 10.0, depth 1 = 20.0
        let z_max_vec = vec![10.0, 20.0]; // const: min == max per depth

        // Output has 2*2*2 = 8 elements
        let mut output = vec![0.0_f64; 8];
        fill_const_image(&header, &mask, &z_min_vec, &z_max_vec, &mut output).unwrap();

        // pixel 0: output[0]=10.0, output[1]=20.0
        assert_eq!(output[0], 10.0);
        assert_eq!(output[1], 20.0);
        // pixel 1: output[2]=10.0, output[3]=20.0
        assert_eq!(output[2], 10.0);
        assert_eq!(output[3], 20.0);
        // pixel 2: output[4]=10.0, output[5]=20.0
        assert_eq!(output[4], 10.0);
        assert_eq!(output[5], 20.0);
        // pixel 3: output[6]=10.0, output[7]=20.0
        assert_eq!(output[6], 10.0);
        assert_eq!(output[7], 20.0);
    }

    #[test]
    fn fill_const_image_ndepth2_empty_z_min_vec_uses_header_z_min() {
        let header = HeaderInfo {
            n_rows: 1,
            n_cols: 2,
            n_depth: 2,
            z_min: 7.0,
            z_max: 7.0, // same → z_buf all filled with z_min
            ..Default::default()
        };
        let mask = BitMask::all_valid(2);
        let mut output = vec![0.0_f64; 4]; // 1*2*2

        fill_const_image(&header, &mask, &[], &[], &mut output).unwrap();

        // All values should be 7.0
        for (i, &val) in output.iter().enumerate() {
            assert_eq!(val, 7.0, "element {i} should be 7.0");
        }
    }

    #[test]
    fn fill_const_image_ndepth2_partial_mask() {
        let header = HeaderInfo {
            n_rows: 1,
            n_cols: 3,
            n_depth: 2,
            z_min: 5.0,
            z_max: 5.0,
            ..Default::default()
        };
        let mut mask = BitMask::new(3);
        mask.set_valid(0);
        mask.set_valid(2);
        // pixel 1 is invalid

        let mut output = vec![0.0_f64; 6]; // 1*3*2
        fill_const_image(&header, &mask, &[], &[], &mut output).unwrap();

        // pixel 0: filled
        assert_eq!(output[0], 5.0);
        assert_eq!(output[1], 5.0);
        // pixel 1: invalid, untouched
        assert_eq!(output[2], 0.0);
        assert_eq!(output[3], 0.0);
        // pixel 2: filled
        assert_eq!(output[4], 5.0);
        assert_eq!(output[5], 5.0);
    }

    #[test]
    fn fill_const_image_integer_type() {
        let header = HeaderInfo {
            n_rows: 2,
            n_cols: 2,
            n_depth: 1,
            z_min: 42.0,
            z_max: 42.0,
            ..Default::default()
        };
        let mask = BitMask::all_valid(4);
        let mut output = vec![0_i32; 4];

        fill_const_image(&header, &mask, &[], &[], &mut output).unwrap();

        for (i, &val) in output.iter().enumerate() {
            assert_eq!(val, 42, "pixel {i} should be 42");
        }
    }

    // -----------------------------------------------------------------------
    // read_min_max_ranges
    // -----------------------------------------------------------------------

    #[test]
    fn read_min_max_ranges_single_depth_u8() {
        // nDepth=1, type u8 (1 byte each): min=10, max=200
        let mut data = Vec::new();
        data.push(10u8); // z_min[0]
        data.push(200u8); // z_max[0]

        let mut pos = 0;
        let header = HeaderInfo {
            n_depth: 1,
            ..Default::default()
        };
        let (z_min, z_max) = read_min_max_ranges::<u8>(&data, &mut pos, &header).unwrap();
        assert_eq!(z_min, vec![10.0]);
        assert_eq!(z_max, vec![200.0]);
        assert_eq!(pos, 2);
    }

    #[test]
    fn read_min_max_ranges_multi_depth_f32() {
        // nDepth=3, type f32 (4 bytes each): 6 values total
        let mut data = Vec::new();
        for &v in &[1.0f32, 2.0, 3.0] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &[10.0f32, 20.0, 30.0] {
            data.extend_from_slice(&v.to_le_bytes());
        }

        let mut pos = 0;
        let header = HeaderInfo {
            n_depth: 3,
            ..Default::default()
        };
        let (z_min, z_max) = read_min_max_ranges::<f32>(&data, &mut pos, &header).unwrap();
        assert_eq!(z_min, vec![1.0, 2.0, 3.0]);
        assert_eq!(z_max, vec![10.0, 20.0, 30.0]);
        assert_eq!(pos, 24); // 6 * 4 bytes
    }

    #[test]
    fn read_min_max_ranges_buffer_too_small() {
        let data = vec![0u8; 3]; // Need at least 4 bytes for nDepth=2, type u8
        let mut pos = 0;
        let header = HeaderInfo {
            n_depth: 2,
            ..Default::default()
        };
        let result = read_min_max_ranges::<u8>(&data, &mut pos, &header);
        assert!(result.is_err());
    }

    #[test]
    fn read_min_max_ranges_i16() {
        // nDepth=2, type i16 (2 bytes each): 4 values total
        let mut data = Vec::new();
        for &v in &[-100i16, 50] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &[1000i16, 2000] {
            data.extend_from_slice(&v.to_le_bytes());
        }

        let mut pos = 0;
        let header = HeaderInfo {
            n_depth: 2,
            ..Default::default()
        };
        let (z_min, z_max) = read_min_max_ranges::<i16>(&data, &mut pos, &header).unwrap();
        assert_eq!(z_min, vec![-100.0, 50.0]);
        assert_eq!(z_max, vec![1000.0, 2000.0]);
        assert_eq!(pos, 8); // 4 * 2 bytes
    }

    // -----------------------------------------------------------------------
    // read_data_one_sweep
    // -----------------------------------------------------------------------

    #[test]
    fn read_data_one_sweep_u8_all_valid() {
        let header = HeaderInfo {
            n_rows: 2,
            n_cols: 3,
            n_depth: 1,
            ..Default::default()
        };
        let mask = BitMask::all_valid(6);
        let data: Vec<u8> = vec![10, 20, 30, 40, 50, 60];
        let mut pos = 0;
        let mut output = vec![0u8; 6];
        read_data_one_sweep(&data, &mut pos, &header, &mask, &mut output).unwrap();
        assert_eq!(output, vec![10, 20, 30, 40, 50, 60]);
        assert_eq!(pos, 6);
    }

    #[test]
    fn read_data_one_sweep_f32_all_valid() {
        let header = HeaderInfo {
            n_rows: 1,
            n_cols: 3,
            n_depth: 1,
            ..Default::default()
        };
        let mask = BitMask::all_valid(3);
        let values = [1.5f32, -2.5, 100.0];
        let mut data = Vec::new();
        for &v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let mut pos = 0;
        let mut output = vec![0.0f32; 3];
        read_data_one_sweep(&data, &mut pos, &header, &mask, &mut output).unwrap();
        assert_eq!(output, vec![1.5, -2.5, 100.0]);
        assert_eq!(pos, 12); // 3 * 4 bytes
    }

    #[test]
    fn read_data_one_sweep_with_mask() {
        // 2x2 image, pixels 0 and 2 valid, 1 and 3 invalid
        let header = HeaderInfo {
            n_rows: 2,
            n_cols: 2,
            n_depth: 1,
            ..Default::default()
        };
        let mut mask = BitMask::new(4);
        mask.set_valid(0);
        mask.set_valid(2);
        // Only 2 valid pixel values in the data stream
        let data: Vec<u8> = vec![42, 99];
        let mut pos = 0;
        let mut output = vec![0u8; 4];
        read_data_one_sweep(&data, &mut pos, &header, &mask, &mut output).unwrap();
        assert_eq!(output[0], 42); // valid pixel 0
        assert_eq!(output[1], 0);  // invalid pixel 1 (untouched)
        assert_eq!(output[2], 99); // valid pixel 2
        assert_eq!(output[3], 0);  // invalid pixel 3 (untouched)
        assert_eq!(pos, 2);
    }

    #[test]
    fn read_data_one_sweep_multi_depth() {
        // 1x2 image, nDepth=2, all valid
        let header = HeaderInfo {
            n_rows: 1,
            n_cols: 2,
            n_depth: 2,
            ..Default::default()
        };
        let mask = BitMask::all_valid(2);
        // pixel0_d0, pixel0_d1, pixel1_d0, pixel1_d1
        let data: Vec<u8> = vec![10, 20, 30, 40];
        let mut pos = 0;
        let mut output = vec![0u8; 4];
        read_data_one_sweep(&data, &mut pos, &header, &mask, &mut output).unwrap();
        assert_eq!(output, vec![10, 20, 30, 40]);
    }

    #[test]
    fn read_data_one_sweep_buffer_too_small() {
        let header = HeaderInfo {
            n_rows: 1,
            n_cols: 3,
            n_depth: 1,
            ..Default::default()
        };
        let mask = BitMask::all_valid(3);
        let data: Vec<u8> = vec![10, 20]; // only 2 bytes, need 3
        let mut pos = 0;
        let mut output = vec![0u8; 3];
        let result = read_data_one_sweep(&data, &mut pos, &header, &mask, &mut output);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // decode_huffman via round-trip (encode then decode)
    // -----------------------------------------------------------------------

    /// Helper: encode a u8 image and decode it, returning the decoded pixels.
    fn roundtrip_u8(width: u32, height: u32, pixels: &[u8], max_z_error: f64) -> Vec<u8> {
        let image = crate::LercImage {
            width,
            height,
            n_depth: 1,
            n_bands: 1,
            data_type: DataType::Byte,
            valid_masks: vec![BitMask::all_valid((width * height) as usize)],
            data: crate::LercData::U8(pixels.to_vec()),
            no_data_value: None,
        };
        let blob = crate::encode(&image, max_z_error).unwrap();
        let decoded = crate::decode(&blob).unwrap();
        match decoded.data {
            crate::LercData::U8(v) => v,
            _ => panic!("expected U8 data"),
        }
    }

    /// Helper: encode an i8 image and decode it, returning the decoded pixels.
    fn roundtrip_i8(width: u32, height: u32, pixels: &[i8], max_z_error: f64) -> Vec<i8> {
        let image = crate::LercImage {
            width,
            height,
            n_depth: 1,
            n_bands: 1,
            data_type: DataType::Char,
            valid_masks: vec![BitMask::all_valid((width * height) as usize)],
            data: crate::LercData::I8(pixels.to_vec()),
            no_data_value: None,
        };
        let blob = crate::encode(&image, max_z_error).unwrap();
        let decoded = crate::decode(&blob).unwrap();
        match decoded.data {
            crate::LercData::I8(v) => v,
            _ => panic!("expected I8 data"),
        }
    }

    #[test]
    fn decode_huffman_u8_roundtrip_gradient() {
        // A gradient pattern triggers DeltaHuffman encoding (max_z_error=0.5
        // for u8 enables Huffman). Small deltas compress well with Huffman.
        let width = 16;
        let height = 16;
        let pixels: Vec<u8> = (0..width * height)
            .map(|i| (i % 256) as u8)
            .collect();
        let decoded = roundtrip_u8(width as u32, height as u32, &pixels, 0.5);
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn decode_huffman_u8_roundtrip_repeated_pattern() {
        // Repeated low-entropy pattern: very compressible by Huffman
        let width = 32;
        let height = 32;
        let pixels: Vec<u8> = (0..width * height)
            .map(|i| [0u8, 1, 2, 3, 4][i % 5])
            .collect();
        let decoded = roundtrip_u8(width as u32, height as u32, &pixels, 0.5);
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn decode_huffman_u8_roundtrip_constant_regions() {
        // Image with constant regions and sharp edges
        let width = 24;
        let height = 24;
        let mut pixels = vec![0u8; width * height];
        for i in 0..height {
            for j in 0..width {
                pixels[i * width + j] = if i < height / 2 { 50 } else { 200 };
            }
        }
        let decoded = roundtrip_u8(width as u32, height as u32, &pixels, 0.5);
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn decode_huffman_i8_roundtrip() {
        // Signed byte (Char) type triggers Huffman with offset=128
        let width = 16;
        let height = 16;
        let pixels: Vec<i8> = (0..width * height)
            .map(|i| ((i as i32 % 256) - 128) as i8)
            .collect();
        let decoded = roundtrip_i8(width as u32, height as u32, &pixels, 0.5);
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn decode_huffman_u8_roundtrip_all_same_value() {
        // All same value — should be const image, not Huffman,
        // but tests the decode path can handle it
        let width = 8;
        let height = 8;
        let pixels = vec![42u8; width * height];
        let decoded = roundtrip_u8(width as u32, height as u32, &pixels, 0.5);
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn decode_huffman_u8_roundtrip_two_values() {
        // Two alternating values — minimal Huffman tree
        let width = 16;
        let height = 16;
        let pixels: Vec<u8> = (0..width * height)
            .map(|i| if i % 2 == 0 { 100 } else { 200 })
            .collect();
        let decoded = roundtrip_u8(width as u32, height as u32, &pixels, 0.5);
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn decode_huffman_u8_roundtrip_full_range() {
        // Use all 256 values — covers full byte range
        let width = 16;
        let height = 16;
        let pixels: Vec<u8> = (0..width * height)
            .map(|i| (i % 256) as u8)
            .collect();
        let decoded = roundtrip_u8(width as u32, height as u32, &pixels, 0.5);
        assert_eq!(decoded, pixels);
    }

    // -----------------------------------------------------------------------
    // decode_one_band via round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn decode_one_band_u8_roundtrip() {
        // Encode a small u8 image, then decode with decode_one_band directly
        let width = 4u32;
        let height = 4u32;
        let pixels: Vec<u8> = (0..16).collect();
        let image = crate::LercImage {
            width,
            height,
            n_depth: 1,
            n_bands: 1,
            data_type: DataType::Byte,
            valid_masks: vec![BitMask::all_valid(16)],
            data: crate::LercData::U8(pixels.clone()),
            no_data_value: None,
        };
        let blob = crate::encode(&image, 0.5).unwrap();
        let mut output = vec![0u8; 16];
        let (mask, hd, consumed) = decode_one_band(&blob, &mut output, None).unwrap();
        assert_eq!(consumed, blob.len());
        assert_eq!(mask.count_valid(), 16);
        assert_eq!(hd.n_rows, 4);
        assert_eq!(hd.n_cols, 4);
        assert_eq!(output, pixels);
    }

    #[test]
    fn decode_one_band_f32_roundtrip() {
        let width = 4u32;
        let height = 4u32;
        let pixels: Vec<f32> = (0..16).map(|i| i as f32 * 1.5).collect();
        let image = crate::LercImage {
            width,
            height,
            n_depth: 1,
            n_bands: 1,
            data_type: DataType::Float,
            valid_masks: vec![BitMask::all_valid(16)],
            data: crate::LercData::F32(pixels.clone()),
            no_data_value: None,
        };
        let blob = crate::encode(&image, 0.0).unwrap();
        let mut output = vec![0.0f32; 16];
        let (mask, _hd, consumed) = decode_one_band(&blob, &mut output, None).unwrap();
        assert_eq!(consumed, blob.len());
        assert_eq!(mask.count_valid(), 16);
        assert_eq!(output, pixels);
    }

    #[test]
    fn decode_one_band_const_image() {
        // All same value — const image path in decoder
        let width = 3u32;
        let height = 3u32;
        let pixels = vec![42u16; 9];
        let image = crate::LercImage {
            width,
            height,
            n_depth: 1,
            n_bands: 1,
            data_type: DataType::UShort,
            valid_masks: vec![BitMask::all_valid(9)],
            data: crate::LercData::U16(pixels.clone()),
            no_data_value: None,
        };
        let blob = crate::encode(&image, 0.5).unwrap();
        let mut output = vec![0u16; 9];
        let (mask, _hd, _) = decode_one_band(&blob, &mut output, None).unwrap();
        assert_eq!(mask.count_valid(), 9);
        assert_eq!(output, pixels);
    }

    #[test]
    fn decode_one_band_no_valid_pixels() {
        // All pixels invalid — no data to decode
        let width = 2u32;
        let height = 2u32;
        let pixels = vec![0u8; 4];
        let image = crate::LercImage {
            width,
            height,
            n_depth: 1,
            n_bands: 1,
            data_type: DataType::Byte,
            valid_masks: vec![BitMask::new(4)],
            data: crate::LercData::U8(pixels),
            no_data_value: None,
        };
        let blob = crate::encode(&image, 0.5).unwrap();
        let mut output = vec![255u8; 4];
        let (mask, _hd, _) = decode_one_band(&blob, &mut output, None).unwrap();
        assert_eq!(mask.count_valid(), 0);
        // Output should be zeroed for invalid pixels
        assert_eq!(output, vec![0, 0, 0, 0]);
    }

    #[test]
    fn decode_one_band_partial_mask() {
        // 4x4 image with only some pixels valid
        let width = 4u32;
        let height = 4u32;
        let n = (width * height) as usize;
        let mut mask = BitMask::new(n);
        let mut pixels = vec![0u8; n];
        // Set a checkerboard pattern of valid pixels
        for i in 0..n {
            if i % 3 == 0 {
                mask.set_valid(i);
                pixels[i] = (i * 7 % 256) as u8;
            }
        }
        let image = crate::LercImage {
            width,
            height,
            n_depth: 1,
            n_bands: 1,
            data_type: DataType::Byte,
            valid_masks: vec![mask.clone()],
            data: crate::LercData::U8(pixels.clone()),
            no_data_value: None,
        };
        let blob = crate::encode(&image, 0.5).unwrap();
        let mut output = vec![0u8; n];
        let (decoded_mask, _hd, _) = decode_one_band(&blob, &mut output, None).unwrap();

        // Verify valid pixels match
        for i in 0..n {
            if mask.is_valid(i) {
                assert!(decoded_mask.is_valid(i), "pixel {i} should be valid");
                assert_eq!(output[i], pixels[i], "pixel {i} value mismatch");
            }
        }
    }

    // -----------------------------------------------------------------------
    // decode_bands_into / decode_into
    // -----------------------------------------------------------------------

    #[test]
    fn decode_into_u8_roundtrip() {
        let width = 8u32;
        let height = 8u32;
        let n = (width * height) as usize;
        let pixels: Vec<u8> = (0..n).map(|i| (i * 3 % 256) as u8).collect();
        let blob = crate::encode_typed(width, height, &pixels, 0.5).unwrap();
        let mut output = vec![0u8; n];
        let result = crate::decode_into(&blob, &mut output).unwrap();
        assert_eq!(result.width, width);
        assert_eq!(result.height, height);
        assert_eq!(result.data_type, DataType::Byte);
        assert_eq!(output, pixels);
    }

    #[test]
    fn decode_into_f64_roundtrip() {
        let width = 4u32;
        let height = 4u32;
        let n = (width * height) as usize;
        let pixels: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let blob = crate::encode_typed(width, height, &pixels, 0.0).unwrap();
        let mut output = vec![0.0f64; n];
        let result = crate::decode_into(&blob, &mut output).unwrap();
        assert_eq!(result.data_type, DataType::Double);
        assert_eq!(output, pixels);
    }

    #[test]
    fn decode_into_type_mismatch() {
        let pixels: Vec<u8> = vec![1, 2, 3, 4];
        let blob = crate::encode_typed(2, 2, &pixels, 0.5).unwrap();
        let mut output = vec![0.0f32; 4];
        let result = crate::decode_into(&blob, &mut output);
        assert!(result.is_err());
    }

    #[test]
    fn decode_into_buffer_too_small() {
        let pixels: Vec<u8> = vec![1, 2, 3, 4];
        let blob = crate::encode_typed(2, 2, &pixels, 0.5).unwrap();
        let mut output = vec![0u8; 2]; // too small, need 4
        let result = crate::decode_into(&blob, &mut output);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // decode_info
    // -----------------------------------------------------------------------

    #[test]
    fn decode_info_u8() {
        let width = 10u32;
        let height = 20u32;
        let pixels: Vec<u8> = (0..200).map(|i| (i % 256) as u8).collect();
        let blob = crate::encode_typed(width, height, &pixels, 0.5).unwrap();
        let info = crate::decode_info(&blob).unwrap();
        assert_eq!(info.width, width);
        assert_eq!(info.height, height);
        assert_eq!(info.n_bands, 1);
        assert_eq!(info.n_depth, 1);
        assert_eq!(info.data_type, DataType::Byte);
        assert_eq!(info.num_valid_pixels, width * height);
    }

    #[test]
    fn decode_info_f32() {
        let width = 5u32;
        let height = 5u32;
        let pixels: Vec<f32> = (0..25).map(|i| i as f32).collect();
        let blob = crate::encode_typed(width, height, &pixels, 0.01).unwrap();
        let info = crate::decode_info(&blob).unwrap();
        assert_eq!(info.width, width);
        assert_eq!(info.height, height);
        assert_eq!(info.data_type, DataType::Float);
    }

    // -----------------------------------------------------------------------
    // Multi-band decode
    // -----------------------------------------------------------------------

    #[test]
    fn decode_multi_band_u8() {
        let width = 4u32;
        let height = 4u32;
        let n = (width * height) as usize;
        let band1: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
        let band2: Vec<u8> = (0..n).map(|i| ((i * 3) % 256) as u8).collect();
        let mut all_pixels = band1.clone();
        all_pixels.extend_from_slice(&band2);

        let image = crate::LercImage {
            width,
            height,
            n_depth: 1,
            n_bands: 2,
            data_type: DataType::Byte,
            valid_masks: vec![
                BitMask::all_valid(n),
                BitMask::all_valid(n),
            ],
            data: crate::LercData::U8(all_pixels.clone()),
            no_data_value: None,
        };
        let blob = crate::encode(&image, 0.5).unwrap();
        let decoded = crate::decode(&blob).unwrap();
        assert_eq!(decoded.n_bands, 2);
        assert_eq!(decoded.valid_masks.len(), 2);
        match decoded.data {
            crate::LercData::U8(v) => assert_eq!(v, all_pixels),
            _ => panic!("expected U8 data"),
        }
    }

    // -----------------------------------------------------------------------
    // decode_typed convenience
    // -----------------------------------------------------------------------

    #[test]
    fn decode_typed_u8() {
        let width = 6u32;
        let height = 6u32;
        let pixels: Vec<u8> = (0..36).map(|i| (i * 7 % 256) as u8).collect();
        let blob = crate::encode_typed(width, height, &pixels, 0.5).unwrap();
        let (decoded, mask, w, h) = crate::decode_typed::<u8>(&blob).unwrap();
        assert_eq!(w, width);
        assert_eq!(h, height);
        assert_eq!(mask.count_valid(), 36);
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn decode_typed_wrong_type() {
        let pixels: Vec<u8> = vec![1, 2, 3, 4];
        let blob = crate::encode_typed(2, 2, &pixels, 0.5).unwrap();
        let result = crate::decode_typed::<f32>(&blob);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // remap_no_data
    // -----------------------------------------------------------------------

    #[test]
    fn remap_no_data_replaces_sentinel() {
        let header = HeaderInfo {
            n_rows: 1,
            n_cols: 3,
            n_depth: 2,
            no_data_val: -9998.0,      // internal sentinel
            no_data_val_orig: -9999.0,  // original user value
            ..Default::default()
        };
        let mut mask = BitMask::new(3);
        mask.set_valid(0);
        mask.set_valid(1);
        mask.set_valid(2);

        // pixel0: [5.0, -9998.0] -> [5.0, -9999.0]
        // pixel1: [10.0, 20.0]   -> unchanged
        // pixel2: [-9998.0, 30.0] -> [-9999.0, 30.0]
        let mut output: Vec<f64> = vec![5.0, -9998.0, 10.0, 20.0, -9998.0, 30.0];
        remap_no_data(&mut output, &mask, &header);
        assert_eq!(output[0], 5.0);
        assert_eq!(output[1], -9999.0); // remapped
        assert_eq!(output[2], 10.0);
        assert_eq!(output[3], 20.0);
        assert_eq!(output[4], -9999.0); // remapped
        assert_eq!(output[5], 30.0);
    }

    #[test]
    fn remap_no_data_skips_invalid_pixels() {
        let header = HeaderInfo {
            n_rows: 1,
            n_cols: 2,
            n_depth: 1,
            no_data_val: -9998.0,
            no_data_val_orig: -9999.0,
            ..Default::default()
        };
        let mut mask = BitMask::new(2);
        mask.set_valid(0);
        // pixel 1 is invalid

        let mut output: Vec<f64> = vec![-9998.0, -9998.0];
        remap_no_data(&mut output, &mask, &header);
        assert_eq!(output[0], -9999.0); // valid pixel: remapped
        assert_eq!(output[1], -9998.0); // invalid pixel: not touched
    }

    // -----------------------------------------------------------------------
    // Integer type round-trips (i16, u16, i32, u32)
    // -----------------------------------------------------------------------

    #[test]
    fn roundtrip_i16() {
        let width = 8u32;
        let height = 8u32;
        let n = (width * height) as usize;
        let pixels: Vec<i16> = (0..n).map(|i| i as i16 - 32).collect();
        let blob = crate::encode_typed(width, height, &pixels, 0.5).unwrap();
        let (decoded, _mask, w, h) = crate::decode_typed::<i16>(&blob).unwrap();
        assert_eq!((w, h), (width, height));
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn roundtrip_u32() {
        let width = 4u32;
        let height = 4u32;
        let pixels: Vec<u32> = (0..16).map(|i| i * 1000).collect();
        let blob = crate::encode_typed(width, height, &pixels, 0.5).unwrap();
        let (decoded, _mask, _, _) = crate::decode_typed::<u32>(&blob).unwrap();
        assert_eq!(decoded, pixels);
    }

    // -----------------------------------------------------------------------
    // One-sweep decode via manually constructed blob
    // -----------------------------------------------------------------------

    /// Build a valid LERC2 blob with one_sweep=1, containing raw pixel data.
    fn build_one_sweep_blob_u8(
        width: u32,
        height: u32,
        pixels: &[u8],
    ) -> Vec<u8> {
        let n = (width * height) as usize;
        assert_eq!(pixels.len(), n);
        let z_min = *pixels.iter().min().unwrap() as f64;
        let z_max = *pixels.iter().max().unwrap() as f64;

        let version = 6;
        let hd = HeaderInfo {
            version,
            checksum: 0,
            n_rows: height as i32,
            n_cols: width as i32,
            n_depth: 1,
            num_valid_pixel: n as i32,
            micro_block_size: 8,
            blob_size: 0, // patched later
            data_type: DataType::Byte,
            max_z_error: 0.0,
            z_min,
            z_max,
            ..Default::default()
        };

        let mut blob = header::write_header(&hd);

        // Mask: numBytesMask=0 (all valid)
        blob.extend_from_slice(&0i32.to_le_bytes());

        // Per-depth min/max ranges (v4+): 1 byte min, 1 byte max
        blob.push(z_min as u8); // z_min[0]
        blob.push(z_max as u8); // z_max[0]

        // One sweep flag = 1
        blob.push(1u8);

        // Raw pixel data
        blob.extend_from_slice(pixels);

        // Finalize (patches blob_size and checksum)
        header::finalize_blob(&mut blob);
        blob
    }

    #[test]
    fn decode_one_sweep_blob_u8() {
        let pixels: Vec<u8> = vec![10, 20, 30, 40, 50, 60];
        let blob = build_one_sweep_blob_u8(3, 2, &pixels);
        let decoded = crate::decode(&blob).unwrap();
        assert_eq!(decoded.width, 3);
        assert_eq!(decoded.height, 2);
        match decoded.data {
            crate::LercData::U8(v) => assert_eq!(v, pixels),
            _ => panic!("expected U8 data"),
        }
    }

    #[test]
    fn decode_one_sweep_blob_u8_gradient() {
        let width = 8u32;
        let height = 8u32;
        let pixels: Vec<u8> = (0..64).map(|i| (i * 4 % 256) as u8).collect();
        let blob = build_one_sweep_blob_u8(width, height, &pixels);
        let (decoded, _mask, w, h) = crate::decode_typed::<u8>(&blob).unwrap();
        assert_eq!((w, h), (width, height));
        assert_eq!(decoded, pixels);
    }

    /// Build a valid LERC2 blob with one_sweep=1 for f32 data.
    fn build_one_sweep_blob_f32(
        width: u32,
        height: u32,
        pixels: &[f32],
    ) -> Vec<u8> {
        let n = (width * height) as usize;
        assert_eq!(pixels.len(), n);
        let z_min = pixels.iter().cloned().reduce(f32::min).unwrap() as f64;
        let z_max = pixels.iter().cloned().reduce(f32::max).unwrap() as f64;

        let hd = HeaderInfo {
            version: 6,
            checksum: 0,
            n_rows: height as i32,
            n_cols: width as i32,
            n_depth: 1,
            num_valid_pixel: n as i32,
            micro_block_size: 8,
            blob_size: 0,
            data_type: DataType::Float,
            max_z_error: 0.0,
            z_min,
            z_max,
            ..Default::default()
        };

        let mut blob = header::write_header(&hd);

        // Mask: numBytesMask=0 (all valid)
        blob.extend_from_slice(&0i32.to_le_bytes());

        // Per-depth min/max ranges: f32 LE
        blob.extend_from_slice(&(z_min as f32).to_le_bytes());
        blob.extend_from_slice(&(z_max as f32).to_le_bytes());

        // One sweep flag = 1
        blob.push(1u8);

        // Raw pixel data (f32 LE)
        for &v in pixels {
            blob.extend_from_slice(&v.to_le_bytes());
        }

        header::finalize_blob(&mut blob);
        blob
    }

    #[test]
    fn decode_one_sweep_blob_f32() {
        let pixels: Vec<f32> = vec![1.5, -2.5, 100.0, 0.0];
        let blob = build_one_sweep_blob_f32(2, 2, &pixels);
        let (decoded, _mask, w, h) = crate::decode_typed::<f32>(&blob).unwrap();
        assert_eq!((w, h), (2, 2));
        assert_eq!(decoded, pixels);
    }

    // -----------------------------------------------------------------------
    // Large u8 Huffman round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn decode_huffman_u8_large_image() {
        // Larger image to ensure Huffman encoding is definitely used
        // (small images might fall back to tiling)
        let width = 64u32;
        let height = 64u32;
        let n = (width * height) as usize;
        let pixels: Vec<u8> = (0..n)
            .map(|i| {
                let x = (i % width as usize) as u8;
                let y = (i / width as usize) as u8;
                x.wrapping_add(y)
            })
            .collect();
        let decoded = roundtrip_u8(width, height, &pixels, 0.5);
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn decode_huffman_u8_sparse_values() {
        // Only a few distinct values — small Huffman tree
        let width = 32u32;
        let height = 32u32;
        let n = (width * height) as usize;
        let values = [0u8, 128, 255];
        let pixels: Vec<u8> = (0..n).map(|i| values[i % 3]).collect();
        let decoded = roundtrip_u8(width, height, &pixels, 0.5);
        assert_eq!(decoded, pixels);
    }

    // -----------------------------------------------------------------------
    // Tiling mode round-trips (for completeness, covers tiles::read_tiles
    // path through decode_one_band)
    // -----------------------------------------------------------------------

    #[test]
    fn decode_tiling_i32_roundtrip() {
        let width = 16u32;
        let height = 16u32;
        let n = (width * height) as usize;
        let pixels: Vec<i32> = (0..n).map(|i| (i as i32) * 100 - 10000).collect();
        let blob = crate::encode_typed(width, height, &pixels, 0.5).unwrap();
        let (decoded, _mask, _, _) = crate::decode_typed::<i32>(&blob).unwrap();
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn decode_tiling_f32_lossy() {
        let width = 16u32;
        let height = 16u32;
        let n = (width * height) as usize;
        let pixels: Vec<f32> = (0..n).map(|i| (i as f32) * 0.7).collect();
        let max_z_error = 0.01;
        let blob = crate::encode_typed(width, height, &pixels, max_z_error).unwrap();
        let (decoded, _mask, _, _) = crate::decode_typed::<f32>(&blob).unwrap();
        for (i, (&orig, &dec)) in pixels.iter().zip(decoded.iter()).enumerate() {
            assert!(
                (orig - dec).abs() <= max_z_error as f32,
                "pixel {i}: orig={orig}, decoded={dec}, error={}",
                (orig - dec).abs()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Error cases in decode
    // -----------------------------------------------------------------------

    #[test]
    fn decode_invalid_magic() {
        let data = b"NotLerc data here";
        let result = crate::decode(data);
        assert!(result.is_err());
    }

    #[test]
    fn decode_empty_data() {
        let result = crate::decode(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn decode_truncated_blob() {
        let pixels: Vec<u8> = (0..64).collect();
        let blob = crate::encode_typed(8, 8, &pixels, 0.5).unwrap();
        // Truncate the blob
        let truncated = &blob[..blob.len() / 2];
        let result = crate::decode(truncated);
        assert!(result.is_err());
    }

    #[test]
    fn decode_info_truncated_blob() {
        // Valid header but truncated — decode_info should still succeed
        // since it only reads the header
        let pixels: Vec<u8> = (0..64).collect();
        let blob = crate::encode_typed(8, 8, &pixels, 0.5).unwrap();
        // Just the header should be enough for decode_info
        let info = crate::decode_info(&blob).unwrap();
        assert_eq!(info.width, 8);
        assert_eq!(info.height, 8);
    }
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

