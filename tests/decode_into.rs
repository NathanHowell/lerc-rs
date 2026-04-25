use lerc::Precision;
use lerc::bitmask::BitMask;
use lerc::{DataType, Image, SampleData};

/// Helper: encode an image, then decode with both `decode()` and `decode_*_into()`,
/// and verify the results match.
macro_rules! test_decode_into_matches {
    ($name:ident, $dt:expr, $variant:ident, $into_fn:path, $ty:ty, $default:expr, $pixels:expr, $width:expr, $height:expr, $precision:expr) => {
        #[test]
        fn $name() {
            let width: u32 = $width;
            let height: u32 = $height;
            let pixels: Vec<$ty> = $pixels;

            let image = Image {
                width,
                height,
                depth: 1,
                bands: 1,
                data_type: $dt,
                valid_masks: vec![BitMask::all_valid((width * height) as usize)],
                data: SampleData::$variant(pixels.clone()),
                no_data_value: None,
            };

            let encoded = lerc::encode(&image, $precision).expect("encode failed");

            // Decode with allocating API
            let decoded = lerc::decode(&encoded).expect("decode failed");
            let expected_pixels = match &decoded.data {
                SampleData::$variant(p) => p.clone(),
                _ => panic!("unexpected data type from decode()"),
            };

            // Decode into pre-allocated buffer
            let total = (width * height) as usize;
            let mut output = vec![$default; total];
            let result = $into_fn(&encoded, &mut output).expect("decode_into failed");

            assert_eq!(result.width, width);
            assert_eq!(result.height, height);
            assert_eq!(result.depth, 1);
            assert_eq!(result.bands, 1);
            assert_eq!(result.data_type, $dt);
            assert_eq!(result.valid_masks.len(), decoded.valid_masks.len());
            assert_eq!(
                output, expected_pixels,
                "decode_into output differs from decode()"
            );
        }
    };
}

test_decode_into_matches!(
    decode_into_u8_matches_decode,
    DataType::Byte,
    U8,
    lerc::decode_into,
    u8,
    0u8,
    (0..64 * 64).map(|i| (i % 256) as u8).collect::<Vec<u8>>(),
    64,
    64,
    Precision::Lossless
);

test_decode_into_matches!(
    decode_into_i8_matches_decode,
    DataType::Char,
    I8,
    lerc::decode_into,
    i8,
    0i8,
    (0..64 * 64)
        .map(|i| ((i % 256) as i16 - 128) as i8)
        .collect::<Vec<i8>>(),
    64,
    64,
    Precision::Lossless
);

test_decode_into_matches!(
    decode_into_i16_matches_decode,
    DataType::Short,
    I16,
    lerc::decode_into,
    i16,
    0i16,
    (0..64 * 64).map(|i| i as i16 - 2000).collect::<Vec<i16>>(),
    64,
    64,
    Precision::Lossless
);

test_decode_into_matches!(
    decode_into_u16_matches_decode,
    DataType::UShort,
    U16,
    lerc::decode_into,
    u16,
    0u16,
    (0..64 * 64)
        .map(|i| (i % 65536) as u16)
        .collect::<Vec<u16>>(),
    64,
    64,
    Precision::Lossless
);

test_decode_into_matches!(
    decode_into_i32_matches_decode,
    DataType::Int,
    I32,
    lerc::decode_into,
    i32,
    0i32,
    (0..32 * 32).map(|i| i * 100 - 50000).collect::<Vec<i32>>(),
    32,
    32,
    Precision::Lossless
);

test_decode_into_matches!(
    decode_into_u32_matches_decode,
    DataType::UInt,
    U32,
    lerc::decode_into,
    u32,
    0u32,
    (0..32 * 32).map(|i| i as u32 * 7).collect::<Vec<u32>>(),
    32,
    32,
    Precision::Lossless
);

test_decode_into_matches!(
    decode_into_f32_matches_decode,
    DataType::Float,
    F32,
    lerc::decode_into,
    f32,
    0.0f32,
    (0..64 * 64).map(|i| i as f32 * 0.1).collect::<Vec<f32>>(),
    64,
    64,
    Precision::Tolerance(0.001)
);

test_decode_into_matches!(
    decode_into_f64_matches_decode,
    DataType::Double,
    F64,
    lerc::decode_into,
    f64,
    0.0f64,
    (0..32 * 32).map(|i| i as f64 * 0.01).collect::<Vec<f64>>(),
    32,
    32,
    Precision::Tolerance(0.0001)
);

#[test]
fn decode_into_buffer_too_small() {
    let width = 32u32;
    let height = 32u32;
    let pixels: Vec<f32> = (0..width * height).map(|i| i as f32).collect();

    let image = Image {
        width,
        height,
        depth: 1,
        bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: SampleData::F32(pixels),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Tolerance(0.001)).expect("encode failed");

    // Buffer is too small
    let mut output = vec![0.0f32; 10];
    let err = lerc::decode_into(&encoded, &mut output).unwrap_err();
    match err {
        lerc::LercError::OutputBufferTooSmall { needed, available } => {
            assert_eq!(needed, (width * height) as usize);
            assert_eq!(available, 10);
        }
        other => panic!("expected OutputBufferTooSmall, got: {other:?}"),
    }
}

#[test]
fn decode_into_type_mismatch() {
    let width = 16u32;
    let height = 16u32;
    let pixels: Vec<f32> = (0..width * height).map(|i| i as f32).collect();

    let image = Image {
        width,
        height,
        depth: 1,
        bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: SampleData::F32(pixels),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Tolerance(0.001)).expect("encode failed");

    // Try to decode f32 data into a u8 buffer
    let mut output = vec![0u8; (width * height) as usize];
    let err = lerc::decode_into(&encoded, &mut output).unwrap_err();
    match err {
        lerc::LercError::TypeMismatch { expected, actual } => {
            assert_eq!(expected, DataType::Float);
            assert_eq!(actual, DataType::Byte);
        }
        other => panic!("expected TypeMismatch, got: {other:?}"),
    }
}

#[test]
fn decode_into_oversized_buffer_ok() {
    let width = 16u32;
    let height = 16u32;
    let pixels: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();

    let image = Image {
        width,
        height,
        depth: 1,
        bands: 1,
        data_type: DataType::Byte,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: SampleData::U8(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Lossless).expect("encode failed");

    // Buffer is larger than needed -- should succeed
    let mut output = vec![0xFFu8; (width * height * 2) as usize];
    let result = lerc::decode_into(&encoded, &mut output).expect("decode_into should succeed");

    assert_eq!(result.width, width);
    assert_eq!(result.height, height);

    // Verify the first (width*height) elements match, rest untouched
    let decoded = lerc::decode(&encoded).expect("decode failed");
    let expected = match &decoded.data {
        SampleData::U8(p) => p,
        _ => panic!("expected U8"),
    };
    let total = (width * height) as usize;
    assert_eq!(&output[..total], expected.as_slice());
    // The trailing portion should be untouched (still 0xFF)
    for &v in &output[total..] {
        assert_eq!(v, 0xFF, "trailing buffer should be untouched");
    }
}

#[test]
fn decode_into_multiband() {
    let width = 32u32;
    let height = 32u32;
    let n_bands = 3u32;
    let band_size = (width * height) as usize;
    let total = band_size * n_bands as usize;

    let pixels: Vec<u8> = (0..total).map(|i| (i % 10) as u8).collect();

    let image = Image {
        width,
        height,
        depth: 1,
        bands: n_bands,
        data_type: DataType::Byte,
        valid_masks: vec![BitMask::all_valid(band_size)],
        data: SampleData::U8(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Lossless).expect("encode failed");

    // Decode with allocating API
    let decoded = lerc::decode(&encoded).expect("decode failed");
    let expected = match &decoded.data {
        SampleData::U8(p) => p.clone(),
        _ => panic!("expected U8"),
    };

    // Decode into pre-allocated buffer
    let mut output = vec![0u8; total];
    let result = lerc::decode_into(&encoded, &mut output).expect("decode_into failed");

    assert_eq!(result.bands, n_bands);
    assert_eq!(result.valid_masks.len(), decoded.valid_masks.len());
    assert_eq!(output, expected);
}

#[test]
fn decode_into_with_mask() {
    let width = 32u32;
    let height = 32u32;
    let total = (width * height) as usize;

    let mut mask = BitMask::new(total);
    for k in 0..total {
        if k % 3 != 0 {
            mask.set_valid(k);
        }
    }

    let pixels: Vec<f32> = (0..total)
        .map(|i| {
            if mask.is_valid(i) {
                i as f32 * 0.5
            } else {
                0.0
            }
        })
        .collect();

    let image = Image {
        width,
        height,
        depth: 1,
        bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![mask.clone()],
        data: SampleData::F32(pixels),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Tolerance(0.001)).expect("encode failed");

    let decoded = lerc::decode(&encoded).expect("decode failed");
    let expected = match &decoded.data {
        SampleData::F32(p) => p.clone(),
        _ => panic!("expected F32"),
    };

    let mut output = vec![0.0f32; total];
    let result = lerc::decode_into(&encoded, &mut output).expect("decode_into failed");

    // Verify masks match
    assert_eq!(result.valid_masks.len(), 1);
    for k in 0..total {
        assert_eq!(
            result.valid_masks[0].is_valid(k),
            decoded.valid_masks[0].is_valid(k),
            "mask mismatch at pixel {k}"
        );
    }

    assert_eq!(output, expected);
}

#[test]
fn decode_into_generic_api() {
    // Test the generic decode_into<T> function directly
    let width = 16u32;
    let height = 16u32;
    let pixels: Vec<u32> = (0..width * height).map(|i| i * 42).collect();

    let image = Image {
        width,
        height,
        depth: 1,
        bands: 1,
        data_type: DataType::UInt,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: SampleData::U32(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Lossless).expect("encode failed");

    let mut output = vec![0u32; (width * height) as usize];
    let result = lerc::decode_into::<u32>(&encoded, &mut output).expect("decode_into failed");

    assert_eq!(result.data_type, DataType::UInt);

    let decoded = lerc::decode(&encoded).expect("decode failed");
    let expected = match &decoded.data {
        SampleData::U32(p) => p,
        _ => panic!("expected U32"),
    };
    assert_eq!(&output, expected);
}

// ---------------------------------------------------------------------------
// decode_into_with_nodata tests
// ---------------------------------------------------------------------------

#[test]
fn decode_into_with_nodata_f32_single_band() {
    let width = 16u32;
    let height = 16u32;
    let total = (width * height) as usize;

    // Mark every 4th pixel as invalid.
    let mut mask = BitMask::all_valid(total);
    let mut invalid_indices = Vec::new();
    for k in 0..total {
        if k % 4 == 0 {
            mask.set_invalid(k);
            invalid_indices.push(k);
        }
    }

    let pixels: Vec<f32> = (0..total)
        .map(|i| {
            if mask.is_valid(i) {
                i as f32 * 0.5
            } else {
                0.0
            }
        })
        .collect();

    let image = Image {
        width,
        height,
        depth: 1,
        bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![mask.clone()],
        data: SampleData::F32(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Tolerance(0.001)).expect("encode failed");

    let mut output = vec![0.0f32; total];
    let result = lerc::decode_into_with_nodata::<f32>(&encoded, &mut output, f32::NAN)
        .expect("decode_into_with_nodata failed");

    // Mask is still returned to the caller.
    assert_eq!(result.valid_masks.len(), 1);
    let result_mask = &result.valid_masks[0];

    for k in 0..total {
        let row = (k / width as usize) as u32;
        let col = (k % width as usize) as u32;
        let label = format!("pixel ({row}, {col}) idx {k}");
        if result_mask.is_valid(k) {
            assert!(
                !output[k].is_nan(),
                "{label} should be valid and not NaN, got {}",
                output[k]
            );
            assert!(
                (output[k] - pixels[k]).abs() <= 0.001,
                "{label}: expected ~{}, got {}",
                pixels[k],
                output[k]
            );
        } else {
            assert!(
                output[k].is_nan(),
                "{label} should be NaN, got {}",
                output[k]
            );
        }
    }
}

#[test]
fn decode_into_with_nodata_i32_single_band() {
    let width = 12u32;
    let height = 8u32;
    let total = (width * height) as usize;

    let mut mask = BitMask::all_valid(total);
    for k in [0usize, 3, 7, 17, 50, total - 1] {
        mask.set_invalid(k);
    }

    let pixels: Vec<i32> = (0..total)
        .map(|i| {
            if mask.is_valid(i) {
                (i as i32) * 100 - 5000
            } else {
                0
            }
        })
        .collect();

    let image = Image {
        width,
        height,
        depth: 1,
        bands: 1,
        data_type: DataType::Int,
        valid_masks: vec![mask.clone()],
        data: SampleData::I32(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Lossless).expect("encode failed");

    let sentinel = i32::MIN;
    let mut output = vec![0i32; total];
    let result = lerc::decode_into_with_nodata::<i32>(&encoded, &mut output, sentinel)
        .expect("decode_into_with_nodata failed");

    let result_mask = &result.valid_masks[0];
    for k in 0..total {
        if result_mask.is_valid(k) {
            assert_eq!(output[k], pixels[k], "valid pixel {k} mismatch");
            assert_ne!(
                output[k], sentinel,
                "valid pixel {k} should not be sentinel"
            );
        } else {
            assert_eq!(output[k], sentinel, "invalid pixel {k} should be sentinel");
        }
    }
}

#[test]
fn decode_into_with_nodata_depth3_writes_all_slices() {
    // Single band, depth = 3. Confirm the sentinel lands in every depth slice
    // of every invalid pixel.
    let width = 8u32;
    let height = 8u32;
    let depth = 3u32;
    let pixel_count = (width * height) as usize;
    let total = pixel_count * depth as usize;

    let mut mask = BitMask::all_valid(pixel_count);
    let invalid: Vec<usize> = vec![0, 5, 11, 33, 63];
    for &k in &invalid {
        mask.set_invalid(k);
    }

    // Build interleaved depth-3 pixels: row-major over (row, col, d).
    let mut pixels = vec![0.0f32; total];
    for i in 0..height as usize {
        for j in 0..width as usize {
            let k = i * width as usize + j;
            let base = (i * width as usize + j) * depth as usize;
            if mask.is_valid(k) {
                pixels[base] = k as f32;
                pixels[base + 1] = k as f32 + 0.25;
                pixels[base + 2] = k as f32 + 0.5;
            }
        }
    }

    let image = Image {
        width,
        height,
        depth,
        bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![mask.clone()],
        data: SampleData::F32(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Tolerance(0.001)).expect("encode failed");

    let sentinel: f32 = -1.0e30;
    let mut output = vec![0.0f32; total];
    let result = lerc::decode_into_with_nodata::<f32>(&encoded, &mut output, sentinel)
        .expect("decode_into_with_nodata failed");

    let result_mask = &result.valid_masks[0];
    for k in 0..pixel_count {
        let base = k * depth as usize;
        if result_mask.is_valid(k) {
            for d in 0..depth as usize {
                assert!(
                    (output[base + d] - pixels[base + d]).abs() <= 0.001,
                    "valid pixel {k}, depth {d}: expected ~{}, got {}",
                    pixels[base + d],
                    output[base + d]
                );
            }
        } else {
            for d in 0..depth as usize {
                assert_eq!(
                    output[base + d],
                    sentinel,
                    "invalid pixel {k}, depth slice {d} should be sentinel"
                );
            }
        }
    }
}

#[test]
fn decode_into_with_nodata_all_valid_does_not_write_sentinel() {
    // When the mask is AllValid, the function must skip the band entirely and
    // never write the sentinel into output.
    let width = 16u32;
    let height = 16u32;
    let total = (width * height) as usize;

    let pixels: Vec<f32> = (0..total).map(|i| i as f32 + 0.125).collect();

    let image = Image {
        width,
        height,
        depth: 1,
        bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![BitMask::all_valid(total)],
        data: SampleData::F32(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Lossless).expect("encode failed");

    let sentinel: f32 = f32::NAN;
    let mut output = vec![0.0f32; total];
    let result = lerc::decode_into_with_nodata::<f32>(&encoded, &mut output, sentinel)
        .expect("decode_into_with_nodata failed");

    // No NaNs should have been introduced.
    for (k, &v) in output.iter().enumerate() {
        assert!(!v.is_nan(), "pixel {k} unexpectedly NaN");
    }
    // And the mask returned should report all-valid.
    assert!(result.valid_masks[0].is_all_valid());
    assert_eq!(output, pixels);
}

#[test]
fn decode_into_with_nodata_multiband_mixed_masks() {
    // Two bands: first all-valid, second with explicit invalid pixels.
    // Ensures band offsetting is correct and that all-valid bands are skipped.
    let width = 8u32;
    let height = 8u32;
    let bands = 2u32;
    let band_size = (width * height) as usize;
    let total = band_size * bands as usize;

    let mut band0_pixels: Vec<u16> = (0..band_size).map(|i| i as u16).collect();
    let mut band1_pixels: Vec<u16> = (0..band_size).map(|i| (i as u16) * 3 + 7).collect();

    let band0_mask = BitMask::all_valid(band_size);
    let mut band1_mask = BitMask::all_valid(band_size);
    let invalid_band1: Vec<usize> = vec![0, 1, 5, 17, 40, band_size - 1];
    for &k in &invalid_band1 {
        band1_mask.set_invalid(k);
    }
    // Concatenate band 0 then band 1.
    let mut pixels = Vec::with_capacity(total);
    pixels.append(&mut band0_pixels);
    pixels.append(&mut band1_pixels);

    let image = Image {
        width,
        height,
        depth: 1,
        bands,
        data_type: DataType::UShort,
        valid_masks: vec![band0_mask, band1_mask.clone()],
        data: SampleData::U16(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Lossless).expect("encode failed");

    let sentinel: u16 = u16::MAX;
    let mut output = vec![0u16; total];
    let result = lerc::decode_into_with_nodata::<u16>(&encoded, &mut output, sentinel)
        .expect("decode_into_with_nodata failed");

    assert_eq!(result.bands, bands);

    // Band 0 should be untouched by the sentinel-fill logic and equal to the originals.
    for k in 0..band_size {
        assert_eq!(output[k], pixels[k], "band 0 pixel {k} mismatch");
    }

    // Band 1 should have sentinel only at the invalid indices.
    let band1_mask_out = &result.valid_masks[1];
    for k in 0..band_size {
        let idx = band_size + k;
        if band1_mask_out.is_valid(k) {
            assert_eq!(output[idx], pixels[idx], "band 1 valid pixel {k} mismatch");
            assert_ne!(
                output[idx], sentinel,
                "band 1 valid pixel {k} should not be sentinel"
            );
        } else {
            assert_eq!(
                output[idx], sentinel,
                "band 1 invalid pixel {k} should be sentinel"
            );
        }
    }
}
