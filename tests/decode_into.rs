use lerc::bitmask::BitMask;
use lerc::{DataType, LercData, LercImage};

/// Helper: encode an image, then decode with both `decode()` and `decode_*_into()`,
/// and verify the results match.
macro_rules! test_decode_into_matches {
    ($name:ident, $dt:expr, $variant:ident, $into_fn:path, $ty:ty, $default:expr, $pixels:expr, $width:expr, $height:expr, $max_z_error:expr) => {
        #[test]
        fn $name() {
            let width: u32 = $width;
            let height: u32 = $height;
            let pixels: Vec<$ty> = $pixels;

            let image = LercImage {
                width,
                height,
                n_depth: 1,
                n_bands: 1,
                data_type: $dt,
                valid_masks: vec![BitMask::all_valid((width * height) as usize)],
                data: LercData::$variant(pixels.clone()),
                no_data_value: None,
            };

            let encoded = lerc::encode(&image, $max_z_error).expect("encode failed");

            // Decode with allocating API
            let decoded = lerc::decode(&encoded).expect("decode failed");
            let expected_pixels = match &decoded.data {
                LercData::$variant(p) => p.clone(),
                _ => panic!("unexpected data type from decode()"),
            };

            // Decode into pre-allocated buffer
            let total = (width * height) as usize;
            let mut output = vec![$default; total];
            let result = $into_fn(&encoded, &mut output).expect("decode_into failed");

            assert_eq!(result.width, width);
            assert_eq!(result.height, height);
            assert_eq!(result.n_depth, 1);
            assert_eq!(result.n_bands, 1);
            assert_eq!(result.data_type, $dt);
            assert_eq!(result.valid_masks.len(), decoded.valid_masks.len());
            assert_eq!(output, expected_pixels, "decode_into output differs from decode()");
        }
    };
}

test_decode_into_matches!(
    decode_into_u8_matches_decode,
    DataType::Byte,
    U8,
    lerc::decode_u8_into,
    u8,
    0u8,
    (0..64 * 64).map(|i| (i % 256) as u8).collect::<Vec<u8>>(),
    64,
    64,
    0.5
);

test_decode_into_matches!(
    decode_into_i8_matches_decode,
    DataType::Char,
    I8,
    lerc::decode_i8_into,
    i8,
    0i8,
    (0..64 * 64)
        .map(|i| ((i % 256) as i16 - 128) as i8)
        .collect::<Vec<i8>>(),
    64,
    64,
    0.5
);

test_decode_into_matches!(
    decode_into_i16_matches_decode,
    DataType::Short,
    I16,
    lerc::decode_i16_into,
    i16,
    0i16,
    (0..64 * 64).map(|i| i as i16 - 2000).collect::<Vec<i16>>(),
    64,
    64,
    0.0
);

test_decode_into_matches!(
    decode_into_u16_matches_decode,
    DataType::UShort,
    U16,
    lerc::decode_u16_into,
    u16,
    0u16,
    (0..64 * 64).map(|i| (i % 65536) as u16).collect::<Vec<u16>>(),
    64,
    64,
    0.0
);

test_decode_into_matches!(
    decode_into_i32_matches_decode,
    DataType::Int,
    I32,
    lerc::decode_i32_into,
    i32,
    0i32,
    (0..32 * 32).map(|i| i as i32 * 100 - 50000).collect::<Vec<i32>>(),
    32,
    32,
    0.0
);

test_decode_into_matches!(
    decode_into_u32_matches_decode,
    DataType::UInt,
    U32,
    lerc::decode_u32_into,
    u32,
    0u32,
    (0..32 * 32).map(|i| i as u32 * 7).collect::<Vec<u32>>(),
    32,
    32,
    0.0
);

test_decode_into_matches!(
    decode_into_f32_matches_decode,
    DataType::Float,
    F32,
    lerc::decode_f32_into,
    f32,
    0.0f32,
    (0..64 * 64).map(|i| i as f32 * 0.1).collect::<Vec<f32>>(),
    64,
    64,
    0.001
);

test_decode_into_matches!(
    decode_into_f64_matches_decode,
    DataType::Double,
    F64,
    lerc::decode_f64_into,
    f64,
    0.0f64,
    (0..32 * 32).map(|i| i as f64 * 0.01).collect::<Vec<f64>>(),
    32,
    32,
    0.0001
);

#[test]
fn decode_into_buffer_too_small() {
    let width = 32u32;
    let height = 32u32;
    let pixels: Vec<f32> = (0..width * height).map(|i| i as f32).collect();

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: LercData::F32(pixels),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, 0.001).expect("encode failed");

    // Buffer is too small
    let mut output = vec![0.0f32; 10];
    let err = lerc::decode_f32_into(&encoded, &mut output).unwrap_err();
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

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: LercData::F32(pixels),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, 0.001).expect("encode failed");

    // Try to decode f32 data into a u8 buffer
    let mut output = vec![0u8; (width * height) as usize];
    let err = lerc::decode_u8_into(&encoded, &mut output).unwrap_err();
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

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Byte,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: LercData::U8(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, 0.5).expect("encode failed");

    // Buffer is larger than needed -- should succeed
    let mut output = vec![0xFFu8; (width * height * 2) as usize];
    let result = lerc::decode_u8_into(&encoded, &mut output).expect("decode_into should succeed");

    assert_eq!(result.width, width);
    assert_eq!(result.height, height);

    // Verify the first (width*height) elements match, rest untouched
    let decoded = lerc::decode(&encoded).expect("decode failed");
    let expected = match &decoded.data {
        LercData::U8(p) => p,
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

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands,
        data_type: DataType::Byte,
        valid_masks: vec![BitMask::all_valid(band_size)],
        data: LercData::U8(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, 0.5).expect("encode failed");

    // Decode with allocating API
    let decoded = lerc::decode(&encoded).expect("decode failed");
    let expected = match &decoded.data {
        LercData::U8(p) => p.clone(),
        _ => panic!("expected U8"),
    };

    // Decode into pre-allocated buffer
    let mut output = vec![0u8; total];
    let result = lerc::decode_u8_into(&encoded, &mut output).expect("decode_into failed");

    assert_eq!(result.n_bands, n_bands);
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
        .map(|i| if mask.is_valid(i) { i as f32 * 0.5 } else { 0.0 })
        .collect();

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![mask.clone()],
        data: LercData::F32(pixels),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, 0.001).expect("encode failed");

    let decoded = lerc::decode(&encoded).expect("decode failed");
    let expected = match &decoded.data {
        LercData::F32(p) => p.clone(),
        _ => panic!("expected F32"),
    };

    let mut output = vec![0.0f32; total];
    let result = lerc::decode_f32_into(&encoded, &mut output).expect("decode_into failed");

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

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::UInt,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: LercData::U32(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, 0.0).expect("encode failed");

    let mut output = vec![0u32; (width * height) as usize];
    let result = lerc::decode_into::<u32>(&encoded, &mut output).expect("decode_into failed");

    assert_eq!(result.data_type, DataType::UInt);

    let decoded = lerc::decode(&encoded).expect("decode failed");
    let expected = match &decoded.data {
        LercData::U32(p) => p,
        _ => panic!("expected U32"),
    };
    assert_eq!(&output, expected);
}
