use lerc::Precision;
use lerc::bitmask::BitMask;

// ---------------------------------------------------------------------------
// Round-trip tests for every data type via encode_slice / decode_slice
// ---------------------------------------------------------------------------

macro_rules! round_trip_test {
    ($test_name:ident, $ty:ty, $values:expr) => {
        #[test]
        fn $test_name() {
            let width = 4u32;
            let height = 4u32;
            let pixels: Vec<$ty> = $values;
            assert_eq!(pixels.len(), (width * height) as usize);

            let blob = lerc::encode_slice(width, height, &pixels, Precision::Lossless)
                .expect("encode failed");
            let (decoded_pixels, mask, w, h) =
                lerc::decode_slice::<$ty>(&blob).expect("decode failed");

            assert_eq!(w, width);
            assert_eq!(h, height);
            assert_eq!(decoded_pixels, pixels);
            // All pixels should be valid
            for k in 0..(width * height) as usize {
                assert!(mask.is_valid(k), "pixel {k} should be valid");
            }
        }
    };
}

round_trip_test!(round_trip_i8, i8, (0..16).map(|i| (i - 8) as i8).collect());
round_trip_test!(round_trip_u8, u8, (0..16).map(|i| i as u8).collect());
round_trip_test!(
    round_trip_i16,
    i16,
    (0..16).map(|i| (i * 100 - 800) as i16).collect()
);
round_trip_test!(
    round_trip_u16,
    u16,
    (0..16).map(|i| (i * 100) as u16).collect()
);
round_trip_test!(
    round_trip_i32,
    i32,
    (0..16).map(|i| i * 1000 - 8000).collect()
);
round_trip_test!(round_trip_u32, u32, (0..16).map(|i| i * 1000).collect());
round_trip_test!(
    round_trip_f32,
    f32,
    (0..16).map(|i| i as f32 * 0.5).collect()
);
round_trip_test!(
    round_trip_f64,
    f64,
    (0..16).map(|i| i as f64 * 0.25).collect()
);

// ---------------------------------------------------------------------------
// Masked encode/decode round-trips
// ---------------------------------------------------------------------------

#[test]
fn round_trip_f32_masked() {
    let width = 8u32;
    let height = 8u32;
    let n = (width * height) as usize;

    let mut mask = BitMask::new(n);
    for k in 0..n {
        if k % 2 == 0 {
            mask.set_valid(k);
        }
    }

    let pixels: Vec<f32> = (0..n as u32).map(|i| i as f32 * 1.5).collect();
    let blob = lerc::encode_slice_masked(width, height, &pixels, &mask, Precision::Lossless)
        .expect("encode failed");
    let (decoded_pixels, decoded_mask, w, h) =
        lerc::decode_slice::<f32>(&blob).expect("decode failed");

    assert_eq!(w, width);
    assert_eq!(h, height);
    for k in 0..n {
        if mask.is_valid(k) {
            assert!(decoded_mask.is_valid(k), "pixel {k} should be valid");
            assert_eq!(decoded_pixels[k], pixels[k], "pixel {k} value mismatch");
        }
    }
}

#[test]
fn round_trip_u8_masked() {
    let width = 4u32;
    let height = 4u32;
    let n = (width * height) as usize;

    let mut mask = BitMask::new(n);
    for k in 0..n {
        if k % 3 == 0 {
            mask.set_valid(k);
        }
    }

    let pixels: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
    let blob = lerc::encode_slice_masked(width, height, &pixels, &mask, Precision::Lossless)
        .expect("encode failed");
    let (decoded_pixels, decoded_mask, w, h) =
        lerc::decode_slice::<u8>(&blob).expect("decode failed");

    assert_eq!(w, width);
    assert_eq!(h, height);
    for k in 0..n {
        if mask.is_valid(k) {
            assert!(decoded_mask.is_valid(k));
            assert_eq!(decoded_pixels[k], pixels[k]);
        }
    }
}

// ---------------------------------------------------------------------------
// Accessor methods on Image via as_typed
// ---------------------------------------------------------------------------

#[test]
fn lerc_image_as_typed() {
    let width = 4u32;
    let height = 4u32;
    let pixels: Vec<f32> = (0..16).map(|i| i as f32).collect();

    let blob =
        lerc::encode_slice(width, height, &pixels, Precision::Lossless).expect("encode failed");
    let image = lerc::decode(&blob).expect("decode failed");

    // Correct type succeeds
    assert_eq!(image.as_typed::<f32>().unwrap(), &pixels[..]);

    // Wrong types return None
    assert!(image.as_typed::<f64>().is_none());
    assert!(image.as_typed::<u8>().is_none());
    assert!(image.as_typed::<i32>().is_none());

    // mask() helper
    let m = image.mask().expect("should have a mask");
    assert!(m.is_valid(0));
}

// ---------------------------------------------------------------------------
// Validation error tests
// ---------------------------------------------------------------------------

#[test]
fn encode_wrong_data_length() {
    // 3x3 = 9 pixels but we provide 10 values
    let result = lerc::encode_slice(3, 3, &[0.0f32; 10], Precision::Lossless);
    assert!(result.is_err(), "should fail with wrong data length");
}

#[test]
fn encode_masked_wrong_mask_length() {
    let mask = BitMask::all_valid(100); // 100 pixels
    let result = lerc::encode_slice_masked(3, 3, &[0.0f32; 9], &mask, Precision::Lossless); // 9 pixels vs 100
    assert!(result.is_err(), "should fail with mismatched mask size");
}

#[test]
fn decode_wrong_type() {
    // Encode as u8 then try to decode as f32
    let blob =
        lerc::encode_slice(2, 2, &[1u8, 2, 3, 4], Precision::Lossless).expect("encode failed");
    let result = lerc::decode_slice::<f32>(&blob);
    assert!(result.is_err(), "should fail with type mismatch");
}

// ---------------------------------------------------------------------------
// Lossy round-trip test
// ---------------------------------------------------------------------------

#[test]
fn lossy_round_trip_f64() {
    let width = 8u32;
    let height = 8u32;
    let max_z_error = 0.5;
    let pixels: Vec<f64> = (0..64).map(|i| i as f64 * 0.1).collect();

    let blob = lerc::encode_slice(width, height, &pixels, Precision::Tolerance(max_z_error))
        .expect("encode failed");
    let (decoded, _mask, w, h) = lerc::decode_slice::<f64>(&blob).expect("decode failed");

    assert_eq!(w, width);
    assert_eq!(h, height);
    for (i, (&orig, &dec)) in pixels.iter().zip(decoded.iter()).enumerate() {
        let diff = (orig - dec).abs();
        assert!(
            diff <= max_z_error,
            "pixel {i}: diff {diff} exceeds max_z_error {max_z_error}"
        );
    }
}

#[test]
fn decode_slice_rejects_multiband() {
    use lerc::{DataType, Image, SampleData};

    let width = 8u32;
    let height = 8u32;
    let band_size = (width * height) as usize;
    let pixels = vec![0u8; band_size * 3];

    let image = Image {
        width,
        height,
        depth: 1,
        bands: 3,
        data_type: DataType::Byte,
        valid_masks: vec![BitMask::all_valid(band_size)],
        data: SampleData::U8(pixels),
        no_data_value: None,
    };

    let blob = lerc::encode(&image, Precision::Lossless).unwrap();
    let result = lerc::decode_slice::<u8>(&blob);
    assert!(
        result.is_err(),
        "decode_slice should reject multi-band blobs"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("single-band"),
        "error should mention single-band: {err}"
    );
}

#[test]
fn decode_slice_rejects_multidepth() {
    use lerc::{DataType, Image, SampleData};

    let width = 8u32;
    let height = 8u32;
    let pixels = vec![0.0f32; (width * height * 3) as usize];

    let image = Image {
        width,
        height,
        depth: 3,
        bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: SampleData::F32(pixels),
        no_data_value: None,
    };

    let blob = lerc::encode(&image, Precision::Tolerance(0.01)).unwrap();
    let result = lerc::decode_slice::<f32>(&blob);
    assert!(
        result.is_err(),
        "decode_slice should reject multi-depth blobs"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("single-depth"),
        "error should mention single-depth: {err}"
    );
}

// ---------------------------------------------------------------------------
// Image ergonomic API tests
// ---------------------------------------------------------------------------

#[test]
fn lerc_image_pixel_accessor() {
    let width = 4u32;
    let height = 3u32;
    let pixels: Vec<f32> = (0..12).map(|i| i as f32 * 2.0).collect();

    let blob =
        lerc::encode_slice(width, height, &pixels, Precision::Lossless).expect("encode failed");
    let image = lerc::decode(&blob).expect("decode failed");

    // Valid coordinates
    assert_eq!(image.pixel::<f32>(0, 0), Some(0.0));
    assert_eq!(image.pixel::<f32>(0, 3), Some(6.0));
    assert_eq!(image.pixel::<f32>(2, 3), Some(22.0));

    // Out of bounds
    assert_eq!(image.pixel::<f32>(3, 0), None);
    assert_eq!(image.pixel::<f32>(0, 4), None);

    // Wrong type
    assert_eq!(image.pixel::<u8>(0, 0), None);
}

#[test]
fn lerc_image_valid_pixels_all_valid() {
    let width = 3u32;
    let height = 2u32;
    let pixels: Vec<u16> = vec![10, 20, 30, 40, 50, 60];

    let blob =
        lerc::encode_slice(width, height, &pixels, Precision::Lossless).expect("encode failed");
    let image = lerc::decode(&blob).expect("decode failed");

    let valid: Vec<(u32, u32, u16)> = image.valid_pixels::<u16>().unwrap().collect();
    assert_eq!(valid.len(), 6);
    assert_eq!(valid[0], (0, 0, 10));
    assert_eq!(valid[1], (0, 1, 20));
    assert_eq!(valid[2], (0, 2, 30));
    assert_eq!(valid[3], (1, 0, 40));
    assert_eq!(valid[4], (1, 1, 50));
    assert_eq!(valid[5], (1, 2, 60));
}

#[test]
fn lerc_image_valid_pixels_with_mask() {
    let width = 4u32;
    let height = 2u32;
    let n = (width * height) as usize;

    let mut mask = BitMask::new(n);
    // Mark only even indices as valid: 0, 2, 4, 6
    for k in (0..n).step_by(2) {
        mask.set_valid(k);
    }

    let pixels: Vec<i32> = (0..n as i32).collect();
    let blob = lerc::encode_slice_masked(width, height, &pixels, &mask, Precision::Lossless)
        .expect("encode failed");
    let image = lerc::decode(&blob).expect("decode failed");

    let valid: Vec<(u32, u32, i32)> = image.valid_pixels::<i32>().unwrap().collect();
    assert_eq!(valid.len(), 4);
    assert_eq!(valid[0], (0, 0, 0)); // idx 0
    assert_eq!(valid[1], (0, 2, 2)); // idx 2
    assert_eq!(valid[2], (1, 0, 4)); // idx 4
    assert_eq!(valid[3], (1, 2, 6)); // idx 6
}

#[test]
fn lerc_image_valid_pixels_wrong_type() {
    let pixels: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let blob = lerc::encode_slice(2, 2, &pixels, Precision::Lossless).expect("encode failed");
    let image = lerc::decode(&blob).expect("decode failed");

    assert!(image.valid_pixels::<u8>().is_none());
}

#[test]
fn lerc_image_dimensions_and_num_pixels() {
    let width = 5u32;
    let height = 7u32;
    let pixels: Vec<u8> = vec![0; (width * height) as usize];

    let blob =
        lerc::encode_slice(width, height, &pixels, Precision::Lossless).expect("encode failed");
    let image = lerc::decode(&blob).expect("decode failed");

    assert_eq!(image.dimensions(), (5, 7));
    assert_eq!(image.num_pixels(), 35);
}

#[test]
fn lerc_image_all_valid_true() {
    let pixels: Vec<f32> = vec![1.0; 16];
    let blob = lerc::encode_slice(4, 4, &pixels, Precision::Lossless).expect("encode failed");
    let image = lerc::decode(&blob).expect("decode failed");

    assert!(image.all_valid());
}

#[test]
fn lerc_image_all_valid_false() {
    let width = 4u32;
    let height = 4u32;
    let n = (width * height) as usize;

    let mut mask = BitMask::new(n);
    for k in 0..n - 1 {
        mask.set_valid(k);
    }
    // Last pixel is invalid

    let pixels: Vec<f32> = vec![1.0; n];
    let blob = lerc::encode_slice_masked(width, height, &pixels, &mask, Precision::Lossless)
        .expect("encode failed");
    let image = lerc::decode(&blob).expect("decode failed");

    assert!(!image.all_valid());
}

#[test]
fn lerc_image_from_pixels() {
    let width = 3u32;
    let height = 2u32;
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let image = lerc::Image::from_pixels(width, height, data.clone()).expect("from_pixels");
    assert_eq!(image.width, width);
    assert_eq!(image.height, height);
    assert_eq!(image.bands, 1);
    assert_eq!(image.depth, 1);
    assert_eq!(image.data_type, lerc::DataType::Float);
    assert!(image.all_valid());
    assert_eq!(image.as_typed::<f32>().unwrap(), &data[..]);

    // Round-trip through encode/decode
    let blob = lerc::encode(&image, Precision::Lossless).expect("encode failed");
    let decoded = lerc::decode(&blob).expect("decode failed");
    assert_eq!(decoded.as_typed::<f32>().unwrap(), &data[..]);
}

#[test]
fn lerc_image_from_pixels_wrong_size() {
    let result = lerc::Image::from_pixels::<u8>(3, 3, vec![0u8; 10]);
    assert!(result.is_err());
}

#[test]
fn lerc_image_from_pixels_all_types() {
    // Verify from_pixels works for every Sample
    assert!(lerc::Image::from_pixels(2, 2, vec![0i8; 4]).is_ok());
    assert!(lerc::Image::from_pixels(2, 2, vec![0u8; 4]).is_ok());
    assert!(lerc::Image::from_pixels(2, 2, vec![0i16; 4]).is_ok());
    assert!(lerc::Image::from_pixels(2, 2, vec![0u16; 4]).is_ok());
    assert!(lerc::Image::from_pixels(2, 2, vec![0i32; 4]).is_ok());
    assert!(lerc::Image::from_pixels(2, 2, vec![0u32; 4]).is_ok());
    assert!(lerc::Image::from_pixels(2, 2, vec![0.0f32; 4]).is_ok());
    assert!(lerc::Image::from_pixels(2, 2, vec![0.0f64; 4]).is_ok());
}

#[test]
fn lerc_image_pixel_multiband_returns_none() {
    use lerc::{DataType, Image, SampleData};

    let image = Image {
        width: 2,
        height: 2,
        depth: 1,
        bands: 2,
        data_type: DataType::Byte,
        valid_masks: vec![BitMask::all_valid(4)],
        data: SampleData::U8(vec![0; 8]),
        no_data_value: None,
    };

    assert_eq!(image.pixel::<u8>(0, 0), None);
    assert!(image.valid_pixels::<u8>().is_none());
}
