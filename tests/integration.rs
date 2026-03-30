use lerc::bitmask::BitMask;
use lerc::{DataType, LercData, LercImage};

#[test]
fn huffman_u8_compression_ratio() {
    // Data with repetitive patterns should compress well with Huffman
    let width = 256u32;
    let height = 256u32;
    // Image where most pixels are 0 or 1, ideal for Huffman
    let pixels: Vec<u8> = (0..width * height)
        .map(|i| if i % 7 == 0 { 1 } else { 0 })
        .collect();

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Byte,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: LercData::U8(pixels.clone()),
    };

    let encoded = lerc::encode(&image, 0.5).expect("encode failed");
    let raw_size = (width * height) as usize;

    // Huffman-encoded output should be much smaller than raw pixel data
    assert!(
        encoded.len() < raw_size,
        "encoded size {} should be less than raw size {}",
        encoded.len(),
        raw_size
    );

    // Verify round-trip
    let decoded = lerc::decode(&encoded).expect("decode failed");
    match &decoded.data {
        LercData::U8(dec_pixels) => {
            assert_eq!(dec_pixels, &pixels, "Huffman u8 round-trip mismatch");
        }
        _ => panic!("expected U8 data"),
    }
}

#[test]
fn round_trip_i8_lossless() {
    let width = 64u32;
    let height = 64u32;
    let pixels: Vec<i8> = (0..width * height)
        .map(|i| ((i % 256) as i16 - 128) as i8)
        .collect();

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Char,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: LercData::I8(pixels.clone()),
    };

    let encoded = lerc::encode(&image, 0.5).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    match &decoded.data {
        LercData::I8(dec_pixels) => {
            assert_eq!(dec_pixels, &pixels, "lossless i8 round-trip mismatch");
        }
        _ => panic!("expected I8 data"),
    }
}

#[test]
fn round_trip_u8_with_mask_huffman() {
    let width = 64u32;
    let height = 64u32;
    let mut mask = BitMask::new((width * height) as usize);

    // Set every other pixel valid
    for k in 0..(width * height) as usize {
        if k % 2 == 0 {
            mask.set_valid(k);
        }
    }

    // Use repetitive data that benefits from Huffman
    let pixels: Vec<u8> = (0..width * height)
        .map(|i| if mask.is_valid(i as usize) { (i % 4) as u8 } else { 0 })
        .collect();

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Byte,
        valid_masks: vec![mask.clone()],
        data: LercData::U8(pixels.clone()),
    };

    let encoded = lerc::encode(&image, 0.5).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    let dec_mask = &decoded.valid_masks[0];
    match &decoded.data {
        LercData::U8(dec_pixels) => {
            for k in 0..(width * height) as usize {
                assert_eq!(mask.is_valid(k), dec_mask.is_valid(k), "mask mismatch at {k}");
                if mask.is_valid(k) {
                    assert_eq!(
                        pixels[k], dec_pixels[k],
                        "pixel mismatch at {k}: expected {}, got {}",
                        pixels[k], dec_pixels[k]
                    );
                }
            }
        }
        _ => panic!("expected U8 data"),
    }
}

#[test]
fn round_trip_u8_multiband_huffman() {
    let width = 64u32;
    let height = 64u32;
    let n_bands = 3u32;
    let band_size = (width * height) as usize;

    let pixels: Vec<u8> = (0..band_size * n_bands as usize)
        .map(|i| (i % 10) as u8)
        .collect();

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands,
        data_type: DataType::Byte,
        valid_masks: vec![BitMask::all_valid(band_size)],
        data: LercData::U8(pixels.clone()),
    };

    let encoded = lerc::encode(&image, 0.5).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    match &decoded.data {
        LercData::U8(dec_pixels) => {
            assert_eq!(dec_pixels, &pixels, "multiband u8 round-trip mismatch");
        }
        _ => panic!("expected U8 data"),
    }
}

#[test]
fn decode_california_float() {
    let data = include_bytes!("../esri-lerc/testData/california_400_400_1_float.lerc2");
    let info = lerc::decode_info(data).expect("decode_info failed");

    assert_eq!(info.width, 400);
    assert_eq!(info.height, 400);
    assert_eq!(info.n_depth, 1);
    assert_eq!(info.n_bands, 1);
    assert_eq!(info.data_type, DataType::Float);
    assert!(info.max_z_error >= 0.0);

    let image = lerc::decode(data).expect("decode failed");
    assert_eq!(image.width, 400);
    assert_eq!(image.height, 400);
    assert_eq!(image.n_depth, 1);
    assert_eq!(image.n_bands, 1);
    assert_eq!(image.data_type, DataType::Float);
    assert_eq!(image.valid_masks.len(), 1);

    match &image.data {
        LercData::F32(pixels) => {
            assert_eq!(pixels.len(), 400 * 400);

            // Check that valid pixels are within the header's z range
            let mask = &image.valid_masks[0];
            let mut min_val = f32::MAX;
            let mut max_val = f32::MIN;
            let mut valid_count = 0u32;

            for k in 0..400 * 400 {
                if mask.is_valid(k) {
                    let v = pixels[k];
                    assert!(!v.is_nan(), "valid pixel {k} is NaN");
                    if v < min_val {
                        min_val = v;
                    }
                    if v > max_val {
                        max_val = v;
                    }
                    valid_count += 1;
                }
            }

            assert_eq!(valid_count, info.num_valid_pixels);
            assert!(valid_count > 0);

            // Decoded values should be within [zMin - maxZError, zMax + maxZError]
            let tolerance = info.max_z_error;
            assert!(
                min_val as f64 >= info.z_min - tolerance,
                "min_val {min_val} < z_min {} - tolerance {tolerance}",
                info.z_min
            );
            assert!(
                max_val as f64 <= info.z_max + tolerance,
                "max_val {max_val} > z_max {} + tolerance {tolerance}",
                info.z_max
            );
        }
        _ => panic!("expected F32 data"),
    }
}

#[test]
fn decode_bluemarble_byte() {
    let data = include_bytes!("../esri-lerc/testData/bluemarble_256_256_3_byte.lerc2");
    let info = lerc::decode_info(data).expect("decode_info failed");

    assert_eq!(info.width, 256);
    assert_eq!(info.height, 256);
    assert_eq!(info.data_type, DataType::Byte);
    // 3-band file: band-interleaved via concatenated blobs
    assert_eq!(info.n_bands, 3);
    assert_eq!(info.n_depth, 1);

    let image = lerc::decode(data).expect("decode failed");
    assert_eq!(image.width, 256);
    assert_eq!(image.height, 256);
    assert_eq!(image.data_type, DataType::Byte);

    match &image.data {
        LercData::U8(pixels) => {
            let expected_len =
                256 * 256 * image.n_depth as usize * image.n_bands as usize;
            assert_eq!(pixels.len(), expected_len);

            // Byte values should be in [0, 255] (trivially true for u8)
            // Just verify we got data
            let nonzero = pixels.iter().filter(|&&v| v > 0).count();
            assert!(nonzero > 0, "all pixels are zero");
        }
        _ => panic!("expected U8 data"),
    }
}

#[test]
fn decode_info_roundtrip_consistency() {
    let data = include_bytes!("../esri-lerc/testData/california_400_400_1_float.lerc2");
    let info = lerc::decode_info(data).unwrap();
    let image = lerc::decode(data).unwrap();

    assert_eq!(info.width, image.width);
    assert_eq!(info.height, image.height);
    assert_eq!(info.n_depth, image.n_depth);
    assert_eq!(info.data_type, image.data_type);
}

#[test]
fn round_trip_u8_lossless() {
    let width = 64u32;
    let height = 64u32;
    let pixels: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Byte,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: LercData::U8(pixels.clone()),
    };

    let encoded = lerc::encode(&image, 0.5).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    match &decoded.data {
        LercData::U8(dec_pixels) => {
            assert_eq!(dec_pixels, &pixels, "lossless round-trip mismatch");
        }
        _ => panic!("expected U8 data"),
    }
}

#[test]
fn round_trip_f32_lossy() {
    let width = 32u32;
    let height = 32u32;
    let max_z_error = 0.01;
    let pixels: Vec<f32> = (0..width * height)
        .map(|i| (i as f32) * 0.1 + 100.0)
        .collect();

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: LercData::F32(pixels.clone()),
    };

    let encoded = lerc::encode(&image, max_z_error).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    match &decoded.data {
        LercData::F32(dec_pixels) => {
            assert_eq!(dec_pixels.len(), pixels.len());
            for (i, (&orig, &dec)) in pixels.iter().zip(dec_pixels).enumerate() {
                let diff = (orig - dec).abs();
                assert!(
                    diff <= max_z_error as f32,
                    "pixel {i}: orig={orig}, decoded={dec}, diff={diff} > max_z_error={max_z_error}"
                );
            }
        }
        _ => panic!("expected F32 data"),
    }
}

#[test]
fn round_trip_i32_lossless() {
    let width = 16u32;
    let height = 16u32;
    let pixels: Vec<i32> = (0..width * height).map(|i| i as i32 * 1000 - 100000).collect();

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Int,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: LercData::I32(pixels.clone()),
    };

    // maxZError = 0.5 is lossless for integers
    let encoded = lerc::encode(&image, 0.5).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    match &decoded.data {
        LercData::I32(dec_pixels) => {
            assert_eq!(dec_pixels, &pixels, "lossless i32 round-trip mismatch");
        }
        _ => panic!("expected I32 data"),
    }
}

#[test]
fn round_trip_with_partial_mask() {
    let width = 32u32;
    let height = 32u32;
    let mut mask = BitMask::new((width * height) as usize);

    // Set a checkerboard pattern valid
    for i in 0..height as usize {
        for j in 0..width as usize {
            if (i + j) % 2 == 0 {
                mask.set_valid(i * width as usize + j);
            }
        }
    }

    let pixels: Vec<f32> = (0..width * height)
        .map(|i| if mask.is_valid(i as usize) { i as f32 * 0.5 } else { 0.0 })
        .collect();

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![mask.clone()],
        data: LercData::F32(pixels.clone()),
    };

    let encoded = lerc::encode(&image, 0.01).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    // Check mask is preserved
    let dec_mask = &decoded.valid_masks[0];
    for k in 0..(width * height) as usize {
        assert_eq!(
            mask.is_valid(k),
            dec_mask.is_valid(k),
            "mask mismatch at pixel {k}"
        );
    }

    // Check valid pixel values
    match &decoded.data {
        LercData::F32(dec_pixels) => {
            for k in 0..(width * height) as usize {
                if mask.is_valid(k) {
                    let diff = (pixels[k] - dec_pixels[k]).abs();
                    assert!(diff <= 0.01, "pixel {k}: diff={diff}");
                }
            }
        }
        _ => panic!("expected F32 data"),
    }
}
#[test]
fn round_trip_f32_lossless() {
    let width = 32u32;
    let height = 32u32;
    let pixels: Vec<f32> = (0..width * height)
        .map(|i| (i as f32) * 0.1 + 100.0)
        .collect();

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: LercData::F32(pixels.clone()),
    };

    let encoded = lerc::encode(&image, 0.0).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    assert_eq!(decoded.data_type, DataType::Float);

    match &decoded.data {
        LercData::F32(dec_pixels) => {
            assert_eq!(dec_pixels.len(), pixels.len());
            for (i, (&orig, &dec)) in pixels.iter().zip(dec_pixels).enumerate() {
                assert_eq!(
                    orig.to_bits(),
                    dec.to_bits(),
                    "pixel {i}: orig={orig} (bits={:#010x}), decoded={dec} (bits={:#010x})",
                    orig.to_bits(),
                    dec.to_bits()
                );
            }
        }
        _ => panic!("expected F32 data"),
    }
}

#[test]
fn round_trip_f32_lossless_varied_data() {
    let width = 64u32;
    let height = 64u32;
    let pixels: Vec<f32> = (0..width * height)
        .map(|i| {
            let x = (i % width) as f32 / width as f32;
            let y = (i / width) as f32 / height as f32;
            (x * std::f32::consts::PI).sin() * (y * std::f32::consts::E).cos() * 1000.0
        })
        .collect();

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: LercData::F32(pixels.clone()),
    };

    let encoded = lerc::encode(&image, 0.0).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    match &decoded.data {
        LercData::F32(dec_pixels) => {
            assert_eq!(dec_pixels.len(), pixels.len());
            for (i, (&orig, &dec)) in pixels.iter().zip(dec_pixels).enumerate() {
                assert_eq!(
                    orig.to_bits(),
                    dec.to_bits(),
                    "pixel {i}: orig={orig}, decoded={dec}"
                );
            }
        }
        _ => panic!("expected F32 data"),
    }
}

#[test]
fn round_trip_f64_lossless() {
    let width = 32u32;
    let height = 32u32;
    let pixels: Vec<f64> = (0..width * height)
        .map(|i| (i as f64) * 0.123456789 + 1000.0)
        .collect();

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Double,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: LercData::F64(pixels.clone()),
    };

    let encoded = lerc::encode(&image, 0.0).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    assert_eq!(decoded.data_type, DataType::Double);

    match &decoded.data {
        LercData::F64(dec_pixels) => {
            assert_eq!(dec_pixels.len(), pixels.len());
            for (i, (&orig, &dec)) in pixels.iter().zip(dec_pixels).enumerate() {
                assert_eq!(
                    orig.to_bits(),
                    dec.to_bits(),
                    "pixel {i}: orig={orig} (bits={:#018x}), decoded={dec} (bits={:#018x})",
                    orig.to_bits(),
                    dec.to_bits()
                );
            }
        }
        _ => panic!("expected F64 data"),
    }
}

#[test]
fn round_trip_f64_lossless_varied() {
    let width = 48u32;
    let height = 48u32;
    let pixels: Vec<f64> = (0..width * height)
        .map(|i| {
            let x = (i % width) as f64 / width as f64;
            let y = (i / width) as f64 / height as f64;
            (x * std::f64::consts::PI).sin() * (y * std::f64::consts::E).cos() * 1e6
        })
        .collect();

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Double,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: LercData::F64(pixels.clone()),
    };

    let encoded = lerc::encode(&image, 0.0).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    match &decoded.data {
        LercData::F64(dec_pixels) => {
            assert_eq!(dec_pixels.len(), pixels.len());
            for (i, (&orig, &dec)) in pixels.iter().zip(dec_pixels).enumerate() {
                assert_eq!(
                    orig.to_bits(),
                    dec.to_bits(),
                    "pixel {i}: orig={orig}, decoded={dec}"
                );
            }
        }
        _ => panic!("expected F64 data"),
    }
}

#[test]
fn round_trip_f32_lossless_constant() {
    // Special case: all same value
    let width = 16u32;
    let height = 16u32;
    let pixels: Vec<f32> = vec![42.5; (width * height) as usize];

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: LercData::F32(pixels.clone()),
    };

    let encoded = lerc::encode(&image, 0.0).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    match &decoded.data {
        LercData::F32(dec_pixels) => {
            assert_eq!(dec_pixels, &pixels);
        }
        _ => panic!("expected F32 data"),
    }
}

#[test]
fn round_trip_f32_lossless_multi_depth() {
    let width = 16u32;
    let height = 16u32;
    let n_depth = 3u32;
    let pixels: Vec<f32> = (0..width * height * n_depth)
        .map(|i| (i as f32) * 0.01 + 1.0)
        .collect();

    let image = LercImage {
        width,
        height,
        n_depth,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: LercData::F32(pixels.clone()),
    };

    let encoded = lerc::encode(&image, 0.0).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.n_depth, n_depth);
    match &decoded.data {
        LercData::F32(dec_pixels) => {
            assert_eq!(dec_pixels.len(), pixels.len());
            for (i, (&orig, &dec)) in pixels.iter().zip(dec_pixels).enumerate() {
                assert_eq!(
                    orig.to_bits(),
                    dec.to_bits(),
                    "pixel {i}: orig={orig}, decoded={dec}"
                );
            }
        }
        _ => panic!("expected F32 data"),
    }
}
