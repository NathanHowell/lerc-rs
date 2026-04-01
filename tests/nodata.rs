use lerc::bitmask::BitMask;
use lerc::{DataType, LercData, LercImage};

/// Helper: create a multi-depth f32 image with some NoData pixels.
/// Returns (pixels, mask, expected_no_data_value).
fn make_f32_ndepth3_with_nodata() -> (Vec<f32>, BitMask, f64) {
    let width = 8u32;
    let height = 8u32;
    let n_depth = 3u32;
    let no_data: f32 = -9999.0;

    let num_pixels = (width * height) as usize;
    let mask = BitMask::all_valid(num_pixels);

    let mut pixels = vec![0.0f32; num_pixels * n_depth as usize];
    for i in 0..height as usize {
        for j in 0..width as usize {
            let k = i * width as usize + j;
            let base = k * n_depth as usize;
            pixels[base] = (i * 10 + j) as f32;
            pixels[base + 1] = (i * 10 + j + 100) as f32;
            pixels[base + 2] = (i * 10 + j + 200) as f32;

            // Mark some pixels as partially NoData (mixed valid/invalid per depth)
            if k % 5 == 0 && k > 0 {
                // depth 2 is NoData, but depths 0 and 1 are valid
                pixels[base + 2] = no_data;
            }
        }
    }

    (pixels, mask, no_data as f64)
}

#[test]
fn round_trip_f32_ndepth3_nodata() {
    let (pixels, mask, no_data_val) = make_f32_ndepth3_with_nodata();
    let width = 8u32;
    let height = 8u32;
    let n_depth = 3u32;

    let image = LercImage {
        width,
        height,
        n_depth,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![mask],
        data: LercData::F32(pixels.clone()),
        no_data_value: Some(no_data_val),
    };

    // Encode with lossy compression
    let encoded = lerc::encode(&image, 0.01).expect("encode failed");

    // Decode
    let decoded = lerc::decode(&encoded).expect("decode failed");

    // Check that no_data_value is preserved
    assert_eq!(
        decoded.no_data_value,
        Some(no_data_val),
        "no_data_value should be preserved through round-trip"
    );

    // Check that NoData values are correctly remapped back
    match &decoded.data {
        LercData::F32(dec_pixels) => {
            for i in 0..height as usize {
                for j in 0..width as usize {
                    let k = i * width as usize + j;
                    let base = k * n_depth as usize;

                    if k % 5 == 0 && k > 0 {
                        // This pixel had NoData at depth 2
                        assert_eq!(
                            dec_pixels[base + 2], no_data_val as f32,
                            "pixel ({}, {}) depth 2 should be NoData ({}), got {}",
                            i, j, no_data_val, dec_pixels[base + 2]
                        );
                    }

                    // Valid depths should be approximately correct (lossy)
                    let expected0 = pixels[base];
                    let expected1 = pixels[base + 1];
                    if expected0 != no_data_val as f32 {
                        assert!(
                            (dec_pixels[base] - expected0).abs() <= 0.01,
                            "pixel ({}, {}) depth 0: expected ~{}, got {}",
                            i, j, expected0, dec_pixels[base]
                        );
                    }
                    if expected1 != no_data_val as f32 {
                        assert!(
                            (dec_pixels[base + 1] - expected1).abs() <= 0.01,
                            "pixel ({}, {}) depth 1: expected ~{}, got {}",
                            i, j, expected1, dec_pixels[base + 1]
                        );
                    }
                }
            }
        }
        _ => panic!("expected F32 data"),
    }
}

#[test]
fn round_trip_f32_ndepth3_nodata_lossless() {
    let (pixels, mask, no_data_val) = make_f32_ndepth3_with_nodata();
    let width = 8u32;
    let height = 8u32;
    let n_depth = 3u32;

    let image = LercImage {
        width,
        height,
        n_depth,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![mask],
        data: LercData::F32(pixels.clone()),
        no_data_value: Some(no_data_val),
    };

    // Encode lossless
    let encoded = lerc::encode(&image, 0.0).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.no_data_value, Some(no_data_val));

    match &decoded.data {
        LercData::F32(dec_pixels) => {
            // Lossless: all valid values should be exact
            for i in 0..height as usize {
                for j in 0..width as usize {
                    let k = i * width as usize + j;
                    let base = k * n_depth as usize;
                    for m in 0..n_depth as usize {
                        let orig = pixels[base + m];
                        let dec = dec_pixels[base + m];
                        if orig == no_data_val as f32 {
                            assert_eq!(
                                dec, orig,
                                "NoData at ({},{},{}) should be exact",
                                i, j, m
                            );
                        } else {
                            assert_eq!(
                                dec, orig,
                                "lossless value at ({},{},{}) should be exact",
                                i, j, m
                            );
                        }
                    }
                }
            }
        }
        _ => panic!("expected F32 data"),
    }
}

#[test]
fn round_trip_f64_ndepth2_nodata() {
    let width = 4u32;
    let height = 4u32;
    let n_depth = 2u32;
    let no_data = -1.0e10;

    let num_pixels = (width * height) as usize;
    let mask = BitMask::all_valid(num_pixels);

    let mut pixels = vec![0.0f64; num_pixels * n_depth as usize];
    for k in 0..num_pixels {
        let base = k * n_depth as usize;
        pixels[base] = (k as f64) * 1.5;
        pixels[base + 1] = if k % 3 == 0 { no_data } else { (k as f64) * 2.5 };
    }

    let image = LercImage {
        width,
        height,
        n_depth,
        n_bands: 1,
        data_type: DataType::Double,
        valid_masks: vec![mask],
        data: LercData::F64(pixels.clone()),
        no_data_value: Some(no_data),
    };

    let encoded = lerc::encode(&image, 0.0).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.no_data_value, Some(no_data));

    match &decoded.data {
        LercData::F64(dec_pixels) => {
            for k in 0..num_pixels {
                let base = k * n_depth as usize;
                assert_eq!(dec_pixels[base], pixels[base], "depth 0 mismatch at pixel {}", k);
                assert_eq!(dec_pixels[base + 1], pixels[base + 1], "depth 1 mismatch at pixel {}", k);
            }
        }
        _ => panic!("expected F64 data"),
    }
}

#[test]
fn round_trip_i32_ndepth2_nodata() {
    let width = 8u32;
    let height = 8u32;
    let n_depth = 2u32;
    let no_data = -9999i32;

    let num_pixels = (width * height) as usize;
    let mask = BitMask::all_valid(num_pixels);

    let mut pixels = vec![0i32; num_pixels * n_depth as usize];
    for k in 0..num_pixels {
        let base = k * n_depth as usize;
        pixels[base] = k as i32 * 10;
        pixels[base + 1] = if k % 4 == 0 { no_data } else { k as i32 * 20 };
    }

    let image = LercImage {
        width,
        height,
        n_depth,
        n_bands: 1,
        data_type: DataType::Int,
        valid_masks: vec![mask],
        data: LercData::I32(pixels.clone()),
        no_data_value: Some(no_data as f64),
    };

    let encoded = lerc::encode(&image, 0.5).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.no_data_value, Some(no_data as f64));

    match &decoded.data {
        LercData::I32(dec_pixels) => {
            for k in 0..num_pixels {
                let base = k * n_depth as usize;
                assert_eq!(dec_pixels[base], pixels[base], "depth 0 mismatch at pixel {}", k);
                assert_eq!(dec_pixels[base + 1], pixels[base + 1], "depth 1 mismatch at pixel {}", k);
            }
        }
        _ => panic!("expected I32 data"),
    }
}

#[test]
fn decode_info_reports_nodata() {
    let (pixels, mask, no_data_val) = make_f32_ndepth3_with_nodata();
    let width = 8u32;
    let height = 8u32;
    let n_depth = 3u32;

    let image = LercImage {
        width,
        height,
        n_depth,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![mask],
        data: LercData::F32(pixels),
        no_data_value: Some(no_data_val),
    };

    let encoded = lerc::encode(&image, 0.01).expect("encode failed");
    let info = lerc::decode_info(&encoded).expect("decode_info failed");

    assert_eq!(
        info.no_data_value,
        Some(no_data_val),
        "LercInfo.no_data_value should report the original NoData value"
    );
}

#[test]
fn no_nodata_when_not_set() {
    let width = 4u32;
    let height = 4u32;
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

    let encoded = lerc::encode(&image, 0.0).expect("encode failed");
    let info = lerc::decode_info(&encoded).expect("decode_info failed");
    assert_eq!(info.no_data_value, None, "no_data_value should be None when not set");

    let decoded = lerc::decode(&encoded).expect("decode failed");
    assert_eq!(decoded.no_data_value, None, "decoded no_data_value should be None");
}

#[test]
fn nodata_ndepth1_not_encoded() {
    // For nDepth == 1, NoData values are handled by the bitmask, not by the nodata mechanism.
    // The encoder should not set pass_no_data_values for nDepth == 1.
    let width = 4u32;
    let height = 4u32;
    let pixels: Vec<f32> = (0..width * height).map(|i| i as f32).collect();

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: LercData::F32(pixels),
        no_data_value: Some(-9999.0),
    };

    let encoded = lerc::encode(&image, 0.0).expect("encode failed");
    let info = lerc::decode_info(&encoded).expect("decode_info failed");

    // For nDepth == 1, noData should NOT be encoded in the blob
    assert_eq!(
        info.no_data_value, None,
        "nDepth==1 should not use noData encoding"
    );
}

#[test]
fn round_trip_multiband_nodata() {
    let width = 4u32;
    let height = 4u32;
    let n_depth = 2u32;
    let n_bands = 2u32;
    let no_data = -9999.0f32;

    let num_pixels = (width * height) as usize;
    let band_size = num_pixels * n_depth as usize;
    let mask = BitMask::all_valid(num_pixels);

    let mut pixels = vec![0.0f32; band_size * n_bands as usize];
    // Band 0
    for k in 0..num_pixels {
        let base = k * n_depth as usize;
        pixels[base] = k as f32;
        pixels[base + 1] = if k % 3 == 0 { no_data } else { (k as f32) + 100.0 };
    }
    // Band 1
    for k in 0..num_pixels {
        let base = band_size + k * n_depth as usize;
        pixels[base] = (k as f32) + 200.0;
        pixels[base + 1] = if k % 4 == 0 { no_data } else { (k as f32) + 300.0 };
    }

    let image = LercImage {
        width,
        height,
        n_depth,
        n_bands,
        data_type: DataType::Float,
        valid_masks: vec![mask.clone(), mask],
        data: LercData::F32(pixels.clone()),
        no_data_value: Some(no_data as f64),
    };

    let encoded = lerc::encode(&image, 0.0).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.no_data_value, Some(no_data as f64));

    match &decoded.data {
        LercData::F32(dec_pixels) => {
            for k in 0..band_size * n_bands as usize {
                assert_eq!(
                    dec_pixels[k], pixels[k],
                    "mismatch at index {} (expected {}, got {})",
                    k, pixels[k], dec_pixels[k]
                );
            }
        }
        _ => panic!("expected F32 data"),
    }
}

#[test]
fn unsigned_u16_nodata_round_trip() {
    // Regression test: unsigned types must be able to use 0 as sentinel
    // when data minimum is > 0.
    let width = 8u32;
    let height = 8u32;
    let n_depth = 2u32;
    let no_data: u16 = 0;
    let mut pixels = Vec::with_capacity((width * height * n_depth) as usize);
    let mask = BitMask::all_valid((width * height) as usize);

    for i in 0..height {
        for j in 0..width {
            let base = (i * width + j) as u16 + 10; // min valid value is 10
            pixels.push(base);
            // depth 1: some pixels are NoData
            if (i + j) % 3 == 0 {
                pixels.push(no_data);
            } else {
                pixels.push(base + 5);
            }
        }
    }

    let image = LercImage {
        width,
        height,
        n_depth,
        n_bands: 1,
        data_type: DataType::UShort,
        valid_masks: vec![mask],
        data: LercData::U16(pixels.clone()),
        no_data_value: Some(no_data as f64),
    };

    let encoded = lerc::encode(&image, 0.5).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.no_data_value, Some(no_data as f64));
    match &decoded.data {
        LercData::U16(dec) => {
            assert_eq!(dec.len(), pixels.len());
            for (i, (&orig, &dec_val)) in pixels.iter().zip(dec.iter()).enumerate() {
                assert_eq!(orig, dec_val, "mismatch at pixel {i}");
            }
        }
        _ => panic!("expected U16 data"),
    }
}

#[test]
fn unsigned_u8_nodata_round_trip() {
    // Regression test: u8 with min valid value > 0 should find sentinel at 0.
    let width = 8u32;
    let height = 8u32;
    let n_depth = 2u32;
    let no_data: u8 = 0;
    let mut pixels = Vec::with_capacity((width * height * n_depth) as usize);
    let mask = BitMask::all_valid((width * height) as usize);

    for i in 0..height {
        for j in 0..width {
            let base = ((i * width + j) % 200) as u8 + 5; // min valid = 5
            pixels.push(base);
            if (i + j) % 4 == 0 {
                pixels.push(no_data);
            } else {
                pixels.push(base.wrapping_add(1));
            }
        }
    }

    let image = LercImage {
        width,
        height,
        n_depth,
        n_bands: 1,
        data_type: DataType::Byte,
        valid_masks: vec![mask],
        data: LercData::U8(pixels.clone()),
        no_data_value: Some(no_data as f64),
    };

    let encoded = lerc::encode(&image, 0.5).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");
    assert_eq!(decoded.no_data_value, Some(no_data as f64));
}
