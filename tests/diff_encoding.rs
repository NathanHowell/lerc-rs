use lerc::Precision;
use lerc::bitmask::BitMask;
use lerc::{DataType, LercData, LercImage};

/// Helper: build an nDepth=3 image with correlated depth slices.
/// depth[0] = base, depth[1] = base + small_delta, depth[2] = base + 2 * small_delta
fn make_correlated_u16(width: u32, height: u32) -> (Vec<u16>, u32) {
    let n_depth = 3u32;
    let num_pixels = (width * height) as usize;
    let mut pixels = vec![0u16; num_pixels * n_depth as usize];
    for i in 0..num_pixels {
        let base = ((i * 37 + 100) % 60000) as u16;
        let d1 = ((i * 3) % 5) as u16;
        let d2 = ((i * 7) % 5) as u16;
        pixels[i * 3] = base;
        pixels[i * 3 + 1] = base.wrapping_add(d1);
        pixels[i * 3 + 2] = base.wrapping_add(d1).wrapping_add(d2);
    }
    (pixels, n_depth)
}

#[test]
fn round_trip_u16_ndepth3_lossless() {
    let width = 32u32;
    let height = 32u32;
    let (pixels, n_depth) = make_correlated_u16(width, height);

    let image = LercImage {
        width,
        height,
        n_depth,
        n_bands: 1,
        data_type: DataType::UShort,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: LercData::U16(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Lossless).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.n_depth, n_depth);
    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);

    match &decoded.data {
        LercData::U16(dec_pixels) => {
            assert_eq!(dec_pixels.len(), pixels.len());
            for (i, (&orig, &dec)) in pixels.iter().zip(dec_pixels).enumerate() {
                assert_eq!(
                    orig, dec,
                    "mismatch at index {i}: orig={orig}, decoded={dec}"
                );
            }
        }
        _ => panic!("expected U16 data"),
    }
}

#[test]
fn round_trip_i32_ndepth3_lossless() {
    let width = 16u32;
    let height = 16u32;
    let n_depth = 3u32;
    let num_pixels = (width * height) as usize;

    // Correlated i32 data: base values with small differences between depths
    let mut pixels = vec![0i32; num_pixels * n_depth as usize];
    for i in 0..num_pixels {
        let base = (i as i32) * 100 - 10000;
        pixels[i * 3] = base;
        pixels[i * 3 + 1] = base + (i as i32 % 10);
        pixels[i * 3 + 2] = base + (i as i32 % 10) + (i as i32 % 7);
    }

    let image = LercImage {
        width,
        height,
        n_depth,
        n_bands: 1,
        data_type: DataType::Int,
        valid_masks: vec![BitMask::all_valid(num_pixels)],
        data: LercData::I32(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Lossless).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.n_depth, n_depth);
    match &decoded.data {
        LercData::I32(dec_pixels) => {
            assert_eq!(
                dec_pixels, &pixels,
                "lossless i32 nDepth=3 round-trip mismatch"
            );
        }
        _ => panic!("expected I32 data"),
    }
}

#[test]
fn round_trip_f32_ndepth3_lossy() {
    let width = 32u32;
    let height = 32u32;
    let n_depth = 3u32;
    let max_z_error = 0.01;
    let num_pixels = (width * height) as usize;

    // Correlated float data
    let mut pixels = vec![0.0f32; num_pixels * n_depth as usize];
    for i in 0..num_pixels {
        let base = (i as f32) * 0.1 + 100.0;
        pixels[i * 3] = base;
        pixels[i * 3 + 1] = base + 0.5;
        pixels[i * 3 + 2] = base + 1.0;
    }

    let image = LercImage {
        width,
        height,
        n_depth,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![BitMask::all_valid(num_pixels)],
        data: LercData::F32(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::MaxError(max_z_error)).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.n_depth, n_depth);
    match &decoded.data {
        LercData::F32(dec_pixels) => {
            assert_eq!(dec_pixels.len(), pixels.len());
            for (i, (&orig, &dec)) in pixels.iter().zip(dec_pixels).enumerate() {
                let diff = (orig - dec).abs();
                assert!(
                    diff <= max_z_error as f32,
                    "pixel {i}: orig={orig}, decoded={dec}, diff={diff} > {max_z_error}"
                );
            }
        }
        _ => panic!("expected F32 data"),
    }
}

#[test]
fn diff_encoding_smaller_than_independent_for_correlated_data() {
    // Correlated data where depth slices differ only by small amounts
    // should compress better with diff encoding.
    let width = 64u32;
    let height = 64u32;
    let n_depth = 3u32;
    let num_pixels = (width * height) as usize;

    // Highly correlated: depths are nearly identical
    let mut correlated = vec![0u16; num_pixels * n_depth as usize];
    for i in 0..num_pixels {
        let base = ((i * 37 + 100) % 60000) as u16;
        correlated[i * 3] = base;
        correlated[i * 3 + 1] = base.wrapping_add(1);
        correlated[i * 3 + 2] = base.wrapping_add(2);
    }

    let image_correlated = LercImage {
        width,
        height,
        n_depth,
        n_bands: 1,
        data_type: DataType::UShort,
        valid_masks: vec![BitMask::all_valid(num_pixels)],
        data: LercData::U16(correlated.clone()),
        no_data_value: None,
    };

    let encoded_correlated =
        lerc::encode(&image_correlated, Precision::Lossless).expect("encode correlated");

    // Uncorrelated: each depth is independent random-ish data
    let mut uncorrelated = vec![0u16; num_pixels * n_depth as usize];
    for i in 0..num_pixels {
        uncorrelated[i * 3] = ((i * 37 + 100) % 60000) as u16;
        uncorrelated[i * 3 + 1] = ((i * 73 + 5000) % 60000) as u16;
        uncorrelated[i * 3 + 2] = ((i * 131 + 30000) % 60000) as u16;
    }

    let image_uncorrelated = LercImage {
        width,
        height,
        n_depth,
        n_bands: 1,
        data_type: DataType::UShort,
        valid_masks: vec![BitMask::all_valid(num_pixels)],
        data: LercData::U16(uncorrelated.clone()),
        no_data_value: None,
    };

    let encoded_uncorrelated =
        lerc::encode(&image_uncorrelated, Precision::Lossless).expect("encode uncorrelated");

    // The correlated data should be noticeably smaller due to diff encoding
    assert!(
        encoded_correlated.len() < encoded_uncorrelated.len(),
        "correlated encoded size ({}) should be smaller than uncorrelated ({})",
        encoded_correlated.len(),
        encoded_uncorrelated.len()
    );

    // Verify both round-trip correctly
    let decoded = lerc::decode(&encoded_correlated).expect("decode correlated");
    match &decoded.data {
        LercData::U16(dec) => {
            assert_eq!(dec, &correlated, "correlated round-trip mismatch");
        }
        _ => panic!("expected U16"),
    }

    let decoded = lerc::decode(&encoded_uncorrelated).expect("decode uncorrelated");
    match &decoded.data {
        LercData::U16(dec) => {
            assert_eq!(dec, &uncorrelated, "uncorrelated round-trip mismatch");
        }
        _ => panic!("expected U16"),
    }
}

#[test]
fn round_trip_u8_ndepth3_lossless() {
    // RGB-like data
    let width = 32u32;
    let height = 32u32;
    let n_depth = 3u32;
    let num_pixels = (width * height) as usize;

    let mut pixels = vec![0u8; num_pixels * n_depth as usize];
    for i in 0..num_pixels {
        pixels[i * 3] = (i % 256) as u8;
        pixels[i * 3 + 1] = ((i + 1) % 256) as u8;
        pixels[i * 3 + 2] = ((i + 2) % 256) as u8;
    }

    let image = LercImage {
        width,
        height,
        n_depth,
        n_bands: 1,
        data_type: DataType::Byte,
        valid_masks: vec![BitMask::all_valid(num_pixels)],
        data: LercData::U8(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Lossless).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.n_depth, n_depth);
    match &decoded.data {
        LercData::U8(dec_pixels) => {
            assert_eq!(
                dec_pixels, &pixels,
                "lossless u8 nDepth=3 round-trip mismatch"
            );
        }
        _ => panic!("expected U8 data"),
    }
}

#[test]
fn round_trip_ndepth3_with_mask() {
    let width = 16u32;
    let height = 16u32;
    let n_depth = 3u32;
    let num_pixels = (width * height) as usize;

    let mut mask = BitMask::new(num_pixels);
    // Set every other pixel valid
    for k in 0..num_pixels {
        if k % 2 == 0 {
            mask.set_valid(k);
        }
    }

    let mut pixels = vec![0i16; num_pixels * n_depth as usize];
    for i in 0..num_pixels {
        if mask.is_valid(i) {
            let base = (i as i16) * 10 - 500;
            pixels[i * 3] = base;
            pixels[i * 3 + 1] = base + 2;
            pixels[i * 3 + 2] = base + 4;
        }
    }

    let image = LercImage {
        width,
        height,
        n_depth,
        n_bands: 1,
        data_type: DataType::Short,
        valid_masks: vec![mask.clone()],
        data: LercData::I16(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Lossless).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.n_depth, n_depth);
    let dec_mask = &decoded.valid_masks[0];

    match &decoded.data {
        LercData::I16(dec_pixels) => {
            for k in 0..num_pixels {
                assert_eq!(
                    mask.is_valid(k),
                    dec_mask.is_valid(k),
                    "mask mismatch at pixel {k}"
                );
                if mask.is_valid(k) {
                    for m in 0..n_depth as usize {
                        let idx = k * n_depth as usize + m;
                        assert_eq!(
                            pixels[idx], dec_pixels[idx],
                            "mismatch at pixel {k}, depth {m}"
                        );
                    }
                }
            }
        }
        _ => panic!("expected I16 data"),
    }
}

#[test]
fn round_trip_identical_depths() {
    // All depths identical - diffs are all zero
    let width = 16u32;
    let height = 16u32;
    let n_depth = 3u32;
    let num_pixels = (width * height) as usize;

    let mut pixels = vec![0i32; num_pixels * n_depth as usize];
    for i in 0..num_pixels {
        let val = (i as i32) * 100;
        pixels[i * 3] = val;
        pixels[i * 3 + 1] = val;
        pixels[i * 3 + 2] = val;
    }

    let image = LercImage {
        width,
        height,
        n_depth,
        n_bands: 1,
        data_type: DataType::Int,
        valid_masks: vec![BitMask::all_valid(num_pixels)],
        data: LercData::I32(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Lossless).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    match &decoded.data {
        LercData::I32(dec_pixels) => {
            assert_eq!(dec_pixels, &pixels, "identical depths round-trip mismatch");
        }
        _ => panic!("expected I32 data"),
    }
}

#[test]
fn round_trip_f64_ndepth2_lossless() {
    let width = 16u32;
    let height = 16u32;
    let n_depth = 2u32;
    let num_pixels = (width * height) as usize;

    let mut pixels = vec![0.0f64; num_pixels * n_depth as usize];
    for i in 0..num_pixels {
        let base = (i as f64) * 1.23456789;
        pixels[i * 2] = base;
        pixels[i * 2 + 1] = base + 0.001;
    }

    let image = LercImage {
        width,
        height,
        n_depth,
        n_bands: 1,
        data_type: DataType::Double,
        valid_masks: vec![BitMask::all_valid(num_pixels)],
        data: LercData::F64(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Lossless).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.n_depth, n_depth);
    match &decoded.data {
        LercData::F64(dec_pixels) => {
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
fn round_trip_u32_ndepth4_lossless() {
    let width = 16u32;
    let height = 16u32;
    let n_depth = 4u32;
    let num_pixels = (width * height) as usize;

    let mut pixels = vec![0u32; num_pixels * n_depth as usize];
    for i in 0..num_pixels {
        let base = (i as u32) * 1000;
        pixels[i * 4] = base;
        pixels[i * 4 + 1] = base + 5;
        pixels[i * 4 + 2] = base + 10;
        pixels[i * 4 + 3] = base + 15;
    }

    let image = LercImage {
        width,
        height,
        n_depth,
        n_bands: 1,
        data_type: DataType::UInt,
        valid_masks: vec![BitMask::all_valid(num_pixels)],
        data: LercData::U32(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Lossless).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.n_depth, n_depth);
    match &decoded.data {
        LercData::U32(dec_pixels) => {
            assert_eq!(dec_pixels, &pixels, "u32 nDepth=4 round-trip mismatch");
        }
        _ => panic!("expected U32 data"),
    }
}

/// Tolerance for lossy float assertions. LERC quantizes with
/// `floor(val / (2*maxZErr) + 0.5)` in float arithmetic, so the
/// difference can slightly exceed `max_z_err` by a value-dependent
/// ULP margin.
fn lossy_tol_f32(orig: f32, dec: f32, max_z_err: f64) -> f64 {
    max_z_err + (orig.abs().max(dec.abs()) as f64) * 4.0 * (f32::EPSILON as f64)
}

fn lossy_tol_f64(orig: f64, dec: f64, max_z_err: f64) -> f64 {
    max_z_err + orig.abs().max(dec.abs()) * 4.0 * f64::EPSILON
}

/// Create multi-depth f32 data with correlated depths:
/// depth[1] = depth[0] * 1.01 + small_noise.
/// This is the pattern described in the task's testing section.
fn make_correlated_f32_lossy(width: u32, height: u32) -> Vec<f32> {
    let n_depth = 3u32;
    let num_pixels = (width * height) as usize;
    let mut pixels = vec![0.0f32; num_pixels * n_depth as usize];
    for i in 0..num_pixels {
        let base = 100.0 + (i as f32) * 0.37 + ((i as f32) * 0.1).sin() * 10.0;
        let noise1 = ((i * 7 + 3) % 13) as f32 * 0.001;
        let noise2 = ((i * 11 + 5) % 17) as f32 * 0.001;
        pixels[i * 3] = base;
        pixels[i * 3 + 1] = base * 1.01 + noise1;
        pixels[i * 3 + 2] = base * 1.02 + noise2;
    }
    pixels
}

#[test]
fn reconstructed_prev_depth_produces_smaller_blobs() {
    // Verify that the encoder produces correct round-trip results for
    // multi-depth lossy f32 data with correlated depths.
    let width = 64u32;
    let height = 64u32;
    let n_depth = 3u32;
    let max_z_error = 0.01;
    let num_pixels = (width * height) as usize;

    let pixels = make_correlated_f32_lossy(width, height);

    let image = LercImage {
        width,
        height,
        n_depth,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![BitMask::all_valid(num_pixels)],
        data: LercData::F32(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::MaxError(max_z_error)).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.n_depth, n_depth);
    match &decoded.data {
        LercData::F32(dec_pixels) => {
            assert_eq!(dec_pixels.len(), pixels.len());
            for (i, (&orig, &dec)) in pixels.iter().zip(dec_pixels).enumerate() {
                let diff = (orig as f64 - dec as f64).abs();
                let tol = lossy_tol_f32(orig, dec, max_z_error);
                assert!(
                    diff <= tol,
                    "pixel {i}: orig={orig}, decoded={dec}, diff={diff} > {tol} (maxZErr={max_z_error})"
                );
            }
        }
        _ => panic!("expected F32 data"),
    }
}

#[test]
fn lossy_multi_depth_f64_round_trip() {
    // Test lossy f64 multi-depth round-trip with reconstruction buffer
    let width = 32u32;
    let height = 32u32;
    let n_depth = 2u32;
    let max_z_error = 0.1;
    let num_pixels = (width * height) as usize;

    let mut pixels = vec![0.0f64; num_pixels * n_depth as usize];
    for i in 0..num_pixels {
        let base = (i as f64) * 1.5 + 50.0;
        pixels[i * 2] = base;
        pixels[i * 2 + 1] = base + 0.3;
    }

    let image = LercImage {
        width,
        height,
        n_depth,
        n_bands: 1,
        data_type: DataType::Double,
        valid_masks: vec![BitMask::all_valid(num_pixels)],
        data: LercData::F64(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::MaxError(max_z_error)).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.n_depth, n_depth);
    match &decoded.data {
        LercData::F64(dec_pixels) => {
            assert_eq!(dec_pixels.len(), pixels.len());
            for (i, (&orig, &dec)) in pixels.iter().zip(dec_pixels).enumerate() {
                let diff = (orig - dec).abs();
                let tol = lossy_tol_f64(orig, dec, max_z_error);
                assert!(
                    diff <= tol,
                    "pixel {i}: orig={orig}, decoded={dec}, diff={diff} > {tol} (maxZErr={max_z_error})"
                );
            }
        }
        _ => panic!("expected F64 data"),
    }
}

#[test]
fn lossy_multi_depth_i32_round_trip() {
    // Test lossy integer multi-depth with max_z_error > 0.5
    let width = 32u32;
    let height = 32u32;
    let n_depth = 3u32;
    let max_z_error = 5.0; // lossy for integers
    let num_pixels = (width * height) as usize;

    let mut pixels = vec![0i32; num_pixels * n_depth as usize];
    for i in 0..num_pixels {
        let base = (i as i32) * 100;
        pixels[i * 3] = base;
        pixels[i * 3 + 1] = base + 3;
        pixels[i * 3 + 2] = base + 6;
    }

    let image = LercImage {
        width,
        height,
        n_depth,
        n_bands: 1,
        data_type: DataType::Int,
        valid_masks: vec![BitMask::all_valid(num_pixels)],
        data: LercData::I32(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::MaxError(max_z_error)).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.n_depth, n_depth);
    match &decoded.data {
        LercData::I32(dec_pixels) => {
            assert_eq!(dec_pixels.len(), pixels.len());
            for (i, (&orig, &dec)) in pixels.iter().zip(dec_pixels).enumerate() {
                let diff = (orig - dec).abs();
                assert!(
                    diff as f64 <= max_z_error,
                    "pixel {i}: orig={orig}, decoded={dec}, diff={diff} > {max_z_error}"
                );
            }
        }
        _ => panic!("expected I32 data"),
    }
}

#[test]
fn lossy_multi_depth_with_mask_round_trip() {
    // Lossy multi-depth with partial validity mask
    let width = 32u32;
    let height = 32u32;
    let n_depth = 3u32;
    let max_z_error = 0.05;
    let num_pixels = (width * height) as usize;

    let mut mask = BitMask::new(num_pixels);
    for k in 0..num_pixels {
        if k % 3 != 2 {
            mask.set_valid(k);
        }
    }

    let mut pixels = vec![0.0f32; num_pixels * n_depth as usize];
    for i in 0..num_pixels {
        if mask.is_valid(i) {
            let base = 50.0 + (i as f32) * 0.25;
            pixels[i * 3] = base;
            pixels[i * 3 + 1] = base * 1.005;
            pixels[i * 3 + 2] = base * 1.01;
        }
    }

    let image = LercImage {
        width,
        height,
        n_depth,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![mask.clone()],
        data: LercData::F32(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::MaxError(max_z_error)).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.n_depth, n_depth);
    let dec_mask = &decoded.valid_masks[0];

    match &decoded.data {
        LercData::F32(dec_pixels) => {
            for k in 0..num_pixels {
                assert_eq!(
                    mask.is_valid(k),
                    dec_mask.is_valid(k),
                    "mask mismatch at pixel {k}"
                );
                if mask.is_valid(k) {
                    for m in 0..n_depth as usize {
                        let idx = k * n_depth as usize + m;
                        let diff = (pixels[idx] as f64 - dec_pixels[idx] as f64).abs();
                        let tol = lossy_tol_f32(pixels[idx], dec_pixels[idx], max_z_error);
                        assert!(
                            diff <= tol,
                            "pixel {k} depth {m}: orig={}, decoded={}, diff={diff} > {tol} (maxZErr={max_z_error})",
                            pixels[idx],
                            dec_pixels[idx]
                        );
                    }
                }
            }
        }
        _ => panic!("expected F32 data"),
    }
}

#[test]
fn lossy_multi_depth_multiband_round_trip() {
    // Test lossy multi-depth with multiple bands
    let width = 16u32;
    let height = 16u32;
    let n_depth = 2u32;
    let n_bands = 2u32;
    let max_z_error = 0.01;
    let num_pixels = (width * height) as usize;
    let band_size = num_pixels * n_depth as usize;

    let mut pixels = vec![0.0f32; band_size * n_bands as usize];
    for band in 0..n_bands as usize {
        for i in 0..num_pixels {
            let base = 10.0 + (i as f32) * 0.5 + (band as f32) * 100.0;
            pixels[band * band_size + i * 2] = base;
            pixels[band * band_size + i * 2 + 1] = base + 0.2;
        }
    }

    let masks = vec![BitMask::all_valid(num_pixels); n_bands as usize];

    let image = LercImage {
        width,
        height,
        n_depth,
        n_bands,
        data_type: DataType::Float,
        valid_masks: masks,
        data: LercData::F32(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::MaxError(max_z_error)).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.n_depth, n_depth);
    assert_eq!(decoded.n_bands, n_bands);
    match &decoded.data {
        LercData::F32(dec_pixels) => {
            assert_eq!(dec_pixels.len(), pixels.len());
            for (i, (&orig, &dec)) in pixels.iter().zip(dec_pixels).enumerate() {
                let diff = (orig as f64 - dec as f64).abs();
                let tol = lossy_tol_f32(orig, dec, max_z_error);
                assert!(
                    diff <= tol,
                    "pixel {i}: orig={orig}, decoded={dec}, diff={diff} > {tol} (maxZErr={max_z_error})"
                );
            }
        }
        _ => panic!("expected F32 data"),
    }
}
