use lerc::Precision;
use lerc::bitmask::BitMask;
use lerc::{DataType, Image, SampleData};

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

    let image = Image {
        width,
        height,
        depth: n_depth,
        bands: 1,
        data_type: DataType::UShort,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: SampleData::U16(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Lossless).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.depth, n_depth);
    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);

    match &decoded.data {
        SampleData::U16(dec_pixels) => {
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

    let image = Image {
        width,
        height,
        depth: n_depth,
        bands: 1,
        data_type: DataType::Int,
        valid_masks: vec![BitMask::all_valid(num_pixels)],
        data: SampleData::I32(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Lossless).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.depth, n_depth);
    match &decoded.data {
        SampleData::I32(dec_pixels) => {
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

    let image = Image {
        width,
        height,
        depth: n_depth,
        bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![BitMask::all_valid(num_pixels)],
        data: SampleData::F32(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Tolerance(max_z_error)).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.depth, n_depth);
    match &decoded.data {
        SampleData::F32(dec_pixels) => {
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

    let image_correlated = Image {
        width,
        height,
        depth: n_depth,
        bands: 1,
        data_type: DataType::UShort,
        valid_masks: vec![BitMask::all_valid(num_pixels)],
        data: SampleData::U16(correlated.clone()),
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

    let image_uncorrelated = Image {
        width,
        height,
        depth: n_depth,
        bands: 1,
        data_type: DataType::UShort,
        valid_masks: vec![BitMask::all_valid(num_pixels)],
        data: SampleData::U16(uncorrelated.clone()),
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
        SampleData::U16(dec) => {
            assert_eq!(dec, &correlated, "correlated round-trip mismatch");
        }
        _ => panic!("expected U16"),
    }

    let decoded = lerc::decode(&encoded_uncorrelated).expect("decode uncorrelated");
    match &decoded.data {
        SampleData::U16(dec) => {
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

    let image = Image {
        width,
        height,
        depth: n_depth,
        bands: 1,
        data_type: DataType::Byte,
        valid_masks: vec![BitMask::all_valid(num_pixels)],
        data: SampleData::U8(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Lossless).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.depth, n_depth);
    match &decoded.data {
        SampleData::U8(dec_pixels) => {
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

    let image = Image {
        width,
        height,
        depth: n_depth,
        bands: 1,
        data_type: DataType::Short,
        valid_masks: vec![mask.clone()],
        data: SampleData::I16(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Lossless).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.depth, n_depth);
    let dec_mask = &decoded.valid_masks[0];

    match &decoded.data {
        SampleData::I16(dec_pixels) => {
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

    let image = Image {
        width,
        height,
        depth: n_depth,
        bands: 1,
        data_type: DataType::Int,
        valid_masks: vec![BitMask::all_valid(num_pixels)],
        data: SampleData::I32(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Lossless).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    match &decoded.data {
        SampleData::I32(dec_pixels) => {
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

    let image = Image {
        width,
        height,
        depth: n_depth,
        bands: 1,
        data_type: DataType::Double,
        valid_masks: vec![BitMask::all_valid(num_pixels)],
        data: SampleData::F64(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Lossless).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.depth, n_depth);
    match &decoded.data {
        SampleData::F64(dec_pixels) => {
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

    let image = Image {
        width,
        height,
        depth: n_depth,
        bands: 1,
        data_type: DataType::UInt,
        valid_masks: vec![BitMask::all_valid(num_pixels)],
        data: SampleData::U32(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Lossless).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.depth, n_depth);
    match &decoded.data {
        SampleData::U32(dec_pixels) => {
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
fn lossy_multi_depth_f32_correlated_round_trip() {
    // Verify that the encoder produces correct round-trip results for
    // multi-depth lossy f32 data with correlated depths.
    let width = 64u32;
    let height = 64u32;
    let n_depth = 3u32;
    let max_z_error = 0.01;
    let num_pixels = (width * height) as usize;

    let pixels = make_correlated_f32_lossy(width, height);

    let image = Image {
        width,
        height,
        depth: n_depth,
        bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![BitMask::all_valid(num_pixels)],
        data: SampleData::F32(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Tolerance(max_z_error)).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.depth, n_depth);
    match &decoded.data {
        SampleData::F32(dec_pixels) => {
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
    // Test lossy f64 multi-depth round-trip (slices coded independently)
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

    let image = Image {
        width,
        height,
        depth: n_depth,
        bands: 1,
        data_type: DataType::Double,
        valid_masks: vec![BitMask::all_valid(num_pixels)],
        data: SampleData::F64(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Tolerance(max_z_error)).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.depth, n_depth);
    match &decoded.data {
        SampleData::F64(dec_pixels) => {
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

    let image = Image {
        width,
        height,
        depth: n_depth,
        bands: 1,
        data_type: DataType::Int,
        valid_masks: vec![BitMask::all_valid(num_pixels)],
        data: SampleData::I32(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Tolerance(max_z_error)).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.depth, n_depth);
    match &decoded.data {
        SampleData::I32(dec_pixels) => {
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

    let image = Image {
        width,
        height,
        depth: n_depth,
        bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![mask.clone()],
        data: SampleData::F32(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Tolerance(max_z_error)).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.depth, n_depth);
    let dec_mask = &decoded.valid_masks[0];

    match &decoded.data {
        SampleData::F32(dec_pixels) => {
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

    let image = Image {
        width,
        height,
        depth: n_depth,
        bands: n_bands,
        data_type: DataType::Float,
        valid_masks: masks,
        data: SampleData::F32(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Tolerance(max_z_error)).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.depth, n_depth);
    assert_eq!(decoded.bands, n_bands);
    match &decoded.data {
        SampleData::F32(dec_pixels) => {
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
fn lossy_multi_depth_after_raw_fallback_round_trip() {
    // A quantized block whose bit-stuffed form is no smaller than raw is
    // stored raw, so the decoder reads its values back exactly. Lossy
    // slices are coded independently (diff coding is for lossless integer
    // data only), so a raw block in one slice must not disturb the next;
    // an encoder that diffed against its own reconstruction once got this
    // wrong by up to 2 × max_z_error.
    //
    // One valid pair per 16×16 region, so every non-empty block holds two
    // valid pixels whichever micro-block size the encoder picks. Spanning
    // ~2e9 with max_z_error = 2, they need 29 bits each: bit-stuffed (flag,
    // offset, bit-stuffer header, 8 bytes of bits) they cost more than the
    // raw i32 values plus the flag byte, so the block is stored raw.
    let width = 64u32;
    let height = 64u32;
    let n_depth = 2u32;
    let max_z_error = 2.0;
    let num_pixels = (width * height) as usize;

    let mut mask = BitMask::new(num_pixels);
    for k in 0..num_pixels {
        let (row, col) = (k / width as usize, k % width as usize);
        if row % 16 == 0 && col % 16 < 2 {
            mask.set_valid(k);
        }
    }

    let mut pixels = vec![0i32; num_pixels * n_depth as usize];
    for k in 0..num_pixels {
        if mask.is_valid(k) {
            // Alternate between the ends of the range so every block spans it.
            let base = if (k / 8 + k) % 2 == 0 {
                -1_000_000_000 + (k as i32 % 7)
            } else {
                1_000_000_000 - (k as i32 % 5)
            };
            pixels[k * 2] = base;
            pixels[k * 2 + 1] = base + (k as i32 % 4);
        }
    }

    let image = Image {
        width,
        height,
        depth: n_depth,
        bands: 1,
        data_type: DataType::Int,
        valid_masks: vec![mask.clone()],
        data: SampleData::I32(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Tolerance(max_z_error)).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    let SampleData::I32(dec_pixels) = &decoded.data else {
        panic!("expected I32 data");
    };
    for k in (0..num_pixels).filter(|&k| mask.is_valid(k)) {
        for m in 0..n_depth as usize {
            let idx = k * n_depth as usize + m;
            let diff = (i64::from(pixels[idx]) - i64::from(dec_pixels[idx])).abs();
            assert!(
                diff as f64 <= max_z_error,
                "pixel {k} depth {m}: orig={}, decoded={}, diff={diff} > {max_z_error}",
                pixels[idx],
                dec_pixels[idx]
            );
        }
    }
}

/// Lossy float depth slices are coded independently, as Esri's encoder codes
/// them (diff coding is for lossless integer data only).
///
/// This 8 × 8 block is two gradient components rounded to a 0.01 lattice.
/// When the encoder diff-coded lossy floats, the second slice's diffs had a
/// minimum of -0.99999997 (an f64 difference); its f32 rounding, -1, was
/// judged integral, the offset was stored as a 16-bit integer and written
/// truncated to 0, and the whole block decoded 1.0 too high. Esri's decoder
/// read the same wrong values, so the blob itself was wrong.
#[test]
fn lossy_float_depth_slices_decode_within_tolerance() {
    const DX: [[f32; 8]; 8] = [
        [0.17, 0.17, 0.22, 0.26, 0.25, 0.2, 0.16, 0.13],
        [0.23, 0.16, 0.2, 0.27, 0.25, 0.19, 0.12, 0.03],
        [0.25, 0.11, 0.14, 0.24, 0.25, 0.21, 0.11, -0.01],
        [0.21, 0.06, 0.09, 0.22, 0.29, 0.26, 0.13, -0.05],
        [0.11, 0.03, 0.06, 0.22, 0.36, 0.33, 0.13, -0.08],
        [-0.03, -0.04, 0.04, 0.21, 0.39, 0.35, 0.11, -0.12],
        [-0.2, -0.16, -0.01, 0.18, 0.33, 0.27, 0.03, -0.18],
        [-0.35, -0.27, -0.05, 0.14, 0.2, 0.08, -0.15, -0.33],
    ];
    const DY: [[f32; 8]; 8] = [
        [0.25, 0.3, 0.32, 0.35, 0.39, 0.39, 0.33, 0.24],
        [0.37, 0.37, 0.32, 0.28, 0.27, 0.27, 0.26, 0.21],
        [0.28, 0.25, 0.18, 0.12, 0.1, 0.13, 0.16, 0.15],
        [-0.01, -0.06, -0.09, -0.11, -0.08, -0.01, 0.03, 0.02],
        [-0.26, -0.34, -0.36, -0.37, -0.33, -0.26, -0.23, -0.26],
        [-0.3, -0.43, -0.5, -0.53, -0.55, -0.56, -0.59, -0.62],
        [-0.22, -0.37, -0.46, -0.51, -0.58, -0.71, -0.85, -0.96],
        [-0.18, -0.29, -0.35, -0.39, -0.49, -0.67, -0.91, -1.12],
    ];
    let pixels: Vec<f32> = (0..64)
        .flat_map(|i| [DX[i / 8][i % 8], DY[i / 8][i % 8]])
        .collect();
    let image = Image {
        width: 8,
        height: 8,
        depth: 2,
        bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![BitMask::all_valid(64)],
        data: SampleData::F32(pixels.clone()),
        no_data_value: None,
    };
    for max_z_error in [1e-6, 1e-3, 0.02] {
        let encoded = lerc::encode(&image, Precision::Tolerance(max_z_error)).expect("encode");
        let SampleData::F32(decoded) = lerc::decode(&encoded).expect("decode").data else {
            panic!("expected F32 data");
        };
        for (i, (&orig, &dec)) in pixels.iter().zip(&decoded).enumerate() {
            let diff = (f64::from(orig) - f64::from(dec)).abs();
            let tol = lossy_tol_f32(orig, dec, max_z_error);
            assert!(
                diff <= tol,
                "maxZErr {max_z_error}: value {i}: {orig} decoded as {dec}"
            );
        }
    }
}
