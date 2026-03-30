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
    };

    let encoded = lerc::encode(&image, 0.5).expect("encode failed");
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
    };

    let encoded = lerc::encode(&image, 0.5).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.n_depth, n_depth);
    match &decoded.data {
        LercData::I32(dec_pixels) => {
            assert_eq!(dec_pixels, &pixels, "lossless i32 nDepth=3 round-trip mismatch");
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
    };

    let encoded = lerc::encode(&image, max_z_error).expect("encode failed");
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
    };

    let encoded_correlated = lerc::encode(&image_correlated, 0.5).expect("encode correlated");

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
    };

    let encoded_uncorrelated = lerc::encode(&image_uncorrelated, 0.5).expect("encode uncorrelated");

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
    };

    let encoded = lerc::encode(&image, 0.5).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.n_depth, n_depth);
    match &decoded.data {
        LercData::U8(dec_pixels) => {
            assert_eq!(dec_pixels, &pixels, "lossless u8 nDepth=3 round-trip mismatch");
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
    };

    let encoded = lerc::encode(&image, 0.5).expect("encode failed");
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
    };

    let encoded = lerc::encode(&image, 0.5).expect("encode failed");
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
    };

    let encoded = lerc::encode(&image, 0.0).expect("encode failed");
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
    };

    let encoded = lerc::encode(&image, 0.5).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.n_depth, n_depth);
    match &decoded.data {
        LercData::U32(dec_pixels) => {
            assert_eq!(dec_pixels, &pixels, "u32 nDepth=4 round-trip mismatch");
        }
        _ => panic!("expected U32 data"),
    }
}
