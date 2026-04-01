use lerc::Precision;
use lerc::bitmask::BitMask;
use lerc::{DataType, LercImage, SampleData};

/// Build a u16 image where the lower `noisy_bits` are random noise and
/// the upper bits form a smooth gradient. With enough pixels this should
/// trigger bit-plane compression when max_z_error is negative.
fn make_noisy_u16(width: u32, height: u32, noisy_bits: u32) -> Vec<u16> {
    let mask = (1u16 << noisy_bits) - 1; // e.g. noisy_bits=3 => mask=0x7
    let num_pixels = (width * height) as usize;
    let mut pixels = Vec::with_capacity(num_pixels);

    // Simple LCG for deterministic pseudo-random noise
    let mut rng: u32 = 12345;
    for i in 0..num_pixels {
        // Smooth upper bits: gradient across the image
        let base = ((i * 100 / num_pixels) as u16) << noisy_bits;
        // Random noise in low bits
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let noise = ((rng >> 16) as u16) & mask;
        pixels.push(base | noise);
    }
    pixels
}

/// Build a u32 image with noisy low bits.
fn make_noisy_u32(width: u32, height: u32, noisy_bits: u32) -> Vec<u32> {
    let mask = (1u32 << noisy_bits) - 1;
    let num_pixels = (width * height) as usize;
    let mut pixels = Vec::with_capacity(num_pixels);

    let mut rng: u32 = 67890;
    for i in 0..num_pixels {
        let base = ((i * 1000 / num_pixels) as u32) << noisy_bits;
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let noise = (rng >> 16) & mask;
        pixels.push(base | noise);
    }
    pixels
}

/// Build a i16 image with noisy low bits.
fn make_noisy_i16(width: u32, height: u32, noisy_bits: u32) -> Vec<i16> {
    let mask = (1u16 << noisy_bits) - 1;
    let num_pixels = (width * height) as usize;
    let mut pixels = Vec::with_capacity(num_pixels);

    let mut rng: u32 = 54321;
    for i in 0..num_pixels {
        // Signed gradient: range from -500 to 500 in upper bits
        let base_val = (i as i32 * 1000 / num_pixels as i32) - 500;
        let base = (base_val as i16) << noisy_bits;
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let noise = ((rng >> 16) as i16) & (mask as i16);
        pixels.push(base.wrapping_add(noise));
    }
    pixels
}

/// Read maxZError from the LERC2 header (version-aware).
fn read_max_z_error(encoded: &[u8]) -> f64 {
    let version = i32::from_le_bytes(encoded[6..10].try_into().unwrap());
    // Compute offset to maxZError based on version:
    // magic(6) + version(4) + checksum(4) + nRows(4) + nCols(4) = 22
    let mut offset: usize = 22;
    if version >= 4 {
        offset += 4; // nDepth
    }
    offset += 4 * 4; // numValidPixel + microBlockSize + blobSize + dataType
    if version >= 6 {
        offset += 4; // nBlobsMore
        offset += 4; // 4 flag bytes (passNoData, isInt, reserved, reserved)
    }
    // Now at maxZError (f64)
    f64::from_le_bytes(encoded[offset..offset + 8].try_into().unwrap())
}

#[test]
fn bit_plane_compression_u16_noisy_low_bits() {
    // Create a 128x128 u16 image with 3 noisy low bits
    let width = 128u32;
    let height = 128u32;
    let noisy_bits = 3u32;
    let pixels = make_noisy_u16(width, height, noisy_bits);

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::UShort,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: SampleData::U16(pixels.clone()),
        no_data_value: None,
    };

    // Negative max_z_error triggers bit-plane compression
    let encoded =
        lerc::encode(&image, Precision::Tolerance(-0.01)).expect("encode with bit-plane failed");

    // The maxZError in the header should be > 0.5 (some bit planes were dropped)
    let header_mze = read_max_z_error(&encoded);
    assert!(
        header_mze > 0.5,
        "expected maxZError > 0.5 from bit-plane compression, got {header_mze}"
    );

    // Compare with lossless encoding
    let lossless_encoded =
        lerc::encode(&image, Precision::Lossless).expect("lossless encode failed");

    // Bit-plane compressed should be smaller
    assert!(
        encoded.len() < lossless_encoded.len(),
        "bit-plane compressed ({} bytes) should be smaller than lossless ({} bytes)",
        encoded.len(),
        lossless_encoded.len()
    );

    // Verify round-trip: decoded values should be within maxZError of originals
    let decoded = lerc::decode(&encoded).expect("decode failed");
    match &decoded.data {
        SampleData::U16(dec_pixels) => {
            assert_eq!(dec_pixels.len(), pixels.len());
            for (i, (&orig, &dec)) in pixels.iter().zip(dec_pixels).enumerate() {
                let diff = (orig as i32 - dec as i32).unsigned_abs();
                assert!(
                    diff as f64 <= header_mze,
                    "pixel {i}: orig={orig}, decoded={dec}, diff={diff} > maxZError={header_mze}"
                );
            }
        }
        _ => panic!("expected U16 data"),
    }
}

#[test]
fn bit_plane_compression_u32_noisy_low_bits() {
    let width = 128u32;
    let height = 128u32;
    let noisy_bits = 4u32;
    let pixels = make_noisy_u32(width, height, noisy_bits);

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::UInt,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: SampleData::U32(pixels.clone()),
        no_data_value: None,
    };

    let encoded =
        lerc::encode(&image, Precision::Tolerance(-0.01)).expect("encode with bit-plane failed");
    let header_mze = read_max_z_error(&encoded);
    assert!(
        header_mze > 0.5,
        "expected maxZError > 0.5 from bit-plane compression, got {header_mze}"
    );

    // Verify round-trip
    let decoded = lerc::decode(&encoded).expect("decode failed");
    match &decoded.data {
        SampleData::U32(dec_pixels) => {
            assert_eq!(dec_pixels.len(), pixels.len());
            for (i, (&orig, &dec)) in pixels.iter().zip(dec_pixels).enumerate() {
                let diff = orig.abs_diff(dec);
                assert!(
                    diff as f64 <= header_mze,
                    "pixel {i}: orig={orig}, decoded={dec}, diff={diff} > maxZError={header_mze}"
                );
            }
        }
        _ => panic!("expected U32 data"),
    }
}

#[test]
fn bit_plane_compression_i16_noisy_low_bits() {
    let width = 128u32;
    let height = 128u32;
    let noisy_bits = 3u32;
    let pixels = make_noisy_i16(width, height, noisy_bits);

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Short,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: SampleData::I16(pixels.clone()),
        no_data_value: None,
    };

    let encoded =
        lerc::encode(&image, Precision::Tolerance(-0.01)).expect("encode with bit-plane failed");
    let header_mze = read_max_z_error(&encoded);
    assert!(
        header_mze > 0.5,
        "expected maxZError > 0.5 from bit-plane compression, got {header_mze}"
    );

    let decoded = lerc::decode(&encoded).expect("decode failed");
    match &decoded.data {
        SampleData::I16(dec_pixels) => {
            assert_eq!(dec_pixels.len(), pixels.len());
            for (i, (&orig, &dec)) in pixels.iter().zip(dec_pixels).enumerate() {
                let diff = (orig as i32 - dec as i32).unsigned_abs();
                assert!(
                    diff as f64 <= header_mze,
                    "pixel {i}: orig={orig}, decoded={dec}, diff={diff} > maxZError={header_mze}"
                );
            }
        }
        _ => panic!("expected I16 data"),
    }
}

#[test]
fn positive_max_z_error_does_not_trigger_bit_plane() {
    // Positive maxZError should NOT trigger bit-plane compression
    let width = 128u32;
    let height = 128u32;
    let pixels = make_noisy_u16(width, height, 3);

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::UShort,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: SampleData::U16(pixels.clone()),
        no_data_value: None,
    };

    // With positive maxZError = 0.5 (lossless for integers), bit-plane should not activate
    let encoded = lerc::encode(&image, Precision::Lossless).expect("encode lossless");
    let header_mze = read_max_z_error(&encoded);
    assert!(
        (header_mze - 0.5).abs() < 1e-10,
        "lossless integer should have maxZError=0.5, got {header_mze}"
    );

    // Verify exact round-trip
    let decoded = lerc::decode(&encoded).expect("decode failed");
    match &decoded.data {
        SampleData::U16(dec_pixels) => {
            assert_eq!(dec_pixels, &pixels, "lossless u16 round-trip mismatch");
        }
        _ => panic!("expected U16 data"),
    }
}

#[test]
fn magic_value_777_triggers_bit_plane() {
    // Magic value 777 should be treated as -0.01 epsilon
    let width = 128u32;
    let height = 128u32;
    let pixels = make_noisy_u16(width, height, 3);

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::UShort,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: SampleData::U16(pixels.clone()),
        no_data_value: None,
    };

    let encoded_777 =
        lerc::encode(&image, Precision::Tolerance(777.0)).expect("encode with magic 777");
    let encoded_neg = lerc::encode(&image, Precision::Tolerance(-0.01)).expect("encode with -0.01");

    // Both should produce the same maxZError in the header
    let mze_777 = read_max_z_error(&encoded_777);
    let mze_neg = read_max_z_error(&encoded_neg);
    assert!(
        (mze_777 - mze_neg).abs() < 1e-10,
        "magic 777 maxZError ({mze_777}) should equal -0.01 maxZError ({mze_neg})"
    );
}

#[test]
fn bit_plane_fallback_to_lossless_for_clean_data() {
    // Data with NO noisy low bits should fall back to lossless (maxZError=0.5)
    let width = 128u32;
    let height = 128u32;
    // Clean data: smooth gradient, no noise
    let pixels: Vec<u16> = (0..(width * height) as usize)
        .map(|i| (i * 100 / (width * height) as usize) as u16)
        .collect();

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::UShort,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: SampleData::U16(pixels.clone()),
        no_data_value: None,
    };

    let encoded =
        lerc::encode(&image, Precision::Tolerance(-0.01)).expect("encode with negative mze");
    let header_mze = read_max_z_error(&encoded);

    // Clean data should result in lossless (0.5 for integers)
    assert!(
        (header_mze - 0.5).abs() < 1e-10,
        "clean data with negative maxZError should fall back to lossless (0.5), got {header_mze}"
    );

    // Verify exact round-trip
    let decoded = lerc::decode(&encoded).expect("decode failed");
    match &decoded.data {
        SampleData::U16(dec_pixels) => {
            assert_eq!(dec_pixels, &pixels, "clean data round-trip mismatch");
        }
        _ => panic!("expected U16 data"),
    }
}

#[test]
fn bit_plane_compression_smaller_than_lossless() {
    // Verify that bit-plane compressed output is actually smaller
    let width = 128u32;
    let height = 128u32;
    let pixels = make_noisy_u16(width, height, 4);

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::UShort,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: SampleData::U16(pixels.clone()),
        no_data_value: None,
    };

    let lossy = lerc::encode(&image, Precision::Tolerance(-0.01)).expect("lossy encode");
    let lossless = lerc::encode(&image, Precision::Lossless).expect("lossless encode");

    let lossy_mze = read_max_z_error(&lossy);
    if lossy_mze > 0.5 {
        // Bit-plane compression kicked in, should be smaller
        assert!(
            lossy.len() < lossless.len(),
            "bit-plane compressed ({} bytes) should be smaller than lossless ({} bytes)",
            lossy.len(),
            lossless.len()
        );
    }
}

#[test]
fn bit_plane_too_few_pixels_falls_back() {
    // With very few pixels, bit-plane compression should not activate (needs minCnt=5000)
    let width = 16u32;
    let height = 16u32; // Only 256 pixels, well below 5000
    let pixels = make_noisy_u16(width, height, 3);

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::UShort,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: SampleData::U16(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Tolerance(-0.01)).expect("encode with bit-plane");
    let header_mze = read_max_z_error(&encoded);

    // With only 256 pixels, should fall back to lossless
    assert!(
        (header_mze - 0.5).abs() < 1e-10,
        "small image should fall back to lossless (0.5), got {header_mze}"
    );
}
