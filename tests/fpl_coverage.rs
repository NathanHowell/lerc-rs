use lerc::Precision;
use lerc::bitmask::BitMask;
use lerc::{DataType, LercData, LercImage};

// ---------------------------------------------------------------------------
// Helper: verify that FPL encoding was used (IEM_DeltaDeltaHuffman = 3)
// ---------------------------------------------------------------------------

/// Parse the blob header and mask to find the one_sweep and IEM flag bytes,
/// verifying that FPL encoding (IEM_DeltaDeltaHuffman = 3) was used.
fn assert_fpl_used(blob: &[u8]) {
    // The header is always version 6 for our encoder. Header size for v6 = 90 bytes.
    let header_size = 90usize;
    let mut pos = header_size;

    // Read mask section: numBytesMask (4-byte LE i32). If 0, mask is all-valid.
    let mask_num_bytes = i32::from_le_bytes(blob[pos..pos + 4].try_into().unwrap());
    pos += 4;
    if mask_num_bytes > 0 {
        pos += mask_num_bytes as usize;
    }

    // Read per-depth min/max ranges (always present for v4+).
    // Layout: nDepth typed zMin values, then nDepth typed zMax values.
    // Read nDepth from header at offset 22.
    let n_depth = i32::from_le_bytes(blob[22..26].try_into().unwrap()) as usize;
    // Read data type from header at offset 38.
    let dt_raw = i32::from_le_bytes(blob[38..42].try_into().unwrap());
    let type_size = match dt_raw {
        6 => 4, // Float
        7 => 8, // Double
        _ => panic!("expected Float(6) or Double(7) data type, got {}", dt_raw),
    };
    pos += n_depth * type_size * 2; // zMin[nDepth] + zMax[nDepth]

    // Now we should have: one_sweep(1 byte) then IEM_flag(1 byte).
    assert!(
        pos + 2 <= blob.len(),
        "blob too short to contain one_sweep + IEM flag at offset {}",
        pos,
    );
    let one_sweep = blob[pos];
    let iem_flag = blob[pos + 1];

    assert_eq!(
        one_sweep, 0,
        "expected one_sweep=0 for FPL, got {}",
        one_sweep,
    );
    assert_eq!(
        iem_flag, 3,
        "expected IEM_DeltaDeltaHuffman=3, got {}",
        iem_flag,
    );
}

// ---------------------------------------------------------------------------
// Helpers: lossless round-trip verification
// ---------------------------------------------------------------------------

/// Encode f32 data losslessly, verify FPL was used, decode and check bit-exact.
fn round_trip_f32_fpl(width: u32, height: u32, pixels: &[f32]) {
    let blob =
        lerc::encode_typed(width, height, pixels, Precision::Lossless).expect("encode failed");

    assert_fpl_used(&blob);

    let info = lerc::decode_info(&blob).expect("decode_info failed");
    assert_eq!(info.data_type, DataType::Float);
    assert_eq!(info.max_z_error, 0.0);
    assert_eq!(info.blob_size as usize, blob.len());

    let (decoded, _mask, dw, dh) = lerc::decode_typed::<f32>(&blob).expect("decode failed");
    assert_eq!(dw, width);
    assert_eq!(dh, height);
    for (i, (&orig, &dec)) in pixels.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(
            orig.to_bits(),
            dec.to_bits(),
            "pixel {}: orig={} (bits={:#010x}), decoded={} (bits={:#010x})",
            i,
            orig,
            orig.to_bits(),
            dec,
            dec.to_bits(),
        );
    }
}

/// Encode f64 data losslessly, verify FPL was used, decode and check bit-exact.
fn round_trip_f64_fpl(width: u32, height: u32, pixels: &[f64]) {
    let blob =
        lerc::encode_typed(width, height, pixels, Precision::Lossless).expect("encode failed");

    assert_fpl_used(&blob);

    let info = lerc::decode_info(&blob).expect("decode_info failed");
    assert_eq!(info.data_type, DataType::Double);
    assert_eq!(info.max_z_error, 0.0);
    assert_eq!(info.blob_size as usize, blob.len());

    let (decoded, _mask, dw, dh) = lerc::decode_typed::<f64>(&blob).expect("decode failed");
    assert_eq!(dw, width);
    assert_eq!(dh, height);
    for (i, (&orig, &dec)) in pixels.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(
            orig.to_bits(),
            dec.to_bits(),
            "pixel {}: orig={} (bits={:#018x}), decoded={} (bits={:#018x})",
            i,
            orig,
            orig.to_bits(),
            dec,
            dec.to_bits(),
        );
    }
}

/// Encode f32 data losslessly, decode and check bit-exact (no FPL assertion).
/// Used for constant data and 1x1 images that take the const-image shortcut.
fn round_trip_f32_lossless(width: u32, height: u32, pixels: &[f32]) {
    let blob =
        lerc::encode_typed(width, height, pixels, Precision::Lossless).expect("encode failed");

    let (decoded, _mask, dw, dh) = lerc::decode_typed::<f32>(&blob).expect("decode failed");
    assert_eq!(dw, width);
    assert_eq!(dh, height);
    for (i, (&orig, &dec)) in pixels.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(
            orig.to_bits(),
            dec.to_bits(),
            "pixel {}: orig={} (bits={:#010x}), decoded={} (bits={:#010x})",
            i,
            orig,
            orig.to_bits(),
            dec,
            dec.to_bits(),
        );
    }
}

/// Encode f64 data losslessly, decode and check bit-exact (no FPL assertion).
fn round_trip_f64_lossless(width: u32, height: u32, pixels: &[f64]) {
    let blob =
        lerc::encode_typed(width, height, pixels, Precision::Lossless).expect("encode failed");

    let (decoded, _mask, dw, dh) = lerc::decode_typed::<f64>(&blob).expect("decode failed");
    assert_eq!(dw, width);
    assert_eq!(dh, height);
    for (i, (&orig, &dec)) in pixels.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(
            orig.to_bits(),
            dec.to_bits(),
            "pixel {}: orig={} (bits={:#018x}), decoded={} (bits={:#018x})",
            i,
            orig,
            orig.to_bits(),
            dec,
            dec.to_bits(),
        );
    }
}

// ---------------------------------------------------------------------------
// Test 1: Basic FPL round-trip with linear ramp
// ---------------------------------------------------------------------------

#[test]
fn fpl_f32_linear_ramp() {
    let w = 32u32;
    let h = 32u32;
    let pixels: Vec<f32> = (0..w * h).map(|i| i as f32 * 0.1).collect();
    round_trip_f32_fpl(w, h, &pixels);
}

#[test]
fn fpl_f64_linear_ramp() {
    let w = 32u32;
    let h = 32u32;
    let pixels: Vec<f64> = (0..w * h).map(|i| i as f64 * 0.123456789).collect();
    round_trip_f64_fpl(w, h, &pixels);
}

// ---------------------------------------------------------------------------
// Test 2: Constant data (takes const-image shortcut, FPL not used)
// ---------------------------------------------------------------------------

#[test]
fn fpl_f32_constant() {
    let w = 16u32;
    let h = 16u32;
    let pixels = vec![42.5f32; (w * h) as usize];
    round_trip_f32_lossless(w, h, &pixels);
}

#[test]
fn fpl_f64_constant() {
    let w = 16u32;
    let h = 16u32;
    let pixels = vec![std::f64::consts::PI; (w * h) as usize];
    round_trip_f64_lossless(w, h, &pixels);
}

// ---------------------------------------------------------------------------
// Test 3: Sin wave
// ---------------------------------------------------------------------------

#[test]
fn fpl_f32_sin_wave() {
    let w = 64u32;
    let h = 64u32;
    let pixels: Vec<f32> = (0..w * h)
        .map(|i| {
            let x = (i % w) as f32 / w as f32;
            let y = (i / w) as f32 / h as f32;
            (x * std::f32::consts::TAU).sin() + (y * std::f32::consts::TAU * 2.0).cos()
        })
        .collect();
    round_trip_f32_fpl(w, h, &pixels);
}

#[test]
fn fpl_f64_sin_wave() {
    let w = 64u32;
    let h = 64u32;
    let pixels: Vec<f64> = (0..w * h)
        .map(|i| {
            let x = (i % w) as f64 / w as f64;
            let y = (i / w) as f64 / h as f64;
            (x * std::f64::consts::TAU).sin() + (y * std::f64::consts::TAU * 2.0).cos()
        })
        .collect();
    round_trip_f64_fpl(w, h, &pixels);
}

// ---------------------------------------------------------------------------
// Test 4: Random data
// ---------------------------------------------------------------------------

#[test]
fn fpl_f32_pseudorandom() {
    let w = 32u32;
    let h = 32u32;
    let mut state: u64 = 0xDEADBEEF;
    let pixels: Vec<f32> = (0..w * h)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = (state >> 32) as u32;
            // Build a value in [1.0, 2.0) + small offset
            let float_bits = 0x3F800000 | (bits & 0x007FFFFF);
            f32::from_bits(float_bits) - 1.0 + (bits >> 23) as f32 * 0.001
        })
        .collect();
    round_trip_f32_fpl(w, h, &pixels);
}

#[test]
fn fpl_f64_pseudorandom() {
    let w = 32u32;
    let h = 32u32;
    let mut state: u64 = 0xCAFEBABE;
    let pixels: Vec<f64> = (0..w * h)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = state;
            let float_bits = 0x3FF0000000000000 | (bits & 0x000FFFFFFFFFFFFF);
            f64::from_bits(float_bits) - 1.0
        })
        .collect();
    round_trip_f64_fpl(w, h, &pixels);
}

// ---------------------------------------------------------------------------
// Test 5: Sparse data (mostly zero)
// ---------------------------------------------------------------------------

#[test]
fn fpl_f32_sparse() {
    let w = 32u32;
    let h = 32u32;
    let pixels: Vec<f32> = (0..w * h)
        .map(|i| if i % 37 == 0 { (i as f32) * 0.01 } else { 0.0 })
        .collect();
    round_trip_f32_fpl(w, h, &pixels);
}

// ---------------------------------------------------------------------------
// Test 6: Extreme values (very large, very small, subnormals)
// ---------------------------------------------------------------------------

#[test]
fn fpl_f32_extreme_values() {
    let w = 8u32;
    let h = 8u32;
    let n = (w * h) as usize;

    // Include a variety of extreme f32 values (no NaN since LERC may not preserve NaN bits).
    // Avoid mixing f32::MAX and f32::MIN in the same image since the extreme range
    // triggers one-sweep encoding instead of FPL. We test those in a separate assertion.
    let mut pixels = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::MIN_POSITIVE, // smallest positive normal
        -f32::MIN_POSITIVE,
        1.0e-38,                    // near subnormal boundary
        1.0e38,                     // large
        f32::from_bits(0x00000001), // smallest subnormal
        f32::from_bits(0x00400000), // a subnormal
        f32::from_bits(0x007FFFFF), // largest subnormal
        f32::EPSILON,
        std::f32::consts::PI,
        std::f32::consts::E,
        0.5,
        -0.5,
    ];
    while pixels.len() < n {
        pixels.push((pixels.len() as f32) * 0.001);
    }

    round_trip_f32_fpl(w, h, &pixels);
}

#[test]
fn fpl_f32_extreme_range() {
    // Extreme range values (MAX/MIN) that may trigger one-sweep instead of FPL.
    // We still verify lossless round-trip.
    let w = 4u32;
    let h = 4u32;
    let pixels = vec![
        f32::MAX,
        f32::MIN,
        f32::MAX,
        f32::MIN,
        1.0,
        -1.0,
        0.0,
        f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
        1.0e38,
        -1.0e38,
        f32::EPSILON,
        std::f32::consts::PI,
        std::f32::consts::E,
        0.0,
    ];
    round_trip_f32_lossless(w, h, &pixels);
}

#[test]
fn fpl_f64_extreme_values() {
    let w = 8u32;
    let h = 8u32;
    let n = (w * h) as usize;

    let mut pixels = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        1.0e-307,
        1.0e307,
        f64::from_bits(0x0000000000000001), // smallest subnormal
        f64::from_bits(0x0008000000000000), // a subnormal
        f64::from_bits(0x000FFFFFFFFFFFFF), // largest subnormal
        f64::EPSILON,
        std::f64::consts::PI,
        std::f64::consts::E,
        0.5,
        -0.5,
    ];
    while pixels.len() < n {
        pixels.push((pixels.len() as f64) * 0.001);
    }

    round_trip_f64_fpl(w, h, &pixels);
}

// ---------------------------------------------------------------------------
// Test 7: Multi-depth FPL (nDepth=3, f32 lossless)
// ---------------------------------------------------------------------------

#[test]
fn fpl_f32_multi_depth() {
    let w = 16u32;
    let h = 16u32;
    let n_depth = 3u32;
    let n = (w * h * n_depth) as usize;

    let pixels: Vec<f32> = (0..n)
        .map(|i| {
            let pixel = i / n_depth as usize;
            let channel = i % n_depth as usize;
            match channel {
                0 => (pixel as f32) * 0.01,
                1 => ((pixel as f32) * 0.02).sin(),
                _ => 1.0 - (pixel as f32) * 0.005,
            }
        })
        .collect();

    let image = LercImage {
        width: w,
        height: h,
        n_depth,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![BitMask::all_valid((w * h) as usize)],
        data: LercData::F32(pixels.clone()),
        no_data_value: None,
    };

    let blob = lerc::encode(&image, Precision::Lossless).expect("encode failed");
    assert_fpl_used(&blob);

    let info = lerc::decode_info(&blob).expect("decode_info failed");
    assert_eq!(info.n_depth, n_depth);
    assert_eq!(info.data_type, DataType::Float);
    assert_eq!(info.max_z_error, 0.0);

    let decoded = lerc::decode(&blob).expect("decode failed");
    assert_eq!(decoded.width, w);
    assert_eq!(decoded.height, h);
    assert_eq!(decoded.n_depth, n_depth);

    let decoded_pixels = decoded.as_typed::<f32>().expect("expected f32 data");
    for (i, (&orig, &dec)) in pixels.iter().zip(decoded_pixels.iter()).enumerate() {
        assert_eq!(
            orig.to_bits(),
            dec.to_bits(),
            "pixel {}: orig={} (bits={:#010x}), decoded={} (bits={:#010x})",
            i,
            orig,
            orig.to_bits(),
            dec,
            dec.to_bits(),
        );
    }
}

// ---------------------------------------------------------------------------
// Test 8: Predictor coverage - data patterns favoring different predictors
// ---------------------------------------------------------------------------

/// Data that is already constant row-by-row (should favor no predictor or trivial delta).
#[test]
fn fpl_f32_predictor_none_candidate() {
    let w = 4u32;
    let h = 4u32;
    let pixels: Vec<f32> = (0..w * h)
        .map(|i| f32::from_bits(0x3F800000 | ((i * 7919 + 104729) & 0x007FFFFF)))
        .collect();
    round_trip_f32_fpl(w, h, &pixels);
}

/// Data with strong row-wise correlation (should favor delta1 predictor).
#[test]
fn fpl_f32_predictor_delta1_candidate() {
    let w = 64u32;
    let h = 64u32;
    let pixels: Vec<f32> = (0..w * h)
        .map(|i| {
            let x = (i % w) as f32;
            x * 0.001
        })
        .collect();
    round_trip_f32_fpl(w, h, &pixels);
}

/// Data with strong 2D correlation (should favor rows_cols predictor).
#[test]
fn fpl_f32_predictor_rows_cols_candidate() {
    let w = 64u32;
    let h = 64u32;
    let pixels: Vec<f32> = (0..w * h)
        .map(|i| {
            let x = (i % w) as f32 / w as f32;
            let y = (i / w) as f32 / h as f32;
            x * 0.5 + y * 0.3 + x * y * 0.1
        })
        .collect();
    round_trip_f32_fpl(w, h, &pixels);
}

// ---------------------------------------------------------------------------
// Test 9: 1x1 and edge-case dimensions
// ---------------------------------------------------------------------------

#[test]
fn fpl_f32_1x1() {
    // 1x1 image has zMin==zMax, so const-image shortcut is used (no FPL).
    round_trip_f32_lossless(1, 1, &[std::f32::consts::PI]);
}

#[test]
fn fpl_f32_1xn() {
    let h = 63u32;
    let pixels: Vec<f32> = (0..h).map(|i| i as f32 * 0.7).collect();
    round_trip_f32_fpl(1, h, &pixels);
}

#[test]
fn fpl_f32_nx1() {
    let w = 63u32;
    let pixels: Vec<f32> = (0..w).map(|i| i as f32 * 0.7).collect();
    round_trip_f32_fpl(w, 1, &pixels);
}

#[test]
fn fpl_f64_1x1() {
    // 1x1 const-image shortcut, no FPL.
    round_trip_f64_lossless(1, 1, &[std::f64::consts::E]);
}

// ---------------------------------------------------------------------------
// Test 10: Negative values and mixed sign data
// ---------------------------------------------------------------------------

#[test]
fn fpl_f32_mixed_sign() {
    let w = 32u32;
    let h = 32u32;
    let pixels: Vec<f32> = (0..w * h)
        .map(|i| {
            let val = (i as f32 - (w * h / 2) as f32) * 0.01;
            if i % 3 == 0 { -val } else { val }
        })
        .collect();
    round_trip_f32_fpl(w, h, &pixels);
}

// ---------------------------------------------------------------------------
// Test 11: f64 multi-depth
// ---------------------------------------------------------------------------

#[test]
fn fpl_f64_multi_depth() {
    let w = 8u32;
    let h = 8u32;
    let n_depth = 3u32;
    let n = (w * h * n_depth) as usize;

    let pixels: Vec<f64> = (0..n)
        .map(|i| {
            let pixel = i / n_depth as usize;
            let channel = i % n_depth as usize;
            (pixel as f64) * 0.1 + (channel as f64) * 100.0
        })
        .collect();

    let image = LercImage {
        width: w,
        height: h,
        n_depth,
        n_bands: 1,
        data_type: DataType::Double,
        valid_masks: vec![BitMask::all_valid((w * h) as usize)],
        data: LercData::F64(pixels.clone()),
        no_data_value: None,
    };

    let blob = lerc::encode(&image, Precision::Lossless).expect("encode failed");
    assert_fpl_used(&blob);

    let decoded = lerc::decode(&blob).expect("decode failed");
    assert_eq!(decoded.n_depth, n_depth);

    let decoded_pixels = decoded.as_typed::<f64>().expect("expected f64 data");
    for (i, (&orig, &dec)) in pixels.iter().zip(decoded_pixels.iter()).enumerate() {
        assert_eq!(
            orig.to_bits(),
            dec.to_bits(),
            "pixel {}: orig={} decoded={}",
            i,
            orig,
            dec,
        );
    }
}
