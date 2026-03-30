use lerc::bitmask::BitMask;
use lerc::DataType;
use proptest::prelude::*;
use proptest::strategy::ValueTree;

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

fn f32_image_strategy() -> impl Strategy<Value = (u32, u32, Vec<f32>, f64)> {
    (1..64u32, 1..64u32, 0.0..1.0f64).prop_flat_map(|(w, h, mze)| {
        let size = (w * h) as usize;
        (
            Just(w),
            Just(h),
            prop::collection::vec(-1000.0f32..1000.0, size),
            Just(mze),
        )
    })
}

fn f64_image_strategy() -> impl Strategy<Value = (u32, u32, Vec<f64>, f64)> {
    (1..64u32, 1..64u32, 0.0..1.0f64).prop_flat_map(|(w, h, mze)| {
        let size = (w * h) as usize;
        (
            Just(w),
            Just(h),
            prop::collection::vec(-1000.0f64..1000.0, size),
            Just(mze),
        )
    })
}

fn u8_image_strategy() -> impl Strategy<Value = (u32, u32, Vec<u8>)> {
    (1..64u32, 1..64u32).prop_flat_map(|(w, h)| {
        let size = (w * h) as usize;
        (
            Just(w),
            Just(h),
            prop::collection::vec(any::<u8>(), size),
        )
    })
}

fn u16_image_strategy() -> impl Strategy<Value = (u32, u32, Vec<u16>)> {
    (1..64u32, 1..64u32).prop_flat_map(|(w, h)| {
        let size = (w * h) as usize;
        (
            Just(w),
            Just(h),
            prop::collection::vec(any::<u16>(), size),
        )
    })
}

fn i32_image_strategy() -> impl Strategy<Value = (u32, u32, Vec<i32>)> {
    (1..64u32, 1..64u32).prop_flat_map(|(w, h)| {
        let size = (w * h) as usize;
        (
            Just(w),
            Just(h),
            prop::collection::vec(-100_000i32..100_000, size),
        )
    })
}

fn mask_strategy(num_pixels: usize) -> impl Strategy<Value = BitMask> {
    prop::collection::vec(any::<bool>(), num_pixels).prop_map(move |bits| {
        let mut mask = BitMask::new(num_pixels);
        for (k, &valid) in bits.iter().enumerate() {
            if valid {
                mask.set_valid(k);
            }
        }
        mask
    })
}

// ---------------------------------------------------------------------------
// Property 1: Lossy f32 round-trip
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn lossy_f32_round_trip((w, h, pixels, mze) in f32_image_strategy()) {
        let blob = lerc::encode_typed(w, h, &pixels, mze).expect("encode failed");
        // The encoder may raise maxZError (e.g. try_raise_max_z_error for float data
        // with limited precision). Use the effective maxZError from the header.
        let info = lerc::decode_info(&blob).expect("decode_info failed");
        let effective_mze = info.max_z_error;
        // Allow tolerance for f32 quantization rounding. The quantize/dequantize
        // cycle can exceed maxZError due to float precision limitations when the
        // data range is large relative to maxZError. f32 quantization can exceed
        // maxZError by a few percent due to rounding (same as C++ NeedToCheckForFltRndErr).
        let tol = effective_mze * 1.05 + 1e-5;
        let (decoded, mask, dw, dh) = lerc::decode_typed::<f32>(&blob).expect("decode failed");
        prop_assert_eq!(dw, w);
        prop_assert_eq!(dh, h);
        for (i, (&orig, &dec)) in pixels.iter().zip(decoded.iter()).enumerate() {
            prop_assert!(
                mask.is_valid(i),
                "pixel {} should be valid",
                i,
            );
            let diff = (orig as f64 - dec as f64).abs();
            prop_assert!(
                diff <= tol,
                "pixel {}: |{} - {}| = {} > tolerance {} (effective maxZError {}, requested {})",
                i, orig, dec, diff, tol, effective_mze, mze,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Property 2: Lossy f64 round-trip
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn lossy_f64_round_trip((w, h, pixels, mze) in f64_image_strategy()) {
        let blob = lerc::encode_typed(w, h, &pixels, mze).expect("encode failed");
        let info = lerc::decode_info(&blob).expect("decode_info failed");
        let effective_mze = info.max_z_error;
        let tol = effective_mze * 1.001;
        let (decoded, mask, dw, dh) = lerc::decode_typed::<f64>(&blob).expect("decode failed");
        prop_assert_eq!(dw, w);
        prop_assert_eq!(dh, h);
        for (i, (&orig, &dec)) in pixels.iter().zip(decoded.iter()).enumerate() {
            prop_assert!(mask.is_valid(i), "pixel {} should be valid", i);
            let diff = (orig - dec).abs();
            prop_assert!(
                diff <= tol,
                "pixel {}: |{} - {}| = {} > tolerance {} (effective maxZError {}, requested {})",
                i, orig, dec, diff, tol, effective_mze, mze,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Property 3: Lossless integer round-trips
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn lossless_u8_round_trip((w, h, pixels) in u8_image_strategy()) {
        let blob = lerc::encode_typed(w, h, &pixels, 0.0).expect("encode failed");
        let (decoded, _mask, dw, dh) = lerc::decode_typed::<u8>(&blob).expect("decode failed");
        prop_assert_eq!(dw, w);
        prop_assert_eq!(dh, h);
        prop_assert_eq!(decoded, pixels);
    }

    #[test]
    fn lossless_u16_round_trip((w, h, pixels) in u16_image_strategy()) {
        let blob = lerc::encode_typed(w, h, &pixels, 0.0).expect("encode failed");
        let (decoded, _mask, dw, dh) = lerc::decode_typed::<u16>(&blob).expect("decode failed");
        prop_assert_eq!(dw, w);
        prop_assert_eq!(dh, h);
        prop_assert_eq!(decoded, pixels);
    }

    #[test]
    fn lossless_i32_round_trip((w, h, pixels) in i32_image_strategy()) {
        let blob = lerc::encode_typed(w, h, &pixels, 0.0).expect("encode failed");
        let (decoded, _mask, dw, dh) = lerc::decode_typed::<i32>(&blob).expect("decode failed");
        prop_assert_eq!(dw, w);
        prop_assert_eq!(dh, h);
        prop_assert_eq!(decoded, pixels);
    }
}

// ---------------------------------------------------------------------------
// Property 4: Lossless float round-trips (bit-exact with maxZError=0)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn lossless_f32_round_trip((w, h, pixels, _mze) in f32_image_strategy()) {
        let blob = lerc::encode_typed(w, h, &pixels, 0.0).expect("encode failed");
        let (decoded, _mask, dw, dh) = lerc::decode_typed::<f32>(&blob).expect("decode failed");
        prop_assert_eq!(dw, w);
        prop_assert_eq!(dh, h);
        for (i, (&orig, &dec)) in pixels.iter().zip(decoded.iter()).enumerate() {
            prop_assert_eq!(
                orig.to_bits(),
                dec.to_bits(),
                "pixel {}: orig bits {:#010x} != decoded bits {:#010x}",
                i,
                orig.to_bits(),
                dec.to_bits(),
            );
        }
    }

    #[test]
    fn lossless_f64_round_trip((w, h, pixels, _mze) in f64_image_strategy()) {
        let blob = lerc::encode_typed(w, h, &pixels, 0.0).expect("encode failed");
        let (decoded, _mask, dw, dh) = lerc::decode_typed::<f64>(&blob).expect("decode failed");
        prop_assert_eq!(dw, w);
        prop_assert_eq!(dh, h);
        for (i, (&orig, &dec)) in pixels.iter().zip(decoded.iter()).enumerate() {
            prop_assert_eq!(
                orig.to_bits(),
                dec.to_bits(),
                "pixel {}: orig bits {:#018x} != decoded bits {:#018x}",
                i,
                orig.to_bits(),
                dec.to_bits(),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Property 5: Mask preservation
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn mask_preservation_f32(
        (w, h) in (1..32u32, 1..32u32),
    ) {
        let n = (w * h) as usize;
        // Use a separate runner for the dependent strategy
        let mut runner = proptest::test_runner::TestRunner::default();
        let mask = mask_strategy(n)
            .new_tree(&mut runner)
            .expect("mask strategy failed")
            .current();

        // Need at least one valid pixel for a meaningful test
        if mask.count_valid() == 0 {
            return Ok(());
        }

        let pixels: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let blob = lerc::encode_typed_masked(w, h, &pixels, &mask, 0.0)
            .expect("encode failed");
        let (decoded, decoded_mask, dw, dh) =
            lerc::decode_typed::<f32>(&blob).expect("decode failed");
        prop_assert_eq!(dw, w);
        prop_assert_eq!(dh, h);

        for k in 0..n {
            prop_assert_eq!(
                mask.is_valid(k),
                decoded_mask.is_valid(k),
                "mask mismatch at pixel {}",
                k,
            );
            if mask.is_valid(k) {
                prop_assert_eq!(
                    pixels[k].to_bits(),
                    decoded[k].to_bits(),
                    "pixel {} value mismatch",
                    k,
                );
            }
        }
    }

    #[test]
    fn mask_preservation_u8(
        (w, h) in (1..32u32, 1..32u32),
    ) {
        let n = (w * h) as usize;
        let mut runner = proptest::test_runner::TestRunner::default();
        let mask = mask_strategy(n)
            .new_tree(&mut runner)
            .expect("mask strategy failed")
            .current();

        if mask.count_valid() == 0 {
            return Ok(());
        }

        let pixels: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
        let blob = lerc::encode_typed_masked(w, h, &pixels, &mask, 0.0)
            .expect("encode failed");
        let (decoded, decoded_mask, dw, dh) =
            lerc::decode_typed::<u8>(&blob).expect("decode failed");
        prop_assert_eq!(dw, w);
        prop_assert_eq!(dh, h);

        for k in 0..n {
            prop_assert_eq!(
                mask.is_valid(k),
                decoded_mask.is_valid(k),
                "mask mismatch at pixel {}",
                k,
            );
            if mask.is_valid(k) {
                prop_assert_eq!(
                    pixels[k], decoded[k],
                    "pixel {} value mismatch",
                    k,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Property 6: Header consistency (decode_info matches decode)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn header_consistency_f32((w, h, pixels, mze) in f32_image_strategy()) {
        let blob = lerc::encode_typed(w, h, &pixels, mze).expect("encode failed");
        let info = lerc::decode_info(&blob).expect("decode_info failed");
        let image = lerc::decode(&blob).expect("decode failed");

        prop_assert_eq!(info.width, image.width);
        prop_assert_eq!(info.height, image.height);
        prop_assert_eq!(info.n_depth, image.n_depth);
        prop_assert_eq!(info.n_bands, image.n_bands);
        prop_assert_eq!(info.data_type, image.data_type);
        prop_assert_eq!(info.data_type, DataType::Float);
    }

    #[test]
    fn header_consistency_u8((w, h, pixels) in u8_image_strategy()) {
        let blob = lerc::encode_typed(w, h, &pixels, 0.0).expect("encode failed");
        let info = lerc::decode_info(&blob).expect("decode_info failed");
        let image = lerc::decode(&blob).expect("decode failed");

        prop_assert_eq!(info.width, image.width);
        prop_assert_eq!(info.height, image.height);
        prop_assert_eq!(info.n_depth, image.n_depth);
        prop_assert_eq!(info.n_bands, image.n_bands);
        prop_assert_eq!(info.data_type, image.data_type);
        prop_assert_eq!(info.data_type, DataType::Byte);
    }
}

// ---------------------------------------------------------------------------
// Property 7: Blob size matches header
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn blob_size_matches_header_f32((w, h, pixels, mze) in f32_image_strategy()) {
        let blob = lerc::encode_typed(w, h, &pixels, mze).expect("encode failed");
        let info = lerc::decode_info(&blob).expect("decode_info failed");
        prop_assert_eq!(
            info.blob_size as usize,
            blob.len(),
            "header blob_size {} != actual {}",
            info.blob_size,
            blob.len(),
        );
    }

    #[test]
    fn blob_size_matches_header_u16((w, h, pixels) in u16_image_strategy()) {
        let blob = lerc::encode_typed(w, h, &pixels, 0.0).expect("encode failed");
        let info = lerc::decode_info(&blob).expect("decode_info failed");
        prop_assert_eq!(
            info.blob_size as usize,
            blob.len(),
            "header blob_size {} != actual {}",
            info.blob_size,
            blob.len(),
        );
    }

    #[test]
    fn blob_size_matches_header_i32((w, h, pixels) in i32_image_strategy()) {
        let blob = lerc::encode_typed(w, h, &pixels, 0.0).expect("encode failed");
        let info = lerc::decode_info(&blob).expect("decode_info failed");
        prop_assert_eq!(
            info.blob_size as usize,
            blob.len(),
            "header blob_size {} != actual {}",
            info.blob_size,
            blob.len(),
        );
    }
}

// ---------------------------------------------------------------------------
// Property 8: Deterministic encoding
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn deterministic_f32((w, h, pixels, mze) in f32_image_strategy()) {
        let blob1 = lerc::encode_typed(w, h, &pixels, mze).expect("encode 1 failed");
        let blob2 = lerc::encode_typed(w, h, &pixels, mze).expect("encode 2 failed");
        prop_assert_eq!(blob1, blob2, "encoding same data twice produced different blobs");
    }

    #[test]
    fn deterministic_u8((w, h, pixels) in u8_image_strategy()) {
        let blob1 = lerc::encode_typed(w, h, &pixels, 0.0).expect("encode 1 failed");
        let blob2 = lerc::encode_typed(w, h, &pixels, 0.0).expect("encode 2 failed");
        prop_assert_eq!(blob1, blob2, "encoding same data twice produced different blobs");
    }

    #[test]
    fn deterministic_f64((w, h, pixels, _mze) in f64_image_strategy()) {
        let blob1 = lerc::encode_typed(w, h, &pixels, 0.0).expect("encode 1 failed");
        let blob2 = lerc::encode_typed(w, h, &pixels, 0.0).expect("encode 2 failed");
        prop_assert_eq!(blob1, blob2, "encoding same data twice produced different blobs");
    }
}
