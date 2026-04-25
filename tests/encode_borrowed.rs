//! Tests for `encode_borrowed`, the zero-copy multi-band encode entry point.
//!
//! These tests verify the borrowed API:
//! - Produces byte-identical blobs to the existing `Image`-based encoders
//!   for inputs that match their single-band shapes.
//! - Round-trips correctly for multi-band layouts.
//! - Validates input shape (data length, mask count/sizes).

use lerc::bitmask::BitMask;
use lerc::{Precision, SampleData, decode, encode_borrowed, encode_slice, encode_slice_masked};

#[test]
fn f32_single_band_all_valid_matches_encode_slice() {
    let width = 16u32;
    let height = 12u32;
    let pixels: Vec<f32> = (0..(width * height)).map(|i| (i as f32) * 0.5).collect();

    let from_slice = encode_slice::<f32>(width, height, &pixels, Precision::Lossless)
        .expect("encode_slice failed");

    let masks = [BitMask::all_valid((width * height) as usize)];
    let from_borrowed = encode_borrowed::<f32>(
        width,
        height,
        1,
        1,
        &pixels,
        &masks,
        None,
        Precision::Lossless,
    )
    .expect("encode_borrowed failed");

    assert_eq!(
        from_borrowed, from_slice,
        "encode_borrowed must produce a byte-identical blob to encode_slice for the same input"
    );
}

#[test]
fn f32_single_band_masked_matches_encode_slice_masked() {
    let width = 8u32;
    let height = 8u32;
    let n = (width * height) as usize;
    let pixels: Vec<f32> = (0..n).map(|i| (i as f32) * 1.25).collect();

    // Invalidate a handful of pixels.
    let mut mask = BitMask::new(n);
    for k in 0..n {
        if k.is_multiple_of(3) {
            // leave invalid
        } else {
            mask.set_valid(k);
        }
    }

    let from_slice = encode_slice_masked::<f32>(width, height, &pixels, &mask, Precision::Lossless)
        .expect("encode_slice_masked failed");

    let masks = [mask];
    let from_borrowed = encode_borrowed::<f32>(
        width,
        height,
        1,
        1,
        &pixels,
        &masks,
        None,
        Precision::Lossless,
    )
    .expect("encode_borrowed failed");

    assert_eq!(
        from_borrowed, from_slice,
        "encode_borrowed (single band, masked) must match encode_slice_masked"
    );
}

#[test]
fn f32_multi_band_round_trip() {
    let width = 6u32;
    let height = 5u32;
    let bands = 2u32;
    let depth = 1u32;
    let n_per_band = (width * height) as usize;

    let band0: Vec<f32> = (0..n_per_band).map(|i| i as f32).collect();
    let band1: Vec<f32> = (0..n_per_band)
        .map(|i| (n_per_band - i) as f32 * 0.5)
        .collect();

    // Build a band-major buffer.
    let mut data: Vec<f32> = Vec::with_capacity(n_per_band * bands as usize);
    data.extend_from_slice(&band0);
    data.extend_from_slice(&band1);

    // Per-band masks: band 0 fully valid; band 1 has a hole at index 4.
    let mask0 = BitMask::all_valid(n_per_band);
    let mut mask1 = BitMask::new(n_per_band);
    for k in 0..n_per_band {
        if k != 4 {
            mask1.set_valid(k);
        }
    }
    let masks = [mask0, mask1.clone()];

    let blob = encode_borrowed::<f32>(
        width,
        height,
        depth,
        bands,
        &data,
        &masks,
        None,
        Precision::Lossless,
    )
    .expect("encode_borrowed multi-band failed");

    let decoded = decode(&blob).expect("decode failed");
    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    assert_eq!(decoded.depth, depth);
    assert_eq!(decoded.bands, bands);

    let pixels = match decoded.data {
        SampleData::F32(v) => v,
        other => panic!("expected F32 data, got {other:?}"),
    };
    assert_eq!(pixels.len(), n_per_band * bands as usize);

    // Band 0 fully valid: all values must round-trip exactly.
    for k in 0..n_per_band {
        assert_eq!(pixels[k], band0[k], "band0 pixel {k} mismatch");
    }
    // Band 1: valid pixels round-trip; the invalid pixel index can hold anything.
    for k in 0..n_per_band {
        if mask1.is_valid(k) {
            assert_eq!(
                pixels[n_per_band + k],
                band1[k],
                "band1 valid pixel {k} mismatch"
            );
        }
    }

    // Per-band masks must reflect what we passed in.
    assert_eq!(decoded.valid_masks.len(), bands as usize);
    for k in 0..n_per_band {
        assert!(
            decoded.valid_masks[0].is_valid(k),
            "band0 should be all valid"
        );
    }
    assert!(
        !decoded.valid_masks[1].is_valid(4),
        "band1 hole should remain invalid"
    );
    for k in 0..n_per_band {
        if k != 4 {
            assert!(
                decoded.valid_masks[1].is_valid(k),
                "band1 valid pixel {k} lost"
            );
        }
    }
}

#[test]
fn i16_single_band_all_valid_matches_encode_slice() {
    let width = 10u32;
    let height = 7u32;
    let pixels: Vec<i16> = (0..(width * height) as i16).map(|i| i * 3 - 100).collect();

    let from_slice = encode_slice::<i16>(width, height, &pixels, Precision::Lossless)
        .expect("encode_slice (i16) failed");

    let masks = [BitMask::all_valid((width * height) as usize)];
    let from_borrowed = encode_borrowed::<i16>(
        width,
        height,
        1,
        1,
        &pixels,
        &masks,
        None,
        Precision::Lossless,
    )
    .expect("encode_borrowed (i16) failed");

    assert_eq!(
        from_borrowed, from_slice,
        "encode_borrowed must produce a byte-identical blob for i16 lossless"
    );
}

#[test]
fn data_length_mismatch_returns_error() {
    let width = 4u32;
    let height = 3u32;
    // Wrong size: should be 12, supply 11.
    let pixels: Vec<f32> = vec![0.0; 11];
    let masks = [BitMask::all_valid((width * height) as usize)];

    let err = encode_borrowed::<f32>(
        width,
        height,
        1,
        1,
        &pixels,
        &masks,
        None,
        Precision::Lossless,
    )
    .expect_err("expected length mismatch error");

    let msg = format!("{err}");
    assert!(
        msg.contains("data length"),
        "expected data length error, got: {msg}"
    );
}

#[test]
fn mask_count_mismatch_returns_error() {
    let width = 4u32;
    let height = 3u32;
    let bands = 2u32;
    let n = (width * height) as usize;
    let pixels: Vec<f32> = vec![0.0; n * bands as usize];
    // Only one mask supplied for two bands.
    let masks = [BitMask::all_valid(n)];

    let err = encode_borrowed::<f32>(
        width,
        height,
        1,
        bands,
        &pixels,
        &masks,
        None,
        Precision::Lossless,
    )
    .expect_err("expected mask count mismatch error");

    let msg = format!("{err}");
    assert!(
        msg.contains("masks length"),
        "expected mask length error, got: {msg}"
    );
}
