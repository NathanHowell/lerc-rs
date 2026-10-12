//! Blocks whose value range is under one quantization step.
//!
//! When a micro-block's range `z_max - z_min` is non-zero but rounds to zero
//! quantization steps (`(z_max - z_min) / (2 * max_z_error) < 0.5`), every
//! value is within `max_z_error` of `z_min`, so the block is coded as a
//! constant at `z_min`. Esri's Lerc2 does this (`NumBytesTile` and
//! `WriteTile` in `Lerc2.cpp`). Storing such a block raw instead costs the
//! full sample width per value — 32 bits per pixel for `f32` — on exactly the
//! data lossy LERC is meant for, such as flat terrain carrying noise below
//! the tolerance.

use lerc::bitmask::BitMask;
use lerc::{DataType, Image, Precision, SampleData};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;
const PIXELS: usize = (WIDTH * HEIGHT) as usize;
const MAX_Z_ERROR: f64 = 0.02;

/// Raw `f32` costs 4 bytes per pixel. A constant block is a one-byte flag
/// plus an offset of at most 4 bytes for 64 or 256 pixels, under 0.1 byte per
/// pixel, so anything over this bound means blocks went out raw or quantized.
const CONSTANT_BLOCK_BYTES_PER_PIXEL: f64 = 0.5;

/// Deterministic noise in `[0, 1)` (splitmix64 finalizer over the index).
fn unit_noise(i: usize) -> f64 {
    let mut z = (i as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// Noise in `[0, fraction × MAX_Z_ERROR)`, seeded by `seed`. A quantization
/// step is `2 × MAX_Z_ERROR`, so any `fraction` below 1 keeps a block's range
/// under half a step — which rounds to zero steps — whatever micro-block
/// size the encoder picks.
fn sub_tolerance_noise(fraction: f64, seed: usize) -> Vec<f64> {
    (0..PIXELS)
        .map(|i| fraction * MAX_Z_ERROR * unit_noise(i + seed * PIXELS))
        .collect()
}

/// Flat ground at 1000 m; the noise spans 0.9 × `MAX_Z_ERROR`, just under
/// the half-step boundary.
fn flat_ground() -> Vec<f32> {
    sub_tolerance_noise(0.9, 0)
        .iter()
        .map(|n| (1000.0 + n) as f32)
        .collect()
}

/// A ramp on the quantization lattice: 0.24 m per pixel is six steps of
/// `2 × MAX_Z_ERROR`, so it quantizes with (up to `f32` rounding) no error
/// and the encoder's reconstruction of it matches the input. A later depth
/// slice's diff against it then carries only the noise added to that slice.
fn lattice_ramp() -> Vec<f32> {
    (0..PIXELS)
        .map(|i| {
            let (x, y) = (i % WIDTH as usize, i / WIDTH as usize);
            (1000.0 + 0.24 * x as f64 + 0.24 * y as f64) as f32
        })
        .collect()
}

/// Three depth slices, pixel-interleaved: the lattice ramp, then twice more
/// with 0.3 × `MAX_Z_ERROR` of noise added per slice. Each diff block is
/// sub-tolerance, including the third slice's: it is diffed against the
/// second slice's reconstruction, so it carries its own noise plus the
/// second slice's sub-step residual, together under 0.6 × `MAX_Z_ERROR`.
fn sub_tolerance_depths() -> Image {
    let base = lattice_ramp();
    let (noise_1, noise_2) = (sub_tolerance_noise(0.3, 1), sub_tolerance_noise(0.3, 2));
    let pixels: Vec<f32> = (0..PIXELS)
        .flat_map(|i| {
            let second = f64::from(base[i]) + noise_1[i];
            let third = second + noise_2[i];
            [base[i], second as f32, third as f32]
        })
        .collect();
    Image {
        width: WIDTH,
        height: HEIGHT,
        depth: 3,
        bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![BitMask::all_valid(PIXELS)],
        data: SampleData::F32(pixels),
        no_data_value: None,
    }
}

fn f32_pixels(image: &Image) -> &[f32] {
    image.as_typed::<f32>().expect("f32 image")
}

/// Every decoded value is within `MAX_Z_ERROR` of the original, allowing the
/// few-ULP margin the `f32` reconstruction `z_min + q × 2 × max_z_error`
/// can add (as `cpp_cross_validation.rs` does).
fn assert_within_tolerance(original: &[f32], decoded: &[f32], ctx: &str) {
    assert_eq!(original.len(), decoded.len(), "{ctx}: length");
    for (i, (&o, &d)) in original.iter().zip(decoded).enumerate() {
        let diff = (f64::from(o) - f64::from(d)).abs();
        let tol = MAX_Z_ERROR + f64::from(o.abs().max(d.abs())) * 4.0 * f64::from(f32::EPSILON);
        assert!(
            diff <= tol,
            "{ctx}: pixel {i}: |{o} - {d}| = {diff} > {tol}"
        );
    }
}

fn bytes_per_pixel(bytes: usize, pixels: usize) -> f64 {
    bytes as f64 / pixels as f64
}

fn encode_single_depth(data: &[f32]) -> Vec<u8> {
    lerc::encode_slice(
        WIDTH,
        HEIGHT,
        data,
        Precision::Tolerance(MAX_Z_ERROR as f32),
    )
    .expect("encode")
}

#[test]
fn sub_tolerance_noise_is_coded_as_constant_blocks() {
    let data = flat_ground();
    let blob = encode_single_depth(&data);

    let bpp = bytes_per_pixel(blob.len(), PIXELS);
    assert!(
        bpp < CONSTANT_BLOCK_BYTES_PER_PIXEL,
        "sub-tolerance blocks went out raw: {bpp:.3} bytes/pixel ({} bytes)",
        blob.len()
    );

    let decoded = lerc::decode_slice::<f32>(&blob).expect("decode");
    assert_within_tolerance(&data, &decoded.pixels, "single depth");
}

/// The encoder diffs each depth slice against its own reconstruction of the
/// previous one, so a constant diff block must be reconstructed the way the
/// decoder rebuilds it; if it were not, the third slice would decode outside
/// the tolerance.
#[test]
fn sub_tolerance_depth_diffs_are_coded_as_constant_blocks() {
    let image = sub_tolerance_depths();
    let blob = lerc::encode(&image, Precision::Tolerance(MAX_Z_ERROR)).expect("encode");

    // The base slice coded on its own prices the quantized ramp; depths 1
    // and 2 should add only constant diff blocks on top of it.
    let base_only = encode_single_depth(&lattice_ramp());
    let extra = blob.len().saturating_sub(base_only.len());
    let extra_bpp = bytes_per_pixel(extra, 2 * PIXELS);
    assert!(
        extra_bpp < CONSTANT_BLOCK_BYTES_PER_PIXEL,
        "sub-tolerance diff blocks were not coded as constants: {extra_bpp:.3} bytes/pixel \
         for the two diff slices (three depths {} bytes, base slice alone {})",
        blob.len(),
        base_only.len()
    );

    let decoded = lerc::decode(&blob).expect("decode");
    assert_eq!(decoded.depth, 3);
    assert_within_tolerance(f32_pixels(&image), f32_pixels(&decoded), "three depths");
}

/// The same blobs against Esri's reference, which must decode them within
/// the tolerance.
#[cfg(feature = "cpp-validation")]
mod cpp_reference {
    use super::*;
    use lerc_cpp_ref::{self as cpp, DT_FLOAT};

    #[test]
    fn reference_decodes_sub_tolerance_constant_blocks() {
        let data = flat_ground();
        let (decoded, _): (Vec<f32>, _) = cpp::decode(
            &encode_single_depth(&data),
            DT_FLOAT,
            WIDTH as i32,
            HEIGHT as i32,
            1,
            1,
        );
        assert_within_tolerance(&data, &decoded, "reference, single depth");
    }

    #[test]
    fn reference_decodes_sub_tolerance_constant_diff_blocks() {
        let image = sub_tolerance_depths();
        let blob = lerc::encode(&image, Precision::Tolerance(MAX_Z_ERROR)).expect("encode");
        let (decoded, _): (Vec<f32>, _) =
            cpp::decode(&blob, DT_FLOAT, WIDTH as i32, HEIGHT as i32, 3, 1);
        assert_within_tolerance(f32_pixels(&image), &decoded, "reference, three depths");
    }
}
