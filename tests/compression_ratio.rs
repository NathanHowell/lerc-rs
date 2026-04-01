#![cfg(feature = "cpp-validation")]

//! Compression ratio regression tests.
//!
//! Each test encodes the same data with both the Rust and C++ encoders and
//! asserts that `rust_size <= cpp_size * MAX_RATIO`. This catches regressions
//! where encoder changes accidentally inflate the output.

use lerc::bitmask::BitMask;
use lerc::{DataType, LercData, LercImage};
use lerc_cpp_ref::{self as cpp, DT_DOUBLE, DT_FLOAT, DT_UCHAR};

/// Maximum allowed ratio of Rust output size to C++ output size.
/// For synthetic FPL data, Huffman tree construction tie-breaking differences
/// can cause up to ~30% larger output on specific patterns. Real-world data
/// (e.g., california DEM) typically compresses as well or better than C++.
const MAX_RATIO_FPL: f64 = 1.35;

/// Tighter bound for non-FPL paths (tiling, Huffman int).
const MAX_RATIO: f64 = 1.02;

// ---------------------------------------------------------------------------
// Test data generators
// ---------------------------------------------------------------------------

fn make_gradient_f32(width: u32, height: u32) -> Vec<f32> {
    (0..width * height)
        .map(|i| {
            let x = (i % width) as f32 / width as f32;
            let y = (i / width) as f32 / height as f32;
            x * 100.0 + y * 200.0 + (x * y * 50.0).sin()
        })
        .collect()
}

fn make_gradient_f64(width: u32, height: u32) -> Vec<f64> {
    (0..width * height)
        .map(|i| {
            let x = (i % width) as f64 / width as f64;
            let y = (i / width) as f64 / height as f64;
            x * 1000.0 + y * 2000.0 + (x * y * 50.0).sin()
        })
        .collect()
}

fn make_ramp_u8(width: u32, height: u32) -> Vec<u8> {
    (0..width * height).map(|i| (i % 256) as u8).collect()
}

fn make_sin_f32(width: u32, height: u32) -> Vec<f32> {
    (0..width * height)
        .map(|i| {
            let x = (i % width) as f32 / width as f32;
            let y = (i / width) as f32 / height as f32;
            ((x * std::f32::consts::TAU).sin() * (y * std::f32::consts::TAU).cos() * 500.0) + 1000.0
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn check_ratio(label: &str, rust_size: usize, cpp_size: usize, max_ratio: f64) {
    let ratio = rust_size as f64 / cpp_size as f64;
    eprintln!("{label}: rust={rust_size} cpp={cpp_size} ratio={ratio:.4}",);
    assert!(
        ratio <= max_ratio,
        "{label}: ratio {ratio:.4} exceeds maximum {max_ratio:.2} \
         (rust={rust_size}, cpp={cpp_size})"
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn ratio_f32_lossy() {
    let (width, height) = (256u32, 256u32);
    let data = make_gradient_f32(width, height);
    let max_z_err = 0.01;

    let rust_blob = lerc::encode_typed(width, height, &data, max_z_err).unwrap();
    let cpp_blob = cpp::encode(
        &data,
        DT_FLOAT,
        width as i32,
        height as i32,
        1,
        1,
        None,
        max_z_err,
    );

    // Lossy tiling should be byte-for-byte identical or very close
    check_ratio("f32_lossy_gradient", rust_blob.len(), cpp_blob.len(), 1.02);
}

#[test]
fn ratio_u8_lossless() {
    let (width, height) = (256u32, 256u32);
    let data = make_ramp_u8(width, height);

    let rust_blob = lerc::encode_typed(width, height, &data, 0.5).unwrap();
    let cpp_blob = cpp::encode(
        &data,
        DT_UCHAR,
        width as i32,
        height as i32,
        1,
        1,
        None,
        0.5,
    );

    // u8 lossless Huffman should be byte-for-byte identical or very close
    check_ratio("u8_lossless_ramp", rust_blob.len(), cpp_blob.len(), 1.02);
}

#[test]
fn ratio_f32_lossless() {
    let (width, height) = (256u32, 256u32);
    let data = make_gradient_f32(width, height);

    let rust_blob = lerc::encode_typed(width, height, &data, 0.0).unwrap();
    let cpp_blob = cpp::encode(
        &data,
        DT_FLOAT,
        width as i32,
        height as i32,
        1,
        1,
        None,
        0.0,
    );

    check_ratio(
        "f32_lossless_gradient",
        rust_blob.len(),
        cpp_blob.len(),
        MAX_RATIO_FPL,
    );
}

#[test]
fn ratio_f32_lossless_sin() {
    let (width, height) = (256u32, 256u32);
    let data = make_sin_f32(width, height);

    let rust_blob = lerc::encode_typed(width, height, &data, 0.0).unwrap();
    let cpp_blob = cpp::encode(
        &data,
        DT_FLOAT,
        width as i32,
        height as i32,
        1,
        1,
        None,
        0.0,
    );

    check_ratio(
        "f32_lossless_sin",
        rust_blob.len(),
        cpp_blob.len(),
        MAX_RATIO_FPL,
    );
}

#[test]
fn ratio_f64_lossless() {
    let (width, height) = (128u32, 128u32);
    let data = make_gradient_f64(width, height);

    let rust_blob = lerc::encode_typed(width, height, &data, 0.0).unwrap();
    let cpp_blob = cpp::encode(
        &data,
        DT_DOUBLE,
        width as i32,
        height as i32,
        1,
        1,
        None,
        0.0,
    );

    check_ratio(
        "f64_lossless_gradient",
        rust_blob.len(),
        cpp_blob.len(),
        MAX_RATIO_FPL,
    );
}

#[test]
fn ratio_reference_california() {
    // Decode the reference california file, re-encode with both Rust and C++
    static CALIFORNIA: &[u8] =
        include_bytes!("../esri-lerc/testData/california_400_400_1_float.lerc2");

    let image = lerc::decode(CALIFORNIA).unwrap();
    let data = image.as_typed::<f32>().unwrap();
    let (width, height) = (image.width, image.height);
    let max_z_err = 0.01; // lossy

    let rust_blob = lerc::encode_typed(width, height, data, max_z_err).unwrap();
    let cpp_blob = cpp::encode(
        data,
        DT_FLOAT,
        width as i32,
        height as i32,
        1,
        1,
        None,
        max_z_err,
    );

    check_ratio("california_lossy", rust_blob.len(), cpp_blob.len(), 1.02);

    // Also check lossless
    let rust_blob_ll = lerc::encode_typed(width, height, data, 0.0).unwrap();
    let cpp_blob_ll = cpp::encode(data, DT_FLOAT, width as i32, height as i32, 1, 1, None, 0.0);

    check_ratio(
        "california_lossless",
        rust_blob_ll.len(),
        cpp_blob_ll.len(),
        MAX_RATIO_FPL,
    );
}

#[test]
fn ratio_reference_bluemarble() {
    // Decode the reference bluemarble file, re-encode with both Rust and C++
    static BLUEMARBLE: &[u8] =
        include_bytes!("../esri-lerc/testData/bluemarble_256_256_3_byte.lerc2");

    let image = lerc::decode(BLUEMARBLE).unwrap();
    let data = image.as_typed::<u8>().unwrap();
    let (width, height, n_bands) = (image.width, image.height, image.n_bands);

    // Build per-band encoding
    let ppb = (width * height) as usize;

    // Encode all bands with Rust
    let masks: Vec<_> = (0..n_bands).map(|_| BitMask::all_valid(ppb)).collect();
    let img = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands,
        data_type: DataType::Byte,
        valid_masks: masks,
        data: LercData::U8(data.to_vec()),
        no_data_value: None,
    };
    let rust_blob = lerc::encode(&img, 0.5).unwrap();

    // Encode all bands with C++
    let cpp_blob = cpp::encode(
        data,
        DT_UCHAR,
        width as i32,
        height as i32,
        1,
        n_bands as i32,
        None,
        0.5,
    );

    check_ratio(
        "bluemarble_u8_lossless",
        rust_blob.len(),
        cpp_blob.len(),
        1.02,
    );
}
