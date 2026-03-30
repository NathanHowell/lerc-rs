#![cfg(feature = "cpp-validation")]
#![allow(clippy::too_many_arguments)]

//! Bidirectional cross-validation tests between the Rust LERC codec and the
//! C++ reference implementation (esri-lerc).
//!
//! These tests verify that:
//! - Blobs encoded by Rust can be decoded by the C++ library
//! - Blobs encoded by C++ can be decoded by the Rust library
//! - Full bidirectional round-trips preserve data within tolerance
//!
//! Note: Lossless float encoding (maxZErr=0) uses the FPL (Float Point
//! Lossless) path in both implementations. For C++ -> Rust lossless float
//! tests we use codec version 5 (pre-FPL) to avoid that path; for
//! Rust -> C++ lossless float tests we use a tiny nonzero maxZErr instead.

use lerc::bitmask::BitMask;
use lerc::{DataType, LercData, LercImage};

// ---------------------------------------------------------------------------
// C API bindings (linked from the C++ LercLib built by build.rs)
// ---------------------------------------------------------------------------

// C++ data type codes
const DT_CHAR: u32 = 0;
const DT_UCHAR: u32 = 1;
const DT_SHORT: u32 = 2;
const DT_USHORT: u32 = 3;
const DT_INT: u32 = 4;
const DT_UINT: u32 = 5;
const DT_FLOAT: u32 = 6;
const DT_DOUBLE: u32 = 7;

// InfoArray indices (from Lerc_types.h InfoArrOrder)
const INFO_DATA_TYPE: usize = 1;
const INFO_N_COLS: usize = 3;
const INFO_N_ROWS: usize = 4;
const INFO_N_BANDS: usize = 5;

// DataRangeArray indices
const RANGE_MAX_Z_ERR_USED: usize = 2;

unsafe extern "C" {
    fn lerc_computeCompressedSize(
        pData: *const std::ffi::c_void,
        dataType: u32,
        nDepth: i32,
        nCols: i32,
        nRows: i32,
        nBands: i32,
        nMasks: i32,
        pValidBytes: *const u8,
        maxZErr: f64,
        numBytes: *mut u32,
    ) -> u32;

    fn lerc_encode(
        pData: *const std::ffi::c_void,
        dataType: u32,
        nDepth: i32,
        nCols: i32,
        nRows: i32,
        nBands: i32,
        nMasks: i32,
        pValidBytes: *const u8,
        maxZErr: f64,
        pOutBuffer: *mut u8,
        outBufferSize: u32,
        nBytesWritten: *mut u32,
    ) -> u32;

    fn lerc_computeCompressedSizeForVersion(
        pData: *const std::ffi::c_void,
        codecVersion: i32,
        dataType: u32,
        nDepth: i32,
        nCols: i32,
        nRows: i32,
        nBands: i32,
        nMasks: i32,
        pValidBytes: *const u8,
        maxZErr: f64,
        numBytes: *mut u32,
    ) -> u32;

    fn lerc_encodeForVersion(
        pData: *const std::ffi::c_void,
        codecVersion: i32,
        dataType: u32,
        nDepth: i32,
        nCols: i32,
        nRows: i32,
        nBands: i32,
        nMasks: i32,
        pValidBytes: *const u8,
        maxZErr: f64,
        pOutBuffer: *mut u8,
        outBufferSize: u32,
        nBytesWritten: *mut u32,
    ) -> u32;

    fn lerc_getBlobInfo(
        pLercBlob: *const u8,
        blobSize: u32,
        infoArray: *mut u32,
        dataRangeArray: *mut f64,
        infoArraySize: i32,
        dataRangeArraySize: i32,
    ) -> u32;

    fn lerc_decode(
        pLercBlob: *const u8,
        blobSize: u32,
        nMasks: i32,
        pValidBytes: *mut u8,
        nDepth: i32,
        nCols: i32,
        nRows: i32,
        nBands: i32,
        dataType: u32,
        pData: *mut std::ffi::c_void,
    ) -> u32;
}

// ---------------------------------------------------------------------------
// Helpers: C++ encode / decode / info
// ---------------------------------------------------------------------------

fn cpp_encode<T: Copy>(
    data: &[T],
    data_type: u32,
    n_cols: i32,
    n_rows: i32,
    n_depth: i32,
    n_bands: i32,
    valid_bytes: Option<&[u8]>,
    max_z_err: f64,
) -> Vec<u8> {
    let n_masks = if valid_bytes.is_some() { 1i32 } else { 0i32 };
    let valid_ptr = valid_bytes
        .map(|v| v.as_ptr())
        .unwrap_or(std::ptr::null());

    let mut num_bytes: u32 = 0;
    let rc = unsafe {
        lerc_computeCompressedSize(
            data.as_ptr() as *const std::ffi::c_void,
            data_type,
            n_depth,
            n_cols,
            n_rows,
            n_bands,
            n_masks,
            valid_ptr,
            max_z_err,
            &mut num_bytes,
        )
    };
    assert_eq!(rc, 0, "lerc_computeCompressedSize failed with code {rc}");
    assert!(num_bytes > 0);

    let mut buffer = vec![0u8; num_bytes as usize];
    let mut bytes_written: u32 = 0;
    let rc = unsafe {
        lerc_encode(
            data.as_ptr() as *const std::ffi::c_void,
            data_type,
            n_depth,
            n_cols,
            n_rows,
            n_bands,
            n_masks,
            valid_ptr,
            max_z_err,
            buffer.as_mut_ptr(),
            num_bytes,
            &mut bytes_written,
        )
    };
    assert_eq!(rc, 0, "lerc_encode failed with code {rc}");
    buffer.truncate(bytes_written as usize);
    buffer
}

fn cpp_encode_for_version<T: Copy>(
    data: &[T],
    codec_version: i32,
    data_type: u32,
    n_cols: i32,
    n_rows: i32,
    n_depth: i32,
    n_bands: i32,
    valid_bytes: Option<&[u8]>,
    max_z_err: f64,
) -> Vec<u8> {
    let n_masks = if valid_bytes.is_some() { 1i32 } else { 0i32 };
    let valid_ptr = valid_bytes
        .map(|v| v.as_ptr())
        .unwrap_or(std::ptr::null());

    let mut num_bytes: u32 = 0;
    let rc = unsafe {
        lerc_computeCompressedSizeForVersion(
            data.as_ptr() as *const std::ffi::c_void,
            codec_version,
            data_type,
            n_depth,
            n_cols,
            n_rows,
            n_bands,
            n_masks,
            valid_ptr,
            max_z_err,
            &mut num_bytes,
        )
    };
    assert_eq!(rc, 0, "lerc_computeCompressedSizeForVersion failed: {rc}");
    assert!(num_bytes > 0);

    let mut buffer = vec![0u8; num_bytes as usize];
    let mut bytes_written: u32 = 0;
    let rc = unsafe {
        lerc_encodeForVersion(
            data.as_ptr() as *const std::ffi::c_void,
            codec_version,
            data_type,
            n_depth,
            n_cols,
            n_rows,
            n_bands,
            n_masks,
            valid_ptr,
            max_z_err,
            buffer.as_mut_ptr(),
            num_bytes,
            &mut bytes_written,
        )
    };
    assert_eq!(rc, 0, "lerc_encodeForVersion failed: {rc}");
    buffer.truncate(bytes_written as usize);
    buffer
}

fn cpp_decode<T: Copy + Default>(
    blob: &[u8],
    data_type: u32,
    n_cols: i32,
    n_rows: i32,
    n_depth: i32,
    n_bands: i32,
) -> (Vec<T>, Vec<u8>) {
    let pixel_count =
        (n_cols as usize) * (n_rows as usize) * (n_depth as usize) * (n_bands as usize);
    let mut data = vec![T::default(); pixel_count];
    let mask_size = (n_cols as usize) * (n_rows as usize);
    let mut valid_bytes = vec![0u8; mask_size];

    let rc = unsafe {
        lerc_decode(
            blob.as_ptr(),
            blob.len() as u32,
            1,
            valid_bytes.as_mut_ptr(),
            n_depth,
            n_cols,
            n_rows,
            n_bands,
            data_type,
            data.as_mut_ptr() as *mut std::ffi::c_void,
        )
    };
    assert_eq!(rc, 0, "lerc_decode failed with code {rc}");
    (data, valid_bytes)
}

fn cpp_get_blob_info(blob: &[u8]) -> ([u32; 11], [f64; 3]) {
    let mut info = [0u32; 11];
    let mut range = [0.0f64; 3];
    let rc = unsafe {
        lerc_getBlobInfo(
            blob.as_ptr(),
            blob.len() as u32,
            info.as_mut_ptr(),
            range.as_mut_ptr(),
            info.len() as i32,
            range.len() as i32,
        )
    };
    assert_eq!(rc, 0, "lerc_getBlobInfo failed with code {rc}");
    (info, range)
}

// ---------------------------------------------------------------------------
// Assertion helpers
// ---------------------------------------------------------------------------

/// Check that two f32 values match within `max_z_err`, accounting for the
/// float quantization pipeline rounding. LERC quantizes with
/// `floor(val / (2*maxZErr) + 0.5)` in float arithmetic, so the difference
/// measured in f64 can slightly exceed `max_z_err` by a value-dependent ULP
/// margin.
fn assert_f32_close(orig: f32, dec: f32, max_z_err: f64, ctx: std::fmt::Arguments<'_>) {
    let diff = (orig as f64 - dec as f64).abs();
    let tol = max_z_err + (orig.abs().max(dec.abs()) as f64) * 4.0 * (f32::EPSILON as f64);
    assert!(
        diff <= tol,
        "{ctx}: |{orig} - {dec}| = {diff} > {tol} (maxZErr={max_z_err})"
    );
}

fn assert_f64_close(orig: f64, dec: f64, max_z_err: f64, ctx: std::fmt::Arguments<'_>) {
    let diff = (orig - dec).abs();
    let tol = max_z_err + orig.abs().max(dec.abs()) * 4.0 * f64::EPSILON;
    assert!(
        diff <= tol,
        "{ctx}: |{orig} - {dec}| = {diff} > {tol} (maxZErr={max_z_err})"
    );
}

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

fn make_ramp_i16(width: u32, height: u32) -> Vec<i16> {
    (0..width * height)
        .map(|i| ((i as i32 % 65536) - 32768) as i16)
        .collect()
}

fn make_ramp_u16(width: u32, height: u32) -> Vec<u16> {
    (0..width * height).map(|i| (i % 65536) as u16).collect()
}

fn make_ramp_u32(width: u32, height: u32) -> Vec<u32> {
    (0..width * height).map(|i| i * 7 + 13).collect()
}

fn make_ramp_i32(width: u32, height: u32) -> Vec<i32> {
    (0..width * height)
        .map(|i| (i as i32) * 7 - 50000)
        .collect()
}

fn make_validity_mask(width: u32, height: u32) -> (BitMask, Vec<u8>) {
    let n = (width * height) as usize;
    let mut mask = BitMask::new(n);
    let mut valid_bytes = vec![0u8; n];
    for (i, vb) in valid_bytes.iter_mut().enumerate().take(n) {
        if (i % 4) != 3 {
            mask.set_valid(i);
            *vb = 1;
        }
    }
    (mask, valid_bytes)
}

// ===========================================================================
// Rust encode -> C++ decode
// ===========================================================================

#[test]
fn rust_encode_cpp_decode_f32_near_lossless() {
    let (width, height) = (64, 48);
    let data = make_gradient_f32(width, height);
    let max_z_err = f32::EPSILON as f64;

    let blob = lerc::encode_typed(width, height, &data, max_z_err).unwrap();

    let (info, _) = cpp_get_blob_info(&blob);
    assert_eq!(info[INFO_DATA_TYPE], DT_FLOAT);
    assert_eq!(info[INFO_N_COLS], width);
    assert_eq!(info[INFO_N_ROWS], height);
    assert_eq!(info[INFO_N_BANDS], 1);

    let (decoded, valid_bytes): (Vec<f32>, _) =
        cpp_decode(&blob, DT_FLOAT, width as i32, height as i32, 1, 1);

    assert!(valid_bytes.iter().all(|&v| v == 1));
    for (i, (&o, &d)) in data.iter().zip(decoded.iter()).enumerate() {
        assert_f32_close(o, d, max_z_err, format_args!("pixel {i}"));
    }
}

#[test]
fn rust_encode_cpp_decode_f32_lossy() {
    let (width, height) = (64, 48);
    let data = make_gradient_f32(width, height);

    for max_z_err in [0.01, 0.1, 1.0] {
        let blob = lerc::encode_typed(width, height, &data, max_z_err).unwrap();

        let (info, range) = cpp_get_blob_info(&blob);
        assert_eq!(info[INFO_DATA_TYPE], DT_FLOAT);
        assert!(
            range[RANGE_MAX_Z_ERR_USED] <= max_z_err + f64::EPSILON,
            "maxZErrUsed {} > requested {max_z_err}",
            range[RANGE_MAX_Z_ERR_USED]
        );

        let (decoded, _): (Vec<f32>, _) =
            cpp_decode(&blob, DT_FLOAT, width as i32, height as i32, 1, 1);

        for (i, (&o, &d)) in data.iter().zip(decoded.iter()).enumerate() {
            assert_f32_close(o, d, max_z_err, format_args!("maxZErr={max_z_err} pixel {i}"));
        }
    }
}

#[test]
fn rust_encode_cpp_decode_f64_near_lossless() {
    let (width, height) = (32, 32);
    let data = make_gradient_f64(width, height);
    let max_z_err = f64::EPSILON;

    let blob = lerc::encode_typed(width, height, &data, max_z_err).unwrap();

    let (info, _) = cpp_get_blob_info(&blob);
    assert_eq!(info[INFO_DATA_TYPE], DT_DOUBLE);

    let (decoded, _): (Vec<f64>, _) =
        cpp_decode(&blob, DT_DOUBLE, width as i32, height as i32, 1, 1);

    for (i, (&o, &d)) in data.iter().zip(decoded.iter()).enumerate() {
        assert_f64_close(o, d, max_z_err, format_args!("pixel {i}"));
    }
}

#[test]
fn rust_encode_cpp_decode_f64_lossy() {
    let (width, height) = (32, 32);
    let data = make_gradient_f64(width, height);
    let max_z_err = 0.5;

    let blob = lerc::encode_typed(width, height, &data, max_z_err).unwrap();

    let (decoded, _): (Vec<f64>, _) =
        cpp_decode(&blob, DT_DOUBLE, width as i32, height as i32, 1, 1);

    for (i, (&o, &d)) in data.iter().zip(decoded.iter()).enumerate() {
        assert_f64_close(o, d, max_z_err, format_args!("pixel {i}"));
    }
}

#[test]
fn rust_encode_cpp_decode_u8() {
    let (width, height) = (64, 64);
    let data = make_ramp_u8(width, height);

    let blob = lerc::encode_typed(width, height, &data, 0.5).unwrap();
    let (info, _) = cpp_get_blob_info(&blob);
    assert_eq!(info[INFO_DATA_TYPE], DT_UCHAR);

    let (decoded, _): (Vec<u8>, _) =
        cpp_decode(&blob, DT_UCHAR, width as i32, height as i32, 1, 1);
    assert_eq!(data, decoded);
}

#[test]
fn rust_encode_cpp_decode_i8() {
    let (width, height) = (64, 64);
    let data: Vec<i8> = (0..(width * height) as usize)
        .map(|i| ((i % 256) as i16 - 128) as i8)
        .collect();

    let blob = lerc::encode_typed(width, height, &data, 0.5).unwrap();
    let (info, _) = cpp_get_blob_info(&blob);
    assert_eq!(info[INFO_DATA_TYPE], DT_CHAR);

    let (decoded, _): (Vec<i8>, _) =
        cpp_decode(&blob, DT_CHAR, width as i32, height as i32, 1, 1);
    assert_eq!(data, decoded);
}

#[test]
fn rust_encode_cpp_decode_i16() {
    let (width, height) = (64, 64);
    let data = make_ramp_i16(width, height);

    let blob = lerc::encode_typed(width, height, &data, 0.0).unwrap();
    let (info, _) = cpp_get_blob_info(&blob);
    assert_eq!(info[INFO_DATA_TYPE], DT_SHORT);

    let (decoded, _): (Vec<i16>, _) =
        cpp_decode(&blob, DT_SHORT, width as i32, height as i32, 1, 1);
    assert_eq!(data, decoded);
}

#[test]
fn rust_encode_cpp_decode_u16() {
    let (width, height) = (64, 64);
    let data = make_ramp_u16(width, height);

    let blob = lerc::encode_typed(width, height, &data, 0.0).unwrap();
    let (info, _) = cpp_get_blob_info(&blob);
    assert_eq!(info[INFO_DATA_TYPE], DT_USHORT);

    let (decoded, _): (Vec<u16>, _) =
        cpp_decode(&blob, DT_USHORT, width as i32, height as i32, 1, 1);
    assert_eq!(data, decoded);
}

#[test]
fn rust_encode_cpp_decode_u32() {
    let (width, height) = (64, 64);
    let data = make_ramp_u32(width, height);

    let blob = lerc::encode_typed(width, height, &data, 0.0).unwrap();
    let (info, _) = cpp_get_blob_info(&blob);
    assert_eq!(info[INFO_DATA_TYPE], DT_UINT);

    let (decoded, _): (Vec<u32>, _) =
        cpp_decode(&blob, DT_UINT, width as i32, height as i32, 1, 1);
    assert_eq!(data, decoded);
}

#[test]
fn rust_encode_cpp_decode_i32() {
    let (width, height) = (64, 64);
    let data = make_ramp_i32(width, height);

    let blob = lerc::encode_typed(width, height, &data, 0.0).unwrap();
    let (info, _) = cpp_get_blob_info(&blob);
    assert_eq!(info[INFO_DATA_TYPE], DT_INT);

    let (decoded, _): (Vec<i32>, _) =
        cpp_decode(&blob, DT_INT, width as i32, height as i32, 1, 1);
    assert_eq!(data, decoded);
}

#[test]
fn rust_encode_cpp_decode_with_mask() {
    let (width, height) = (64, 48);
    let data = make_gradient_f32(width, height);
    let (mask, valid_bytes) = make_validity_mask(width, height);
    let max_z_err = 0.01;

    let blob = lerc::encode_typed_masked(width, height, &data, &mask, max_z_err).unwrap();

    let (info, _) = cpp_get_blob_info(&blob);
    assert_eq!(info[INFO_DATA_TYPE], DT_FLOAT);

    let (decoded, dec_valid): (Vec<f32>, _) =
        cpp_decode(&blob, DT_FLOAT, width as i32, height as i32, 1, 1);

    let n = (width * height) as usize;
    for i in 0..n {
        assert_eq!(valid_bytes[i], dec_valid[i], "mask mismatch at pixel {i}");
        if valid_bytes[i] != 0 {
            assert_f32_close(data[i], decoded[i], max_z_err, format_args!("valid pixel {i}"));
        }
    }
}

// ===========================================================================
// C++ encode -> Rust decode
// ===========================================================================

#[test]
fn cpp_encode_rust_decode_f32_lossless() {
    // Use codec version 5 to avoid the FPL path.
    let (width, height) = (64, 48);
    let data = make_gradient_f32(width, height);

    let blob =
        cpp_encode_for_version(&data, 5, DT_FLOAT, width as i32, height as i32, 1, 1, None, 0.0);

    let image = lerc::decode(&blob).unwrap();
    assert_eq!(image.width, width);
    assert_eq!(image.height, height);
    assert_eq!(image.data_type, DataType::Float);

    let decoded = image.as_typed::<f32>().unwrap();
    for (i, (&o, &d)) in data.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(o.to_bits(), d.to_bits(), "pixel {i}: {o} vs {d}");
    }
}

#[test]
fn cpp_encode_rust_decode_f32_lossy() {
    let (width, height) = (64, 48);
    let data = make_gradient_f32(width, height);

    for max_z_err in [0.01, 0.1, 1.0] {
        let blob = cpp_encode(
            &data,
            DT_FLOAT,
            width as i32,
            height as i32,
            1,
            1,
            None,
            max_z_err,
        );

        let image = lerc::decode(&blob).unwrap();
        let decoded = image.as_typed::<f32>().unwrap();

        for (i, (&o, &d)) in data.iter().zip(decoded.iter()).enumerate() {
            assert_f32_close(o, d, max_z_err, format_args!("maxZErr={max_z_err} pixel {i}"));
        }
    }
}

#[test]
fn cpp_encode_rust_decode_f64_lossless() {
    let (width, height) = (32, 32);
    let data = make_gradient_f64(width, height);

    let blob = cpp_encode_for_version(
        &data, 5, DT_DOUBLE, width as i32, height as i32, 1, 1, None, 0.0,
    );

    let image = lerc::decode(&blob).unwrap();
    let decoded = image.as_typed::<f64>().unwrap();

    for (i, (&o, &d)) in data.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(o.to_bits(), d.to_bits(), "pixel {i}: {o} vs {d}");
    }
}

#[test]
fn cpp_encode_rust_decode_u8() {
    let (width, height) = (64, 64);
    let data = make_ramp_u8(width, height);

    let blob = cpp_encode(&data, DT_UCHAR, width as i32, height as i32, 1, 1, None, 0.5);

    let image = lerc::decode(&blob).unwrap();
    assert_eq!(image.data_type, DataType::Byte);
    assert_eq!(data.as_slice(), image.as_typed::<u8>().unwrap());
}

#[test]
fn cpp_encode_rust_decode_i8() {
    let (width, height): (i32, i32) = (64, 64);
    let data: Vec<i8> = (0..(width * height) as usize)
        .map(|i| ((i % 256) as i16 - 128) as i8)
        .collect();

    let blob = cpp_encode(&data, DT_CHAR, width, height, 1, 1, None, 0.5);

    let image = lerc::decode(&blob).unwrap();
    assert_eq!(image.data_type, DataType::Char);
    assert_eq!(data.as_slice(), image.as_typed::<i8>().unwrap());
}

#[test]
fn cpp_encode_rust_decode_i16() {
    let (width, height) = (64, 64);
    let data = make_ramp_i16(width, height);

    let blob = cpp_encode(&data, DT_SHORT, width as i32, height as i32, 1, 1, None, 0.0);

    let image = lerc::decode(&blob).unwrap();
    assert_eq!(image.data_type, DataType::Short);
    assert_eq!(data.as_slice(), image.as_typed::<i16>().unwrap());
}

#[test]
fn cpp_encode_rust_decode_u16() {
    let (width, height) = (64, 64);
    let data = make_ramp_u16(width, height);

    let blob = cpp_encode(&data, DT_USHORT, width as i32, height as i32, 1, 1, None, 0.0);

    let image = lerc::decode(&blob).unwrap();
    assert_eq!(image.data_type, DataType::UShort);
    assert_eq!(data.as_slice(), image.as_typed::<u16>().unwrap());
}

#[test]
fn cpp_encode_rust_decode_u32() {
    let (width, height) = (64, 64);
    let data = make_ramp_u32(width, height);

    let blob = cpp_encode(&data, DT_UINT, width as i32, height as i32, 1, 1, None, 0.0);

    let image = lerc::decode(&blob).unwrap();
    assert_eq!(image.data_type, DataType::UInt);
    assert_eq!(data.as_slice(), image.as_typed::<u32>().unwrap());
}

#[test]
fn cpp_encode_rust_decode_i32() {
    let (width, height) = (64, 64);
    let data = make_ramp_i32(width, height);

    let blob = cpp_encode(&data, DT_INT, width as i32, height as i32, 1, 1, None, 0.0);

    let image = lerc::decode(&blob).unwrap();
    assert_eq!(image.data_type, DataType::Int);
    assert_eq!(data.as_slice(), image.as_typed::<i32>().unwrap());
}

#[test]
fn cpp_encode_rust_decode_with_mask() {
    let (width, height) = (64, 48);
    let data = make_gradient_f32(width, height);
    let (_mask, valid_bytes) = make_validity_mask(width, height);
    let max_z_err = 0.01;

    let blob = cpp_encode(
        &data,
        DT_FLOAT,
        width as i32,
        height as i32,
        1,
        1,
        Some(&valid_bytes),
        max_z_err,
    );

    let image = lerc::decode(&blob).unwrap();
    let decoded = image.as_typed::<f32>().unwrap();
    let rust_mask = image.mask().unwrap();

    let n = (width * height) as usize;
    for i in 0..n {
        let cpp_valid = valid_bytes[i] != 0;
        let rust_valid = rust_mask.is_valid(i);
        assert_eq!(cpp_valid, rust_valid, "mask mismatch at pixel {i}");
        if cpp_valid {
            assert_f32_close(data[i], decoded[i], max_z_err, format_args!("valid pixel {i}"));
        }
    }
}

// ===========================================================================
// Multi-band
// ===========================================================================

#[test]
fn rust_encode_cpp_decode_multiband_f32() {
    let (width, height, n_bands) = (32u32, 32u32, 3u32);
    let ppb = (width * height) as usize;

    let mut data = Vec::with_capacity(ppb * n_bands as usize);
    let band0 = make_gradient_f32(width, height);
    let band1: Vec<f32> = band0.iter().map(|&v| 300.0 - v).collect();
    let band2 = vec![42.0f32; ppb];
    data.extend_from_slice(&band0);
    data.extend_from_slice(&band1);
    data.extend_from_slice(&band2);

    let masks: Vec<_> = (0..n_bands)
        .map(|_| BitMask::all_valid(ppb))
        .collect();

    let max_z_err = 0.01;
    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands,
        data_type: DataType::Float,
        valid_masks: masks,
        data: LercData::F32(data.clone()),
        no_data_value: None,
    };

    let blob = lerc::encode(&image, max_z_err).unwrap();

    let (info, _) = cpp_get_blob_info(&blob);
    assert_eq!(info[INFO_DATA_TYPE], DT_FLOAT);
    assert_eq!(info[INFO_N_COLS], width);
    assert_eq!(info[INFO_N_ROWS], height);
    assert_eq!(info[INFO_N_BANDS], n_bands);

    let (decoded, _): (Vec<f32>, _) = cpp_decode(
        &blob,
        DT_FLOAT,
        width as i32,
        height as i32,
        1,
        n_bands as i32,
    );

    assert_eq!(decoded.len(), data.len());
    for (i, (&o, &d)) in data.iter().zip(decoded.iter()).enumerate() {
        assert_f32_close(o, d, max_z_err, format_args!("pixel {i}"));
    }
}

#[test]
fn cpp_encode_rust_decode_multiband_u8() {
    let (w, h, nb) = (32i32, 32i32, 3i32);
    let ppb = (w * h) as usize;

    let mut data = Vec::with_capacity(ppb * nb as usize);
    for band in 0..nb {
        for i in 0..ppb {
            data.push(((i + band as usize * 80) % 256) as u8);
        }
    }

    let blob = cpp_encode(&data, DT_UCHAR, w, h, 1, nb, None, 0.5);

    let image = lerc::decode(&blob).unwrap();
    assert_eq!(image.data_type, DataType::Byte);
    assert_eq!(image.n_bands, nb as u32);
    assert_eq!(data.as_slice(), image.as_typed::<u8>().unwrap());
}

// ===========================================================================
// Bidirectional round-trips
// ===========================================================================

#[test]
fn roundtrip_rust_cpp_rust_f32() {
    let (width, height) = (64, 48);
    let original = make_gradient_f32(width, height);
    let max_z_err = 0.01;

    let rust_blob = lerc::encode_typed(width, height, &original, max_z_err).unwrap();
    let (cpp_decoded, _): (Vec<f32>, _) =
        cpp_decode(&rust_blob, DT_FLOAT, width as i32, height as i32, 1, 1);
    let cpp_blob = cpp_encode(
        &cpp_decoded,
        DT_FLOAT,
        width as i32,
        height as i32,
        1,
        1,
        None,
        max_z_err,
    );
    let final_image = lerc::decode(&cpp_blob).unwrap();
    let final_decoded = final_image.as_typed::<f32>().unwrap();

    for (i, (&o, &d)) in original.iter().zip(final_decoded.iter()).enumerate() {
        assert_f32_close(o, d, 2.0 * max_z_err, format_args!("pixel {i}"));
    }
}

#[test]
fn roundtrip_cpp_rust_cpp_f32() {
    let (width, height) = (64, 48);
    let original = make_gradient_f32(width, height);
    let max_z_err = 0.01;

    let cpp_blob = cpp_encode(
        &original,
        DT_FLOAT,
        width as i32,
        height as i32,
        1,
        1,
        None,
        max_z_err,
    );
    let rust_image = lerc::decode(&cpp_blob).unwrap();
    let rust_decoded: Vec<f32> = rust_image.as_typed::<f32>().unwrap().to_vec();
    let rust_blob = lerc::encode_typed(width, height, &rust_decoded, max_z_err).unwrap();
    let (final_decoded, _): (Vec<f32>, _) =
        cpp_decode(&rust_blob, DT_FLOAT, width as i32, height as i32, 1, 1);

    for (i, (&o, &d)) in original.iter().zip(final_decoded.iter()).enumerate() {
        assert_f32_close(o, d, 2.0 * max_z_err, format_args!("pixel {i}"));
    }
}

#[test]
fn roundtrip_rust_cpp_rust_u8_lossless() {
    let (width, height) = (64, 64);
    let original = make_ramp_u8(width, height);

    let rust_blob = lerc::encode_typed(width, height, &original, 0.5).unwrap();
    let (cpp_decoded, _): (Vec<u8>, _) =
        cpp_decode(&rust_blob, DT_UCHAR, width as i32, height as i32, 1, 1);
    assert_eq!(original, cpp_decoded);

    let cpp_blob = cpp_encode(
        &cpp_decoded,
        DT_UCHAR,
        width as i32,
        height as i32,
        1,
        1,
        None,
        0.5,
    );
    let final_image = lerc::decode(&cpp_blob).unwrap();
    assert_eq!(original.as_slice(), final_image.as_typed::<u8>().unwrap());
}

#[test]
fn roundtrip_cpp_rust_cpp_u8_lossless() {
    let (width, height) = (64, 64);
    let original = make_ramp_u8(width, height);

    let cpp_blob = cpp_encode(
        &original,
        DT_UCHAR,
        width as i32,
        height as i32,
        1,
        1,
        None,
        0.5,
    );
    let rust_image = lerc::decode(&cpp_blob).unwrap();
    let rust_decoded: Vec<u8> = rust_image.as_typed::<u8>().unwrap().to_vec();
    assert_eq!(original, rust_decoded);

    let rust_blob = lerc::encode_typed(width, height, &rust_decoded, 0.5).unwrap();
    let (final_decoded, _): (Vec<u8>, _) =
        cpp_decode(&rust_blob, DT_UCHAR, width as i32, height as i32, 1, 1);
    assert_eq!(original, final_decoded);
}

#[test]
fn roundtrip_with_mask_f32() {
    let (width, height) = (64, 48);
    let original = make_gradient_f32(width, height);
    let (mask, valid_bytes) = make_validity_mask(width, height);
    let max_z_err = 0.01;

    let rust_blob =
        lerc::encode_typed_masked(width, height, &original, &mask, max_z_err).unwrap();
    let (cpp_decoded, cpp_valid): (Vec<f32>, _) =
        cpp_decode(&rust_blob, DT_FLOAT, width as i32, height as i32, 1, 1);

    for i in 0..(width * height) as usize {
        assert_eq!(valid_bytes[i], cpp_valid[i], "mask mismatch at pixel {i}");
    }

    let cpp_blob = cpp_encode(
        &cpp_decoded,
        DT_FLOAT,
        width as i32,
        height as i32,
        1,
        1,
        Some(&cpp_valid),
        max_z_err,
    );

    let final_image = lerc::decode(&cpp_blob).unwrap();
    let final_decoded = final_image.as_typed::<f32>().unwrap();
    let final_mask = final_image.mask().unwrap();

    let n = (width * height) as usize;
    for i in 0..n {
        assert_eq!(
            mask.is_valid(i),
            final_mask.is_valid(i),
            "mask mismatch at pixel {i} after roundtrip"
        );
        if mask.is_valid(i) {
            assert_f32_close(
                original[i],
                final_decoded[i],
                2.0 * max_z_err,
                format_args!("pixel {i}"),
            );
        }
    }
}

// ===========================================================================
// Larger images
// ===========================================================================

#[test]
fn rust_encode_cpp_decode_large_f32() {
    let (width, height) = (256, 256);
    let data = make_gradient_f32(width, height);
    let max_z_err = 0.001;

    let blob = lerc::encode_typed(width, height, &data, max_z_err).unwrap();
    let (decoded, _): (Vec<f32>, _) =
        cpp_decode(&blob, DT_FLOAT, width as i32, height as i32, 1, 1);

    for (i, (&o, &d)) in data.iter().zip(decoded.iter()).enumerate() {
        assert_f32_close(o, d, max_z_err, format_args!("pixel {i}"));
    }
}

#[test]
fn cpp_encode_rust_decode_large_f32() {
    let (width, height) = (256, 256);
    let data = make_gradient_f32(width, height);
    let max_z_err = 0.001;

    let blob = cpp_encode(
        &data,
        DT_FLOAT,
        width as i32,
        height as i32,
        1,
        1,
        None,
        max_z_err,
    );

    let image = lerc::decode(&blob).unwrap();
    let decoded = image.as_typed::<f32>().unwrap();

    for (i, (&o, &d)) in data.iter().zip(decoded.iter()).enumerate() {
        assert_f32_close(o, d, max_z_err, format_args!("pixel {i}"));
    }
}

// ===========================================================================
// Edge cases
// ===========================================================================

#[test]
fn rust_encode_cpp_decode_constant_f32() {
    let (width, height) = (32, 32);
    let data = vec![42.5f32; (width * height) as usize];

    let blob = lerc::encode_typed(width, height, &data, 0.0).unwrap();
    let (decoded, _): (Vec<f32>, _) =
        cpp_decode(&blob, DT_FLOAT, width as i32, height as i32, 1, 1);
    assert_eq!(data, decoded);
}

#[test]
fn cpp_encode_rust_decode_constant_f32() {
    let (width, height): (i32, i32) = (32, 32);
    let data = vec![42.5f32; (width * height) as usize];

    let blob = cpp_encode(&data, DT_FLOAT, width, height, 1, 1, None, 0.0);
    let image = lerc::decode(&blob).unwrap();
    assert_eq!(data.as_slice(), image.as_typed::<f32>().unwrap());
}

#[test]
fn rust_encode_cpp_decode_all_zeros_u8() {
    let (width, height) = (32, 32);
    let data = vec![0u8; (width * height) as usize];

    let blob = lerc::encode_typed(width, height, &data, 0.5).unwrap();
    let (decoded, _): (Vec<u8>, _) =
        cpp_decode(&blob, DT_UCHAR, width as i32, height as i32, 1, 1);
    assert_eq!(data, decoded);
}
