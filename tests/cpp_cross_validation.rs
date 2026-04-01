#![cfg(feature = "cpp-validation")]
#![allow(clippy::too_many_arguments)]

use lerc::bitmask::BitMask;
use lerc::{DataType, LercData, LercImage};
use lerc_cpp_ref::{
    self as cpp, DT_CHAR, DT_DOUBLE, DT_FLOAT, DT_INT, DT_SHORT, DT_UCHAR, DT_UINT, DT_USHORT,
    INFO_DATA_TYPE, INFO_N_BANDS, INFO_N_COLS, INFO_N_ROWS, RANGE_MAX_Z_ERR_USED,
};

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

    let (info, _) = cpp::get_blob_info(&blob);
    assert_eq!(info[INFO_DATA_TYPE], DT_FLOAT);
    assert_eq!(info[INFO_N_COLS], width);
    assert_eq!(info[INFO_N_ROWS], height);
    assert_eq!(info[INFO_N_BANDS], 1);

    let (decoded, valid_bytes): (Vec<f32>, _) =
        cpp::decode(&blob, DT_FLOAT, width as i32, height as i32, 1, 1);

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

        let (info, range) = cpp::get_blob_info(&blob);
        assert_eq!(info[INFO_DATA_TYPE], DT_FLOAT);
        assert!(
            range[RANGE_MAX_Z_ERR_USED] <= max_z_err + f64::EPSILON,
            "maxZErrUsed {} > requested {max_z_err}",
            range[RANGE_MAX_Z_ERR_USED]
        );

        let (decoded, _): (Vec<f32>, _) =
            cpp::decode(&blob, DT_FLOAT, width as i32, height as i32, 1, 1);

        for (i, (&o, &d)) in data.iter().zip(decoded.iter()).enumerate() {
            assert_f32_close(
                o,
                d,
                max_z_err,
                format_args!("maxZErr={max_z_err} pixel {i}"),
            );
        }
    }
}

#[test]
fn rust_encode_cpp_decode_f64_near_lossless() {
    let (width, height) = (32, 32);
    let data = make_gradient_f64(width, height);
    let max_z_err = f64::EPSILON;

    let blob = lerc::encode_typed(width, height, &data, max_z_err).unwrap();

    let (info, _) = cpp::get_blob_info(&blob);
    assert_eq!(info[INFO_DATA_TYPE], DT_DOUBLE);

    let (decoded, _): (Vec<f64>, _) =
        cpp::decode(&blob, DT_DOUBLE, width as i32, height as i32, 1, 1);

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
        cpp::decode(&blob, DT_DOUBLE, width as i32, height as i32, 1, 1);

    for (i, (&o, &d)) in data.iter().zip(decoded.iter()).enumerate() {
        assert_f64_close(o, d, max_z_err, format_args!("pixel {i}"));
    }
}

#[test]
fn rust_encode_cpp_decode_u8() {
    let (width, height) = (64, 64);
    let data = make_ramp_u8(width, height);

    let blob = lerc::encode_typed(width, height, &data, 0.5).unwrap();
    let (info, _) = cpp::get_blob_info(&blob);
    assert_eq!(info[INFO_DATA_TYPE], DT_UCHAR);

    let (decoded, _): (Vec<u8>, _) =
        cpp::decode(&blob, DT_UCHAR, width as i32, height as i32, 1, 1);
    assert_eq!(data, decoded);
}

#[test]
fn rust_encode_cpp_decode_i8() {
    let (width, height) = (64, 64);
    let data: Vec<i8> = (0..(width * height) as usize)
        .map(|i| ((i % 256) as i16 - 128) as i8)
        .collect();

    let blob = lerc::encode_typed(width, height, &data, 0.5).unwrap();
    let (info, _) = cpp::get_blob_info(&blob);
    assert_eq!(info[INFO_DATA_TYPE], DT_CHAR);

    let (decoded, _): (Vec<i8>, _) = cpp::decode(&blob, DT_CHAR, width as i32, height as i32, 1, 1);
    assert_eq!(data, decoded);
}

#[test]
fn rust_encode_cpp_decode_i16() {
    let (width, height) = (64, 64);
    let data = make_ramp_i16(width, height);

    let blob = lerc::encode_typed(width, height, &data, 0.0).unwrap();
    let (info, _) = cpp::get_blob_info(&blob);
    assert_eq!(info[INFO_DATA_TYPE], DT_SHORT);

    let (decoded, _): (Vec<i16>, _) =
        cpp::decode(&blob, DT_SHORT, width as i32, height as i32, 1, 1);
    assert_eq!(data, decoded);
}

#[test]
fn rust_encode_cpp_decode_u16() {
    let (width, height) = (64, 64);
    let data = make_ramp_u16(width, height);

    let blob = lerc::encode_typed(width, height, &data, 0.0).unwrap();
    let (info, _) = cpp::get_blob_info(&blob);
    assert_eq!(info[INFO_DATA_TYPE], DT_USHORT);

    let (decoded, _): (Vec<u16>, _) =
        cpp::decode(&blob, DT_USHORT, width as i32, height as i32, 1, 1);
    assert_eq!(data, decoded);
}

#[test]
fn rust_encode_cpp_decode_u32() {
    let (width, height) = (64, 64);
    let data = make_ramp_u32(width, height);

    let blob = lerc::encode_typed(width, height, &data, 0.0).unwrap();
    let (info, _) = cpp::get_blob_info(&blob);
    assert_eq!(info[INFO_DATA_TYPE], DT_UINT);

    let (decoded, _): (Vec<u32>, _) =
        cpp::decode(&blob, DT_UINT, width as i32, height as i32, 1, 1);
    assert_eq!(data, decoded);
}

#[test]
fn rust_encode_cpp_decode_i32() {
    let (width, height) = (64, 64);
    let data = make_ramp_i32(width, height);

    let blob = lerc::encode_typed(width, height, &data, 0.0).unwrap();
    let (info, _) = cpp::get_blob_info(&blob);
    assert_eq!(info[INFO_DATA_TYPE], DT_INT);

    let (decoded, _): (Vec<i32>, _) = cpp::decode(&blob, DT_INT, width as i32, height as i32, 1, 1);
    assert_eq!(data, decoded);
}

#[test]
fn rust_encode_cpp_decode_with_mask() {
    let (width, height) = (64, 48);
    let data = make_gradient_f32(width, height);
    let (mask, valid_bytes) = make_validity_mask(width, height);
    let max_z_err = 0.01;

    let blob = lerc::encode_typed_masked(width, height, &data, &mask, max_z_err).unwrap();

    let (info, _) = cpp::get_blob_info(&blob);
    assert_eq!(info[INFO_DATA_TYPE], DT_FLOAT);

    let (decoded, dec_valid): (Vec<f32>, _) =
        cpp::decode(&blob, DT_FLOAT, width as i32, height as i32, 1, 1);

    let n = (width * height) as usize;
    for i in 0..n {
        assert_eq!(valid_bytes[i], dec_valid[i], "mask mismatch at pixel {i}");
        if valid_bytes[i] != 0 {
            assert_f32_close(
                data[i],
                decoded[i],
                max_z_err,
                format_args!("valid pixel {i}"),
            );
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

    let blob = cpp::encode_for_version(
        &data,
        5,
        DT_FLOAT,
        width as i32,
        height as i32,
        1,
        1,
        None,
        0.0,
    );

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
        let blob = cpp::encode(
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
            assert_f32_close(
                o,
                d,
                max_z_err,
                format_args!("maxZErr={max_z_err} pixel {i}"),
            );
        }
    }
}

#[test]
fn cpp_encode_rust_decode_f64_lossless() {
    let (width, height) = (32, 32);
    let data = make_gradient_f64(width, height);

    let blob = cpp::encode_for_version(
        &data,
        5,
        DT_DOUBLE,
        width as i32,
        height as i32,
        1,
        1,
        None,
        0.0,
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

    let blob = cpp::encode(
        &data,
        DT_UCHAR,
        width as i32,
        height as i32,
        1,
        1,
        None,
        0.5,
    );

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

    let blob = cpp::encode(&data, DT_CHAR, width, height, 1, 1, None, 0.5);

    let image = lerc::decode(&blob).unwrap();
    assert_eq!(image.data_type, DataType::Char);
    assert_eq!(data.as_slice(), image.as_typed::<i8>().unwrap());
}

#[test]
fn cpp_encode_rust_decode_i16() {
    let (width, height) = (64, 64);
    let data = make_ramp_i16(width, height);

    let blob = cpp::encode(
        &data,
        DT_SHORT,
        width as i32,
        height as i32,
        1,
        1,
        None,
        0.0,
    );

    let image = lerc::decode(&blob).unwrap();
    assert_eq!(image.data_type, DataType::Short);
    assert_eq!(data.as_slice(), image.as_typed::<i16>().unwrap());
}

#[test]
fn cpp_encode_rust_decode_u16() {
    let (width, height) = (64, 64);
    let data = make_ramp_u16(width, height);

    let blob = cpp::encode(
        &data,
        DT_USHORT,
        width as i32,
        height as i32,
        1,
        1,
        None,
        0.0,
    );

    let image = lerc::decode(&blob).unwrap();
    assert_eq!(image.data_type, DataType::UShort);
    assert_eq!(data.as_slice(), image.as_typed::<u16>().unwrap());
}

#[test]
fn cpp_encode_rust_decode_u32() {
    let (width, height) = (64, 64);
    let data = make_ramp_u32(width, height);

    let blob = cpp::encode(&data, DT_UINT, width as i32, height as i32, 1, 1, None, 0.0);

    let image = lerc::decode(&blob).unwrap();
    assert_eq!(image.data_type, DataType::UInt);
    assert_eq!(data.as_slice(), image.as_typed::<u32>().unwrap());
}

#[test]
fn cpp_encode_rust_decode_i32() {
    let (width, height) = (64, 64);
    let data = make_ramp_i32(width, height);

    let blob = cpp::encode(&data, DT_INT, width as i32, height as i32, 1, 1, None, 0.0);

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

    let blob = cpp::encode(
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
            assert_f32_close(
                data[i],
                decoded[i],
                max_z_err,
                format_args!("valid pixel {i}"),
            );
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

    let masks: Vec<_> = (0..n_bands).map(|_| BitMask::all_valid(ppb)).collect();

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

    let (info, _) = cpp::get_blob_info(&blob);
    assert_eq!(info[INFO_DATA_TYPE], DT_FLOAT);
    assert_eq!(info[INFO_N_COLS], width);
    assert_eq!(info[INFO_N_ROWS], height);
    assert_eq!(info[INFO_N_BANDS], n_bands);

    let (decoded, _): (Vec<f32>, _) = cpp::decode(
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

    let blob = cpp::encode(&data, DT_UCHAR, w, h, 1, nb, None, 0.5);

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
        cpp::decode(&rust_blob, DT_FLOAT, width as i32, height as i32, 1, 1);
    let cpp_blob = cpp::encode(
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

    let cpp_blob = cpp::encode(
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
        cpp::decode(&rust_blob, DT_FLOAT, width as i32, height as i32, 1, 1);

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
        cpp::decode(&rust_blob, DT_UCHAR, width as i32, height as i32, 1, 1);
    assert_eq!(original, cpp_decoded);

    let cpp_blob = cpp::encode(
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

    let cpp_blob = cpp::encode(
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
        cpp::decode(&rust_blob, DT_UCHAR, width as i32, height as i32, 1, 1);
    assert_eq!(original, final_decoded);
}

#[test]
fn roundtrip_with_mask_f32() {
    let (width, height) = (64, 48);
    let original = make_gradient_f32(width, height);
    let (mask, valid_bytes) = make_validity_mask(width, height);
    let max_z_err = 0.01;

    let rust_blob = lerc::encode_typed_masked(width, height, &original, &mask, max_z_err).unwrap();
    let (cpp_decoded, cpp_valid): (Vec<f32>, _) =
        cpp::decode(&rust_blob, DT_FLOAT, width as i32, height as i32, 1, 1);

    for i in 0..(width * height) as usize {
        assert_eq!(valid_bytes[i], cpp_valid[i], "mask mismatch at pixel {i}");
    }

    let cpp_blob = cpp::encode(
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
        cpp::decode(&blob, DT_FLOAT, width as i32, height as i32, 1, 1);

    for (i, (&o, &d)) in data.iter().zip(decoded.iter()).enumerate() {
        assert_f32_close(o, d, max_z_err, format_args!("pixel {i}"));
    }
}

#[test]
fn cpp_encode_rust_decode_large_f32() {
    let (width, height) = (256, 256);
    let data = make_gradient_f32(width, height);
    let max_z_err = 0.001;

    let blob = cpp::encode(
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
        cpp::decode(&blob, DT_FLOAT, width as i32, height as i32, 1, 1);
    assert_eq!(data, decoded);
}

#[test]
fn cpp_encode_rust_decode_constant_f32() {
    let (width, height): (i32, i32) = (32, 32);
    let data = vec![42.5f32; (width * height) as usize];

    let blob = cpp::encode(&data, DT_FLOAT, width, height, 1, 1, None, 0.0);
    let image = lerc::decode(&blob).unwrap();
    assert_eq!(data.as_slice(), image.as_typed::<f32>().unwrap());
}

#[test]
fn rust_encode_cpp_decode_all_zeros_u8() {
    let (width, height) = (32, 32);
    let data = vec![0u8; (width * height) as usize];

    let blob = lerc::encode_typed(width, height, &data, 0.5).unwrap();
    let (decoded, _): (Vec<u8>, _) =
        cpp::decode(&blob, DT_UCHAR, width as i32, height as i32, 1, 1);
    assert_eq!(data, decoded);
}

// ===========================================================================
// Multi-depth lossy cross-validation
// ===========================================================================

#[test]
fn rust_encode_cpp_decode_multi_depth_f32_lossy() {
    let (width, height) = (64u32, 64u32);
    let n_depth = 3u32;
    let max_z_err = 0.01;
    let num_pixels = (width * height) as usize;

    // Correlated multi-depth data: depth[1] ~ depth[0] * 1.01
    let mut data = vec![0.0f32; num_pixels * n_depth as usize];
    for i in 0..num_pixels {
        let base = 100.0 + (i as f32) * 0.37 + ((i as f32) * 0.1).sin() * 10.0;
        let noise1 = ((i * 7 + 3) % 13) as f32 * 0.001;
        let noise2 = ((i * 11 + 5) % 17) as f32 * 0.001;
        data[i * 3] = base;
        data[i * 3 + 1] = base * 1.01 + noise1;
        data[i * 3 + 2] = base * 1.02 + noise2;
    }

    let image = LercImage {
        width,
        height,
        n_depth,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![BitMask::all_valid(num_pixels)],
        data: LercData::F32(data.clone()),
        no_data_value: None,
    };

    let blob = lerc::encode(&image, max_z_err).unwrap();

    // C++ decoder should accept our blob
    let (decoded, _): (Vec<f32>, _) = cpp::decode(
        &blob,
        DT_FLOAT,
        width as i32,
        height as i32,
        n_depth as i32,
        1,
    );

    assert_eq!(decoded.len(), data.len());
    for (i, (&o, &d)) in data.iter().zip(decoded.iter()).enumerate() {
        assert_f32_close(o, d, max_z_err, format_args!("multi-depth lossy pixel {i}"));
    }
}

#[test]
fn cpp_encode_rust_decode_multi_depth_f32_lossy() {
    let (width, height) = (64u32, 64u32);
    let n_depth = 3i32;
    let max_z_err = 0.01;
    let num_pixels = (width * height) as usize;

    let mut data = vec![0.0f32; num_pixels * n_depth as usize];
    for i in 0..num_pixels {
        let base = 100.0 + (i as f32) * 0.37 + ((i as f32) * 0.1).sin() * 10.0;
        data[i * 3] = base;
        data[i * 3 + 1] = base * 1.01;
        data[i * 3 + 2] = base * 1.02;
    }

    let blob = cpp::encode(
        &data,
        DT_FLOAT,
        width as i32,
        height as i32,
        n_depth,
        1,
        None,
        max_z_err,
    );

    let image = lerc::decode(&blob).unwrap();
    assert_eq!(image.n_depth, n_depth as u32);
    let decoded = image.as_typed::<f32>().unwrap();

    for (i, (&o, &d)) in data.iter().zip(decoded.iter()).enumerate() {
        assert_f32_close(
            o,
            d,
            max_z_err,
            format_args!("cpp multi-depth lossy pixel {i}"),
        );
    }
}

#[test]
fn roundtrip_rust_cpp_multi_depth_f32_lossy() {
    // Rust encode -> C++ decode -> C++ encode -> Rust decode
    let (width, height) = (32u32, 32u32);
    let n_depth = 3u32;
    let max_z_err = 0.05;
    let num_pixels = (width * height) as usize;

    let mut data = vec![0.0f32; num_pixels * n_depth as usize];
    for i in 0..num_pixels {
        let base = 50.0 + (i as f32) * 0.5;
        data[i * 3] = base;
        data[i * 3 + 1] = base + 1.0;
        data[i * 3 + 2] = base + 2.0;
    }

    let image = LercImage {
        width,
        height,
        n_depth,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![BitMask::all_valid(num_pixels)],
        data: LercData::F32(data.clone()),
        no_data_value: None,
    };

    // Rust encode
    let rust_blob = lerc::encode(&image, max_z_err).unwrap();

    // C++ decode
    let (cpp_decoded, _): (Vec<f32>, _) = cpp::decode(
        &rust_blob,
        DT_FLOAT,
        width as i32,
        height as i32,
        n_depth as i32,
        1,
    );

    // C++ re-encode
    let cpp_blob = cpp::encode(
        &cpp_decoded,
        DT_FLOAT,
        width as i32,
        height as i32,
        n_depth as i32,
        1,
        None,
        max_z_err,
    );

    // Rust decode
    let final_image = lerc::decode(&cpp_blob).unwrap();
    let final_decoded = final_image.as_typed::<f32>().unwrap();

    for (i, (&o, &d)) in data.iter().zip(final_decoded.iter()).enumerate() {
        assert_f32_close(
            o,
            d,
            2.0 * max_z_err,
            format_args!("roundtrip multi-depth pixel {i}"),
        );
    }
}

// ===========================================================================
// FPL path cross-validation (maxZErr=0 triggers FPL for float/double)
// ===========================================================================

#[test]
fn rust_encode_cpp_decode_f32_lossless_fpl() {
    let (width, height) = (64, 48);
    let data = make_gradient_f32(width, height);

    // maxZErr=0 triggers the FPL (floating-point lossless) path
    let blob = lerc::encode_typed(width, height, &data, 0.0).unwrap();

    let (info, _) = cpp::get_blob_info(&blob);
    assert_eq!(info[INFO_DATA_TYPE], DT_FLOAT);
    assert_eq!(info[INFO_N_COLS], width);
    assert_eq!(info[INFO_N_ROWS], height);

    let (decoded, valid_bytes): (Vec<f32>, _) =
        cpp::decode(&blob, DT_FLOAT, width as i32, height as i32, 1, 1);

    assert!(valid_bytes.iter().all(|&v| v == 1));
    for (i, (&o, &d)) in data.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(o.to_bits(), d.to_bits(), "pixel {i}: {o} vs {d}");
    }
}

#[test]
fn cpp_encode_rust_decode_f32_lossless_fpl() {
    let (width, height) = (64, 48);
    let data = make_gradient_f32(width, height);

    // Use cpp::encode (not encode_for_version with v5) to get the actual FPL path
    let blob = cpp::encode(
        &data,
        DT_FLOAT,
        width as i32,
        height as i32,
        1,
        1,
        None,
        0.0,
    );

    let image = lerc::decode(&blob).unwrap();
    assert_eq!(image.width, width as u32);
    assert_eq!(image.height, height as u32);
    assert_eq!(image.data_type, DataType::Float);

    let decoded = image.as_typed::<f32>().unwrap();
    for (i, (&o, &d)) in data.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(o.to_bits(), d.to_bits(), "pixel {i}: {o} vs {d}");
    }
}

#[test]
fn rust_encode_cpp_decode_f64_lossless_fpl() {
    let (width, height) = (32, 32);
    let data = make_gradient_f64(width, height);

    let blob = lerc::encode_typed(width, height, &data, 0.0).unwrap();

    let (info, _) = cpp::get_blob_info(&blob);
    assert_eq!(info[INFO_DATA_TYPE], DT_DOUBLE);

    let (decoded, valid_bytes): (Vec<f64>, _) =
        cpp::decode(&blob, DT_DOUBLE, width as i32, height as i32, 1, 1);

    assert!(valid_bytes.iter().all(|&v| v == 1));
    for (i, (&o, &d)) in data.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(o.to_bits(), d.to_bits(), "pixel {i}: {o} vs {d}");
    }
}

#[test]
fn cpp_encode_rust_decode_f64_lossless_fpl() {
    let (width, height) = (32, 32);
    let data = make_gradient_f64(width, height);

    let blob = cpp::encode(
        &data,
        DT_DOUBLE,
        width as i32,
        height as i32,
        1,
        1,
        None,
        0.0,
    );

    let image = lerc::decode(&blob).unwrap();
    assert_eq!(image.data_type, DataType::Double);

    let decoded = image.as_typed::<f64>().unwrap();
    for (i, (&o, &d)) in data.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(o.to_bits(), d.to_bits(), "pixel {i}: {o} vs {d}");
    }
}

#[test]
fn fpl_cross_validation_constant_byte_plane() {
    // Data crafted so all values share the same exponent byte, which triggers
    // RLE encoding on that byte plane. This specifically exercises the RLE
    // byte-order fix (value before count, matching C++ format).
    let (width, height) = (64, 64);

    // All values are in [1.0, 2.0), so they share the same IEEE 754 exponent
    // byte (0x3F for the high byte). This guarantees at least one byte plane
    // is constant and will be RLE-encoded.
    let data: Vec<f32> = (0..(width * height) as usize)
        .map(|i| 1.0 + (i as f32) / ((width * height) as f32))
        .collect();

    // Rust encode -> C++ decode
    let rust_blob = lerc::encode_typed(width, height, &data, 0.0).unwrap();
    let (cpp_decoded, _): (Vec<f32>, _) =
        cpp::decode(&rust_blob, DT_FLOAT, width as i32, height as i32, 1, 1);

    for (i, (&o, &d)) in data.iter().zip(cpp_decoded.iter()).enumerate() {
        assert_eq!(
            o.to_bits(),
            d.to_bits(),
            "rust->cpp constant-plane pixel {i}: {o} vs {d}"
        );
    }

    // C++ encode -> Rust decode
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

    let image = lerc::decode(&cpp_blob).unwrap();
    let rust_decoded = image.as_typed::<f32>().unwrap();

    for (i, (&o, &d)) in data.iter().zip(rust_decoded.iter()).enumerate() {
        assert_eq!(
            o.to_bits(),
            d.to_bits(),
            "cpp->rust constant-plane pixel {i}: {o} vs {d}"
        );
    }
}
