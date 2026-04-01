//! Cross-validation tests verifying interoperability between the Rust
//! LERC encoder/decoder and the C++ reference implementation.
//!
//! These tests exercise:
//! - Detailed pixel-level validation of reference file decodes
//! - Round-trip re-encoding of reference data
//! - Byte-level header and structural validation of our encoder output
//! - Deterministic encoding and self-consistency

use lerc::Precision;
use lerc::bitmask::BitMask;
use lerc::{DataType, LercImage, SampleData};

// ---------------------------------------------------------------------------
// Reference test data from the C++ LERC SDK
// ---------------------------------------------------------------------------

static CALIFORNIA: &[u8] = include_bytes!("../esri-lerc/testData/california_400_400_1_float.lerc2");
static BLUEMARBLE: &[u8] = include_bytes!("../esri-lerc/testData/bluemarble_256_256_3_byte.lerc2");

// ---------------------------------------------------------------------------
// Minimal header parser (duplicates the crate-internal logic so we can
// inspect raw blobs from integration tests without making internals public)
// ---------------------------------------------------------------------------

/// Parsed LERC2 header fields needed for validation.
#[derive(Debug)]
struct RawHeader {
    version: i32,
    checksum: u32,
    n_rows: i32,
    n_cols: i32,
    n_depth: i32,
    num_valid_pixel: i32,
    micro_block_size: i32,
    blob_size: i32,
    data_type: i32,
    n_blobs_more: i32,
    max_z_error: f64,
    z_min: f64,
    z_max: f64,
}

fn read_le_i32(data: &[u8], off: usize) -> i32 {
    i32::from_le_bytes(data[off..off + 4].try_into().unwrap())
}

fn read_le_u32(data: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(data[off..off + 4].try_into().unwrap())
}

fn read_le_f64(data: &[u8], off: usize) -> f64 {
    f64::from_le_bytes(data[off..off + 8].try_into().unwrap())
}

/// Parse a LERC2 header from raw bytes (version-aware).  Panics on anything unexpected.
fn parse_header(data: &[u8]) -> RawHeader {
    assert!(data.len() >= 58, "blob too small for LERC2 header");
    assert_eq!(&data[0..6], b"Lerc2 ", "bad magic");

    let version = read_le_i32(data, 6);
    assert!((2..=6).contains(&version), "unsupported version {version}");

    let mut off = 10;

    let checksum = if version >= 3 {
        let c = read_le_u32(data, off);
        off += 4;
        c
    } else {
        0
    };

    let n_rows = read_le_i32(data, off);
    off += 4;
    let n_cols = read_le_i32(data, off);
    off += 4;

    let n_depth = if version >= 4 {
        let d = read_le_i32(data, off);
        off += 4;
        d
    } else {
        1
    };

    let num_valid_pixel = read_le_i32(data, off);
    off += 4;
    let micro_block_size = read_le_i32(data, off);
    off += 4;
    let blob_size = read_le_i32(data, off);
    off += 4;
    let data_type = read_le_i32(data, off);
    off += 4;

    let n_blobs_more = if version >= 6 {
        let nbm = read_le_i32(data, off);
        off += 4;
        off += 4; // 4 flag bytes
        nbm
    } else {
        0
    };

    let max_z_error = read_le_f64(data, off);
    off += 8;
    let z_min = read_le_f64(data, off);
    off += 8;
    let z_max = read_le_f64(data, off);
    let _ = off + 8; // suppress unused-assignment warning

    RawHeader {
        version,
        checksum,
        n_rows,
        n_cols,
        n_depth,
        num_valid_pixel,
        micro_block_size,
        blob_size,
        data_type,
        n_blobs_more,
        max_z_error,
        z_min,
        z_max,
    }
}

/// Compute Fletcher-32 checksum matching the LERC C++ implementation.
fn fletcher32(data: &[u8]) -> u32 {
    let mut sum1: u32 = 0xffff;
    let mut sum2: u32 = 0xffff;

    let mut i = 0;
    let len = data.len();

    while i < len.saturating_sub(1) {
        let batch_end = (i + 359 * 2).min(len.saturating_sub(1));
        while i < batch_end {
            sum1 += (data[i] as u32) << 8;
            sum1 += data[i + 1] as u32;
            sum2 += sum1;
            i += 2;
        }
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }

    if i < len {
        sum1 += (data[i] as u32) << 8;
        sum2 += sum1;
    }

    sum1 = (sum1 & 0xffff) + (sum1 >> 16);
    sum2 = (sum2 & 0xffff) + (sum2 >> 16);

    (sum2 << 16) | sum1
}

/// Validate raw header structural properties.  Panics with a descriptive
/// message on any inconsistency.
fn validate_header(blob: &[u8], expected_dt: i32, expected_rows: i32, expected_cols: i32) {
    let h = parse_header(blob);

    assert!(
        (3..=6).contains(&h.version),
        "expected encoder version 3-6, got {}",
        h.version
    );
    assert_eq!(h.n_rows, expected_rows);
    assert_eq!(h.n_cols, expected_cols);
    assert_eq!(h.data_type, expected_dt);
    assert!(
        h.micro_block_size == 8 || h.micro_block_size == 16,
        "micro_block_size must be 8 or 16, got {}",
        h.micro_block_size
    );
    assert!(
        h.blob_size as usize <= blob.len(),
        "blobSize {} exceeds actual blob length {}",
        h.blob_size,
        blob.len()
    );
    assert!(h.max_z_error >= 0.0, "maxZError must be non-negative");
    assert!(
        h.z_min <= h.z_max,
        "zMin ({}) > zMax ({})",
        h.z_min,
        h.z_max
    );

    // Verify checksum
    let computed = fletcher32(&blob[14..h.blob_size as usize]);
    assert_eq!(
        computed, h.checksum,
        "checksum mismatch: computed {computed:#010x}, header {:#010x}",
        h.checksum
    );
}

// =========================================================================
// 1. Detailed decode of C++ reference files
// =========================================================================

#[test]
fn california_detailed_pixel_statistics() {
    let image = lerc::decode(CALIFORNIA).expect("decode california failed");
    let info = lerc::decode_info(CALIFORNIA).expect("decode_info failed");

    assert_eq!(image.width, 400);
    assert_eq!(image.height, 400);
    assert_eq!(image.data_type, DataType::Float);
    assert_eq!(image.n_bands, 1);
    assert_eq!(image.n_depth, 1);

    let pixels = match &image.data {
        SampleData::F32(p) => p,
        _ => panic!("expected F32 data"),
    };

    let mask = &image.valid_masks[0];
    let total = 400 * 400;

    // Gather statistics over valid pixels
    let mut sum: f64 = 0.0;
    let mut sum_sq: f64 = 0.0;
    let mut valid_count: usize = 0;
    let mut invalid_count: usize = 0;
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;

    for (k, &v) in pixels.iter().enumerate().take(total) {
        if mask.is_valid(k) {
            assert!(!v.is_nan(), "valid pixel {k} is NaN");
            assert!(v.is_finite(), "valid pixel {k} is not finite");
            sum += v as f64;
            sum_sq += (v as f64) * (v as f64);
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
            valid_count += 1;
        } else {
            invalid_count += 1;
        }
    }

    assert_eq!(valid_count as u32, info.num_valid_pixels);
    assert!(valid_count > 0, "no valid pixels");
    assert!(invalid_count > 0, "expected some invalid (ocean) pixels");

    // California elevation: expect mix of sea-level/low and mountains
    let mean = sum / valid_count as f64;
    let variance = sum_sq / valid_count as f64 - mean * mean;
    assert!(
        mean > -500.0 && mean < 5000.0,
        "mean elevation {mean} outside expected range for California"
    );
    assert!(
        variance > 0.0,
        "variance should be positive for elevation data, got {variance}"
    );

    // z range should match header within tolerance
    let tolerance = info.max_z_error;
    assert!(
        min_val as f64 >= info.z_min - tolerance,
        "min_val {min_val} below z_min {} - tolerance {tolerance}",
        info.z_min
    );
    assert!(
        max_val as f64 <= info.z_max + tolerance,
        "max_val {max_val} above z_max {} + tolerance {tolerance}",
        info.z_max
    );
}

#[test]
fn california_invalid_pixels_are_ocean() {
    let image = lerc::decode(CALIFORNIA).expect("decode failed");
    let mask = &image.valid_masks[0];
    let total = 400 * 400;

    // Count invalid pixels grouped by row to verify they form contiguous
    // regions (ocean along the coast, not scattered noise).
    let mut invalid_rows = 0u32;
    for row in 0..400 {
        let row_invalid: usize = (0..400)
            .filter(|&col| !mask.is_valid(row * 400 + col))
            .count();
        if row_invalid > 0 {
            invalid_rows += 1;
        }
    }

    let invalid_count = total - mask.count_valid();
    assert!(
        invalid_count > 100,
        "expected substantial ocean area, only got {invalid_count} invalid pixels"
    );
    // Invalid pixels should span multiple contiguous rows (coast)
    assert!(
        invalid_rows > 10,
        "expected invalid pixels across many rows (ocean), only found {invalid_rows} rows with invalids"
    );
}

#[test]
fn bluemarble_three_bands_detailed() {
    let image = lerc::decode(BLUEMARBLE).expect("decode bluemarble failed");
    let info = lerc::decode_info(BLUEMARBLE).expect("decode_info failed");

    assert_eq!(image.width, 256);
    assert_eq!(image.height, 256);
    assert_eq!(image.n_bands, 3);
    assert_eq!(image.n_depth, 1);
    assert_eq!(image.data_type, DataType::Byte);

    let pixels = match &image.data {
        SampleData::U8(p) => p,
        _ => panic!("expected U8 data"),
    };

    let band_size = 256 * 256;
    assert_eq!(pixels.len(), band_size * 3);

    // Verify each band has different statistical properties (R/G/B are distinct)
    let mut band_sums = [0u64; 3];
    let mut band_nonzero = [0usize; 3];
    let mut band_histograms = [[0u32; 256]; 3];

    for band in 0..3 {
        let offset = band * band_size;
        for k in 0..band_size {
            let v = pixels[offset + k];
            band_sums[band] += v as u64;
            if v > 0 {
                band_nonzero[band] += 1;
            }
            band_histograms[band][v as usize] += 1;
        }
    }

    // Each band should have nonzero pixels
    for (band, &count) in band_nonzero.iter().enumerate() {
        assert!(count > 0, "band {band} has no nonzero pixels");
    }

    // Bands should not all have the same mean (they represent R, G, B)
    let band_means: Vec<f64> = band_sums
        .iter()
        .map(|&s| s as f64 / band_size as f64)
        .collect();
    let all_same_mean =
        (band_means[0] - band_means[1]).abs() < 1.0 && (band_means[1] - band_means[2]).abs() < 1.0;
    assert!(
        !all_same_mean,
        "all three bands have nearly identical means ({:?}), expected different R/G/B channels",
        band_means
    );

    // Valid pixel count per band: from decode_info (first blob header)
    assert!(
        info.num_valid_pixels > 0 && info.num_valid_pixels <= (256 * 256) as u32,
        "num_valid_pixels {} out of expected range",
        info.num_valid_pixels
    );

    // Verify the validity mask from the decoded image
    let mask = &image.valid_masks[0];
    let valid_count = mask.count_valid();
    assert_eq!(
        valid_count, info.num_valid_pixels as usize,
        "mask valid count does not match header"
    );
}

// =========================================================================
// 2. Round-trip re-encoding of reference data
// =========================================================================

#[test]
fn california_round_trip_lossy() {
    let image = lerc::decode(CALIFORNIA).expect("decode failed");
    let info = lerc::decode_info(CALIFORNIA).expect("decode_info failed");
    let max_z_error = info.max_z_error;

    let pixels_orig = match &image.data {
        SampleData::F32(p) => p.clone(),
        _ => panic!("expected F32"),
    };
    let mask_orig = image.valid_masks[0].clone();

    // Re-encode with the same maxZError
    let reencoded =
        lerc::encode(&image, Precision::Tolerance(max_z_error)).expect("re-encode failed");

    // Decode the re-encoded blob
    let image2 = lerc::decode(&reencoded).expect("decode re-encoded failed");

    assert_eq!(image2.width, 400);
    assert_eq!(image2.height, 400);
    assert_eq!(image2.data_type, DataType::Float);

    let pixels2 = match &image2.data {
        SampleData::F32(p) => p,
        _ => panic!("expected F32"),
    };
    let mask2 = &image2.valid_masks[0];

    // Masks must match exactly
    for k in 0..400 * 400 {
        assert_eq!(
            mask_orig.is_valid(k),
            mask2.is_valid(k),
            "mask mismatch at pixel {k}"
        );
    }

    // Valid pixel values within 2*maxZError (one encode/decode cycle each way)
    let tolerance = 2.0 * max_z_error;
    for k in 0..400 * 400 {
        if mask_orig.is_valid(k) {
            let diff = (pixels_orig[k] - pixels2[k]).abs() as f64;
            assert!(
                diff <= tolerance,
                "pixel {k}: orig={}, reencoded={}, diff={diff} > 2*maxZError={tolerance}",
                pixels_orig[k],
                pixels2[k]
            );
        }
    }
}

#[test]
fn bluemarble_round_trip_lossless() {
    let image = lerc::decode(BLUEMARBLE).expect("decode failed");

    let pixels_orig = match &image.data {
        SampleData::U8(p) => p.clone(),
        _ => panic!("expected U8"),
    };

    // Re-encode lossless (maxZError = 0 for byte data means lossless;
    // for integer types 0.5 is also lossless)
    let reencoded = lerc::encode(&image, Precision::Lossless).expect("re-encode failed");

    let image2 = lerc::decode(&reencoded).expect("decode re-encoded failed");

    assert_eq!(image2.width, 256);
    assert_eq!(image2.height, 256);
    assert_eq!(image2.n_bands, 3);
    assert_eq!(image2.data_type, DataType::Byte);

    let pixels2 = match &image2.data {
        SampleData::U8(p) => p,
        _ => panic!("expected U8"),
    };

    assert_eq!(
        pixels_orig, *pixels2,
        "lossless byte round-trip should produce exact match"
    );
}

// =========================================================================
// 3. Structural / byte-level header validation
// =========================================================================

#[test]
fn reference_california_header_valid() {
    let h = parse_header(CALIFORNIA);
    assert_eq!(&CALIFORNIA[0..6], b"Lerc2 ");
    assert!(h.version >= 2 && h.version <= 6);
    assert_eq!(h.n_rows, 400);
    assert_eq!(h.n_cols, 400);
    assert_eq!(h.data_type, 6); // Float
    assert!(h.blob_size > 0);
    assert!(h.max_z_error >= 0.0);

    // Verify checksum of the reference file
    if h.version >= 3 {
        let computed = fletcher32(&CALIFORNIA[14..h.blob_size as usize]);
        assert_eq!(
            computed, h.checksum,
            "reference file checksum mismatch: computed {computed:#010x}, header {:#010x}",
            h.checksum
        );
    }
}

#[test]
fn reference_bluemarble_header_valid() {
    // Bluemarble is 3-band: concatenated blobs with n_blobs_more = 2 on the first
    let h = parse_header(BLUEMARBLE);
    assert_eq!(&BLUEMARBLE[0..6], b"Lerc2 ");
    assert!(h.version >= 2 && h.version <= 6);
    assert_eq!(h.n_rows, 256);
    assert_eq!(h.n_cols, 256);
    assert_eq!(h.data_type, 1); // Byte

    if h.version >= 3 {
        let computed = fletcher32(&BLUEMARBLE[14..h.blob_size as usize]);
        assert_eq!(
            computed, h.checksum,
            "reference file checksum mismatch: computed {computed:#010x}, header {:#010x}",
            h.checksum
        );
    }
}

#[test]
fn encoded_single_band_header_validation() {
    let width = 64u32;
    let height = 48u32;
    let pixels: Vec<f32> = (0..width * height)
        .map(|i| (i as f32) * 0.1 + 100.0)
        .collect();

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: SampleData::F32(pixels),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Tolerance(0.01)).expect("encode failed");
    validate_header(&encoded, 6, height as i32, width as i32);

    let h = parse_header(&encoded);
    assert_eq!(h.n_depth, 1);
    assert_eq!(h.num_valid_pixel, (width * height) as i32);

    // blobSize must match total encoded length (single band = single blob)
    assert_eq!(
        h.blob_size as usize,
        encoded.len(),
        "blobSize field does not match actual blob length"
    );
}

#[test]
fn encoded_multiband_three_concatenated_blobs() {
    let width = 32u32;
    let height = 32u32;
    let band_size = (width * height) as usize;
    let n_bands = 3u32;

    let pixels: Vec<u8> = (0..band_size * n_bands as usize)
        .map(|i| ((i * 37) % 256) as u8)
        .collect();

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands,
        data_type: DataType::Byte,
        valid_masks: vec![BitMask::all_valid(band_size)],
        data: SampleData::U8(pixels),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Lossless).expect("encode failed");

    // Walk through the concatenated blobs
    let mut offset = 0usize;
    let mut blobs_found = 0u32;

    while offset < encoded.len() {
        assert!(
            encoded.len() - offset >= 90,
            "not enough bytes for header at offset {offset}"
        );
        assert_eq!(
            &encoded[offset..offset + 6],
            b"Lerc2 ",
            "bad magic at blob {blobs_found}, offset {offset}"
        );

        let h = parse_header(&encoded[offset..]);

        // Validate each blob's checksum independently
        let blob_bytes = &encoded[offset..offset + h.blob_size as usize];
        if h.version >= 3 {
            let computed = fletcher32(&blob_bytes[14..]);
            assert_eq!(
                computed, h.checksum,
                "checksum mismatch in blob {blobs_found}"
            );
        }

        assert_eq!(h.n_rows, height as i32);
        assert_eq!(h.n_cols, width as i32);
        assert_eq!(h.data_type, 1); // Byte

        // Each blob's n_blobs_more should count the remaining bands
        let expected_more = (n_bands - 1 - blobs_found) as i32;
        assert_eq!(
            h.n_blobs_more, expected_more,
            "blob {blobs_found}: n_blobs_more should be {expected_more}, got {}",
            h.n_blobs_more
        );

        offset += h.blob_size as usize;
        blobs_found += 1;
    }

    assert_eq!(
        blobs_found, n_bands,
        "expected {n_bands} blobs, found {blobs_found}"
    );
    assert_eq!(
        offset,
        encoded.len(),
        "total blob sizes should sum to encoded length"
    );
}

#[test]
fn encoded_blob_size_matches_header() {
    // Verify for several data types that blobSize == actual blob length
    let width = 24u32;
    let height = 24u32;
    let n = (width * height) as usize;

    // f32
    {
        let pixels: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let img = LercImage {
            width,
            height,
            n_depth: 1,
            n_bands: 1,
            data_type: DataType::Float,
            valid_masks: vec![BitMask::all_valid(n)],
            data: SampleData::F32(pixels),
            no_data_value: None,
        };
        let enc = lerc::encode(&img, Precision::Lossless).unwrap();
        let h = parse_header(&enc);
        assert_eq!(h.blob_size as usize, enc.len());
    }

    // u8
    {
        let pixels: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
        let img = LercImage {
            width,
            height,
            n_depth: 1,
            n_bands: 1,
            data_type: DataType::Byte,
            valid_masks: vec![BitMask::all_valid(n)],
            data: SampleData::U8(pixels),
            no_data_value: None,
        };
        let enc = lerc::encode(&img, Precision::Lossless).unwrap();
        let h = parse_header(&enc);
        assert_eq!(h.blob_size as usize, enc.len());
    }

    // i32
    {
        let pixels: Vec<i32> = (0..n).map(|i| i as i32 - 300).collect();
        let img = LercImage {
            width,
            height,
            n_depth: 1,
            n_bands: 1,
            data_type: DataType::Int,
            valid_masks: vec![BitMask::all_valid(n)],
            data: SampleData::I32(pixels),
            no_data_value: None,
        };
        let enc = lerc::encode(&img, Precision::Lossless).unwrap();
        let h = parse_header(&enc);
        assert_eq!(h.blob_size as usize, enc.len());
    }

    // f64
    {
        let pixels: Vec<f64> = (0..n).map(|i| i as f64 * 0.001).collect();
        let img = LercImage {
            width,
            height,
            n_depth: 1,
            n_bands: 1,
            data_type: DataType::Double,
            valid_masks: vec![BitMask::all_valid(n)],
            data: SampleData::F64(pixels),
            no_data_value: None,
        };
        let enc = lerc::encode(&img, Precision::Lossless).unwrap();
        let h = parse_header(&enc);
        assert_eq!(h.blob_size as usize, enc.len());
    }
}

#[test]
fn encoded_checksum_verified_manually() {
    let width = 50u32;
    let height = 50u32;
    let n = (width * height) as usize;
    let pixels: Vec<f32> = (0..n).map(|i| (i as f32).sqrt()).collect();

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![BitMask::all_valid(n)],
        data: SampleData::F32(pixels),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Tolerance(0.001)).expect("encode failed");
    let h = parse_header(&encoded);

    // Manually compute checksum over everything after the checksum field (offset 14)
    let checksum_data = &encoded[14..h.blob_size as usize];
    let computed = fletcher32(checksum_data);

    assert_eq!(
        computed, h.checksum,
        "manually computed Fletcher32 ({computed:#010x}) != header checksum ({:#010x})",
        h.checksum
    );
}

#[test]
fn encoded_data_type_codes_correct() {
    let width = 8u32;
    let height = 8u32;
    let n = (width * height) as usize;

    let cases: Vec<(LercImage, i32)> = vec![
        (
            LercImage {
                width,
                height,
                n_depth: 1,
                n_bands: 1,
                data_type: DataType::Byte,
                valid_masks: vec![BitMask::all_valid(n)],
                data: SampleData::U8(vec![0u8; n]),
                no_data_value: None,
            },
            1, // Byte
        ),
        (
            LercImage {
                width,
                height,
                n_depth: 1,
                n_bands: 1,
                data_type: DataType::Short,
                valid_masks: vec![BitMask::all_valid(n)],
                data: SampleData::I16(vec![0i16; n]),
                no_data_value: None,
            },
            2, // Short
        ),
        (
            LercImage {
                width,
                height,
                n_depth: 1,
                n_bands: 1,
                data_type: DataType::UShort,
                valid_masks: vec![BitMask::all_valid(n)],
                data: SampleData::U16(vec![0u16; n]),
                no_data_value: None,
            },
            3, // UShort
        ),
        (
            LercImage {
                width,
                height,
                n_depth: 1,
                n_bands: 1,
                data_type: DataType::Int,
                valid_masks: vec![BitMask::all_valid(n)],
                data: SampleData::I32(vec![0i32; n]),
                no_data_value: None,
            },
            4, // Int
        ),
        (
            LercImage {
                width,
                height,
                n_depth: 1,
                n_bands: 1,
                data_type: DataType::UInt,
                valid_masks: vec![BitMask::all_valid(n)],
                data: SampleData::U32(vec![0u32; n]),
                no_data_value: None,
            },
            5, // UInt
        ),
        (
            LercImage {
                width,
                height,
                n_depth: 1,
                n_bands: 1,
                data_type: DataType::Float,
                valid_masks: vec![BitMask::all_valid(n)],
                data: SampleData::F32(vec![0.0f32; n]),
                no_data_value: None,
            },
            6, // Float
        ),
        (
            LercImage {
                width,
                height,
                n_depth: 1,
                n_bands: 1,
                data_type: DataType::Double,
                valid_masks: vec![BitMask::all_valid(n)],
                data: SampleData::F64(vec![0.0f64; n]),
                no_data_value: None,
            },
            7, // Double
        ),
    ];

    for (image, expected_dt) in cases {
        let encoded = lerc::encode(&image, Precision::Lossless).expect("encode failed");
        let h = parse_header(&encoded);
        assert_eq!(
            h.data_type, expected_dt,
            "data type code mismatch for {:?}: expected {expected_dt}, got {}",
            image.data_type, h.data_type
        );
    }
}

// =========================================================================
// 4. Self-consistency and determinism
// =========================================================================

#[test]
fn deterministic_encoding_f32() {
    let width = 64u32;
    let height = 64u32;
    let pixels: Vec<f32> = (0..width * height)
        .map(|i| {
            let x = (i % width) as f32 / width as f32;
            let y = (i / width) as f32 / height as f32;
            (x * std::f32::consts::PI).sin() * (y * 2.0).cos() * 500.0
        })
        .collect();

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![BitMask::all_valid((width * height) as usize)],
        data: SampleData::F32(pixels),
        no_data_value: None,
    };

    let encoded1 = lerc::encode(&image, Precision::Tolerance(0.01)).expect("first encode failed");
    let encoded2 = lerc::encode(&image, Precision::Tolerance(0.01)).expect("second encode failed");

    assert_eq!(
        encoded1, encoded2,
        "encoding the same data twice should produce byte-identical output"
    );
}

#[test]
fn deterministic_encoding_u8_multiband() {
    let width = 32u32;
    let height = 32u32;
    let band_size = (width * height) as usize;
    let pixels: Vec<u8> = (0..band_size * 3)
        .map(|i| ((i * 41 + 7) % 256) as u8)
        .collect();

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 3,
        data_type: DataType::Byte,
        valid_masks: vec![BitMask::all_valid(band_size)],
        data: SampleData::U8(pixels),
        no_data_value: None,
    };

    let encoded1 = lerc::encode(&image, Precision::Lossless).expect("first encode failed");
    let encoded2 = lerc::encode(&image, Precision::Lossless).expect("second encode failed");

    assert_eq!(
        encoded1, encoded2,
        "multiband encoding should be deterministic"
    );
}

#[test]
fn deterministic_encoding_with_mask() {
    let width = 48u32;
    let height = 48u32;
    let n = (width * height) as usize;
    let mut mask = BitMask::new(n);
    for k in 0..n {
        if k % 3 != 0 {
            mask.set_valid(k);
        }
    }

    let pixels: Vec<f32> = (0..n)
        .map(|i| {
            if mask.is_valid(i) {
                (i as f32) * 0.7
            } else {
                0.0
            }
        })
        .collect();

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![mask],
        data: SampleData::F32(pixels),
        no_data_value: None,
    };

    let encoded1 = lerc::encode(&image, Precision::Tolerance(0.001)).expect("first encode failed");
    let encoded2 = lerc::encode(&image, Precision::Tolerance(0.001)).expect("second encode failed");

    assert_eq!(
        encoded1, encoded2,
        "masked encoding should be deterministic"
    );
}

#[test]
fn self_consistency_all_invalid() {
    // Edge case: all pixels invalid
    let width = 16u32;
    let height = 16u32;
    let n = (width * height) as usize;
    let mask = BitMask::new(n); // all invalid

    let pixels: Vec<f32> = vec![0.0; n];
    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![mask],
        data: SampleData::F32(pixels),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Lossless).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);

    let dec_mask = &decoded.valid_masks[0];
    for k in 0..n {
        assert!(!dec_mask.is_valid(k), "pixel {k} should be invalid");
    }
}

#[test]
fn self_consistency_single_valid_pixel() {
    // Edge case: only one valid pixel
    let width = 16u32;
    let height = 16u32;
    let n = (width * height) as usize;
    let mut mask = BitMask::new(n);
    mask.set_valid(100); // only pixel 100 is valid

    let mut pixels: Vec<f32> = vec![0.0; n];
    pixels[100] = 42.5;

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![mask.clone()],
        data: SampleData::F32(pixels),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Lossless).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    let dec_mask = &decoded.valid_masks[0];
    for k in 0..n {
        assert_eq!(
            mask.is_valid(k),
            dec_mask.is_valid(k),
            "mask mismatch at {k}"
        );
    }

    match &decoded.data {
        SampleData::F32(dec_pixels) => {
            assert_eq!(
                dec_pixels[100].to_bits(),
                42.5f32.to_bits(),
                "single valid pixel value mismatch"
            );
        }
        _ => panic!("expected F32"),
    }
}

#[test]
fn self_consistency_constant_image() {
    // Edge case: all pixels have the same value
    let width = 32u32;
    let height = 32u32;
    let n = (width * height) as usize;
    let pixels: Vec<f32> = vec![std::f32::consts::PI; n];

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![BitMask::all_valid(n)],
        data: SampleData::F32(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Lossless).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    match &decoded.data {
        SampleData::F32(dec_pixels) => {
            for (k, (&orig, &dec)) in pixels.iter().zip(dec_pixels.iter()).enumerate() {
                assert_eq!(
                    orig.to_bits(),
                    dec.to_bits(),
                    "pixel {k}: constant image mismatch"
                );
            }
        }
        _ => panic!("expected F32"),
    }
}

#[test]
fn self_consistency_i16_large_range() {
    // Edge case: i16 with full value range
    let width = 16u32;
    let height = 16u32;
    let n = (width * height) as usize;
    let pixels: Vec<i16> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            (t * (i16::MAX as f64 - i16::MIN as f64) + i16::MIN as f64) as i16
        })
        .collect();

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Short,
        valid_masks: vec![BitMask::all_valid(n)],
        data: SampleData::I16(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Lossless).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    match &decoded.data {
        SampleData::I16(dec_pixels) => {
            assert_eq!(*dec_pixels, pixels, "i16 full-range round-trip mismatch");
        }
        _ => panic!("expected I16"),
    }
}

#[test]
fn self_consistency_u32_sparse_mask() {
    // Edge case: u32 with a sparse validity mask
    let width = 32u32;
    let height = 32u32;
    let n = (width * height) as usize;
    let mut mask = BitMask::new(n);

    // Only every 10th pixel valid
    for k in (0..n).step_by(10) {
        mask.set_valid(k);
    }

    let pixels: Vec<u32> = (0..n).map(|i| (i as u32) * 1000).collect();

    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::UInt,
        valid_masks: vec![mask.clone()],
        data: SampleData::U32(pixels.clone()),
        no_data_value: None,
    };

    let encoded = lerc::encode(&image, Precision::Lossless).expect("encode failed");
    let decoded = lerc::decode(&encoded).expect("decode failed");

    let dec_mask = &decoded.valid_masks[0];
    match &decoded.data {
        SampleData::U32(dec_pixels) => {
            for k in 0..n {
                assert_eq!(
                    mask.is_valid(k),
                    dec_mask.is_valid(k),
                    "mask mismatch at {k}"
                );
                if mask.is_valid(k) {
                    assert_eq!(
                        pixels[k], dec_pixels[k],
                        "pixel {k}: expected {}, got {}",
                        pixels[k], dec_pixels[k]
                    );
                }
            }
        }
        _ => panic!("expected U32"),
    }
}
