//! Malformed blobs must produce an error: no panic, hang or runaway
//! allocation. Each of the first tests is a case the `fuzz/` decode target
//! found; the last mutates valid blobs at random as a quick stand-in for it.

use lerc::bitmask::BitMask;
use lerc::checksum::fletcher32;
use lerc::{LercError, Precision};

#[derive(Clone, Copy, Debug)]
enum Field {
    Rows,
    Cols,
    Depth,
    MicroBlockSize,
    BlobSize,
    BlobsMore,
}

fn version(blob: &[u8]) -> i32 {
    i32::from_le_bytes(blob[6..10].try_into().unwrap())
}

/// Byte offset of a header field: magic(6), version, checksum, nRows, nCols,
/// [nDepth from version 4], numValidPixel, microBlockSize, blobSize,
/// dataType, [nBlobsMore in version 6].
fn offset(blob: &[u8], field: Field) -> usize {
    let v = version(blob);
    let depth = if v >= 4 { 4 } else { 0 };
    match field {
        Field::Rows => 14,
        Field::Cols => 18,
        Field::Depth => {
            assert!(v >= 4, "version {v} has no depth field");
            22
        }
        Field::MicroBlockSize => 22 + depth + 4,
        Field::BlobSize => 22 + depth + 8,
        Field::BlobsMore => {
            assert!(v >= 6, "version {v} has no band count");
            22 + depth + 16
        }
    }
}

fn set(blob: &mut [u8], field: Field, v: i32) {
    let at = offset(blob, field);
    blob[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

/// Recompute the checksum, so that a patched header reaches the decoder
/// instead of failing the checksum.
fn fix_checksum(blob: &mut [u8]) {
    if version(blob) < 3 {
        return;
    }
    let size = i32::from_le_bytes(
        blob[offset(blob, Field::BlobSize)..][..4]
            .try_into()
            .unwrap(),
    );
    let size = (size.max(14) as usize).min(blob.len());
    let sum = fletcher32(&blob[14..size]);
    blob[10..14].copy_from_slice(&sum.to_le_bytes());
}

/// One blob of the oldest version the encoder writes for integers, and one of
/// the newest, which has the depth and band count fields.
fn sample_blobs() -> [Vec<u8>; 2] {
    let ints: Vec<i16> = (0..24 * 24).map(|i| (i * 7 % 300) as i16).collect();
    let floats: Vec<f32> = (0..24 * 24).map(|i| (i as f32 * 0.37).sin()).collect();
    let blobs = [
        lerc::encode_slice(24, 24, &ints, Precision::Lossless).unwrap(),
        lerc::encode_slice(24, 24, &floats, Precision::Lossless).unwrap(),
    ];
    assert_eq!([version(&blobs[0]), version(&blobs[1])], [3, 6]);
    blobs
}

fn patched(blob: &[u8], field: Field, v: i32) -> Vec<u8> {
    let mut blob = blob.to_vec();
    set(&mut blob, field, v);
    fix_checksum(&mut blob);
    blob
}

fn assert_invalid(blob: &[u8]) {
    assert!(lerc::decode(blob).is_err());
}

/// A 152-byte version 3 blob whose header says `blobSize = 0`. Walking the
/// concatenated blobs in `decode_info` never advanced.
#[test]
fn zero_blob_size_does_not_hang() {
    #[rustfmt::skip]
    let blob: &[u8] = &[
    0x4c, 0x65, 0x72, 0x63, 0x32, 0x20, 0x03, 0x00, 0x00, 0x00, 0x9a, 0xea, 0x15, 0xaa, 0x05, 0x00,
    0x00, 0x00, 0x07, 0x00, 0x00, 0x7f, 0x1a, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0xfc, 0xa9, 0xf1, 0xd2, 0x4d, 0x62, 0x50, 0x3f, 0xbe, 0xae,
    0x75, 0xf5, 0xa0, 0x36, 0x8f, 0xc0, 0x48, 0x7e, 0xaf, 0x16, 0xa6, 0x0f, 0x8f, 0x40, 0x09, 0x00,
    0x00, 0x00, 0x05, 0x00, 0x6f, 0xcf, 0xbd, 0x7e, 0xa0, 0x00, 0x80, 0x00, 0x01, 0xbe, 0xae, 0x75,
    0xf5, 0xa0, 0x36, 0x8f, 0xc0, 0x94, 0x1a, 0x28, 0x34, 0x1f, 0x0e, 0xf1, 0xee, 0x85, 0xde, 0xfe,
    0xe1, 0xca, 0xa5, 0x2d, 0x90, 0xd1, 0x46, 0x7b, 0xfc, 0xe7, 0xbc, 0x8e, 0x83, 0xc9, 0x0e, 0x8b,
    0xfa, 0xda, 0xe7, 0x44, 0x70, 0x80, 0x2f, 0xb6, 0x5b, 0x49, 0xb8, 0xd5, 0xc3, 0x16, 0x32, 0x04,
    0x7b, 0xf2, 0xfb, 0x15, 0x8f, 0x91, 0x00, 0xaa, 0x04, 0x65, 0x1a, 0xa0, 0x16, 0x00, 0x00, 0x00,
    0xa0, 0x62, 0x01, 0xe3, 0x87, 0x90, 0x11, 0x15,
    ];
    assert!(lerc::decode_info(blob).is_err());
    assert!(lerc::decode(blob).is_err());
}

#[test]
fn blob_size_smaller_than_header_is_an_error() {
    for sample in sample_blobs() {
        for size in [-1, 0, 1, 13, 14, 20, 60] {
            let blob = patched(&sample, Field::BlobSize, size);
            assert!(lerc::decode_info(&blob).is_err(), "blob size {size}");
            assert_invalid(&blob);
        }
    }
}

#[test]
fn blob_size_larger_than_data_is_an_error() {
    for sample in sample_blobs() {
        let blob = patched(&sample, Field::BlobSize, i32::MAX);
        assert!(matches!(
            lerc::decode(&blob),
            Err(LercError::BufferTooSmall { .. })
        ));
    }
}

/// Header dimensions size the output buffer, so absurd ones must be rejected
/// before allocating, not abort the process.
#[test]
fn huge_dimensions_are_an_error() {
    for sample in sample_blobs() {
        let mut cases = vec![
            (Field::Rows, i32::MAX),
            (Field::Cols, i32::MAX),
            (Field::Rows, 0),
            (Field::Cols, -1),
        ];
        if version(&sample) >= 4 {
            cases.push((Field::Depth, 0));
        }
        for (field, v) in cases {
            let blob = patched(&sample, field, v);
            assert!(lerc::decode_info(&blob).is_err(), "{field:?} = {v}");
            assert_invalid(&blob);
        }
    }

    // 576 pixels x 2^31 depth is describable but not allocatable.
    assert_invalid(&patched(&sample_blobs()[1], Field::Depth, i32::MAX));

    // 46340 x 46340 pixels is within the i32 pixel limit; the samples are not
    // allocatable.
    let mut blob = patched(&sample_blobs()[1], Field::Rows, 46340);
    set(&mut blob, Field::Cols, 46340);
    set(&mut blob, Field::Depth, 1 << 20);
    fix_checksum(&mut blob);
    assert_invalid(&blob);
}

#[test]
fn zero_micro_block_size_is_an_error() {
    for sample in sample_blobs() {
        for v in [0, -1] {
            assert_invalid(&patched(&sample, Field::MicroBlockSize, v));
        }
    }
}

#[test]
fn more_bands_than_the_blob_can_hold_is_an_error() {
    let blob = patched(&sample_blobs()[1], Field::BlobsMore, 1_000_000);
    assert!(lerc::decode_info(&blob).is_err());
    assert_invalid(&blob);

    // A band that is claimed but is not there.
    let blob = patched(&sample_blobs()[1], Field::BlobsMore, 1);
    assert_invalid(&blob);
}

/// A second band of another size or data type must not be decoded into the
/// buffer sized for the first.
#[test]
fn bands_must_match_the_first() {
    let small = lerc::encode_slice(4, 4, &[7.5f32; 16], Precision::Lossless).unwrap();
    let big = lerc::encode_slice(30, 30, &[9.5f32; 900], Precision::Lossless).unwrap();
    let other_type = lerc::encode_slice(4, 4, &[7u8; 16], Precision::Lossless).unwrap();
    assert_eq!(version(&small), 6);

    for second in [&big, &other_type] {
        let mut blob = small.clone();
        set(&mut blob, Field::BlobsMore, 1);
        fix_checksum(&mut blob);
        blob.extend_from_slice(second);
        assert_invalid(&blob);
        let mut out = [0f32; 32];
        assert!(lerc::decode_into::<f32>(&blob, &mut out).is_err());
    }
}

/// Tiles of a masked image carry one value per valid pixel; a mask that says
/// otherwise indexed past the tile's values.
#[test]
fn damaged_mask_or_tiles_do_not_panic() {
    let n = 16 * 16;
    let pixels: Vec<u16> = (0..n).map(|i| (i * 5 % 1000) as u16).collect();
    let mut mask = BitMask::all_valid(n);
    for k in (0..n).step_by(3) {
        mask.set_invalid(k);
    }
    let blob = lerc::encode_slice_masked(16, 16, &pixels, &mask, Precision::Tolerance(2)).unwrap();
    assert!(lerc::decode(&blob).is_ok());

    // Every single-byte change must be an error or a decode, not a panic.
    for at in 20..blob.len() {
        for flip in [0xff, 0x01, 0x80] {
            let mut bad = blob.clone();
            bad[at] ^= flip;
            fix_checksum(&mut bad);
            if lerc::decode_info(&bad).is_ok_and(|i| i.width * i.height > 1 << 22) {
                continue;
            }
            let _ = lerc::decode(&bad);
        }
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn corpus() -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    for (w, h) in [(7u32, 5u32), (33, 20)] {
        let n = (w * h) as usize;
        let mut mask = BitMask::all_valid(n);
        for k in 0..n {
            if k % 7 == 3 || k % 11 == 0 {
                mask.set_invalid(k);
            }
        }
        let u8s: Vec<u8> = (0..n).map(|i| (i * 7 % 251) as u8).collect();
        let i16s: Vec<i16> = (0..n).map(|i| ((i * 13) % 4000) as i16 - 2000).collect();
        let f32s: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin() * 100.0).collect();
        let f64s: Vec<f64> = (0..n).map(|i| (i as f64 * 0.11).cos() * 1e3).collect();
        for m in [BitMask::all_valid(n), mask] {
            out.push(lerc::encode_slice_masked(w, h, &u8s, &m, Precision::Lossless).unwrap());
            out.push(lerc::encode_slice_masked(w, h, &i16s, &m, Precision::Lossless).unwrap());
            out.push(lerc::encode_slice_masked(w, h, &i16s, &m, Precision::Tolerance(3)).unwrap());
            out.push(lerc::encode_slice_masked(w, h, &f32s, &m, Precision::Lossless).unwrap());
            out.push(
                lerc::encode_slice_masked(w, h, &f32s, &m, Precision::Tolerance(0.01)).unwrap(),
            );
            out.push(
                lerc::encode_slice_masked(w, h, &f64s, &m, Precision::Tolerance(0.001)).unwrap(),
            );
        }
    }
    out
}

/// Random mutations of valid blobs, checksum repaired so they get past it.
/// This is a quick, fixed-seed version of `cargo fuzz run decode`.
#[test]
fn mutated_blobs_do_not_panic() {
    let corpus = corpus();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for _ in 0..30_000 {
        let mut blob = corpus[rng.below(corpus.len())].clone();
        for _ in 0..1 + rng.below(4) {
            let at = if rng.below(2) == 0 {
                rng.below(blob.len().min(100))
            } else {
                rng.below(blob.len())
            };
            match rng.below(4) {
                0 => blob[at] = rng.next() as u8,
                1 => blob[at] ^= 1 << rng.below(8),
                2 => blob[at] = [0, 0xff, 0x7f, 0x80][rng.below(4)],
                _ => blob.truncate(1 + rng.below(blob.len())),
            }
        }
        if blob.len() > 40 && rng.below(10) < 9 {
            fix_checksum(&mut blob);
        }
        // A few header bytes can describe a huge constant image; decoding it
        // would only measure the machine.
        if let Ok(info) = lerc::decode_info(&blob) {
            let samples = u64::from(info.width)
                * u64::from(info.height)
                * u64::from(info.depth)
                * u64::from(info.bands);
            if samples > 1 << 22 {
                continue;
            }
        }
        let _ = lerc::decode(&blob);
    }
}
