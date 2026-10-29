//! Feeds arbitrary bytes to the decoder. It must return `Ok` or `Err`: no
//! panic, overflow, hang or runaway allocation.
//!
//! The first input byte selects the mode: when bit 0 is set, the Fletcher-32
//! checksum of a LERC2 blob is recomputed after the fuzzer's mutations, so
//! inputs get past the checksum and reach the tile, mask and Huffman parsers.
#![no_main]

use libfuzzer_sys::fuzz_target;

/// Decoded images larger than this many samples are not decoded: a few header
/// bytes can legitimately describe a huge constant image, and decoding it
/// only measures the fuzzer's memory.
const MAX_SAMPLES: u64 = 1 << 24;

fn fix_checksum(blob: &mut [u8]) {
    // magic(6) + version(4) + checksum(4), then nRows, nCols, [nDepth],
    // numValidPixel, microBlockSize, blobSize.
    if blob.len() < 14 || &blob[..6] != b"Lerc2 " {
        return;
    }
    let version = i32::from_le_bytes(blob[6..10].try_into().unwrap());
    if version < 3 {
        return;
    }
    let size_at = 14 + 8 + if version >= 4 { 4 } else { 0 } + 8;
    let Some(bytes) = blob.get(size_at..size_at + 4) else {
        return;
    };
    let blob_size = i32::from_le_bytes(bytes.try_into().unwrap());
    if blob_size >= 14 && blob_size as usize <= blob.len() {
        let checksum = lerc::checksum::fletcher32(&blob[14..blob_size as usize]);
        blob[10..14].copy_from_slice(&checksum.to_le_bytes());
    }
}

fuzz_target!(|input: &[u8]| {
    let Some((&mode, blob)) = input.split_first() else {
        return;
    };
    let mut blob = blob.to_vec();
    if mode & 1 != 0 {
        fix_checksum(&mut blob);
    }

    // `decode` allocates what the header says before it reads the body, so a
    // blob `decode_info` rejects can still ask for a huge buffer.
    let Ok(info) = lerc::decode_info(&blob) else {
        return;
    };
    let samples = u64::from(info.width)
        * u64::from(info.height)
        * u64::from(info.depth)
        * u64::from(info.bands);
    if samples <= MAX_SAMPLES {
        let _ = lerc::decode(&blob);
    }
});
