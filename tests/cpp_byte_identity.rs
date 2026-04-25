#![cfg(feature = "cpp-validation")]
//! Wire-format regression tests for the encoder.
//!
//! Two distinct invariants are pinned here:
//!
//! 1. **AllValid fast-path canonicalization (Rust-vs-Rust).** The encoder
//!    skips the per-pixel mask block when every pixel is valid, regardless
//!    of whether the caller passed `BitMask::AllValid` or a
//!    `BitMask::Explicit` mask with every bit set. A future refactor that
//!    started serializing a mask block when given an Explicit-all-set mask
//!    would still round-trip but break this canonicalization. The
//!    `byte_identity_explicit_all_set_matches_all_valid_u8_32x32` test
//!    catches that.
//!
//! 2. **Float lossless byte-identity vs. the C++ reference.** Rust and the
//!    C++ reference encoder both emit version 6 for the FPL (lossless
//!    float) path and produce identical byte planes (after PR #19 fixed
//!    the small-plane delta-level regression). Pinning byte equality here
//!    catches encoder changes that silently shift the FPL bitstuffer or
//!    Huffman path away from the reference.
//!
//! Other data-type / precision combinations are intentionally NOT pinned
//! against the C++ reference. Rust's `encode_one_band` emits the
//! minimum-required codec version (3 / 5 / 6 depending on features used),
//! while the C++ reference always emits version 6. Both implementations
//! decode each other's blobs without issue (see `cpp_cross_validation.rs`
//! for full round-trip coverage), so the version difference is a stylistic
//! divergence, not a correctness bug.

use lerc::Precision;
use lerc::bitmask::BitMask;
use lerc::{DataType, Image, SampleData};
use lerc_cpp_ref::{self as cpp, DT_DOUBLE, DT_FLOAT};

// ---------------------------------------------------------------------------
// Byte-equality helper
// ---------------------------------------------------------------------------

fn assert_blobs_equal(rust_blob: &[u8], cpp_blob: &[u8], ctx: &str) {
    if rust_blob == cpp_blob {
        return;
    }

    // Find first differing offset, if any (one blob may simply be longer).
    let common = rust_blob.len().min(cpp_blob.len());
    let first_diff = (0..common).find(|&i| rust_blob[i] != cpp_blob[i]);

    let head = |b: &[u8]| {
        let n = b.len().min(32);
        let mut s = String::new();
        for (i, byte) in b[..n].iter().enumerate() {
            if i > 0 {
                s.push(' ');
            }
            s.push_str(&format!("{:02x}", byte));
        }
        s
    };

    panic!(
        "{ctx}: blob mismatch\n  \
         rust len = {} cpp len = {}\n  \
         first differing offset = {}\n  \
         rust[0..32] = {}\n  \
         cpp [0..32] = {}",
        rust_blob.len(),
        cpp_blob.len(),
        first_diff
            .map(|i| i.to_string())
            .unwrap_or_else(|| format!("(none in common prefix; lengths differ at {common})")),
        head(rust_blob),
        head(cpp_blob),
    );
}

// ---------------------------------------------------------------------------
// Deterministic ramp generators (single-band, single-depth)
// ---------------------------------------------------------------------------

fn ramp_u8(width: u32, height: u32) -> Vec<u8> {
    (0..width * height).map(|i| (i % 251) as u8).collect()
}

fn ramp_f32(width: u32, height: u32) -> Vec<f32> {
    (0..width * height)
        .map(|i| {
            let x = (i % width) as f32 / width as f32;
            let y = (i / width) as f32 / height as f32;
            x * 100.0 + y * 200.0
        })
        .collect()
}

fn ramp_f64(width: u32, height: u32) -> Vec<f64> {
    (0..width * height)
        .map(|i| {
            let x = (i % width) as f64 / width as f64;
            let y = (i / width) as f64 / height as f64;
            x * 1000.0 + y * 2000.0
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Test geometry
// ---------------------------------------------------------------------------

const W: u32 = 32;
const H: u32 = 32;

// ===========================================================================
// f32 lossless (FPL path)
// ===========================================================================

#[test]
fn byte_identity_f32_lossless_all_valid_32x32() {
    let data = ramp_f32(W, H);
    let rust_blob = lerc::encode_slice(W, H, &data, Precision::Lossless).unwrap();
    let cpp_blob = cpp::encode(&data, DT_FLOAT, W as i32, H as i32, 1, 1, None, 0.0);
    assert_blobs_equal(&rust_blob, &cpp_blob, "f32 lossless (FPL) 32x32");
}

// ===========================================================================
// f64 lossless (FPL path)
// ===========================================================================

#[test]
fn byte_identity_f64_lossless_all_valid_32x32() {
    let data = ramp_f64(W, H);
    let rust_blob = lerc::encode_slice(W, H, &data, Precision::Lossless).unwrap();
    let cpp_blob = cpp::encode(&data, DT_DOUBLE, W as i32, H as i32, 1, 1, None, 0.0);
    assert_blobs_equal(&rust_blob, &cpp_blob, "f64 lossless (FPL) 32x32");
}

// ===========================================================================
// Canonicalization: Explicit-all-set must encode identically to AllValid.
//
// The encoder's AllValid fast path is gated on `mask.count_valid() >=
// width*height`, which is true for *both* `BitMask::AllValid` and a
// `BitMask::Explicit` mask with every bit set. This Rust-vs-Rust test pins
// that the canonicalization happens at encode time — i.e. constructing an
// Explicit mask with all bits set does not cause a mask block to be
// serialized. A future change that writes a mask block when given an
// Explicit-all-set mask would break this test.
// ===========================================================================

#[test]
fn byte_identity_explicit_all_set_matches_all_valid_u8_32x32() {
    let data = ramp_u8(W, H);
    let n = (W * H) as usize;

    // AllValid: canonical, no allocation.
    let all_valid_blob = lerc::encode_slice(W, H, &data, Precision::Lossless).unwrap();

    // Explicit mask with every bit set — semantically identical.
    let mut explicit = BitMask::new(n);
    for k in 0..n {
        explicit.set_valid(k);
    }
    assert!(matches!(explicit, BitMask::Explicit(_)));
    assert_eq!(explicit.count_valid(), n);

    let explicit_image = Image {
        width: W,
        height: H,
        depth: 1,
        bands: 1,
        data_type: DataType::Byte,
        valid_masks: vec![explicit],
        data: SampleData::U8(data.clone()),
        no_data_value: None,
    };
    let explicit_blob = lerc::encode(&explicit_image, Precision::Lossless).unwrap();

    assert_blobs_equal(
        &explicit_blob,
        &all_valid_blob,
        "Explicit-all-set vs AllValid (u8 32x32, lossless)",
    );
}
