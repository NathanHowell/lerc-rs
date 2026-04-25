# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-04-24

### Changed (breaking)

- `decode_slice<T>` now returns `DecodedSlice<T> { pixels, mask, width, height }`
  instead of a `(Vec<T>, BitMask, u32, u32)` 4-tuple. The named struct removes
  positional ambiguity between `width` and `height`. Migration: replace tuple
  destructuring with field access (e.g. `let result = decode_slice(...)?;
  result.pixels` / `result.width`).

## [0.2.1] - 2026-04-24

### Added

- `decode_into_with_nodata<T>` convenience that fills invalid pixels with a
  caller-supplied sentinel.
- `encode_borrowed<T>` zero-copy multi-band encode entry point that avoids
  the buffer clone forced by the `Image`-based API.

### Added (tests)

- `tests/cpp_byte_identity.rs` — wire-format regression tests: an
  `Explicit`-mask-with-every-bit-set canonicalization pin (AllValid fast
  path against itself) and byte-identity pins for `f32`/`f64` lossless
  encodes against the C++ reference encoder.

### Fixed

- FPL (lossless float) byte-plane level selection on small planes (< 8192 bytes).
  When the snippet-sampling path produced no snippets, Rust fell through to a
  full-plane delta search that could pick a non-zero level on data already
  flattened by the row+column predictor (typical for smooth gradients), making
  the byte plane harder to compress. The C++ reference's empty-snippets path
  short-circuits to level 0; mirror that. Eliminates 7-22% blob-size regressions
  vs. the C++ reference on smooth float ramps at sub-8192-byte plane sizes; no
  effect on larger planes or non-ramp data.

## [0.2.0] - 2026-04-24

### Changed (breaking)

- `BitMask` is now an enum with `AllValid(usize)` and `Explicit(BitVec)` variants.
  The decoder produces `AllValid` directly when the blob header reports full
  validity, so `mask.is_all_valid()` is O(1) and does not allocate or scan.
  Previously, "all pixels valid" was expressed by an `Explicit` mask with every
  bit set, which required an O(n) `count_valid() == num_pixels()` check at the
  consumer.
- `BitMask::as_bytes` / `as_bytes_mut` now return `Option<&[u8]>` /
  `Option<&mut [u8]>`; they return `None` for `AllValid` (no bytes stored).

### Added

- `BitMask::is_all_valid` — O(1) for `AllValid`, popcount fallback for `Explicit`.
- `BitMask::AllValid(n)` pattern matches in downstream code.

## [0.1.1] - 2026-04-15

### Changed

- Switch to crates.io trusted publishing via OIDC
- Add iai-callgrind regression benchmarks

## [0.1.0] - 2026-03-31

### Added

- Pure Rust LERC (Limited Error Raster Compression) encoder and decoder
- Support for LERC2 format (versions 2-6)
- Lossless and lossy compression for integer and floating-point raster data
- `no_std` support with optional `alloc` and `std` features
- Support for all LERC pixel types: `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `f32`, `f64`
- Multi-band and masked raster encoding/decoding
- Bit-plane compression, Huffman coding, and FPL (Floating Point LERC) encoding
