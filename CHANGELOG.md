# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Lossy encoding codes a micro-block whose value range is under half a
  quantization step (`(z_max - z_min) / (2 * max_z_error)` rounds to 0) as a
  constant at `z_min`, as Esri's Lerc2 does; every value is within
  `max_z_error` of it. Such blocks were stored as raw values instead — 32 bits
  per pixel for `f32` — so near-flat data with noise below the tolerance
  compressed several times worse than with the reference encoder (220× on a
  synthetic flat `f32` raster, ~3× on real flat terrain tiles). The
  block-coding decision now lives in one function shared by the encoder and
  its multi-depth reconstruction, so the two cannot drift apart again.
- The micro-block size is chosen as Esri's Lerc2 chooses it: encode at
  8×8, and when that comes to under 1.5 bits per pixel encode again at 16×16
  and keep it if no larger. The encoder used to sample eight 8×8 and eight
  16×16 blocks and compare the byte totals, which priced four times as many
  pixels at 16 and so favoured 8 — on flat data, where every block of either
  size is one constant, the totals tied and 8×8 won at four times the size.
  Output for data above 1.5 bits per pixel is unchanged on the reference
  rasters, and encode time is within ±2% on the benchmarks (a new
  `lossy_flat_0.02` case covers the low-bit-rate path). The u8 lossless
  Huffman comparison now reuses the chosen tiling instead of encoding the
  tiles a second time.
- Lossy multi-depth encoding no longer lets a depth slice decode up to
  2 × `max_z_error` off. The encoder diffs each slice against its own
  reconstruction of the previous one, and reconstructed every quantized
  block as quantized — including one whose bit-stuffed form was no smaller
  than raw and was stored raw, which the decoder reads back exactly. Small or
  sparsely valid blocks with a wide range hit this. The reconstruction now
  follows the coding the encoder actually wrote.

## [0.5.0] - 2026-08-12

### Fixed

- The `no_std` build (`--no-default-features`, optionally with `alloc`)
  actually compiles now. It previously failed: several modules used
  `alloc::boxed::Box`/`alloc::vec::Vec`/`vec!` without importing them
  (papered over by `std`'s prelude), and `f64::floor`/`round`/`log2`/`sqrt`
  aren't `core` methods (`std` links the platform's libm for them). A new
  internal `mathutils` module dispatches to `std`'s methods when the `std`
  feature is enabled and to the pure-Rust `libm` crate otherwise.
- `cargo build --target wasm32-unknown-unknown --all-targets` actually
  compiles now. The `[[bench]]` targets weren't gated for wasm32, but their
  implementations depend on `criterion`/`lerc-cpp-ref`, which are only
  available as dev-dependencies on non-wasm32 targets. Each bench's real
  implementation moved under `benches/impl/`, with a thin wasm32-aware
  entry point left in its place.

### Changed (internal)

- Removed the blanket `#[allow(dead_code)]` module annotations, along with
  the dead code they had been masking (some genuinely unused, some
  superseded duplicate helpers). No public API changes.
- CI now covers the configurations that let the above two issues through:
  a `no-std` job runs the test suite under `--no-default-features` and
  `--no-default-features --features alloc`, `test-wasm` checks
  `--all-targets` (including the alloc-only build) instead of just the
  library, and `clippy` runs as a matrix over default,
  `--no-default-features`, alloc-only, and `--all-features` so
  feature-gated code (e.g. `cpp-validation`-only tests) is actually linted.
- Updated dependencies to their latest semver-compatible versions.

## [0.4.0] - 2026-04-24

### Changed (breaking)

- `encode_borrowed<T>` now takes `Precision<T>` instead of `Precision<f64>`,
  matching `encode_slice<T>` and `encode_slice_masked<T>`. Tolerances are
  expressed in the pixel type and widened to `f64` internally. Migration:
  drop the `as f64` cast on the tolerance value (e.g.
  `Precision::Tolerance(max_z_error as f64)` → `Precision::Tolerance(max_z_error)`
  when `max_z_error: T`).

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
