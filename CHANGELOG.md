# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-31

### Added

- Pure Rust LERC (Limited Error Raster Compression) encoder and decoder
- Support for LERC2 format (versions 2-6)
- Lossless and lossy compression for integer and floating-point raster data
- `no_std` support with optional `alloc` and `std` features
- Support for all LERC pixel types: `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `f32`, `f64`
- Multi-band and masked raster encoding/decoding
- Bit-plane compression, Huffman coding, and FPL (Floating Point LERC) encoding
