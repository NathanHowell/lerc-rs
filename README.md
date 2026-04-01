# lerc-rs

Pure Rust implementation of [LERC](https://github.com/Esri/lerc) (Limited Error Raster Compression), Esri's open format for fast, efficient raster encoding with controlled precision loss.

Supports encoding and decoding of LERC2 (v2–v6) and decoding of legacy LERC1, with all 8 data types, validity masks, multi-band, multi-depth, and NoData values.

## Features

- **Pure Rust** — no C/C++ dependencies, no unsafe code
- **`no_std` + `alloc` compatible** — works on native and `wasm32-unknown-unknown`
- **All data types** — `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `f32`, `f64`
- **Lossless and lossy** — configurable `Precision` per encode
- **Complete codec** — Huffman coding, float-point lossless (FPL) with byte-plane predictors, tiled micro-block encoding, bit-plane compression, diff encoding for multi-depth, validity masks, NoData
- **Byte-for-byte identical** output to the C++ reference at the same settings
- **Competitive with C++** reference implementation on encode and decode

## Performance

Preliminary benchmarks on a single synthetic dataset (512×512, Apple M-series, `--release`). Performance will vary with data characteristics; these numbers are indicative, not comprehensive.

| Path | Rust | C++ reference |
|------|------|---------------|
| encode f32 lossy (maxZErr=0.01) | 1.9 ms | 3.1 ms |
| encode f32 lossless (FPL) | 8.3 ms | 17 ms |
| encode u8 lossless (Huffman) | 2.3 ms | 2.7 ms |
| decode f32 lossy | 0.77 ms | 0.76 ms |

Compression output is byte-for-byte identical to the C++ reference at the same settings. Run `cargo bench --bench codec` to reproduce (requires a C++ toolchain for the reference comparison).

## Usage

```toml
[dependencies]
lerc-rs = "0.1"
```

### Encode

```rust
use lerc::{encode_typed, decode_typed, Precision};
use lerc::bitmask::BitMask;

// Encode a 256×256 f32 raster with 0.01 precision
let pixels: Vec<f32> = vec![0.0; 256 * 256]; // your data here
let blob = lerc::encode_typed(256, 256, &pixels, Precision::MaxError(0.01f32)).unwrap();

// Lossless encoding (works for any type)
let bytes: Vec<u8> = vec![0; 128 * 128];
let blob = lerc::encode_typed(128, 128, &bytes, Precision::Lossless).unwrap();
```

### Decode

```rust
// Decode to typed data
let (pixels, mask, width, height) = lerc::decode_typed::<f32>(&blob).unwrap();

// Or decode to LercImage for full metadata
let image = lerc::decode(&blob).unwrap();
println!("{}×{}, {:?}, {} bands", image.width, image.height, image.data_type, image.n_bands);
```

### Zero-copy decode

```rust
// Decode into a pre-allocated buffer
let info = lerc::decode_info(&blob).unwrap();
let mut output = vec![0.0f32; (info.width * info.height) as usize];
let result = lerc::decode_f32_into(&blob, &mut output).unwrap();
```

### With validity mask

```rust
use lerc::Precision;
use lerc::bitmask::BitMask;

let mut mask = BitMask::new(256 * 256);
mask.set_valid(0); // mark specific pixels as valid
// ... set more pixels valid ...

let blob = lerc::encode_typed_masked(256, 256, &pixels, &mask, Precision::MaxError(0.01f32)).unwrap();
```

## Supported formats

| Format | Encode | Decode |
|--------|--------|--------|
| LERC2 v6 (current) | Yes | Yes |
| LERC2 v3–v5 | Yes (auto-selects minimum version) | Yes |
| LERC2 v2 | — | Yes |
| LERC1 (legacy) | — | Yes |

## Encoder features

- Tiled micro-block encoding with adaptive block size (8×8 or 16×16)
- Huffman coding for 8-bit types with delta prediction
- Float-point lossless (FPL) with byte-plane Huffman and predictors
- Diff encoding between depth slices for multi-depth data
- Bit-plane compression for noisy integer data (opt-in via negative `max_z_error`)
- `TryRaiseMaxZError` for float data with limited precision
- NoData value encoding for mixed valid/invalid multi-depth pixels

## Building

```sh
cargo build --release
cargo test
cargo bench --bench codec  # requires C++ toolchain for comparison benchmarks
```

### wasm

Tests run on `wasm32-wasip1` via [wasmtime](https://wasmtime.dev/):

```sh
rustup target add wasm32-wasip1
cargo test --target wasm32-wasip1       # lib + integration tests
```

Compile-only check for `wasm32-unknown-unknown` (no test runner):

```sh
cargo check --target wasm32-unknown-unknown
```

### C++ cross-validation

The test suite includes bidirectional cross-validation against the C++ reference (compiled via the `lerc-cpp-ref` workspace crate):

```sh
cargo test --features cpp-validation
```

## License

Apache-2.0, matching the [Esri LERC](https://github.com/Esri/lerc) reference implementation.
