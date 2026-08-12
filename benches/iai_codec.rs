//! Callgrind-based instruction-count benchmarks for regression gating in CI.
//!
//! Unlike the Criterion wall-clock benches in `codec.rs`, these use
//! `iai-callgrind` to produce deterministic instruction counts via Valgrind.
//! This makes them suitable for CI regression thresholds, but they require
//! `valgrind` and a version-matched `iai-callgrind-runner` binary on PATH.
//!
//! Run with: `cargo bench --bench iai_codec`
//!
//! The real implementation lives in an inline `mod real` (rather than a
//! separate file brought in via `include!`) because `iai-callgrind`'s
//! `main!` macro embeds `file!()` and matches it against the `src_path`
//! Cargo declares for this `[[bench]]` target -- `include!`-d code reports
//! the *included* file's path, which would break that lookup. `criterion`
//! (see `codec.rs`, `profile_encode.rs`) has no such requirement.
//!
//! criterion and iai-callgrind are only available as dev-dependencies on
//! non-wasm32 targets (see Cargo.toml), so this compiles an empty stub on
//! wasm32 instead -- `cargo build/check --target wasm32-unknown-unknown
//! --all-targets` still needs this crate to produce a valid binary.

#[cfg(not(target_arch = "wasm32"))]
mod real {
    use iai_callgrind::{library_benchmark, library_benchmark_group};
    use lerc::bitmask::BitMask;
    use lerc::{DataType, Image, Precision, SampleData};
    use std::hint::black_box;

    // -----------------------------------------------------------------------
    // Reference files (real-world test data)
    // -----------------------------------------------------------------------

    static CALIFORNIA: &[u8] =
        include_bytes!("../esri-lerc/testData/california_400_400_1_float.lerc2");
    static BLUEMARBLE: &[u8] =
        include_bytes!("../esri-lerc/testData/bluemarble_256_256_3_byte.lerc2");

    // -----------------------------------------------------------------------
    // Synthetic-input builders (run once by iai `setup`, outside measurement)
    // -----------------------------------------------------------------------

    const SIZE: usize = 256;

    fn make_f32_image() -> Image {
        let pixels: Vec<f32> = (0..SIZE * SIZE)
            .map(|i| {
                let x = (i % SIZE) as f32 / SIZE as f32;
                let y = (i / SIZE) as f32 / SIZE as f32;
                x * 1000.0 + y * 500.0 + (x * 31.4).sin() * 50.0
            })
            .collect();
        Image {
            width: SIZE as u32,
            height: SIZE as u32,
            depth: 1,
            bands: 1,
            data_type: DataType::Float,
            valid_masks: vec![BitMask::all_valid(SIZE * SIZE)],
            data: SampleData::F32(pixels),
            ..Default::default()
        }
    }

    fn make_u8_image() -> Image {
        let pixels: Vec<u8> = (0..SIZE * SIZE)
            .map(|i| ((i * 7 + 13) % 256) as u8)
            .collect();
        Image {
            width: SIZE as u32,
            height: SIZE as u32,
            depth: 1,
            bands: 1,
            data_type: DataType::Byte,
            valid_masks: vec![BitMask::all_valid(SIZE * SIZE)],
            data: SampleData::U8(pixels),
            ..Default::default()
        }
    }

    fn make_f64_image() -> Image {
        let pixels: Vec<f64> = (0..SIZE * SIZE)
            .map(|i| {
                let x = (i % SIZE) as f64 / SIZE as f64;
                let y = (i / SIZE) as f64 / SIZE as f64;
                (x * std::f64::consts::PI).sin() * (y * std::f64::consts::E).cos() * 3000.0
            })
            .collect();
        Image {
            width: SIZE as u32,
            height: SIZE as u32,
            depth: 1,
            bands: 1,
            data_type: DataType::Double,
            valid_masks: vec![BitMask::all_valid(SIZE * SIZE)],
            data: SampleData::F64(pixels),
            ..Default::default()
        }
    }

    // -----------------------------------------------------------------------
    // Decode benches (real-world reference data)
    // -----------------------------------------------------------------------

    #[library_benchmark]
    #[bench::california(CALIFORNIA)]
    #[bench::bluemarble(BLUEMARBLE)]
    fn decode(blob: &[u8]) -> Image {
        black_box(lerc::decode(black_box(blob)).unwrap())
    }

    // -----------------------------------------------------------------------
    // Encode benches (synthetic inputs, built in setup)
    // -----------------------------------------------------------------------

    #[library_benchmark]
    #[bench::lossy_0_01(setup = make_f32_image)]
    fn encode_f32_lossy(image: Image) -> Vec<u8> {
        black_box(lerc::encode(&image, Precision::Tolerance(0.01)).unwrap())
    }

    #[library_benchmark]
    #[bench::lossless(setup = make_f32_image)]
    fn encode_f32_lossless(image: Image) -> Vec<u8> {
        black_box(lerc::encode(&image, Precision::Lossless).unwrap())
    }

    #[library_benchmark]
    #[bench::lossless(setup = make_u8_image)]
    fn encode_u8_lossless(image: Image) -> Vec<u8> {
        black_box(lerc::encode(&image, Precision::Lossless).unwrap())
    }

    #[library_benchmark]
    #[bench::lossless(setup = make_f64_image)]
    fn encode_f64_lossless(image: Image) -> Vec<u8> {
        black_box(lerc::encode(&image, Precision::Lossless).unwrap())
    }

    // -----------------------------------------------------------------------
    // Group
    // -----------------------------------------------------------------------

    library_benchmark_group!(
        name = codec;
        benchmarks =
            decode,
            encode_f32_lossy,
            encode_f32_lossless,
            encode_u8_lossless,
            encode_f64_lossless,
    );
}

#[cfg(not(target_arch = "wasm32"))]
use real::codec;

#[cfg(not(target_arch = "wasm32"))]
iai_callgrind::main!(library_benchmark_groups = codec);

#[cfg(target_arch = "wasm32")]
fn main() {}
