use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use lerc::Precision;
use lerc::bitmask::BitMask;
use lerc::{DataType, Image, SampleData};
use lerc_cpp_ref as cpp;

// ---------------------------------------------------------------------------
// Test data generators
// ---------------------------------------------------------------------------

fn make_f32_gradient(width: usize, height: usize) -> Vec<f32> {
    (0..width * height)
        .map(|i| {
            let x = (i % width) as f32 / width as f32;
            let y = (i / width) as f32 / height as f32;
            x * 1000.0 + y * 500.0 + (x * 31.4).sin() * 50.0
        })
        .collect()
}

fn make_u8_image(width: usize, height: usize) -> Vec<u8> {
    (0..width * height)
        .map(|i| ((i * 7 + 13) % 256) as u8)
        .collect()
}

fn make_f64_terrain(width: usize, height: usize) -> Vec<f64> {
    (0..width * height)
        .map(|i| {
            let x = (i % width) as f64 / width as f64;
            let y = (i / width) as f64 / height as f64;
            (x * std::f64::consts::PI).sin() * (y * std::f64::consts::E).cos() * 3000.0
        })
        .collect()
}

fn make_valid_bytes(width: usize, height: usize) -> Vec<u8> {
    vec![1u8; width * height]
}

// ---------------------------------------------------------------------------
// Reference file data (pre-loaded)
// ---------------------------------------------------------------------------

static CALIFORNIA: &[u8] = include_bytes!("../esri-lerc/testData/california_400_400_1_float.lerc2");
static BLUEMARBLE: &[u8] = include_bytes!("../esri-lerc/testData/bluemarble_256_256_3_byte.lerc2");

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_decode_reference(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_reference");

    group.throughput(Throughput::Bytes(CALIFORNIA.len() as u64));
    group.bench_function("california_400x400_f32/rust", |b| {
        b.iter(|| lerc::decode(CALIFORNIA).unwrap());
    });

    group.throughput(Throughput::Bytes(BLUEMARBLE.len() as u64));
    group.bench_function("bluemarble_256x256x3_u8/rust", |b| {
        b.iter(|| lerc::decode(BLUEMARBLE).unwrap());
    });

    // C++ decode of the same files
    let ca_info = lerc::decode_info(CALIFORNIA).unwrap();
    group.throughput(Throughput::Bytes(CALIFORNIA.len() as u64));
    group.bench_function("california_400x400_f32/cpp", |b| {
        b.iter(|| {
            cpp::decode::<f32>(
                CALIFORNIA,
                cpp::DT_FLOAT,
                ca_info.width as i32,
                ca_info.height as i32,
                ca_info.n_depth as i32,
                ca_info.n_bands as i32,
            )
        });
    });

    let bm_info = lerc::decode_info(BLUEMARBLE).unwrap();
    group.throughput(Throughput::Bytes(BLUEMARBLE.len() as u64));
    group.bench_function("bluemarble_256x256x3_u8/cpp", |b| {
        b.iter(|| {
            cpp::decode::<u8>(
                BLUEMARBLE,
                cpp::DT_UCHAR,
                bm_info.width as i32,
                bm_info.height as i32,
                bm_info.n_depth as i32,
                bm_info.n_bands as i32,
            )
        });
    });

    group.finish();
}

fn bench_encode_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_f32");

    for &size in &[64, 256, 512, 1024] {
        let pixels = make_f32_gradient(size, size);
        let valid = make_valid_bytes(size, size);
        let pixel_bytes = (size * size * 4) as u64;

        let image = Image {
            width: size as u32,
            height: size as u32,
            n_depth: 1,
            n_bands: 1,
            data_type: DataType::Float,
            valid_masks: vec![BitMask::all_valid(size * size)],
            data: SampleData::F32(pixels.clone()),
            ..Default::default()
        };

        // Lossy
        group.throughput(Throughput::Bytes(pixel_bytes));
        group.bench_with_input(
            BenchmarkId::new("lossy_0.01/rust", size),
            &image,
            |b, img| {
                b.iter(|| lerc::encode(img, Precision::Tolerance(0.01)).unwrap());
            },
        );

        group.bench_with_input(
            BenchmarkId::new("lossy_0.01/cpp", size),
            &pixels,
            |b, px| {
                b.iter(|| {
                    cpp::encode(
                        px,
                        cpp::DT_FLOAT,
                        size as i32,
                        size as i32,
                        1,
                        1,
                        Some(&valid),
                        0.01,
                    )
                });
            },
        );

        // Lossless (FPL)
        group.bench_with_input(BenchmarkId::new("lossless/rust", size), &image, |b, img| {
            b.iter(|| lerc::encode(img, Precision::Lossless).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("lossless/cpp", size), &pixels, |b, px| {
            b.iter(|| {
                cpp::encode(
                    px,
                    cpp::DT_FLOAT,
                    size as i32,
                    size as i32,
                    1,
                    1,
                    Some(&valid),
                    0.0,
                )
            });
        });
    }

    group.finish();
}

fn bench_encode_u8(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_u8");

    for &size in &[64, 256, 512, 1024] {
        let pixels = make_u8_image(size, size);
        let valid = make_valid_bytes(size, size);
        let pixel_bytes = (size * size) as u64;

        let image = Image {
            width: size as u32,
            height: size as u32,
            n_depth: 1,
            n_bands: 1,
            data_type: DataType::Byte,
            valid_masks: vec![BitMask::all_valid(size * size)],
            data: SampleData::U8(pixels.clone()),
            ..Default::default()
        };

        group.throughput(Throughput::Bytes(pixel_bytes));
        group.bench_with_input(BenchmarkId::new("lossless/rust", size), &image, |b, img| {
            b.iter(|| lerc::encode(img, Precision::Lossless).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("lossless/cpp", size), &pixels, |b, px| {
            b.iter(|| {
                cpp::encode(
                    px,
                    cpp::DT_UCHAR,
                    size as i32,
                    size as i32,
                    1,
                    1,
                    Some(&valid),
                    0.5,
                )
            });
        });
    }

    group.finish();
}

fn bench_decode_synthetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_synthetic");

    for &size in &[64, 256, 512, 1024] {
        let pixels = make_f32_gradient(size, size);
        let valid = make_valid_bytes(size, size);
        let pixel_bytes = (size * size * 4) as u64;

        // Encode with Rust, then benchmark decode
        let image = Image {
            width: size as u32,
            height: size as u32,
            n_depth: 1,
            n_bands: 1,
            data_type: DataType::Float,
            valid_masks: vec![BitMask::all_valid(size * size)],
            data: SampleData::F32(pixels.clone()),
            ..Default::default()
        };
        let rust_blob = lerc::encode(&image, Precision::Tolerance(0.01)).unwrap();

        // Encode with C++, then benchmark decode
        let cpp_blob = cpp::encode(
            &pixels,
            cpp::DT_FLOAT,
            size as i32,
            size as i32,
            1,
            1,
            Some(&valid),
            0.01,
        );

        group.throughput(Throughput::Bytes(pixel_bytes));

        // Rust decode of Rust-encoded blob
        group.bench_with_input(
            BenchmarkId::new("f32_lossy/rust_blob/rust", size),
            &rust_blob,
            |b, blob| {
                b.iter(|| lerc::decode(blob).unwrap());
            },
        );

        // Rust decode of C++-encoded blob
        group.bench_with_input(
            BenchmarkId::new("f32_lossy/cpp_blob/rust", size),
            &cpp_blob,
            |b, blob| {
                b.iter(|| lerc::decode(blob).unwrap());
            },
        );

        // C++ decode of Rust-encoded blob
        group.bench_with_input(
            BenchmarkId::new("f32_lossy/rust_blob/cpp", size),
            &rust_blob,
            |b, blob| {
                b.iter(|| cpp::decode::<f32>(blob, cpp::DT_FLOAT, size as i32, size as i32, 1, 1));
            },
        );

        // C++ decode of C++-encoded blob
        group.bench_with_input(
            BenchmarkId::new("f32_lossy/cpp_blob/cpp", size),
            &cpp_blob,
            |b, blob| {
                b.iter(|| cpp::decode::<f32>(blob, cpp::DT_FLOAT, size as i32, size as i32, 1, 1));
            },
        );
    }

    group.finish();
}

fn bench_decode_f64_lossless(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_f64_lossless");

    for &size in &[64, 256, 512] {
        let pixels = make_f64_terrain(size, size);
        let valid = make_valid_bytes(size, size);
        let pixel_bytes = (size * size * 8) as u64;

        let image = Image {
            width: size as u32,
            height: size as u32,
            n_depth: 1,
            n_bands: 1,
            data_type: DataType::Double,
            valid_masks: vec![BitMask::all_valid(size * size)],
            data: SampleData::F64(pixels.clone()),
            ..Default::default()
        };
        let rust_blob = lerc::encode(&image, Precision::Lossless).unwrap();
        let cpp_blob = cpp::encode(
            &pixels,
            cpp::DT_DOUBLE,
            size as i32,
            size as i32,
            1,
            1,
            Some(&valid),
            0.0,
        );

        group.throughput(Throughput::Bytes(pixel_bytes));

        group.bench_with_input(
            BenchmarkId::new("rust_blob/rust", size),
            &rust_blob,
            |b, blob| {
                b.iter(|| lerc::decode(blob).unwrap());
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cpp_blob/rust", size),
            &cpp_blob,
            |b, blob| {
                b.iter(|| lerc::decode(blob).unwrap());
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cpp_blob/cpp", size),
            &cpp_blob,
            |b, blob| {
                b.iter(|| cpp::decode::<f64>(blob, cpp::DT_DOUBLE, size as i32, size as i32, 1, 1));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_decode_reference,
    bench_encode_f32,
    bench_encode_u8,
    bench_decode_synthetic,
    bench_decode_f64_lossless,
);
criterion_main!(benches);
