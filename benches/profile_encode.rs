use criterion::{Criterion, criterion_group, criterion_main};
use lerc::Precision;
use lerc::bitmask::BitMask;
use lerc::{DataType, LercData, LercImage};

fn make_f32_gradient(width: usize, height: usize) -> Vec<f32> {
    (0..width * height)
        .map(|i| {
            let x = (i % width) as f32 / width as f32;
            let y = (i / width) as f32 / height as f32;
            x * 1000.0 + y * 500.0 + (x * 31.4).sin() * 50.0
        })
        .collect()
}

fn bench_encode_512_lossy(c: &mut Criterion) {
    let size = 512;
    let pixels = make_f32_gradient(size, size);
    let image = LercImage {
        width: size as u32,
        height: size as u32,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![BitMask::all_valid(size * size)],
        data: LercData::F32(pixels),
        ..Default::default()
    };

    c.bench_function("encode_f32_512_lossy", |b| {
        b.iter(|| lerc::encode(&image, Precision::Tolerance(0.01)).unwrap());
    });
}

fn bench_encode_512_lossless(c: &mut Criterion) {
    let size = 512;
    let pixels = make_f32_gradient(size, size);
    let image = LercImage {
        width: size as u32,
        height: size as u32,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![BitMask::all_valid(size * size)],
        data: LercData::F32(pixels),
        ..Default::default()
    };

    c.bench_function("encode_f32_512_lossless", |b| {
        b.iter(|| lerc::encode(&image, Precision::Lossless).unwrap());
    });
}

fn bench_encode_u8_512(c: &mut Criterion) {
    let size = 512;
    let pixels: Vec<u8> = (0..size * size)
        .map(|i| ((i * 7 + 13) % 256) as u8)
        .collect();
    let image = LercImage {
        width: size as u32,
        height: size as u32,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Byte,
        valid_masks: vec![BitMask::all_valid(size * size)],
        data: LercData::U8(pixels),
        ..Default::default()
    };

    c.bench_function("encode_u8_512_lossless", |b| {
        b.iter(|| lerc::encode(&image, Precision::Lossless).unwrap());
    });
}

fn bench_decode_512_lossy(c: &mut Criterion) {
    let size = 512;
    let pixels = make_f32_gradient(size, size);
    let image = LercImage {
        width: size as u32,
        height: size as u32,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![BitMask::all_valid(size * size)],
        data: LercData::F32(pixels),
        ..Default::default()
    };
    let blob = lerc::encode(&image, Precision::Tolerance(0.01)).unwrap();

    c.bench_function("decode_f32_512_lossy", |b| {
        b.iter(|| lerc::decode(&blob).unwrap());
    });
}

criterion_group!(
    benches,
    bench_encode_512_lossy,
    bench_encode_512_lossless,
    bench_encode_u8_512,
    bench_decode_512_lossy,
);
criterion_main!(benches);
