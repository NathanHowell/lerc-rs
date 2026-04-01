use lerc::Precision;
use lerc::bitmask::BitMask;
use lerc::{DataType, LercImage, SampleData};
use std::hint::black_box;
use std::time::Instant;

fn make_f32_gradient(width: usize, height: usize) -> Vec<f32> {
    (0..width * height)
        .map(|i| {
            let x = (i % width) as f32 / width as f32;
            let y = (i / width) as f32 / height as f32;
            x * 1000.0 + y * 500.0 + (x * 31.4).sin() * 50.0
        })
        .collect()
}

fn main() {
    let size = 512;
    let pixels = make_f32_gradient(size, size);
    let image = LercImage {
        width: size as u32,
        height: size as u32,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Float,
        valid_masks: vec![BitMask::all_valid(size * size)],
        data: SampleData::F32(pixels.clone()),
        ..Default::default()
    };

    let u8_pixels: Vec<u8> = (0..size * size)
        .map(|i| ((i * 7 + 13) % 256) as u8)
        .collect();
    let u8_image = LercImage {
        width: size as u32,
        height: size as u32,
        n_depth: 1,
        n_bands: 1,
        data_type: DataType::Byte,
        valid_masks: vec![BitMask::all_valid(size * size)],
        data: SampleData::U8(u8_pixels),
        ..Default::default()
    };

    let mode = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    let iters = 100;

    if mode == "all" || mode == "encode-lossy" {
        let start = Instant::now();
        for _ in 0..iters {
            black_box(lerc::encode(black_box(&image), Precision::Tolerance(0.01)).unwrap());
        }
        let elapsed = start.elapsed();
        eprintln!(
            "encode f32 512x512 lossy: {:.2} ms/iter ({iters} iters)",
            elapsed.as_secs_f64() / iters as f64 * 1000.0
        );
    }

    if mode == "all" || mode == "encode-lossless" {
        let start = Instant::now();
        for _ in 0..iters {
            black_box(lerc::encode(black_box(&image), Precision::Lossless).unwrap());
        }
        let elapsed = start.elapsed();
        eprintln!(
            "encode f32 512x512 lossless: {:.2} ms/iter ({iters} iters)",
            elapsed.as_secs_f64() / iters as f64 * 1000.0
        );
    }

    if mode == "all" || mode == "encode-u8" {
        let start = Instant::now();
        for _ in 0..iters {
            black_box(lerc::encode(black_box(&u8_image), Precision::Lossless).unwrap());
        }
        let elapsed = start.elapsed();
        eprintln!(
            "encode u8 512x512 lossless: {:.2} ms/iter ({iters} iters)",
            elapsed.as_secs_f64() / iters as f64 * 1000.0
        );
    }

    if mode == "all" || mode == "decode" {
        let blob = lerc::encode(&image, Precision::Tolerance(0.01)).unwrap();
        let start = Instant::now();
        for _ in 0..iters {
            black_box(lerc::decode(black_box(&blob)).unwrap());
        }
        let elapsed = start.elapsed();
        eprintln!(
            "decode f32 512x512 lossy: {:.2} ms/iter ({iters} iters)",
            elapsed.as_secs_f64() / iters as f64 * 1000.0
        );
    }
}
