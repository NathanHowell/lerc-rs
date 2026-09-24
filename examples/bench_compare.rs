//! One-off decode-speed comparison against the ahenshaw/lerc-rs README table,
//! using the same 1024×1024 fixture sizes. Not part of the permanent bench
//! suite -- just `cargo run --release --example bench_compare`.
//!
//! Uses lerc-cpp-ref, a non-wasm32-only dev-dependency (see Cargo.toml), so
//! this compiles an empty stub on wasm32 the same way benches/ does.

#[cfg(not(target_arch = "wasm32"))]
mod real {
    use lerc::Precision;
    use lerc::bitmask::BitMask;
    use lerc::{DataType, Image, SampleData};
    use lerc_cpp_ref as cpp;
    use std::hint::black_box;
    use std::time::Instant;

    const SIZE: usize = 1024;
    const ITERS: u32 = 30;

    fn make_u8(size: usize) -> Vec<u8> {
        (0..size * size)
            .map(|i| ((i * 7 + 13) % 256) as u8)
            .collect()
    }

    fn make_i16(size: usize) -> Vec<i16> {
        (0..size * size)
            .map(|i| (((i * 37 + 11) % 60000) as i32 - 30000) as i16)
            .collect()
    }

    fn make_f32(size: usize) -> Vec<f32> {
        (0..size * size)
            .map(|i| {
                let x = (i % size) as f32 / size as f32;
                let y = (i / size) as f32 / size as f32;
                x * 1000.0 + y * 500.0 + (x * 31.4).sin() * 50.0
            })
            .collect()
    }

    fn make_f64(size: usize) -> Vec<f64> {
        (0..size * size)
            .map(|i| {
                let x = (i % size) as f64 / size as f64;
                let y = (i / size) as f64 / size as f64;
                (x * std::f64::consts::PI).sin() * (y * std::f64::consts::E).cos() * 3000.0
            })
            .collect()
    }

    fn valid(size: usize) -> Vec<u8> {
        vec![1u8; size * size]
    }

    /// Median of `ITERS` timed runs of `f`.
    fn time_ms<F: FnMut()>(mut f: F) -> f64 {
        let mut samples = Vec::with_capacity(ITERS as usize);
        for _ in 0..ITERS {
            let start = Instant::now();
            f();
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        samples[samples.len() / 2]
    }

    fn report(label: &str, rust_ms: f64, cpp_ms: f64) {
        println!(
            "{label:<32} rust={rust_ms:8.3}ms  cpp={cpp_ms:8.3}ms  ratio(rust/cpp)={:.2}x",
            rust_ms / cpp_ms
        );
    }

    pub fn main() {
        println!("Decode-only comparison, {SIZE}x{SIZE}, median of {ITERS} runs\n");

        // u8 lossless, single band
        {
            let px = make_u8(SIZE);
            let v = valid(SIZE);
            let image = Image {
                width: SIZE as u32,
                height: SIZE as u32,
                depth: 1,
                bands: 1,
                data_type: DataType::Byte,
                valid_masks: vec![BitMask::all_valid(SIZE * SIZE)],
                data: SampleData::U8(px.clone()),
                ..Default::default()
            };
            let blob = lerc::encode(&image, Precision::Lossless).unwrap();
            let cpp_blob = cpp::encode(
                &px,
                cpp::DT_UCHAR,
                SIZE as i32,
                SIZE as i32,
                1,
                1,
                Some(&v),
                0.5,
            );

            let rust_ms = time_ms(|| {
                black_box(lerc::decode(black_box(&blob)).unwrap());
            });
            let cpp_ms = time_ms(|| {
                black_box(cpp::decode::<u8>(
                    black_box(&cpp_blob),
                    cpp::DT_UCHAR,
                    SIZE as i32,
                    SIZE as i32,
                    1,
                    1,
                ));
            });
            report("u8 1MP lossless", rust_ms, cpp_ms);
        }

        // i16 lossless, single band
        {
            let px = make_i16(SIZE);
            let v = valid(SIZE);
            let image = Image {
                width: SIZE as u32,
                height: SIZE as u32,
                depth: 1,
                bands: 1,
                data_type: DataType::Short,
                valid_masks: vec![BitMask::all_valid(SIZE * SIZE)],
                data: SampleData::I16(px.clone()),
                ..Default::default()
            };
            let blob = lerc::encode(&image, Precision::Lossless).unwrap();
            let cpp_blob = cpp::encode(
                &px,
                cpp::DT_SHORT,
                SIZE as i32,
                SIZE as i32,
                1,
                1,
                Some(&v),
                0.5,
            );

            let rust_ms = time_ms(|| {
                black_box(lerc::decode(black_box(&blob)).unwrap());
            });
            let cpp_ms = time_ms(|| {
                black_box(cpp::decode::<i16>(
                    black_box(&cpp_blob),
                    cpp::DT_SHORT,
                    SIZE as i32,
                    SIZE as i32,
                    1,
                    1,
                ));
            });
            report("i16 1MP lossless", rust_ms, cpp_ms);
        }

        // f32 lossy 0.01
        {
            let px = make_f32(SIZE);
            let v = valid(SIZE);
            let image = Image {
                width: SIZE as u32,
                height: SIZE as u32,
                depth: 1,
                bands: 1,
                data_type: DataType::Float,
                valid_masks: vec![BitMask::all_valid(SIZE * SIZE)],
                data: SampleData::F32(px.clone()),
                ..Default::default()
            };
            let blob = lerc::encode(&image, Precision::Tolerance(0.01)).unwrap();
            let cpp_blob = cpp::encode(
                &px,
                cpp::DT_FLOAT,
                SIZE as i32,
                SIZE as i32,
                1,
                1,
                Some(&v),
                0.01,
            );

            let rust_ms = time_ms(|| {
                black_box(lerc::decode(black_box(&blob)).unwrap());
            });
            let cpp_ms = time_ms(|| {
                black_box(cpp::decode::<f32>(
                    black_box(&cpp_blob),
                    cpp::DT_FLOAT,
                    SIZE as i32,
                    SIZE as i32,
                    1,
                    1,
                ));
            });
            report("f32 1MP lossy (0.01)", rust_ms, cpp_ms);
        }

        // f32 lossless
        {
            let px = make_f32(SIZE);
            let v = valid(SIZE);
            let image = Image {
                width: SIZE as u32,
                height: SIZE as u32,
                depth: 1,
                bands: 1,
                data_type: DataType::Float,
                valid_masks: vec![BitMask::all_valid(SIZE * SIZE)],
                data: SampleData::F32(px.clone()),
                ..Default::default()
            };
            let blob = lerc::encode(&image, Precision::Lossless).unwrap();
            let cpp_blob = cpp::encode(
                &px,
                cpp::DT_FLOAT,
                SIZE as i32,
                SIZE as i32,
                1,
                1,
                Some(&v),
                0.0,
            );

            let rust_ms = time_ms(|| {
                black_box(lerc::decode(black_box(&blob)).unwrap());
            });
            let cpp_ms = time_ms(|| {
                black_box(cpp::decode::<f32>(
                    black_box(&cpp_blob),
                    cpp::DT_FLOAT,
                    SIZE as i32,
                    SIZE as i32,
                    1,
                    1,
                ));
            });
            report("f32 1MP lossless", rust_ms, cpp_ms);
        }

        // f64 lossy 0.01
        {
            let px = make_f64(SIZE);
            let v = valid(SIZE);
            let image = Image {
                width: SIZE as u32,
                height: SIZE as u32,
                depth: 1,
                bands: 1,
                data_type: DataType::Double,
                valid_masks: vec![BitMask::all_valid(SIZE * SIZE)],
                data: SampleData::F64(px.clone()),
                ..Default::default()
            };
            let blob = lerc::encode(&image, Precision::Tolerance(0.01)).unwrap();
            let cpp_blob = cpp::encode(
                &px,
                cpp::DT_DOUBLE,
                SIZE as i32,
                SIZE as i32,
                1,
                1,
                Some(&v),
                0.01,
            );

            let rust_ms = time_ms(|| {
                black_box(lerc::decode(black_box(&blob)).unwrap());
            });
            let cpp_ms = time_ms(|| {
                black_box(cpp::decode::<f64>(
                    black_box(&cpp_blob),
                    cpp::DT_DOUBLE,
                    SIZE as i32,
                    SIZE as i32,
                    1,
                    1,
                ));
            });
            report("f64 1MP lossy (0.01)", rust_ms, cpp_ms);
        }

        // u8, 3 bands, lossless
        {
            let bands = 3;
            let mut all_px = Vec::with_capacity(SIZE * SIZE * bands);
            for b in 0..bands {
                all_px.extend(
                    make_u8(SIZE)
                        .into_iter()
                        .map(|v| v.wrapping_add(b as u8 * 17)),
                );
            }
            let masks: Vec<_> = (0..bands)
                .map(|_| BitMask::all_valid(SIZE * SIZE))
                .collect();
            let image = Image {
                width: SIZE as u32,
                height: SIZE as u32,
                depth: 1,
                bands: bands as u32,
                data_type: DataType::Byte,
                valid_masks: masks,
                data: SampleData::U8(all_px.clone()),
                ..Default::default()
            };
            let blob = lerc::encode(&image, Precision::Lossless).unwrap();
            let v = valid(SIZE);
            let cpp_blob = cpp::encode(
                &all_px,
                cpp::DT_UCHAR,
                SIZE as i32,
                SIZE as i32,
                1,
                bands as i32,
                Some(&v),
                0.5,
            );

            let rust_ms = time_ms(|| {
                black_box(lerc::decode(black_box(&blob)).unwrap());
            });
            let cpp_ms = time_ms(|| {
                black_box(cpp::decode::<u8>(
                    black_box(&cpp_blob),
                    cpp::DT_UCHAR,
                    SIZE as i32,
                    SIZE as i32,
                    1,
                    bands as i32,
                ));
            });
            report("u8 1MP x3 bands lossless", rust_ms, cpp_ms);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    real::main();
}

#[cfg(target_arch = "wasm32")]
fn main() {}
