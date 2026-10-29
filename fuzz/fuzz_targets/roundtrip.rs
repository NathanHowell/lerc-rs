//! Encodes arbitrary small images and checks the decoder returns them: exactly
//! for lossless, within the tolerance for lossy.
#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use lerc::bitmask::BitMask;
use lerc::{DataType, Image, Precision, SampleData};
use libfuzzer_sys::fuzz_target;

#[derive(Debug)]
struct Case {
    width: u32,
    height: u32,
    depth: u32,
    data_type: DataType,
    /// `None` is lossless.
    tolerance: Option<f64>,
    invalid: Vec<bool>,
    raw: Vec<u64>,
}

impl<'a> Arbitrary<'a> for Case {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let width = u.int_in_range(1..=40)?;
        let height = u.int_in_range(1..=40)?;
        let depth = u.int_in_range(1..=3)?;
        let data_type = DataType::from_i32(u.int_in_range(0..=7)?).unwrap();
        let tolerance = if u.arbitrary()? {
            Some([0.01, 0.5, 1.0, 3.0][u.int_in_range(0..=3)?])
        } else {
            None
        };
        let pixels = (width * height) as usize;
        let invalid = if u.arbitrary()? {
            (0..pixels)
                .map(|_| u.arbitrary())
                .collect::<Result<_, _>>()?
        } else {
            vec![false; pixels]
        };
        // Few distinct values so that constant blocks, narrow ranges and
        // wide ranges all turn up.
        let wide: bool = u.arbitrary()?;
        let raw = (0..pixels * depth as usize)
            .map(|_| {
                let v: u64 = u.arbitrary()?;
                Ok(if wide { v } else { v % 17 })
            })
            .collect::<arbitrary::Result<_>>()?;
        Ok(Case {
            width,
            height,
            depth,
            data_type,
            tolerance,
            invalid,
            raw,
        })
    }
}

fn to_data(dt: DataType, raw: &[u64]) -> SampleData {
    match dt {
        DataType::Char => SampleData::I8(raw.iter().map(|&v| v as i8).collect()),
        DataType::Byte => SampleData::U8(raw.iter().map(|&v| v as u8).collect()),
        DataType::Short => SampleData::I16(raw.iter().map(|&v| v as i16).collect()),
        DataType::UShort => SampleData::U16(raw.iter().map(|&v| v as u16).collect()),
        DataType::Int => SampleData::I32(raw.iter().map(|&v| v as i32).collect()),
        DataType::UInt => SampleData::U32(raw.iter().map(|&v| v as u32).collect()),
        // Finite values only: the format has no NaN or infinity.
        DataType::Float => SampleData::F32(raw.iter().map(|&v| (v as i32 as f32) / 64.0).collect()),
        DataType::Double => {
            SampleData::F64(raw.iter().map(|&v| (v as i64 as f64) / 64.0).collect())
        }
    }
}

fn as_f64(d: &SampleData) -> Vec<f64> {
    match d {
        SampleData::I8(v) => v.iter().map(|&x| x as f64).collect(),
        SampleData::U8(v) => v.iter().map(|&x| x as f64).collect(),
        SampleData::I16(v) => v.iter().map(|&x| x as f64).collect(),
        SampleData::U16(v) => v.iter().map(|&x| x as f64).collect(),
        SampleData::I32(v) => v.iter().map(|&x| x as f64).collect(),
        SampleData::U32(v) => v.iter().map(|&x| x as f64).collect(),
        SampleData::F32(v) => v.iter().map(|&x| x as f64).collect(),
        SampleData::F64(v) => v.clone(),
    }
}

fuzz_target!(|case: Case| {
    let pixels = (case.width * case.height) as usize;
    let mut mask = BitMask::all_valid(pixels);
    for (k, &bad) in case.invalid.iter().enumerate() {
        if bad {
            mask.set_invalid(k);
        }
    }
    let image = Image {
        width: case.width,
        height: case.height,
        depth: case.depth,
        bands: 1,
        data_type: case.data_type,
        valid_masks: vec![mask.clone()],
        data: to_data(case.data_type, &case.raw),
        no_data_value: None,
    };
    let precision = match case.tolerance {
        Some(t) => Precision::Tolerance(t),
        None => Precision::Lossless,
    };
    let Ok(blob) = lerc::encode(&image, precision) else {
        return;
    };
    let decoded = lerc::decode(&blob).expect("a blob we encoded must decode");
    assert_eq!(
        (decoded.width, decoded.height, decoded.depth, decoded.bands),
        (case.width, case.height, case.depth, 1)
    );

    let tolerance = match case.tolerance {
        // Integer data is lossless at any tolerance below 1.
        None if case.data_type.is_integer() => 0.0,
        None => 0.0,
        Some(t) => t,
    };
    let (want, got) = (as_f64(&image.data), as_f64(&decoded.data));
    let depth = case.depth as usize;
    for k in 0..pixels {
        if !mask.is_valid(k) {
            continue;
        }
        for m in 0..depth {
            let (a, b) = (want[k * depth + m], got[k * depth + m]);
            // The decoder rounds to the sample type, so allow one unit in the
            // last place of a float.
            let slack = match case.data_type {
                DataType::Float => a.abs() * f64::from(f32::EPSILON),
                DataType::Double => a.abs() * f64::EPSILON,
                _ => 0.0,
            };
            assert!(
                (a - b).abs() <= tolerance + slack + 1e-9,
                "pixel {k} depth {m}: encoded {a}, decoded {b}, tolerance {tolerance}"
            );
        }
    }
});
