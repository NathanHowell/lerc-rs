/// Field-separated subtraction for transformed f32 values.
/// Mantissa (23 bits) and sign+exponent (9 bits) wrap independently.
/// Matches C++ `SUB32_BIT_FLT`.
fn sub32_bit_flt(a: u32, b: u32) -> u32 {
    const FLT_MANT_MASK: u32 = 0x007F_FFFF;
    const FLT_9BIT_MASK: u32 = 0xFF80_0000;

    let ret = (a.wrapping_sub(b)) & FLT_MANT_MASK;

    let ae = ((a & FLT_9BIT_MASK) >> 23) & 0x1FF;
    let be = ((b & FLT_9BIT_MASK) >> 23) & 0x1FF;

    ret | (((ae.wrapping_sub(be)) & 0x1FF) << 23)
}

/// Field-separated addition for transformed f32 values.
/// Matches C++ `ADD32_BIT_FLT`.
fn add32_bit_flt(a: u32, b: u32) -> u32 {
    const FLT_MANT_MASK: u32 = 0x007F_FFFF;
    const FLT_9BIT_MASK: u32 = 0xFF80_0000;

    let ret = (a.wrapping_add(b)) & FLT_MANT_MASK;

    let ae = ((a & FLT_9BIT_MASK) >> 23) & 0x1FF;
    let be = ((b & FLT_9BIT_MASK) >> 23) & 0x1FF;

    ret | (((ae.wrapping_add(be)) & 0x1FF) << 23)
}

/// Field-separated subtraction for transformed f64 values.
/// Mantissa (52 bits) and sign+exponent (12 bits) wrap independently.
/// Matches C++ `SUB64_BIT_DBL`.
fn sub64_bit_dbl(a: u64, b: u64) -> u64 {
    const DBL_MANT_MASK: u64 = 0x000F_FFFF_FFFF_FFFF;
    const DBL_12BIT_MASK: u64 = 0xFFF0_0000_0000_0000;

    let ret = (a.wrapping_sub(b)) & DBL_MANT_MASK;

    let ae = ((a & DBL_12BIT_MASK) >> 52) & 0xFFF;
    let be = ((b & DBL_12BIT_MASK) >> 52) & 0xFFF;

    ret | (((ae.wrapping_sub(be)) & 0xFFF) << 52)
}

/// Field-separated addition for transformed f64 values.
/// Matches C++ `ADD64_BIT_DBL`.
fn add64_bit_dbl(a: u64, b: u64) -> u64 {
    const DBL_MANT_MASK: u64 = 0x000F_FFFF_FFFF_FFFF;
    const DBL_12BIT_MASK: u64 = 0xFFF0_0000_0000_0000;

    let ret = (a.wrapping_add(b)) & DBL_MANT_MASK;

    let ae = ((a & DBL_12BIT_MASK) >> 52) & 0xFFF;
    let be = ((b & DBL_12BIT_MASK) >> 52) & 0xFFF;

    ret | (((ae.wrapping_add(be)) & 0xFFF) << 52)
}

/// Read a u32 from a LE byte slice at the given offset.
#[inline]
fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}

/// Write a u32 to a LE byte slice at the given offset.
#[inline]
fn write_u32_le(data: &mut [u8], offset: usize, val: u32) {
    data[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
}

/// Read a u64 from a LE byte slice at the given offset.
#[inline]
fn read_u64_le(data: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap())
}

/// Write a u64 to a LE byte slice at the given offset.
#[inline]
fn write_u64_le(data: &mut [u8], offset: usize, val: u64) {
    data[offset..offset + 8].copy_from_slice(&val.to_le_bytes());
}

/// Restore derivative sequence for a byte plane.
/// This undoes the delta encoding applied during compression.
/// The delta is applied linearly over the entire plane (1D), matching the C++ `restoreSequence`.
/// Levels are processed in reverse order (level down to 1) to correctly invert
/// the forward delta which applies levels 1 up to level.
pub(super) fn restore_sequence(data: &mut [u8], _width: usize, level: u8) {
    if data.is_empty() || level == 0 {
        return;
    }

    let size = data.len();

    // C++ restoreSequence: for (int l = level; l > 0; l--)
    for l in (1..=level as usize).rev() {
        for i in l..size {
            data[i] = data[i].wrapping_add(data[i - 1]);
        }
    }
}

/// Restore cross (2D) prediction on interleaved unit data.
/// Uses field-separated arithmetic matching the C++ `restoreCrossBytesFloat` / `restoreCrossBytesDouble`.
/// First restores column deltas, then row deltas.
pub(super) fn restore_cross_bytes(data: &mut [u8], width: usize, height: usize, unit_size: usize) {
    match unit_size {
        4 => restore_cross_bytes_f32(data, width, height),
        8 => restore_cross_bytes_f64(data, width, height),
        _ => unreachable!("unexpected unit_size {unit_size}"),
    }
}

fn restore_cross_bytes_f32(data: &mut [u8], width: usize, height: usize) {
    // Restore column deltas (matches C++ restoreCrossBytesFloat delta==2 block)
    for col in 0..width {
        let mut offset = width; // start at row 1
        for _row in 1..height {
            let prev_off = (offset - width) * 4 + col * 4;
            let cur_off = offset * 4 + col * 4;
            let prev = read_u32_le(data, prev_off);
            let cur = read_u32_le(data, cur_off);
            write_u32_le(data, cur_off, add32_bit_flt(cur, prev));
            offset += width;
        }
    }

    // Restore row deltas
    for row in 0..height {
        for col in 1..width {
            let prev_off = (row * width + col - 1) * 4;
            let cur_off = (row * width + col) * 4;
            let prev = read_u32_le(data, prev_off);
            let cur = read_u32_le(data, cur_off);
            write_u32_le(data, cur_off, add32_bit_flt(cur, prev));
        }
    }
}

fn restore_cross_bytes_f64(data: &mut [u8], width: usize, height: usize) {
    // Restore column deltas
    for col in 0..width {
        let mut offset = width;
        for _row in 1..height {
            let prev_off = (offset - width) * 8 + col * 8;
            let cur_off = offset * 8 + col * 8;
            let prev = read_u64_le(data, prev_off);
            let cur = read_u64_le(data, cur_off);
            write_u64_le(data, cur_off, add64_bit_dbl(cur, prev));
            offset += width;
        }
    }

    // Restore row deltas
    for row in 0..height {
        for col in 1..width {
            let prev_off = (row * width + col - 1) * 8;
            let cur_off = (row * width + col) * 8;
            let prev = read_u64_le(data, prev_off);
            let cur = read_u64_le(data, cur_off);
            write_u64_le(data, cur_off, add64_bit_dbl(cur, prev));
        }
    }
}

/// Restore byte-order prediction (1D delta) on interleaved unit data.
/// Uses field-separated arithmetic matching the C++ `restoreBlockSequenceFloat` / `restoreBlockSequenceDouble`.
pub(super) fn restore_byte_order(data: &mut [u8], width: usize, height: usize, unit_size: usize) {
    match unit_size {
        4 => restore_byte_order_f32(data, width, height),
        8 => restore_byte_order_f64(data, width, height),
        _ => unreachable!("unexpected unit_size {unit_size}"),
    }
}

fn restore_byte_order_f32(data: &mut [u8], width: usize, height: usize) {
    // Matches C++ restoreBlockSequenceFloat with nDelta=1:
    // for each row, cumulative add from col 1..width
    for row in 0..height {
        for col in 1..width {
            let prev_off = (row * width + col - 1) * 4;
            let cur_off = (row * width + col) * 4;
            let prev = read_u32_le(data, prev_off);
            let cur = read_u32_le(data, cur_off);
            write_u32_le(data, cur_off, add32_bit_flt(cur, prev));
        }
    }
}

fn restore_byte_order_f64(data: &mut [u8], width: usize, height: usize) {
    for row in 0..height {
        for col in 1..width {
            let prev_off = (row * width + col - 1) * 8;
            let cur_off = (row * width + col) * 8;
            let prev = read_u64_le(data, prev_off);
            let cur = read_u64_le(data, cur_off);
            write_u64_le(data, cur_off, add64_bit_dbl(cur, prev));
        }
    }
}

/// Apply forward delta sequence for a byte plane (inverse of restore_sequence).
/// The delta is applied linearly over the entire plane (1D), matching the C++ `setDerivative`.
/// Levels are processed in forward order (1 up to level), which is the inverse
/// of restore_sequence that processes levels in reverse order (level down to 1).
pub(super) fn apply_sequence(data: &mut [u8], _width: usize, level: u8) {
    if data.is_empty() || level == 0 {
        return;
    }

    let size = data.len();

    // C++ setDerivative: for (int l = 1; l <= level; l++)
    for l in 1..=level as usize {
        for i in (l..size).rev() {
            data[i] = data[i].wrapping_sub(data[i - 1]);
        }
    }
}

/// Apply forward cross (2D) prediction on interleaved unit data (inverse of restore_cross_bytes).
/// Uses field-separated arithmetic matching the C++ `setCrossDerivativeFloat` / `setCrossDerivativeDouble`.
/// First applies row deltas (right to left within each row),
/// then applies column deltas (bottom to top within each column).
pub(super) fn apply_cross_bytes(data: &mut [u8], width: usize, height: usize, unit_size: usize) {
    match unit_size {
        4 => apply_cross_bytes_f32(data, width, height),
        8 => apply_cross_bytes_f64(data, width, height),
        _ => unreachable!("unexpected unit_size {unit_size}"),
    }
}

fn apply_cross_bytes_f32(data: &mut [u8], width: usize, height: usize) {
    // Forward row deltas (right to left) -- matches C++ setCrossDerivativeFloat phase 0/1
    for row in 0..height {
        for col in (1..width).rev() {
            let prev_off = (row * width + col - 1) * 4;
            let cur_off = (row * width + col) * 4;
            let prev = read_u32_le(data, prev_off);
            let cur = read_u32_le(data, cur_off);
            write_u32_le(data, cur_off, sub32_bit_flt(cur, prev));
        }
    }

    // Forward column deltas (bottom to top) -- matches C++ setCrossDerivativeFloat phase 0/2
    for col in 0..width {
        let mut offset = (height - 1) * width;
        for _row in (1..height).rev() {
            let cur_off = (offset + col) * 4;
            let prev_off = (offset - width + col) * 4;
            let prev = read_u32_le(data, prev_off);
            let cur = read_u32_le(data, cur_off);
            write_u32_le(data, cur_off, sub32_bit_flt(cur, prev));
            offset -= width;
        }
    }
}

fn apply_cross_bytes_f64(data: &mut [u8], width: usize, height: usize) {
    // Forward row deltas (right to left)
    for row in 0..height {
        for col in (1..width).rev() {
            let prev_off = (row * width + col - 1) * 8;
            let cur_off = (row * width + col) * 8;
            let prev = read_u64_le(data, prev_off);
            let cur = read_u64_le(data, cur_off);
            write_u64_le(data, cur_off, sub64_bit_dbl(cur, prev));
        }
    }

    // Forward column deltas (bottom to top)
    for col in 0..width {
        let mut offset = (height - 1) * width;
        for _row in (1..height).rev() {
            let cur_off = (offset + col) * 8;
            let prev_off = (offset - width + col) * 8;
            let prev = read_u64_le(data, prev_off);
            let cur = read_u64_le(data, cur_off);
            write_u64_le(data, cur_off, sub64_bit_dbl(cur, prev));
            offset -= width;
        }
    }
}

/// Apply only the column delta (phase 2) of cross-byte prediction on interleaved
/// unit data. This is used by the predictor selector: first `apply_byte_order`
/// is called (row deltas), then this adds column deltas on top.
/// Matches C++ `setCrossDerivative` with phase=2.
pub(super) fn apply_cross_bytes_phase2(
    data: &mut [u8],
    width: usize,
    height: usize,
    unit_size: usize,
) {
    match unit_size {
        4 => apply_cross_bytes_phase2_f32(data, width, height),
        8 => apply_cross_bytes_phase2_f64(data, width, height),
        _ => unreachable!("unexpected unit_size {unit_size}"),
    }
}

fn apply_cross_bytes_phase2_f32(data: &mut [u8], width: usize, height: usize) {
    for col in 0..width {
        let mut offset = (height - 1) * width;
        for _row in (1..height).rev() {
            let cur_off = (offset + col) * 4;
            let prev_off = (offset - width + col) * 4;
            let prev = read_u32_le(data, prev_off);
            let cur = read_u32_le(data, cur_off);
            write_u32_le(data, cur_off, sub32_bit_flt(cur, prev));
            offset -= width;
        }
    }
}

fn apply_cross_bytes_phase2_f64(data: &mut [u8], width: usize, height: usize) {
    for col in 0..width {
        let mut offset = (height - 1) * width;
        for _row in (1..height).rev() {
            let cur_off = (offset + col) * 8;
            let prev_off = (offset - width + col) * 8;
            let prev = read_u64_le(data, prev_off);
            let cur = read_u64_le(data, cur_off);
            write_u64_le(data, cur_off, sub64_bit_dbl(cur, prev));
            offset -= width;
        }
    }
}

/// Apply forward byte-order prediction (1D delta) on interleaved unit data (inverse of restore_byte_order).
/// Uses field-separated arithmetic matching the C++ `setRowsDerivativeFloat` / `setRowsDerivativeDouble`.
pub(super) fn apply_byte_order(data: &mut [u8], width: usize, height: usize, unit_size: usize) {
    match unit_size {
        4 => apply_byte_order_f32(data, width, height),
        8 => apply_byte_order_f64(data, width, height),
        _ => unreachable!("unexpected unit_size {unit_size}"),
    }
}

fn apply_byte_order_f32(data: &mut [u8], width: usize, height: usize) {
    // Matches C++ setRowsDerivativeFloat with phase=1 (start_level=1, end_level=1):
    // for each row, right-to-left delta subtract from col (width-1) down to 1
    for row in 0..height {
        for col in (1..width).rev() {
            let prev_off = (row * width + col - 1) * 4;
            let cur_off = (row * width + col) * 4;
            let prev = read_u32_le(data, prev_off);
            let cur = read_u32_le(data, cur_off);
            write_u32_le(data, cur_off, sub32_bit_flt(cur, prev));
        }
    }
}

fn apply_byte_order_f64(data: &mut [u8], width: usize, height: usize) {
    for row in 0..height {
        for col in (1..width).rev() {
            let prev_off = (row * width + col - 1) * 8;
            let cur_off = (row * width + col) * 8;
            let prev = read_u64_le(data, prev_off);
            let cur = read_u64_le(data, cur_off);
            write_u64_le(data, cur_off, sub64_bit_dbl(cur, prev));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sub_add_32_bit_flt_known_values() {
        // identity: sub(a, a) == 0
        assert_eq!(sub32_bit_flt(0x3F80_0000, 0x3F80_0000), 0);
        // sub(a, 0) == a
        assert_eq!(sub32_bit_flt(0x3F80_0000, 0), 0x3F80_0000);
        // add(0, b) == b
        assert_eq!(add32_bit_flt(0, 0x3F80_0000), 0x3F80_0000);
    }

    #[test]
    fn sub_add_64_bit_dbl_known_values() {
        let one_f64 = 1.0f64.to_bits();
        assert_eq!(sub64_bit_dbl(one_f64, one_f64), 0);
        assert_eq!(sub64_bit_dbl(one_f64, 0), one_f64);
        assert_eq!(add64_bit_dbl(0, one_f64), one_f64);
    }

    #[test]
    fn apply_sequence_level_zero_is_noop() {
        let original = vec![1u8, 2, 3, 4, 5];
        let mut data = original.clone();
        apply_sequence(&mut data, 5, 0);
        assert_eq!(data, original);
    }

    #[test]
    fn apply_sequence_empty_is_noop() {
        let mut data: Vec<u8> = vec![];
        apply_sequence(&mut data, 0, 3);
        assert!(data.is_empty());
    }

    #[test]
    fn apply_sequence_level1_known() {
        // Level 1: each element replaced by element - predecessor
        let mut data = vec![10u8, 20, 35, 55];
        apply_sequence(&mut data, 4, 1);
        // differences: 10, 10, 15, 20
        assert_eq!(data, vec![10, 10, 15, 20]);
    }

    // ---- apply_cross_bytes / restore_cross_bytes round-trip for f32 ----

    #[test]
    fn apply_restore_cross_bytes_f32_round_trip() {
        let width = 4;
        let height = 4;
        // Create a 4x4 f32 image with known values
        let floats: Vec<f32> = (0..16).map(|i| i as f32 * 1.5 + 0.25).collect();
        let mut data: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        let original = data.clone();

        apply_cross_bytes(&mut data, width, height, 4);
        // After applying, data should differ from original (unless trivial)
        assert_ne!(data, original, "forward transform should modify data");
        restore_cross_bytes(&mut data, width, height, 4);
        assert_eq!(data, original, "round-trip should recover original");
    }

    #[test]
    fn apply_restore_cross_bytes_f64_round_trip() {
        let width = 3;
        let height = 3;
        let doubles: Vec<f64> = (0..9).map(|i| i as f64 * 2.5 - 1.0).collect();
        let mut data: Vec<u8> = doubles.iter().flat_map(|d| d.to_le_bytes()).collect();
        let original = data.clone();

        apply_cross_bytes(&mut data, width, height, 8);
        assert_ne!(data, original, "forward transform should modify data");
        restore_cross_bytes(&mut data, width, height, 8);
        assert_eq!(data, original, "round-trip should recover original");
    }

    // ---- apply_byte_order / restore_byte_order round-trip ----

    #[test]
    fn apply_restore_byte_order_f32_round_trip() {
        let width = 4;
        let height = 4;
        let floats: Vec<f32> = (0..16).map(|i| (i as f32).sin() * 100.0).collect();
        let mut data: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        let original = data.clone();

        apply_byte_order(&mut data, width, height, 4);
        assert_ne!(data, original, "forward transform should modify data");
        restore_byte_order(&mut data, width, height, 4);
        assert_eq!(data, original, "round-trip should recover original");
    }

    #[test]
    fn apply_restore_byte_order_f64_round_trip() {
        let width = 3;
        let height = 3;
        let doubles: Vec<f64> = (0..9).map(|i| (i as f64).cos() * 200.0).collect();
        let mut data: Vec<u8> = doubles.iter().flat_map(|d| d.to_le_bytes()).collect();
        let original = data.clone();

        apply_byte_order(&mut data, width, height, 8);
        assert_ne!(data, original, "forward transform should modify data");
        restore_byte_order(&mut data, width, height, 8);
        assert_eq!(data, original, "round-trip should recover original");
    }

    // ---- apply_cross_bytes_phase2 only applies column deltas ----

    #[test]
    fn apply_cross_bytes_phase2_only_column_deltas_f32() {
        let width = 3;
        let height = 3;
        // 3x3 f32 grid:
        // row0: [1.0, 2.0, 3.0]
        // row1: [4.0, 5.0, 6.0]
        // row2: [7.0, 8.0, 9.0]
        let floats: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut data: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();

        // phase2 only applies column deltas (bottom-to-top subtract)
        apply_cross_bytes_phase2(&mut data, width, height, 4);

        // Extract resulting f32 values
        let result: Vec<u32> = (0..9).map(|i| read_u32_le(&data, i * 4)).collect();

        // Row 0 should be unchanged (no row above to subtract)
        let row0_original: Vec<u32> = vec![1.0f32, 2.0, 3.0].iter().map(|f| f.to_bits()).collect();
        assert_eq!(&result[0..3], &row0_original[..]);

        // Row 1 and Row 2 should be modified by column deltas
        // Row 2 values = sub(row2[col], row1[col]) for each col
        // Row 1 values = sub(row1[col], row0[col]) for each col
        for col in 0..width {
            let r1_val = result[width + col];
            let expected_r1 = sub32_bit_flt(floats[width + col].to_bits(), floats[col].to_bits());
            assert_eq!(r1_val, expected_r1, "row1 col{col} should be col delta");

            let r2_val = result[2 * width + col];
            let expected_r2 = sub32_bit_flt(
                floats[2 * width + col].to_bits(),
                floats[width + col].to_bits(),
            );
            assert_eq!(r2_val, expected_r2, "row2 col{col} should be col delta");
        }
    }

    #[test]
    fn apply_cross_bytes_phase2_f64_only_column_deltas() {
        let width = 2;
        let height = 3;
        let doubles: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut data: Vec<u8> = doubles.iter().flat_map(|d| d.to_le_bytes()).collect();

        apply_cross_bytes_phase2(&mut data, width, height, 8);

        let result: Vec<u64> = (0..6).map(|i| read_u64_le(&data, i * 8)).collect();

        // Row 0 unchanged
        for col in 0..width {
            assert_eq!(result[col], doubles[col].to_bits());
        }

        // Row 1: sub(row1, row0)
        for col in 0..width {
            let expected = sub64_bit_dbl(doubles[width + col].to_bits(), doubles[col].to_bits());
            assert_eq!(result[width + col], expected);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        // ---- Field-separated arithmetic round-trips ----

        proptest! {
            #[test]
            fn sub_add_32_bit_flt_round_trip(a: u32, b: u32) {
                let diff = sub32_bit_flt(a, b);
                let recovered = add32_bit_flt(diff, b);
                prop_assert_eq!(recovered, a);
            }

            #[test]
            fn sub_add_64_bit_dbl_round_trip(a: u64, b: u64) {
                let diff = sub64_bit_dbl(a, b);
                let recovered = add64_bit_dbl(diff, b);
                prop_assert_eq!(recovered, a);
            }
        }

        // ---- apply_sequence / restore_sequence round-trip ----

        proptest! {
            #[test]
            fn apply_restore_sequence_round_trip(level in 1..5u8, len in 10..200usize) {
                let original: Vec<u8> = (0..len).map(|i| (i * 7 + 13) as u8).collect();
                let mut data = original.clone();
                apply_sequence(&mut data, len, level);
                restore_sequence(&mut data, len, level);
                prop_assert_eq!(data, original);
            }
        }

        // ---- Larger proptest round-trips for cross_bytes and byte_order ----

        proptest! {
            #[test]
            fn apply_restore_cross_bytes_f32_prop(
                width in 2..8usize,
                height in 2..8usize,
                seed in 0u32..1000,
            ) {
                let n = width * height;
                let floats: Vec<f32> = (0..n).map(|i| ((i as u32).wrapping_mul(seed.wrapping_add(1))) as f32).collect();
                let mut data: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
                let original = data.clone();

                apply_cross_bytes(&mut data, width, height, 4);
                restore_cross_bytes(&mut data, width, height, 4);
                prop_assert_eq!(data, original);
            }

            #[test]
            fn apply_restore_byte_order_f32_prop(
                width in 2..8usize,
                height in 2..8usize,
                seed in 0u32..1000,
            ) {
                let n = width * height;
                let floats: Vec<f32> = (0..n).map(|i| ((i as u32).wrapping_mul(seed.wrapping_add(1))) as f32).collect();
                let mut data: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
                let original = data.clone();

                apply_byte_order(&mut data, width, height, 4);
                restore_byte_order(&mut data, width, height, 4);
                prop_assert_eq!(data, original);
            }
        }
    }
}
