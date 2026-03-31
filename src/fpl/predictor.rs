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
pub(super) fn restore_cross_bytes(
    data: &mut [u8],
    width: usize,
    height: usize,
    unit_size: usize,
) {
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
pub(super) fn restore_byte_order(
    data: &mut [u8],
    width: usize,
    height: usize,
    unit_size: usize,
) {
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
pub(super) fn apply_cross_bytes(
    data: &mut [u8],
    width: usize,
    height: usize,
    unit_size: usize,
) {
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
pub(super) fn apply_byte_order(
    data: &mut [u8],
    width: usize,
    height: usize,
    unit_size: usize,
) {
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
