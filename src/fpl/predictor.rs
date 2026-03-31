/// Restore derivative sequence for a byte plane.
/// This undoes the delta encoding applied during compression.
/// The delta is applied linearly over the entire plane (1D), matching the C++ FPL format.
pub(super) fn restore_sequence(data: &mut [u8], _width: usize, level: u8) {
    if data.is_empty() || level == 0 {
        return;
    }

    let size = data.len();

    // Apply inverse delta for each level (1D linear over entire buffer)
    for l in 1..=level as usize {
        for i in l..size {
            data[i] = data[i].wrapping_add(data[i - 1]);
        }
    }
}

/// Restore cross (2D) prediction on interleaved unit data.
/// Undoes column deltas then row deltas.
pub(super) fn restore_cross_bytes(
    data: &mut [u8],
    width: usize,
    height: usize,
    unit_size: usize,
) {
    let stride = width * unit_size;

    // Restore column deltas
    for row in 1..height {
        for col in 0..width {
            for b in 0..unit_size {
                let idx = row * stride + col * unit_size + b;
                let prev = (row - 1) * stride + col * unit_size + b;
                data[idx] = data[idx].wrapping_add(data[prev]);
            }
        }
    }

    // Restore row deltas
    for row in 0..height {
        for col in 1..width {
            for b in 0..unit_size {
                let idx = row * stride + col * unit_size + b;
                let prev = row * stride + (col - 1) * unit_size + b;
                data[idx] = data[idx].wrapping_add(data[prev]);
            }
        }
    }
}

/// Restore byte-order prediction (1D delta) on interleaved unit data.
pub(super) fn restore_byte_order(
    data: &mut [u8],
    width: usize,
    height: usize,
    unit_size: usize,
) {
    let stride = width * unit_size;

    for row in 0..height {
        for col in 1..width {
            for b in 0..unit_size {
                let idx = row * stride + col * unit_size + b;
                let prev = row * stride + (col - 1) * unit_size + b;
                data[idx] = data[idx].wrapping_add(data[prev]);
            }
        }
    }
}

/// Apply forward delta sequence for a byte plane (inverse of restore_sequence).
/// The delta is applied linearly over the entire plane (1D), matching the C++ FPL format.
pub(super) fn apply_sequence(data: &mut [u8], _width: usize, level: u8) {
    if data.is_empty() || level == 0 {
        return;
    }

    let size = data.len();

    for l in 1..=level as usize {
        // Forward delta: from end to start (reverse of restore)
        for i in (l..size).rev() {
            data[i] = data[i].wrapping_sub(data[i - 1]);
        }
    }
}

/// Apply forward cross (2D) prediction on interleaved unit data (inverse of restore_cross_bytes).
/// First applies row deltas (right to left within each row),
/// then applies column deltas (bottom to top within each column).
pub(super) fn apply_cross_bytes(
    data: &mut [u8],
    width: usize,
    height: usize,
    unit_size: usize,
) {
    let stride = width * unit_size;

    // Forward row deltas (right to left)
    for row in 0..height {
        for col in (1..width).rev() {
            for b in 0..unit_size {
                let idx = row * stride + col * unit_size + b;
                let prev = row * stride + (col - 1) * unit_size + b;
                data[idx] = data[idx].wrapping_sub(data[prev]);
            }
        }
    }

    // Forward column deltas (bottom to top)
    for row in (1..height).rev() {
        for col in 0..width {
            for b in 0..unit_size {
                let idx = row * stride + col * unit_size + b;
                let prev = (row - 1) * stride + col * unit_size + b;
                data[idx] = data[idx].wrapping_sub(data[prev]);
            }
        }
    }
}

/// Apply forward byte-order prediction (1D delta) on interleaved unit data (inverse of restore_byte_order).
pub(super) fn apply_byte_order(
    data: &mut [u8],
    width: usize,
    height: usize,
    unit_size: usize,
) {
    let stride = width * unit_size;

    for row in 0..height {
        for col in (1..width).rev() {
            for b in 0..unit_size {
                let idx = row * stride + col * unit_size + b;
                let prev = row * stride + (col - 1) * unit_size + b;
                data[idx] = data[idx].wrapping_sub(data[prev]);
            }
        }
    }
}
