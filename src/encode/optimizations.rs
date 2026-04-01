use alloc::vec;
use alloc::vec::Vec;

use crate::bitmask::BitMask;
use crate::types::{DataType, LercDataType};

/// Try to raise `max_z_error` for float data without introducing additional loss.
///
/// When float values have limited precision (e.g., stored as "%.2f"), raising
/// maxZError to align with that precision loses nothing because the data's
/// inherent rounding error is already bounded by the original maxZError.
///
/// This mirrors the C++ `Lerc2::TryRaiseMaxZError` algorithm.
pub(super) fn try_raise_max_z_error<T: LercDataType>(
    data: &[T],
    mask: &BitMask,
    width: usize,
    height: usize,
    n_depth: usize,
    max_z_error: &mut f64,
) -> bool {
    if T::is_integer() || *max_z_error <= 0.0 {
        return false;
    }

    let num_valid = mask.count_valid().min(width * height);
    if num_valid == 0 {
        return false;
    }

    // Candidate precision levels (zErrCand) and their reciprocal multipliers (zFacCand).
    // zErr = zErrCand / 2 is the half-quantum; zFac = 1 / (2 * zErr) = zFacCand.
    let z_err_cand: [f64; 9] = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001];
    let z_fac_cand: [i32; 9] = [1, 2, 10, 20, 100, 200, 1000, 2000, 10000];

    // Keep only candidates where zErrCand[i] / 2 > maxZError (i.e., strictly coarser).
    let mut z_err: Vec<f64> = Vec::new();
    let mut z_fac: Vec<i32> = Vec::new();
    let mut round_err: Vec<f64> = Vec::new();

    for i in 0..z_err_cand.len() {
        if z_err_cand[i] / 2.0 > *max_z_error {
            z_err.push(z_err_cand[i] / 2.0);
            z_fac.push(z_fac_cand[i]);
            round_err.push(0.0);
        }
    }

    if z_err.is_empty() {
        return false;
    }

    let all_valid = num_valid == width * height;

    if n_depth == 1 && all_valid {
        // Optimized common case: single depth, all pixels valid
        for i in 0..height {
            let n_cand = z_err.len();

            for j in 0..width {
                let k = i * width + j;
                let x = data[k].to_f64();

                for n in 0..n_cand {
                    let z = x * z_fac[n] as f64;
                    // If z is already an integer, this candidate (and all coarser
                    // ones before it) have zero error for this value.
                    if z == (z as i64) as f64 {
                        break;
                    }
                    let delta = (z + 0.5).floor() - z;
                    let delta = delta.abs();
                    if delta > round_err[n] {
                        round_err[n] = delta;
                    }
                }
            }

            if !prune_candidates(&mut round_err, &mut z_err, &mut z_fac, *max_z_error) {
                return false;
            }
        }
    } else {
        // General case: nDepth > 1 or not all pixels valid
        for i in 0..height {
            let n_cand = z_err.len();

            for j in 0..width {
                let k = i * width + j;
                if !all_valid && !mask.is_valid(k) {
                    continue;
                }
                for m in 0..n_depth {
                    let x = data[k * n_depth + m].to_f64();

                    for n in 0..n_cand {
                        let z = x * z_fac[n] as f64;
                        if z == (z as i64) as f64 {
                            break;
                        }
                        let delta = (z + 0.5).floor() - z;
                        let delta = delta.abs();
                        if delta > round_err[n] {
                            round_err[n] = delta;
                        }
                    }
                }
            }

            if !prune_candidates(&mut round_err, &mut z_err, &mut z_fac, *max_z_error) {
                return false;
            }
        }
    }

    // Pick the first remaining candidate whose actual rounding error fits.
    for n in 0..z_err.len() {
        if round_err[n] / z_fac[n] as f64 <= *max_z_error / 2.0 {
            *max_z_error = z_err[n];
            return true;
        }
    }

    false
}

/// Remove candidates whose accumulated rounding error exceeds the budget.
/// Returns false if no candidates remain.
fn prune_candidates(
    round_err: &mut Vec<f64>,
    z_err: &mut Vec<f64>,
    z_fac: &mut Vec<i32>,
    max_z_error: f64,
) -> bool {
    let n_cand = z_err.len();
    if n_cand == 0 || max_z_error <= 0.0 {
        return false;
    }

    // Walk backwards so removal indices stay valid.
    for n in (0..n_cand).rev() {
        if round_err[n] / z_fac[n] as f64 > max_z_error / 2.0 {
            round_err.remove(n);
            z_err.remove(n);
            z_fac.remove(n);
        }
    }

    !z_err.is_empty()
}

/// Count set bits per bit-plane in an unsigned XOR difference value.
/// Equivalent to C++ `Lerc2::AddUIntToCounts`.
#[inline]
fn add_uint_to_counts(counts: &mut [i32], mut val: u32, n_bits: usize) {
    counts[0] += (val & 1) as i32;
    for count in &mut counts[1..n_bits] {
        val >>= 1;
        *count += (val & 1) as i32;
    }
}

/// Count set bits per bit-plane in a signed XOR difference value.
/// Equivalent to C++ `Lerc2::AddIntToCounts`.
#[inline]
fn add_int_to_counts(counts: &mut [i32], mut val: i32, n_bits: usize) {
    counts[0] += val & 1;
    for count in &mut counts[1..n_bits] {
        val >>= 1;
        *count += val & 1;
    }
}

/// Try bit-plane compression for integer data. Analyzes XOR differences between
/// neighboring pixels to find bit planes that look like random noise (1-bit
/// frequency close to 0.5). Returns `Some(new_max_z_error)` if noisy bit planes
/// were found, or `None` if the data doesn't qualify.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_bit_plane_compression<T: LercDataType>(
    data: &[T],
    valid_masks: &[BitMask],
    width: usize,
    height: usize,
    n_depth: usize,
    n_bands: usize,
    eps: f64,
    data_type: DataType,
) -> Option<f64> {
    if eps <= 0.0 {
        return None;
    }

    let max_shift = data_type.size() * 8;
    let min_cnt: usize = 5000;
    let is_signed = data_type.is_signed();
    let band_size = width * height * n_depth;
    let mut overall_result: Option<f64> = None;

    for band in 0..n_bands {
        let band_data = &data[band * band_size..(band + 1) * band_size];
        let mask = if band < valid_masks.len() {
            &valid_masks[band]
        } else {
            &valid_masks[0]
        };

        let num_valid = mask.count_valid().min(width * height);
        if num_valid < min_cnt {
            continue;
        }

        let mut cnt_diff_vec = vec![0i32; n_depth * max_shift];
        let mut cnt: usize = 0;
        let all_valid = num_valid == width * height;

        if n_depth == 1 && all_valid {
            if !is_signed {
                for i in 0..height - 1 {
                    for j in 0..width - 1 {
                        let k = i * width + j;
                        let c = (band_data[k].to_f64() as u32)
                            ^ (band_data[k + 1].to_f64() as u32);
                        add_uint_to_counts(&mut cnt_diff_vec, c, max_shift);
                        cnt += 1;
                        let c = (band_data[k].to_f64() as u32)
                            ^ (band_data[k + width].to_f64() as u32);
                        add_uint_to_counts(&mut cnt_diff_vec, c, max_shift);
                        cnt += 1;
                    }
                }
            } else {
                for i in 0..height - 1 {
                    for j in 0..width - 1 {
                        let k = i * width + j;
                        let c = (band_data[k].to_f64() as i32)
                            ^ (band_data[k + 1].to_f64() as i32);
                        add_int_to_counts(&mut cnt_diff_vec, c, max_shift);
                        cnt += 1;
                        let c = (band_data[k].to_f64() as i32)
                            ^ (band_data[k + width].to_f64() as i32);
                        add_int_to_counts(&mut cnt_diff_vec, c, max_shift);
                        cnt += 1;
                    }
                }
            }
        } else {
            if !is_signed {
                for i in 0..height {
                    for j in 0..width {
                        let k = i * width + j;
                        let m0 = k * n_depth;
                        if all_valid || mask.is_valid(k) {
                            if j < width - 1 && (all_valid || mask.is_valid(k + 1)) {
                                for m in 0..n_depth {
                                    let s0 = m * max_shift;
                                    let c = (band_data[m0 + m].to_f64() as u32)
                                        ^ (band_data[m0 + m + n_depth].to_f64() as u32);
                                    add_uint_to_counts(&mut cnt_diff_vec[s0..], c, max_shift);
                                }
                                cnt += 1;
                            }
                            if i < height - 1 && (all_valid || mask.is_valid(k + width)) {
                                for m in 0..n_depth {
                                    let s0 = m * max_shift;
                                    let c = (band_data[m0 + m].to_f64() as u32)
                                        ^ (band_data[m0 + m + n_depth * width].to_f64() as u32);
                                    add_uint_to_counts(&mut cnt_diff_vec[s0..], c, max_shift);
                                }
                                cnt += 1;
                            }
                        }
                    }
                }
            } else {
                for i in 0..height {
                    for j in 0..width {
                        let k = i * width + j;
                        let m0 = k * n_depth;
                        if all_valid || mask.is_valid(k) {
                            if j < width - 1 && (all_valid || mask.is_valid(k + 1)) {
                                for m in 0..n_depth {
                                    let s0 = m * max_shift;
                                    let c = (band_data[m0 + m].to_f64() as i32)
                                        ^ (band_data[m0 + m + n_depth].to_f64() as i32);
                                    add_int_to_counts(&mut cnt_diff_vec[s0..], c, max_shift);
                                }
                                cnt += 1;
                            }
                            if i < height - 1 && (all_valid || mask.is_valid(k + width)) {
                                for m in 0..n_depth {
                                    let s0 = m * max_shift;
                                    let c = (band_data[m0 + m].to_f64() as i32)
                                        ^ (band_data[m0 + m + n_depth * width].to_f64() as i32);
                                    add_int_to_counts(&mut cnt_diff_vec[s0..], c, max_shift);
                                }
                                cnt += 1;
                            }
                        }
                    }
                }
            }
        }

        if cnt < min_cnt {
            continue;
        }

        let mut n_cut_found = 0i32;
        let mut last_plane_kept = 0i32;

        for s in (0..max_shift as i32).rev() {
            let mut b_crit = true;
            for i_depth in 0..n_depth {
                let x = cnt_diff_vec[i_depth * max_shift + s as usize] as f64;
                let n = cnt as f64;
                let m = x / n;
                if (1.0 - 2.0 * m).abs() >= eps {
                    b_crit = false;
                }
            }

            if b_crit && n_cut_found < 2 {
                if n_cut_found == 0 {
                    last_plane_kept = s;
                }
                if n_cut_found == 1 && s < last_plane_kept - 1 {
                    last_plane_kept = s;
                    n_cut_found = 0;
                }
                n_cut_found += 1;
            }
        }

        last_plane_kept = last_plane_kept.max(0);
        let new_max_z_error = ((1u64 << last_plane_kept as u64) >> 1) as f64;

        overall_result = Some(match overall_result {
            Some(existing) => existing.min(new_max_z_error),
            None => new_max_z_error,
        });
    }

    match overall_result {
        Some(v) if v > 0.0 => Some(v),
        _ => None,
    }
}

/// Compute an internal noData sentinel value that lies below the valid data range.
/// This mirrors the C++ approach of picking a value below `zMin - 2 * maxZError`.
pub(super) fn compute_no_data_sentinel<T: LercDataType>(
    min_val: f64,
    max_z_error: f64,
) -> Option<f64> {
    let is_int = T::is_integer();

    if is_int {
        // For integer types: pick candidates below minVal - 2*maxZErr, must be integer
        let max_z_err = max_z_error;
        let threshold = min_val - 2.0 * max_z_err;
        let candidates: &[f64] = &[
            threshold - 1.0,
            threshold - 10.0,
            threshold - 100.0,
            threshold - 1000.0,
        ];

        let low_limit = T::min_representable();

        for &c in candidates {
            let c = c.floor();
            if c >= low_limit && c < threshold && c == (c as i64) as f64 {
                return Some(c);
            }
        }
        // No sentinel found — cannot represent NoData for this type/range combination
        None
    } else {
        // For float/double: pick candidates below minVal - 2*maxZErr
        let max_z_err = max_z_error;
        let threshold = min_val - 2.0 * max_z_err;
        let dist_candidates: &[f64] = &[
            4.0 * max_z_err,
            0.0001,
            0.001,
            0.01,
            0.1,
            1.0,
            10.0,
            100.0,
            1000.0,
            10000.0,
        ];

        let is_f32 = T::DATA_TYPE == DataType::Float;
        let lowest_val: f64 = if is_f32 {
            -(f32::MAX as f64)
        } else {
            -f64::MAX
        };

        let mut candidates: Vec<f64> = Vec::new();
        for &dist in dist_candidates {
            candidates.push(min_val - dist);
        }
        // candidate for large min values
        let cand = if min_val > 0.0 {
            (min_val / 2.0).floor()
        } else {
            min_val * 2.0
        };
        candidates.push(cand);

        // Sort descending (pick closest one that's far enough from valid range)
        candidates.sort_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));

        for c in candidates {
            if c > lowest_val && c < threshold {
                return Some(c);
            }
        }
        None
    }
}
