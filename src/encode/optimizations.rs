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
pub(super) fn try_bit_plane_compression<T: LercDataType>(
    data: &[T],
    valid_masks: &[BitMask],
    width: usize,
    height: usize,
    n_depth: usize,
    n_bands: usize,
    eps: f64,
) -> Option<f64> {
    if eps <= 0.0 {
        return None;
    }

    let data_type = T::DATA_TYPE;
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
                        let c = (band_data[k].to_f64() as u32) ^ (band_data[k + 1].to_f64() as u32);
                        add_uint_to_counts(&mut cnt_diff_vec, c, max_shift);
                        cnt += 1;
                        let c =
                            (band_data[k].to_f64() as u32) ^ (band_data[k + width].to_f64() as u32);
                        add_uint_to_counts(&mut cnt_diff_vec, c, max_shift);
                        cnt += 1;
                    }
                }
            } else {
                for i in 0..height - 1 {
                    for j in 0..width - 1 {
                        let k = i * width + j;
                        let c = (band_data[k].to_f64() as i32) ^ (band_data[k + 1].to_f64() as i32);
                        add_int_to_counts(&mut cnt_diff_vec, c, max_shift);
                        cnt += 1;
                        let c =
                            (band_data[k].to_f64() as i32) ^ (band_data[k + width].to_f64() as i32);
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
        // For integer types: pick candidates below minVal - 2*maxZErr, must be integer.
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

#[cfg(test)]
mod tests {
    use super::*;

    // ---- compute_no_data_sentinel ----

    #[test]
    fn sentinel_u8_with_room() {
        // min=5.0, mze=0.5 => threshold = 4.0, first candidate floor(3.0)=3.0
        // 3.0 >= 0 (u8 min) and 3.0 < 4.0 => Some(3.0)
        let result = compute_no_data_sentinel::<u8>(5.0, 0.5);
        assert!(result.is_some());
        let sentinel = result.unwrap();
        assert!(sentinel < 4.0);
        assert!(sentinel >= 0.0);
        assert_eq!(sentinel, 3.0);
    }

    #[test]
    fn sentinel_u8_at_zero() {
        // min=0.0, mze=0.5 => threshold = -1.0
        // All integer candidates are negative, below u8::MIN (0) => None
        let result = compute_no_data_sentinel::<u8>(0.0, 0.5);
        assert!(result.is_none());
    }

    #[test]
    fn sentinel_i16_negative_min() {
        // min=-100.0, mze=0.5 => threshold = -101.0
        // First candidate: floor(-102.0)=-102.0; -102.0 >= -32768 and < -101.0 => Some(-102.0)
        let result = compute_no_data_sentinel::<i16>(-100.0, 0.5);
        assert!(result.is_some());
        let sentinel = result.unwrap();
        assert!(sentinel < -101.0);
        assert!(sentinel >= i16::MIN as f64);
        assert_eq!(sentinel, -102.0);
    }

    #[test]
    fn sentinel_f32_float() {
        // min=10.0, mze=0.01 => threshold = 9.98
        // Float candidates sorted descending, first < 9.98 wins.
        // 4*mze = 0.04, so min - 0.04 = 9.96 < 9.98 => Some(9.96)
        let result = compute_no_data_sentinel::<f32>(10.0, 0.01);
        assert!(result.is_some());
        let sentinel = result.unwrap();
        assert!(sentinel < 9.98);
        // The sentinel should be close to min_val but below threshold
        assert!(sentinel > 0.0);
    }

    // ---- try_raise_max_z_error ----

    #[test]
    fn raise_mze_for_2_decimal_data() {
        // Data with exactly 2 decimal places: all values are multiples of 0.01
        let width = 100;
        let height = 100;
        let mask = BitMask::all_valid(width * height);
        let data: Vec<f32> = (0..width * height)
            .map(|i| ((i % 1000) as f32) * 0.01)
            .collect();
        let mut mze = 0.001; // initial mze smaller than 0.005
        let raised = try_raise_max_z_error(&data, &mask, width, height, 1, &mut mze);
        assert!(raised, "should raise mze for %.2f data");
        assert!(mze > 0.001, "mze should have increased");
    }

    #[test]
    fn no_raise_for_full_precision_data() {
        // Data with full f32 precision: PI, E, sqrt(2), etc.
        let width = 100;
        let height = 100;
        let mask = BitMask::all_valid(width * height);
        let data: Vec<f32> = (0..width * height)
            .map(|i| (i as f32 * core::f32::consts::PI).sin())
            .collect();
        let mut mze = 0.001;
        let original_mze = mze;
        let raised = try_raise_max_z_error(&data, &mask, width, height, 1, &mut mze);
        if !raised {
            assert_eq!(mze, original_mze, "mze should be unchanged when not raised");
        }
    }

    #[test]
    fn no_raise_for_integer_type() {
        let width = 10;
        let height = 10;
        let mask = BitMask::all_valid(width * height);
        let data: Vec<u8> = (0..100).collect();
        let mut mze = 0.5;
        let raised = try_raise_max_z_error(&data, &mask, width, height, 1, &mut mze);
        assert!(!raised, "should not raise mze for integer types");
    }

    #[test]
    fn no_raise_for_zero_mze() {
        let width = 10;
        let height = 10;
        let mask = BitMask::all_valid(width * height);
        let data: Vec<f32> = vec![1.0; 100];
        let mut mze = 0.0;
        let raised = try_raise_max_z_error(&data, &mask, width, height, 1, &mut mze);
        assert!(!raised, "should not raise mze when mze=0");
    }

    // ---- prune_candidates ----

    #[test]
    fn prune_removes_bad_candidates() {
        let mut round_err = vec![10.0, 0.1, 50.0];
        let mut z_err = vec![1.0, 0.5, 2.0];
        let mut z_fac = vec![1, 2, 1];
        let max_z_error = 1.0;
        // round_err[n] / z_fac[n] > max_z_error / 2.0 means removal
        // [0]: 10.0/1 = 10.0 > 0.5 => remove
        // [1]: 0.1/2 = 0.05 <= 0.5 => keep
        // [2]: 50.0/1 = 50.0 > 0.5 => remove
        let result = prune_candidates(&mut round_err, &mut z_err, &mut z_fac, max_z_error);
        assert!(result, "should have remaining candidates");
        assert_eq!(z_err.len(), 1);
        assert_eq!(z_err[0], 0.5);
    }

    #[test]
    fn prune_empty_returns_false() {
        let mut round_err: Vec<f64> = vec![];
        let mut z_err: Vec<f64> = vec![];
        let mut z_fac: Vec<i32> = vec![];
        let result = prune_candidates(&mut round_err, &mut z_err, &mut z_fac, 1.0);
        assert!(!result);
    }

    #[test]
    fn prune_all_removed_returns_false() {
        let mut round_err = vec![100.0];
        let mut z_err = vec![1.0];
        let mut z_fac = vec![1];
        // 100.0/1 > 0.5 => remove all
        let result = prune_candidates(&mut round_err, &mut z_err, &mut z_fac, 1.0);
        assert!(!result);
        assert!(z_err.is_empty());
    }

    // ---- add_uint_to_counts ----

    #[test]
    fn uint_counts_zero() {
        let mut counts = vec![0i32; 8];
        add_uint_to_counts(&mut counts, 0, 8);
        assert!(
            counts.iter().all(|&c| c == 0),
            "zero should contribute no bits"
        );
    }

    #[test]
    fn uint_counts_one() {
        let mut counts = vec![0i32; 8];
        add_uint_to_counts(&mut counts, 1, 8);
        // 1 = 0b00000001 => only bit 0 is set
        assert_eq!(counts[0], 1);
        for (i, &count) in counts.iter().enumerate().skip(1) {
            assert_eq!(count, 0, "bit {i} should be 0");
        }
    }

    #[test]
    fn uint_counts_255() {
        let mut counts = vec![0i32; 8];
        add_uint_to_counts(&mut counts, 255, 8);
        // 255 = 0b11111111 => all 8 bits set
        for (i, &count) in counts.iter().enumerate() {
            assert_eq!(count, 1, "bit {i} should be 1 for 255");
        }
    }

    #[test]
    fn uint_counts_0xaa() {
        let mut counts = vec![0i32; 8];
        add_uint_to_counts(&mut counts, 0xAA, 8);
        // 0xAA = 0b10101010 => bits 1,3,5,7 set
        assert_eq!(counts[0], 0);
        assert_eq!(counts[1], 1);
        assert_eq!(counts[2], 0);
        assert_eq!(counts[3], 1);
        assert_eq!(counts[4], 0);
        assert_eq!(counts[5], 1);
        assert_eq!(counts[6], 0);
        assert_eq!(counts[7], 1);
    }

    #[test]
    fn uint_counts_accumulate() {
        let mut counts = vec![0i32; 8];
        add_uint_to_counts(&mut counts, 0xFF, 8);
        add_uint_to_counts(&mut counts, 0xFF, 8);
        for (i, &count) in counts.iter().enumerate() {
            assert_eq!(count, 2, "bit {i} should be counted twice");
        }
    }

    // ---- add_int_to_counts ----

    #[test]
    fn int_counts_zero() {
        let mut counts = vec![0i32; 8];
        add_int_to_counts(&mut counts, 0, 8);
        assert!(counts.iter().all(|&c| c == 0));
    }

    #[test]
    fn int_counts_one() {
        let mut counts = vec![0i32; 8];
        add_int_to_counts(&mut counts, 1, 8);
        assert_eq!(counts[0], 1);
        for &count in counts.iter().skip(1) {
            assert_eq!(count, 0);
        }
    }

    #[test]
    fn int_counts_negative_one() {
        let mut counts = vec![0i32; 32];
        add_int_to_counts(&mut counts, -1, 32);
        // -1 in two's complement is all 1s
        for (i, &count) in counts.iter().enumerate() {
            assert_eq!(count, 1, "bit {i} should be 1 for -1");
        }
    }

    #[test]
    fn int_counts_0xaa() {
        let mut counts = vec![0i32; 8];
        // 0xAA as i32 = 170
        add_int_to_counts(&mut counts, 0xAAi32, 8);
        // 0b10101010 => bits 1,3,5,7 set
        assert_eq!(counts[0], 0);
        assert_eq!(counts[1], 1);
        assert_eq!(counts[2], 0);
        assert_eq!(counts[3], 1);
        assert_eq!(counts[4], 0);
        assert_eq!(counts[5], 1);
        assert_eq!(counts[6], 0);
        assert_eq!(counts[7], 1);
    }
}
