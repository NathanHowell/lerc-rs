use alloc::vec;
use alloc::vec::Vec;

use crate::bitmask::BitMask;
use crate::error::{LercError, Result};
use crate::fpl;
use crate::header::{self, HeaderInfo};
use crate::huffman::HuffmanCodec;
use crate::rle;
use crate::tiles;
use crate::types::{DataType, ImageEncodeMode, LercDataType, TileCompressionMode, tile_flags};
use crate::{LercData, LercImage};

pub fn encode(image: &LercImage, max_z_error: f64) -> Result<Vec<u8>> {
    match &image.data {
        LercData::I8(d) => encode_typed(image, d, max_z_error),
        LercData::U8(d) => encode_typed(image, d, max_z_error),
        LercData::I16(d) => encode_typed(image, d, max_z_error),
        LercData::U16(d) => encode_typed(image, d, max_z_error),
        LercData::I32(d) => encode_typed(image, d, max_z_error),
        LercData::U32(d) => encode_typed(image, d, max_z_error),
        LercData::F32(d) => encode_typed(image, d, max_z_error),
        LercData::F64(d) => encode_typed(image, d, max_z_error),
    }
}

fn encode_typed<T: LercDataType>(
    image: &LercImage,
    data: &[T],
    max_z_error: f64,
) -> Result<Vec<u8>> {
    let width = image.width as usize;
    let height = image.height as usize;
    let n_depth = image.n_depth as usize;
    let n_bands = image.n_bands as usize;
    let band_size = width * height * n_depth;

    let max_z_error = if T::is_integer() {
        // Magic value 777 means "use default bit-plane epsilon of 0.01"
        let max_z_error = if max_z_error == 777.0 {
            -0.01
        } else {
            max_z_error
        };

        // Negative maxZError signals bit-plane compression request
        let max_z_error = if max_z_error < 0.0 {
            let eps = -max_z_error;
            try_bit_plane_compression::<T>(
                data,
                &image.valid_masks,
                image.width as usize,
                image.height as usize,
                n_depth,
                n_bands,
                eps,
                T::DATA_TYPE,
            )
            .unwrap_or(0.0)
        } else {
            max_z_error
        };

        max_z_error.floor().max(0.5)
    } else {
        max_z_error.max(0.0)
    };

    // For float types with maxZError > 0, try to raise maxZError without extra loss.
    // If float data has limited precision (e.g. values stored as "%.2f"), the actual
    // rounding error is already bounded and we can safely use a coarser quantization.
    let max_z_error = if !T::is_integer() && max_z_error > 0.0 {
        let mut raised = max_z_error;
        // We need any band's mask; use the first one for the scan.
        let mask = if !image.valid_masks.is_empty() {
            &image.valid_masks[0]
        } else {
            // Should not happen for a valid image, but be safe.
            &image.valid_masks[0]
        };
        if try_raise_max_z_error(
            data,
            mask,
            width,
            height,
            n_depth,
            &mut raised,
        ) {
            raised
        } else {
            max_z_error
        }
    } else {
        max_z_error
    };

    let mut result = Vec::new();

    for band in 0..n_bands {
        let band_data = &data[band * band_size..(band + 1) * band_size];
        let mask = if band < image.valid_masks.len() {
            &image.valid_masks[band]
        } else {
            &image.valid_masks[0]
        };

        let blobs_more = (n_bands - 1 - band) as i32;
        let blob = encode_one_band(band_data, mask, image, max_z_error, blobs_more)?;
        result.extend_from_slice(&blob);
    }

    Ok(result)
}

/// Try to raise `max_z_error` for float data without introducing additional loss.
///
/// When float values have limited precision (e.g., stored as "%.2f"), raising
/// maxZError to align with that precision loses nothing because the data's
/// inherent rounding error is already bounded by the original maxZError.
///
/// This mirrors the C++ `Lerc2::TryRaiseMaxZError` algorithm.
fn try_raise_max_z_error<T: LercDataType>(
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
fn try_bit_plane_compression<T: LercDataType>(
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
fn compute_no_data_sentinel<T: LercDataType>(
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

/// Quick check whether 8-bit integer data has high entropy (nearly all 256
/// byte values present with roughly uniform distribution). When this is true,
/// Huffman encoding cannot beat tiling, so we skip the expensive Huffman attempt.
fn is_high_entropy_u8<T: LercDataType>(
    data: &[T],
    mask: &BitMask,
    header: &HeaderInfo,
) -> bool {
    let width = header.n_cols as usize;
    let height = header.n_rows as usize;
    let n_depth = header.n_depth as usize;
    let offset: i32 = if header.data_type == DataType::Char { 128 } else { 0 };
    let all_valid = header.num_valid_pixel == header.n_rows * header.n_cols;

    // Build a quick histogram
    let mut histo = [0u32; 256];
    let mut total = 0u32;

    // Fast path for u8, all-valid, single-depth: avoid to_f64 and mask checks
    if T::DATA_TYPE == DataType::Byte && all_valid && n_depth == 1 {
        debug_assert_eq!(offset, 0);
        let u8_data: &[u8] = unsafe {
            core::slice::from_raw_parts(data.as_ptr() as *const u8, data.len())
        };
        for &val in u8_data {
            histo[val as usize] += 1;
        }
        total = u8_data.len() as u32;
    } else {
        for i in 0..height {
            for j in 0..width {
                let k = i * width + j;
                if all_valid || mask.is_valid(k) {
                    for m in 0..n_depth {
                        let val = data[k * n_depth + m].to_f64() as i32;
                        let bin = (val + offset) as usize;
                        histo[bin] += 1;
                        total += 1;
                    }
                }
            }
        }
    }

    if total == 0 {
        return false;
    }

    let n_distinct = histo.iter().filter(|&&c| c > 0).count();

    // If fewer than 248 distinct values, Huffman may still help.
    if n_distinct < 248 {
        return false;
    }

    let avg = total as f64 / 256.0;
    let max_count = *histo.iter().max().unwrap_or(&0) as f64;

    // If the distribution isn't roughly uniform, Huffman may help.
    if max_count >= avg * 2.0 {
        return false;
    }

    // Direct histogram is uniform, but delta-encoding might still compress well.
    // Quick check: compute a delta histogram on a sample of rows.
    let mut delta_histo = [0u32; 256];
    let sample_rows = 8.min(height);
    let row_step = height / sample_rows;
    let mut delta_total = 0u32;

    if T::DATA_TYPE == DataType::Byte && all_valid && n_depth == 1 {
        let u8_data: &[u8] = unsafe {
            core::slice::from_raw_parts(data.as_ptr() as *const u8, data.len())
        };
        for row_idx in 0..sample_rows {
            let i = row_idx * row_step;
            let row_start = i * width;
            let mut prev_val: u8 = 0;
            for j in 0..width {
                let val = u8_data[row_start + j];
                let delta = if j > 0 { val.wrapping_sub(prev_val) } else { val };
                delta_histo[delta as usize] += 1;
                delta_total += 1;
                prev_val = val;
            }
        }
    } else {
        for row_idx in 0..sample_rows {
            let i = row_idx * row_step;
            let mut prev_val: i32 = 0;
            for j in 0..width {
                let k = i * width + j;
                if all_valid || mask.is_valid(k) {
                    let val = data[k * n_depth].to_f64() as i32;
                    let delta = if j > 0 { val.wrapping_sub(prev_val) } else { val };
                    delta_histo[(delta + offset) as u8 as usize] += 1;
                    delta_total += 1;
                    prev_val = val;
                }
            }
        }
    }

    if delta_total == 0 {
        return true;
    }

    // If delta histogram has few distinct values, delta-Huffman will compress well.
    let delta_distinct = delta_histo.iter().filter(|&&c| c > 0).count();
    if delta_distinct < 64 {
        return false; // Don't skip — delta-Huffman will likely win
    }

    // Both direct and delta are high entropy — safe to skip Huffman
    true
}

/// Iterate over all valid pixel values, calling a visitor for each.
#[inline(always)]
fn for_each_valid_pixel<T: LercDataType>(
    data: &[T],
    mask: &BitMask,
    width: usize,
    height: usize,
    n_depth: usize,
    mut f: impl FnMut(usize, usize, f64), // (pixel_k, depth_m, value)
) {
    let all_valid = mask.count_valid() >= width * height;
    if all_valid && n_depth == 1 {
        for (k, val) in data.iter().enumerate() {
            f(k, 0, val.to_f64());
        }
    } else {
        for i in 0..height {
            for j in 0..width {
                let k = i * width + j;
                if all_valid || mask.is_valid(k) {
                    for m in 0..n_depth {
                        f(k, m, data[k * n_depth + m].to_f64());
                    }
                }
            }
        }
    }
}

/// Per-band statistics gathered in a single pass over valid pixels.
struct BandStats {
    z_min_vec: Vec<f64>,
    z_max_vec: Vec<f64>,
    overall_min: f64,
    overall_max: f64,
    /// Minimum value excluding NoData (for sentinel computation).
    valid_min: f64,
    /// Whether any valid pixel has mixed NoData across depths.
    needs_no_data: bool,
}

/// Gather per-depth min/max, overall range, valid-only min, and mixed NoData
/// detection in a single traversal of valid pixels.
fn compute_band_stats<T: LercDataType>(
    data: &[T],
    mask: &BitMask,
    width: usize,
    height: usize,
    n_depth: usize,
    num_valid: usize,
    nd_f64: Option<f64>,
) -> BandStats {
    let mut stats = BandStats {
        z_min_vec: vec![f64::MAX; n_depth],
        z_max_vec: vec![f64::MIN; n_depth],
        overall_min: f64::MAX,
        overall_max: f64::MIN,
        valid_min: f64::MAX,
        needs_no_data: false,
    };

    let mut prev_k = usize::MAX;
    let mut nd_count_at_pixel = 0usize;
    let mut depth_count_at_pixel = 0usize;

    for_each_valid_pixel(data, mask, width, height, n_depth, |k, m, val| {
        if val < stats.z_min_vec[m] { stats.z_min_vec[m] = val; }
        if val > stats.z_max_vec[m] { stats.z_max_vec[m] = val; }
        if val < stats.overall_min { stats.overall_min = val; }
        if val > stats.overall_max { stats.overall_max = val; }

        if let Some(nd) = nd_f64 {
            if k != prev_k {
                if prev_k != usize::MAX
                    && nd_count_at_pixel > 0
                    && nd_count_at_pixel < depth_count_at_pixel
                {
                    stats.needs_no_data = true;
                }
                prev_k = k;
                nd_count_at_pixel = 0;
                depth_count_at_pixel = 0;
            }
            depth_count_at_pixel += 1;
            if val == nd {
                nd_count_at_pixel += 1;
            } else if val < stats.valid_min {
                stats.valid_min = val;
            }
        }
    });

    // Check the last pixel
    if nd_f64.is_some()
        && prev_k != usize::MAX
        && nd_count_at_pixel > 0
        && nd_count_at_pixel < depth_count_at_pixel
    {
        stats.needs_no_data = true;
    }

    if num_valid == 0 {
        stats.overall_min = 0.0;
        stats.overall_max = 0.0;
    }
    if stats.valid_min == f64::MAX {
        stats.valid_min = stats.overall_min;
    }

    stats
}

/// Result of NoData sentinel processing.
struct NoDataResult<T> {
    pass_no_data: bool,
    no_data_val_internal: f64,
    no_data_val_orig: f64,
    /// Remapped data buffer (if sentinel differs from original NoData value).
    remapped_data: Option<Vec<T>>,
}

/// Compute the NoData sentinel and remap data if needed.
fn process_no_data<T: LercDataType>(
    data: &[T],
    mask: &BitMask,
    width: usize,
    height: usize,
    n_depth: usize,
    num_valid: usize,
    image_no_data: Option<f64>,
    nd_f64: Option<f64>,
    stats: &mut BandStats,
    max_z_error: f64,
) -> Result<NoDataResult<T>> {
    let mut result = NoDataResult {
        pass_no_data: false,
        no_data_val_internal: 0.0,
        no_data_val_orig: 0.0,
        remapped_data: None,
    };

    let (Some(nd_orig), Some(nd_val)) = (image_no_data, nd_f64) else {
        return Ok(result);
    };
    if n_depth <= 1 || num_valid == 0 {
        return Ok(result);
    }

    if stats.needs_no_data {
        let sentinel = compute_no_data_sentinel::<T>(stats.valid_min, max_z_error)
            .ok_or_else(|| LercError::InvalidData(
                "cannot find a NoData sentinel value below the valid data range for this type".into(),
            ))?;
        result.pass_no_data = true;
        result.no_data_val_orig = nd_orig;

        if sentinel != nd_val {
            result.no_data_val_internal = sentinel;
            let mut buf = data.to_vec();
            for i in 0..height {
                for j in 0..width {
                    let k = i * width + j;
                    if mask.is_valid(k) {
                        let base = k * n_depth;
                        for m in 0..n_depth {
                            if buf[base + m].to_f64() == nd_val {
                                buf[base + m] = T::from_f64(sentinel);
                            }
                        }
                    }
                }
            }
            stats.overall_min = stats.overall_min.min(sentinel);
            for val in &mut stats.z_min_vec[..n_depth] {
                *val = val.min(sentinel);
            }
            result.remapped_data = Some(buf);
        } else {
            result.no_data_val_internal = nd_orig;
        }
    } else {
        // NoData value provided but not needed — pass through for metadata.
        result.pass_no_data = true;
        result.no_data_val_orig = nd_orig;
        result.no_data_val_internal = nd_orig;
    }

    Ok(result)
}

/// Write the blob payload: mask, ranges, and encoded pixel data.
fn write_blob_payload<T: LercDataType>(
    blob: &mut Vec<u8>,
    encode_data: &[T],
    mask: &BitMask,
    hd: &mut HeaderInfo,
    stats: &BandStats,
    try_huffman_flt: bool,
) -> Result<()> {
    let width = hd.n_cols as usize;
    let height = hd.n_rows as usize;
    let n_depth = hd.n_depth as usize;
    let num_valid = hd.num_valid_pixel as usize;

    // Write mask
    let num_pixels = width * height;
    if num_valid == 0 || num_valid == num_pixels {
        blob.extend_from_slice(&0i32.to_le_bytes());
    } else {
        let mask_compressed = rle::compress(mask.as_bytes());
        let mask_size = mask_compressed.len() as i32;
        blob.extend_from_slice(&mask_size.to_le_bytes());
        blob.extend_from_slice(&mask_compressed);
    }

    if num_valid == 0 || stats.overall_min == stats.overall_max {
        return Ok(());
    }

    // Write per-depth min/max ranges (v4+)
    if hd.version >= 4 {
        for val in &stats.z_min_vec[..n_depth] {
            T::from_f64(*val).extend_le_bytes(blob);
        }
        for val in &stats.z_max_vec[..n_depth] {
            T::from_f64(*val).extend_le_bytes(blob);
        }
    }

    if stats.z_min_vec == stats.z_max_vec {
        return Ok(());
    }

    // One sweep flag = 0
    blob.push(0u8);

    // Encoding mode dispatch
    let try_huffman_int = matches!(T::DATA_TYPE, DataType::Char | DataType::Byte)
        && hd.max_z_error == 0.5;

    if try_huffman_int {
        let skip_huffman = is_high_entropy_u8(encode_data, mask, hd);
        if !skip_huffman {
            if let Some(huffman_blob) = try_encode_huffman_int(encode_data, mask, hd) {
                let mut tiling_blob = Vec::new();
                encode_tiles(
                    &mut tiling_blob, encode_data, mask, hd,
                    &stats.z_min_vec, &stats.z_max_vec,
                )?;
                if huffman_blob.len() < tiling_blob.len() {
                    blob.extend_from_slice(&huffman_blob);
                    return Ok(());
                }
            }
        }
        blob.push(ImageEncodeMode::Tiling as u8);
    } else if try_huffman_flt {
        let is_double = T::DATA_TYPE == DataType::Double;
        let fpl_data =
            fpl::encode_huffman_flt(encode_data, is_double, width, height, n_depth)?;
        blob.push(ImageEncodeMode::DeltaDeltaHuffman as u8);
        blob.extend_from_slice(&fpl_data);
        return Ok(());
    }

    encode_tiles(blob, encode_data, mask, hd, &stats.z_min_vec, &stats.z_max_vec)?;
    Ok(())
}

fn encode_one_band<T: LercDataType>(
    data: &[T],
    mask: &BitMask,
    image: &LercImage,
    max_z_error: f64,
    n_blobs_more: i32,
) -> Result<Vec<u8>> {
    let width = image.width as usize;
    let height = image.height as usize;
    let n_depth = image.n_depth as usize;
    let num_valid = mask.count_valid().min(width * height);

    let nd_f64 = if n_depth > 1 {
        image.no_data_value.map(|nd| T::from_f64(nd).to_f64())
    } else {
        None
    };

    // 1. Gather statistics in a single pass.
    let mut stats =
        compute_band_stats(data, mask, width, height, n_depth, num_valid, nd_f64);

    // 2. Handle NoData sentinel computation and data remapping.
    let nd_result = process_no_data(
        data, mask, width, height, n_depth, num_valid,
        image.no_data_value, nd_f64, &mut stats, max_z_error,
    )?;
    let encode_data: &[T] = nd_result.remapped_data.as_deref().unwrap_or(data);

    // 3. Determine minimum required version and build header.
    let try_huffman_flt =
        matches!(T::DATA_TYPE, DataType::Float | DataType::Double) && max_z_error == 0.0;
    let min_version = if nd_result.pass_no_data || n_blobs_more > 0 || try_huffman_flt {
        6
    } else if n_depth > 1 {
        5
    } else {
        3
    };

    let mut hd = HeaderInfo {
        version: min_version,
        checksum: 0,
        n_rows: height as i32,
        n_cols: width as i32,
        n_depth: n_depth as i32,
        num_valid_pixel: num_valid as i32,
        micro_block_size: 8,
        blob_size: 0,
        data_type: T::DATA_TYPE,
        n_blobs_more,
        pass_no_data_values: nd_result.pass_no_data,
        is_int: false,
        max_z_error,
        z_min: stats.overall_min,
        z_max: stats.overall_max,
        no_data_val: nd_result.no_data_val_internal,
        no_data_val_orig: nd_result.no_data_val_orig,
    };

    // 4. Select block size (skip for FPL path).
    if !try_huffman_flt {
        let best_block_size =
            select_block_size::<T>(encode_data, mask, &hd, &stats.z_min_vec, &stats.z_max_vec)?;
        hd.micro_block_size = best_block_size;
    }

    // 5. Write header + payload, finalize checksum.
    let mut blob = header::write_header(&hd);
    write_blob_payload(&mut blob, encode_data, mask, &mut hd, &stats, try_huffman_flt)?;
    header::finalize_blob(&mut blob);
    Ok(blob)
}

/// Pick representative tile positions for sampling at a given block size.
fn sample_tile_positions(
    n_rows: usize,
    n_cols: usize,
    mb_size: usize,
) -> Vec<(usize, usize, usize, usize)> {
    let num_tv = n_rows.div_ceil(mb_size);
    let num_th = n_cols.div_ceil(mb_size);

    let mut positions = vec![
        (0, 0),
        (0, num_th.saturating_sub(1)),
        (num_tv.saturating_sub(1), 0),
        (num_tv.saturating_sub(1), num_th.saturating_sub(1)),
        (num_tv / 2, num_th / 2),
        (0, num_th / 2),
        (num_tv / 2, 0),
        (num_tv.saturating_sub(1), num_th / 2),
    ];

    positions.sort();
    positions.dedup();

    positions
        .into_iter()
        .map(|(it, jt)| {
            let i0 = it * mb_size;
            let i1 = (i0 + mb_size).min(n_rows);
            let j0 = jt * mb_size;
            let j1 = (j0 + mb_size).min(n_cols);
            (i0, i1, j0, j1)
        })
        .collect()
}

/// Fast block size selection for u8 lossless, all-valid, single-depth images.
fn select_block_size_u8_fast(data: &[u8], hd: &HeaderInfo) -> i32 {
    let n_rows = hd.n_rows as usize;
    let n_cols = hd.n_cols as usize;
    let version = hd.version;
    let pattern: u8 = if version >= 5 { 14 } else { 15 };
    let mut quant_buf = [0u32; 256];

    let mut size8 = 0usize;
    let mut buf8 = Vec::with_capacity(512);
    for &(i0, i1, j0, j1) in &sample_tile_positions(n_rows, n_cols, 8) {
        buf8.clear();
        let integrity = ((j0 as u8 >> 3) & pattern) << 2;
        encode_single_u8_tile(&mut buf8, data, n_cols, i0, i1, j0, j1, integrity, &mut quant_buf);
        size8 += buf8.len();
    }

    let mut size16 = 0usize;
    let mut buf16 = Vec::with_capacity(512);
    for &(i0, i1, j0, j1) in &sample_tile_positions(n_rows, n_cols, 16) {
        buf16.clear();
        let integrity = ((j0 as u8 >> 3) & pattern) << 2;
        encode_single_u8_tile(&mut buf16, data, n_cols, i0, i1, j0, j1, integrity, &mut quant_buf);
        size16 += buf16.len();
    }

    if size16 < size8 { 16 } else { 8 }
}

/// Encode a single u8 tile (used by select_block_size_u8_fast and encode_tiles_u8_fast).
///
/// Combines min/max scan, quantization, histogram building, LUT decision, and encoding
/// into minimal passes over the data.
#[inline]
#[allow(clippy::too_many_arguments)]
fn encode_single_u8_tile(
    buf: &mut Vec<u8>,
    data: &[u8],
    n_cols: usize,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    integrity: u8,
    quant_buf: &mut [u32; 256],
) {
    let tile_w = j1 - j0;
    let num_valid = (i1 - i0) * tile_w;

    // Single pass: compute min, max, and copy values into quant_buf (as-is, not yet quantized)
    let mut z_min: u8 = 255;
    let mut z_max: u8 = 0;
    let mut idx = 0;
    for i in i0..i1 {
        let row_start = i * n_cols + j0;
        for &val in &data[row_start..row_start + tile_w] {
            if val < z_min { z_min = val; }
            if val > z_max { z_max = val; }
            // Store raw value; we'll subtract z_min below if needed
            quant_buf[idx] = val as u32;
            idx += 1;
        }
    }

    if z_min == 0 && z_max == 0 {
        buf.push(TileCompressionMode::ConstZero as u8 | integrity);
        return;
    }

    if z_min == z_max {
        buf.push(TileCompressionMode::ConstOffset as u8 | integrity);
        buf.push(z_min);
        return;
    }

    // Quantize in-place: subtract z_min
    let max_quant = (z_max - z_min) as u32;
    let quant_data = &mut quant_buf[..num_valid];
    let z_min_u32 = z_min as u32;
    for v in quant_data.iter_mut() {
        *v -= z_min_u32;
    }

    buf.push(TileCompressionMode::BitStuffed as u8 | integrity);
    buf.push(z_min);

    let num_bits = crate::bitstuffer::num_bits_needed(max_quant);
    let num_elem = num_valid as u32;

    // Quick heuristic: if the value range is large relative to the tile size,
    // LUT can't help (not enough duplicates to save bits). Skip the histogram.
    // For a tile of N values with max_quant >= N/2, the maximum possible number
    // of distinct values is N, and n_bits_lut >= log2(N/2) which is close to
    // num_bits, so LUT overhead eats any savings.
    let try_lut = max_quant < num_elem / 2;

    if !try_lut {
        // Simple encoding (most common for high-entropy data)
        crate::bitstuffer::encode_simple_into(buf, quant_data, max_quant);
    } else {
        // Build histogram and check if LUT helps
        let mut counts = [0u16; 256];
        for &v in quant_data.iter() {
            counts[v as usize] += 1;
        }

        // Count distinct values (only up to max_quant)
        let mut n_distinct = 0u32;
        for &c in &counts[..=max_quant as usize] {
            if c > 0 {
                n_distinct += 1;
            }
        }

        // Check if LUT is beneficial
        let use_lut = if (2..=255).contains(&n_distinct) {
            let n_lut = n_distinct - 1;
            let mut n_bits_lut = 0u32;
            while (n_lut >> n_bits_lut) != 0 { n_bits_lut += 1; }
            let n_bytes = crate::bitstuffer::num_bytes_uint(num_elem);
            let num_bytes_simple = 1 + n_bytes as u32 + ((num_elem * num_bits + 7) >> 3);
            let num_bytes_lut = 1 + n_bytes as u32 + 1
                + ((n_lut * num_bits + 7) >> 3)
                + ((num_elem * n_bits_lut + 7) >> 3);
            num_bytes_lut < num_bytes_simple
        } else {
            false
        };

        if !use_lut {
            crate::bitstuffer::encode_simple_into(buf, quant_data, max_quant);
        } else {
            // Build LUT mapping and encode using stack-allocated arrays
            let n_lut = n_distinct - 1;
            let mut n_bits_lut = 0u32;
            while (n_lut >> n_bits_lut) != 0 { n_bits_lut += 1; }

            let mut value_to_index = [0u8; 256];
            let mut lut_values = [0u32; 255];
            let mut lut_count = 0u32;
            let mut lut_idx = 0u8;
            for v in 0..=max_quant as usize {
                if counts[v] > 0 {
                    value_to_index[v] = lut_idx;
                    if lut_idx > 0 {
                        lut_values[lut_count as usize] = v as u32;
                        lut_count += 1;
                    }
                    lut_idx += 1;
                }
            }

            // Build index array on the stack
            let mut index_arr = [0u32; 256];
            for (i, &val) in quant_data.iter().enumerate() {
                index_arr[i] = value_to_index[val as usize] as u32;
            }
            let index_data = &index_arr[..num_valid];
            let lut_data = &lut_values[..lut_count as usize];

            // Write LUT header and bit-stuffed data directly
            let n = crate::bitstuffer::num_bytes_uint(num_elem);
            let bits67 = if n == 4 { 0u8 } else { (3 - n) as u8 };
            let mut header_byte = num_bits as u8;
            header_byte |= bits67 << 6;
            header_byte |= 1 << 5; // LUT mode

            buf.push(header_byte);
            crate::bitstuffer::encode_uint_pub(buf, num_elem, n);
            buf.push((lut_count + 1) as u8);

            if num_bits > 0 {
                crate::bitstuffer::bit_stuff_append(buf, lut_data, num_bits);
            }
            if n_bits_lut > 0 {
                crate::bitstuffer::bit_stuff_append(buf, index_data, n_bits_lut);
            }
        }
    }
}

/// Try encoding a sample of tiles at both block sizes 8 and 16, return whichever
/// produces smaller output. Instead of encoding ALL tiles twice, we sample a
/// small number of representative tiles to make the decision.
fn select_block_size<T: LercDataType>(
    data: &[T],
    mask: &BitMask,
    hd: &HeaderInfo,
    z_min_vec: &[f64],
    z_max_vec: &[f64],
) -> Result<i32> {
    let n_rows = hd.n_rows as usize;
    let n_cols = hd.n_cols as usize;
    let n_depth = hd.n_depth as usize;

    // Fast path for u8 lossless, all-valid, single-depth
    let all_valid = hd.num_valid_pixel == hd.n_rows * hd.n_cols;
    if T::DATA_TYPE == DataType::Byte && hd.max_z_error == 0.5 && all_valid && n_depth == 1 {
        debug_assert_eq!(T::BYTES, 1);
        let u8_data: &[u8] = unsafe {
            core::slice::from_raw_parts(data.as_ptr() as *const u8, data.len())
        };
        return Ok(select_block_size_u8_fast(u8_data, hd));
    }

    // For very small images, just try both fully
    if n_rows <= 16 && n_cols <= 16 {
        let mut hd8 = hd.clone();
        hd8.micro_block_size = 8;
        let mut buf8 = Vec::new();
        encode_tiles(&mut buf8, data, mask, &hd8, z_min_vec, z_max_vec)?;

        let mut hd16 = hd.clone();
        hd16.micro_block_size = 16;
        let mut buf16 = Vec::new();
        encode_tiles(&mut buf16, data, mask, &hd16, z_min_vec, z_max_vec)?;

        return if buf16.len() < buf8.len() {
            Ok(16)
        } else {
            Ok(8)
        };
    }

    let max_val_to_quantize: f64 = match hd.data_type {
        DataType::Char | DataType::Byte | DataType::Short | DataType::UShort => {
            ((1u32 << 15) - 1) as f64
        }
        _ => ((1u32 << 30) - 1) as f64,
    };
    let max_z_error = hd.max_z_error;

    let mut scratch = ScratchBuffers::new();

    // Encode sampled tiles at block size 8
    let mut size8 = 0usize;
    let hd8 = {
        let mut h = hd.clone();
        h.micro_block_size = 8;
        h
    };
    for &(i0, i1, j0, j1) in &sample_tile_positions(n_rows, n_cols, 8) {
        for i_depth in 0..n_depth {
            size8 += encode_tile_to_count::<T>(
                data,
                mask,
                &hd8,
                z_max_vec,
                max_z_error,
                max_val_to_quantize,
                i0,
                i1,
                j0,
                j1,
                i_depth,
                n_depth > 1 && i_depth > 0,
                &mut scratch,
            )?;
        }
    }

    // Encode sampled tiles at block size 16
    let mut size16 = 0usize;
    let hd16 = {
        let mut h = hd.clone();
        h.micro_block_size = 16;
        h
    };
    for &(i0, i1, j0, j1) in &sample_tile_positions(n_rows, n_cols, 16) {
        for i_depth in 0..n_depth {
            size16 += encode_tile_to_count::<T>(
                data,
                mask,
                &hd16,
                z_max_vec,
                max_z_error,
                max_val_to_quantize,
                i0,
                i1,
                j0,
                j1,
                i_depth,
                n_depth > 1 && i_depth > 0,
                &mut scratch,
            )?;
        }
    }

    if size16 < size8 {
        Ok(16)
    } else {
        Ok(8)
    }
}

/// Encode a single tile and return its byte count (without appending to blob).
/// This is used by `select_block_size` for sampling.
#[allow(clippy::too_many_arguments)]
fn encode_tile_to_count<T: LercDataType>(
    data: &[T],
    mask: &BitMask,
    header: &HeaderInfo,
    z_max_vec: &[f64],
    max_z_error: f64,
    max_val_to_quantize: f64,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    i_depth: usize,
    try_diff: bool,
    scratch: &mut ScratchBuffers,
) -> Result<usize> {
    let mut tmp = Vec::new();
    encode_tile(
        &mut tmp,
        data,
        mask,
        header,
        z_max_vec,
        max_z_error,
        max_val_to_quantize,
        i0,
        i1,
        j0,
        j1,
        i_depth,
        try_diff,
        scratch,
        None, // No reconstruction buffer needed for size estimation
    )?;
    Ok(tmp.len())
}

/// Compute histograms for Huffman encoding of 8-bit data.
/// Returns (direct_histogram, delta_histogram) each of size 256.
/// The offset is 128 for i8 (DT_Char) and 0 for u8 (DT_Byte).
fn compute_histo_for_huffman<T: LercDataType>(
    data: &[T],
    mask: &BitMask,
    header: &HeaderInfo,
) -> ([i32; 256], [i32; 256]) {
    let width = header.n_cols as usize;
    let height = header.n_rows as usize;
    let n_depth = header.n_depth as usize;
    let offset: i32 = if header.data_type == DataType::Char {
        128
    } else {
        0
    };
    let all_valid = header.num_valid_pixel == header.n_rows * header.n_cols;

    let mut histo = [0i32; 256];
    let mut delta_histo = [0i32; 256];

    // Direct histogram
    for i in 0..height {
        for j in 0..width {
            let k = i * width + j;
            if all_valid || mask.is_valid(k) {
                for m in 0..n_depth {
                    let val = data[k * n_depth + m].to_f64() as i32;
                    let bin = (val + offset) as usize;
                    histo[bin] += 1;
                }
            }
        }
    }

    // Delta histogram
    for i_depth in 0..n_depth {
        let mut prev_val: i32 = 0;
        for i in 0..height {
            for j in 0..width {
                let k = i * width + j;
                if all_valid || mask.is_valid(k) {
                    let val = data[k * n_depth + i_depth].to_f64() as i32;
                    let predictor = if j > 0 && (all_valid || mask.is_valid(k - 1)) {
                        prev_val
                    } else if i > 0 && (all_valid || mask.is_valid(k - width)) {
                        data[(k - width) * n_depth + i_depth].to_f64() as i32
                    } else {
                        prev_val
                    };
                    let delta = val.wrapping_sub(predictor);
                    let bin = (delta + offset) as u8 as usize;
                    delta_histo[bin] += 1;
                    prev_val = val;
                }
            }
        }
    }

    (histo, delta_histo)
}

/// Try Huffman encoding for 8-bit integer data (u8/i8).
/// Returns Some(bytes) with the mode flag + Huffman data if beneficial, or None.
fn try_encode_huffman_int<T: LercDataType>(
    data: &[T],
    mask: &BitMask,
    header: &HeaderInfo,
) -> Option<Vec<u8>> {
    let width = header.n_cols as usize;
    let height = header.n_rows as usize;
    let n_depth = header.n_depth as usize;
    let offset: i32 = if header.data_type == DataType::Char {
        128
    } else {
        0
    };
    let all_valid = header.num_valid_pixel == header.n_rows * header.n_cols;

    let (histo, delta_histo) = compute_histo_for_huffman(data, mask, header);

    // Try delta Huffman
    let mut delta_codec = HuffmanCodec::new();
    let delta_ok = delta_codec.compute_codes(&delta_histo);
    let delta_size = if delta_ok {
        delta_codec
            .compute_compressed_size(&delta_histo)
            .map(|(bytes, _)| bytes)
    } else {
        None
    };

    // Try direct Huffman (IEM_Huffman, requires v4+)
    let mut direct_codec = HuffmanCodec::new();
    let direct_ok = header.version >= 4 && direct_codec.compute_codes(&histo);
    let direct_size = if direct_ok {
        direct_codec
            .compute_compressed_size(&histo)
            .map(|(bytes, _)| bytes)
    } else {
        None
    };

    // Pick the best: delta Huffman or direct Huffman
    enum HuffMode {
        Delta,
        Direct,
    }

    let (mode, codec) = match (delta_size, direct_size) {
        (Some(ds), Some(hs)) => {
            if ds <= hs {
                (HuffMode::Delta, delta_codec)
            } else {
                (HuffMode::Direct, direct_codec)
            }
        }
        (Some(_), None) => (HuffMode::Delta, delta_codec),
        (None, Some(_)) => (HuffMode::Direct, direct_codec),
        (None, None) => return None,
    };

    // Write the Huffman-encoded blob
    let mut buf = Vec::new();

    // Write mode flag
    match mode {
        HuffMode::Delta => buf.push(ImageEncodeMode::DeltaHuffman as u8),
        HuffMode::Direct => buf.push(ImageEncodeMode::Huffman as u8),
    }

    // Write code table
    let code_table_bytes = codec.write_code_table(header.version).ok()?;
    buf.extend_from_slice(&code_table_bytes);

    // Compute total bits needed for the data
    let code_table = codec.code_table();

    let mut total_bits = 0u64;
    match mode {
        HuffMode::Delta => {
            for i_depth in 0..n_depth {
                let mut prev_val: i32 = 0;
                for i in 0..height {
                    for j in 0..width {
                        let k = i * width + j;
                        if all_valid || mask.is_valid(k) {
                            let val = data[k * n_depth + i_depth].to_f64() as i32;
                            let predictor =
                                if j > 0 && (all_valid || mask.is_valid(k - 1)) {
                                    prev_val
                                } else if i > 0
                                    && (all_valid || mask.is_valid(k - width))
                                {
                                    data[(k - width) * n_depth + i_depth].to_f64() as i32
                                } else {
                                    prev_val
                                };
                            let delta = val.wrapping_sub(predictor);
                            let bin = (delta + offset) as u8 as usize;
                            total_bits += code_table[bin].0 as u64;
                            prev_val = val;
                        }
                    }
                }
            }
        }
        HuffMode::Direct => {
            for i in 0..height {
                for j in 0..width {
                    let k = i * width + j;
                    if all_valid || mask.is_valid(k) {
                        for m in 0..n_depth {
                            let val = data[k * n_depth + m].to_f64() as i32;
                            let bin = (val + offset) as usize;
                            total_bits += code_table[bin].0 as u64;
                        }
                    }
                }
            }
        }
    }

    // Allocate buffer for Huffman-encoded data
    // numUInts = (bitPos > 0 ? 1 : 0) + 1 for padding (decode read-ahead)
    let num_uints_data = total_bits.div_ceil(32) as usize;
    let num_uints_total = num_uints_data + 1; // +1 for decode read-ahead padding
    let mut encoded = vec![0u8; num_uints_total * 4];

    // Use a u64 accumulator (MSB-aligned) instead of per-symbol push_value
    // calls. This avoids repeated u32 reads/writes from the output buffer and
    // is the same pattern used in the FPL Huffman encoder.
    let mut accum: u64 = 0;
    let mut accum_bits: u32 = 0;
    let mut out_idx: usize = 0;

    // Encode data
    match mode {
        HuffMode::Delta => {
            for i_depth in 0..n_depth {
                let mut prev_val: i32 = 0;
                for i in 0..height {
                    for j in 0..width {
                        let k = i * width + j;
                        if all_valid || mask.is_valid(k) {
                            let val = data[k * n_depth + i_depth].to_f64() as i32;
                            let predictor =
                                if j > 0 && (all_valid || mask.is_valid(k - 1)) {
                                    prev_val
                                } else if i > 0
                                    && (all_valid || mask.is_valid(k - width))
                                {
                                    data[(k - width) * n_depth + i_depth].to_f64() as i32
                                } else {
                                    prev_val
                                };
                            let delta = val.wrapping_sub(predictor);
                            let bin = (delta + offset) as u8 as usize;
                            let (len, code) = code_table[bin];
                            let len = len as u32;
                            accum |= (code as u64) << (64 - accum_bits - len);
                            accum_bits += len;
                            if accum_bits >= 32 {
                                let word = (accum >> 32) as u32;
                                encoded[out_idx..out_idx + 4]
                                    .copy_from_slice(&word.to_le_bytes());
                                out_idx += 4;
                                accum <<= 32;
                                accum_bits -= 32;
                            }
                            prev_val = val;
                        }
                    }
                }
            }
        }
        HuffMode::Direct => {
            for i in 0..height {
                for j in 0..width {
                    let k = i * width + j;
                    if all_valid || mask.is_valid(k) {
                        for m in 0..n_depth {
                            let val = data[k * n_depth + m].to_f64() as i32;
                            let bin = (val + offset) as usize;
                            let (len, code) = code_table[bin];
                            let len = len as u32;
                            accum |= (code as u64) << (64 - accum_bits - len);
                            accum_bits += len;
                            if accum_bits >= 32 {
                                let word = (accum >> 32) as u32;
                                encoded[out_idx..out_idx + 4]
                                    .copy_from_slice(&word.to_le_bytes());
                                out_idx += 4;
                                accum <<= 32;
                                accum_bits -= 32;
                            }
                        }
                    }
                }
            }
        }
    }

    // Flush remaining bits
    if accum_bits > 0 {
        let word = (accum >> 32) as u32;
        encoded[out_idx..out_idx + 4].copy_from_slice(&word.to_le_bytes());
        out_idx += 4;
    }

    // Compute final size with padding: round up to next u32 boundary + 1 extra
    // u32 for decode read-ahead
    let num_uints_final = (out_idx / 4) + 1; // +1 for decode read-ahead padding
    encoded.truncate(num_uints_final * 4);

    buf.extend_from_slice(&encoded);
    Some(buf)
}

/// Scratch buffers reused across tiles to avoid per-tile allocations.
struct ScratchBuffers {
    valid_data: Vec<f64>,
    diff_data: Vec<f64>,
    quant_vec: Vec<u32>,
    /// Holds the quantized values from the non-diff encoding attempt,
    /// saved before the diff encoding attempt overwrites `quant_vec`.
    non_diff_quant: Vec<u32>,
    tile_buf: Vec<u8>,
}

impl ScratchBuffers {
    fn new() -> Self {
        Self {
            valid_data: Vec::new(),
            diff_data: Vec::new(),
            quant_vec: Vec::new(),
            non_diff_quant: Vec::new(),
            tile_buf: Vec::new(),
        }
    }
}

/// Information about how a tile was encoded, used to compute the decoder's
/// reconstructed values for the previous-depth reconstruction buffer.
#[derive(Clone, Copy)]
enum TileReconInfo {
    /// All valid pixels in the tile have the same reconstructed value.
    Constant(f64),
    /// Values were quantized: reconstructed[i] = offset + quant[i] * inv_scale
    /// (clamped to z_max).
    Quantized {
        offset: f64,
        inv_scale: f64,
        z_max: f64,
    },
    /// Raw binary: reconstruction matches the original values exactly.
    RawBinary,
}

fn encode_tiles<T: LercDataType>(
    blob: &mut Vec<u8>,
    data: &[T],
    mask: &BitMask,
    header: &HeaderInfo,
    _z_min_vec: &[f64],
    z_max_vec: &[f64],
) -> Result<()> {
    let mb_size = header.micro_block_size as usize;
    let n_depth = header.n_depth as usize;
    let n_rows = header.n_rows as usize;
    let n_cols = header.n_cols as usize;
    let max_z_error = header.max_z_error;

    // Fast path: u8 lossless, all-valid, single depth.
    // This avoids f64 conversions, uses stack-allocated arrays, and merges
    // quantization + LUT decision + encoding in a single pass per tile.
    let all_valid = header.num_valid_pixel == header.n_rows * header.n_cols;
    if T::DATA_TYPE == DataType::Byte && max_z_error == 0.5 && all_valid && n_depth == 1 {
        // T is u8 (verified by DATA_TYPE == Byte and T::BYTES == 1).
        // Safe reinterpret: u8 has size 1, alignment 1, same as T when T=u8.
        debug_assert_eq!(T::BYTES, 1);
        debug_assert_eq!(core::mem::size_of::<T>(), 1);
        debug_assert_eq!(core::mem::align_of::<T>(), 1);
        let u8_data: &[u8] = unsafe {
            core::slice::from_raw_parts(data.as_ptr() as *const u8, data.len())
        };
        let mut quant_buf = [0u32; 256]; // max tile size is 16x16=256
        encode_tiles_u8_fast(blob, u8_data, header, &mut quant_buf);
        return Ok(());
    }

    let num_tiles_vert = n_rows.div_ceil(mb_size);
    let num_tiles_hori = n_cols.div_ceil(mb_size);

    let max_val_to_quantize: f64 = match header.data_type {
        DataType::Char | DataType::Byte | DataType::Short | DataType::UShort => {
            ((1u32 << 15) - 1) as f64
        }
        _ => ((1u32 << 30) - 1) as f64,
    };

    let mut scratch = ScratchBuffers::new();

    // Allocate a per-pixel reconstruction buffer when multi-depth lossy encoding
    // is active. This mirrors the C++ ScaleBack approach: after encoding each
    // depth slice, we store the values as the decoder would reconstruct them,
    // so that subsequent depth-slice diffs are computed from reconstructed (not
    // original) values.
    let needs_recon = n_depth > 1 && max_z_error > 0.0;
    let mut recon_buf: Vec<f64> = if needs_recon {
        vec![0.0; n_rows * n_cols]
    } else {
        Vec::new()
    };

    for i_tile in 0..num_tiles_vert {
        let i0 = i_tile * mb_size;
        let i1 = (i0 + mb_size).min(n_rows);

        for j_tile in 0..num_tiles_hori {
            let j0 = j_tile * mb_size;
            let j1 = (j0 + mb_size).min(n_cols);

            for i_depth in 0..n_depth {
                encode_tile(
                    blob,
                    data,
                    mask,
                    header,
                    z_max_vec,
                    max_z_error,
                    max_val_to_quantize,
                    i0,
                    i1,
                    j0,
                    j1,
                    i_depth,
                    n_depth > 1 && i_depth > 0,
                    &mut scratch,
                    if needs_recon {
                        Some(&mut recon_buf)
                    } else {
                        None
                    },
                )?;
            }
        }
    }

    Ok(())
}

/// Specialized tile encoding for u8 lossless, all-valid, single-depth images.
/// Avoids all f64 conversions and Vec allocations in the inner loop.
fn encode_tiles_u8_fast(
    blob: &mut Vec<u8>,
    data: &[u8],
    header: &HeaderInfo,
    quant_buf: &mut [u32; 256],
) {
    let mb_size = header.micro_block_size as usize;
    let n_rows = header.n_rows as usize;
    let n_cols = header.n_cols as usize;
    let version = header.version;

    let num_tiles_vert = n_rows.div_ceil(mb_size);
    let num_tiles_hori = n_cols.div_ceil(mb_size);

    // Pre-allocate: worst case is ~1 byte overhead per tile + data bytes.
    // For high-entropy u8, output is roughly the same size as input.
    blob.reserve(data.len() + num_tiles_vert * num_tiles_hori * 4);

    let pattern: u8 = if version >= 5 { 14 } else { 15 };

    for i_tile in 0..num_tiles_vert {
        let i0 = i_tile * mb_size;
        let i1 = (i0 + mb_size).min(n_rows);

        for j_tile in 0..num_tiles_hori {
            let j0 = j_tile * mb_size;
            let j1 = (j0 + mb_size).min(n_cols);

            let integrity = ((j0 as u8 >> 3) & pattern) << 2;
            encode_single_u8_tile(blob, data, n_cols, i0, i1, j0, j1, integrity, quant_buf);
        }
    }
}

/// Encode a single tile block (one depth slice of one micro-block).
/// When `try_diff` is true (depth > 0 with nDepth > 1), the encoder tries
/// diff (delta) encoding relative to the previous depth slice and picks
/// whichever representation is smaller.
///
/// When `recon_buf` is `Some`, it holds the decoder-reconstructed values for the
/// previous depth slice (indexed by pixel index `k = row * n_cols + col`).
/// Diffs are computed against these reconstructed values instead of the original
/// data. After encoding, the buffer is updated with this depth slice's
/// reconstructed values so it is ready for the next depth.
#[allow(clippy::too_many_arguments)]
fn encode_tile<T: LercDataType>(
    blob: &mut Vec<u8>,
    data: &[T],
    mask: &BitMask,
    header: &HeaderInfo,
    z_max_vec: &[f64],
    max_z_error: f64,
    max_val_to_quantize: f64,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    i_depth: usize,
    try_diff: bool,
    scratch: &mut ScratchBuffers,
    recon_buf: Option<&mut Vec<f64>>,
) -> Result<()> {
    let n_cols = header.n_cols as usize;
    let n_depth = header.n_depth as usize;

    // Collect valid pixels for this depth slice (reusing scratch buffer)
    scratch.valid_data.clear();
    let mut z_min_f = f64::MAX;
    let mut z_max_f = f64::MIN;

    for i in i0..i1 {
        let mut k = i * n_cols + j0;
        for _j in j0..j1 {
            if mask.is_valid(k) {
                let val = data[k * n_depth + i_depth].to_f64();
                scratch.valid_data.push(val);
                if val < z_min_f {
                    z_min_f = val;
                }
                if val > z_max_f {
                    z_max_f = val;
                }
            }
            k += 1;
        }
    }

    let num_valid = scratch.valid_data.len();

    // Integrity check bits (pattern depends on version)
    let pattern: u8 = if header.version >= 5 { 14 } else { 15 };
    let integrity = ((j0 as u8 >> 3) & pattern) << 2;

    // Compute diff values if applicable.
    // When a reconstruction buffer is provided, use the reconstructed (quantized)
    // previous-depth values instead of the original data. This matches what the
    // C++ encoder does via ScaleBack and what the decoder uses for reconstruction.
    let has_diff = if try_diff {
        scratch.diff_data.clear();
        let mut overflow = false;
        let mut idx = 0;
        for i in i0..i1 {
            let mut k = i * n_cols + j0;
            for _j in j0..j1 {
                if mask.is_valid(k) {
                    let cur = data[k * n_depth + i_depth].to_f64();
                    let prev = if let Some(ref rb) = recon_buf {
                        rb[k]
                    } else {
                        data[k * n_depth + i_depth - 1].to_f64()
                    };
                    let diff = cur - prev;
                    // Check for integer overflow: for integer types, the diff
                    // must be representable as i32
                    if T::is_integer()
                        && (diff < i32::MIN as f64 || diff > i32::MAX as f64)
                    {
                        overflow = true;
                        break;
                    }
                    scratch.diff_data.push(diff);
                    idx += 1;
                }
                k += 1;
            }
            if overflow {
                break;
            }
        }
        !overflow && idx == num_valid
    } else {
        false
    };

    // Encode without diff
    scratch.tile_buf.clear();
    encode_tile_inner::<T>(
        &mut scratch.tile_buf,
        &scratch.valid_data,
        &mut scratch.quant_vec,
        num_valid,
        z_min_f,
        z_max_f,
        max_z_error,
        max_val_to_quantize,
        integrity,
        header.data_type,
        false,
    );
    let non_diff_len = scratch.tile_buf.len();

    // Save the non-diff quantized values before the diff attempt overwrites quant_vec
    scratch.non_diff_quant.clear();
    scratch.non_diff_quant.extend_from_slice(&scratch.quant_vec);
    let non_diff_z_min = z_min_f;
    let non_diff_z_max = z_max_f;

    // Encode with diff (if available)
    let mut used_diff = false;
    let mut diff_z_min = 0.0f64;
    let mut diff_z_max = 0.0f64;
    if has_diff {
        let mut diff_min = f64::MAX;
        let mut diff_max = f64::MIN;
        for &d in scratch.diff_data.iter() {
            if d < diff_min {
                diff_min = d;
            }
            if d > diff_max {
                diff_max = d;
            }
        }
        diff_z_min = diff_min;
        diff_z_max = diff_max;
        // Append diff encoding after the non-diff data in tile_buf
        let diff_start = scratch.tile_buf.len();
        encode_tile_inner::<T>(
            &mut scratch.tile_buf,
            &scratch.diff_data,
            &mut scratch.quant_vec,
            num_valid,
            diff_min,
            diff_max,
            max_z_error,
            max_val_to_quantize,
            integrity,
            header.data_type,
            true,
        );
        let diff_len = scratch.tile_buf.len() - diff_start;

        // Pick the smaller encoding
        if diff_len < non_diff_len {
            blob.extend_from_slice(&scratch.tile_buf[diff_start..]);
            used_diff = true;
        } else {
            blob.extend_from_slice(&scratch.tile_buf[..non_diff_len]);
        }
    } else {
        blob.extend_from_slice(&scratch.tile_buf);
    }

    // Update the reconstruction buffer with what the decoder would produce for
    // this depth slice.
    if let Some(rb) = recon_buf {
        let z_max_depth = if n_depth > 1 {
            z_max_vec[i_depth]
        } else {
            header.z_max
        };

        // Determine reconstruction info based on the chosen encoding path
        let recon_info = if used_diff {
            // Diff encoding was chosen; reconstruct from diff quantized values
            classify_recon(
                &scratch.diff_data,
                &scratch.quant_vec,
                num_valid,
                diff_z_min,
                diff_z_max,
                max_z_error,
                max_val_to_quantize,
                z_max_depth,
                header.data_type,
                true,
            )
        } else {
            // Non-diff encoding was chosen; reconstruct from original quantized values
            classify_recon(
                &scratch.valid_data,
                &scratch.non_diff_quant,
                num_valid,
                non_diff_z_min,
                non_diff_z_max,
                max_z_error,
                max_val_to_quantize,
                z_max_depth,
                header.data_type,
                false,
            )
        };

        // Apply reconstruction to the buffer
        let mut valid_idx = 0;
        for i in i0..i1 {
            let mut k = i * n_cols + j0;
            for _j in j0..j1 {
                if mask.is_valid(k) {
                    let recon_val = match recon_info {
                        TileReconInfo::Constant(c) => {
                            if used_diff {
                                // diff constant: decoder adds offset to prev
                                (c + rb[k]).min(z_max_depth)
                            } else {
                                c
                            }
                        }
                        TileReconInfo::Quantized {
                            offset,
                            inv_scale,
                            z_max,
                        } => {
                            let q = if used_diff {
                                scratch.quant_vec[valid_idx] as f64
                            } else {
                                scratch.non_diff_quant[valid_idx] as f64
                            };
                            let z = offset + q * inv_scale;
                            if used_diff {
                                (z + rb[k]).min(z_max)
                            } else {
                                z.min(z_max)
                            }
                        }
                        TileReconInfo::RawBinary => {
                            // Raw binary: the original values are stored exactly
                            scratch.valid_data[valid_idx]
                        }
                    };
                    rb[k] = recon_val;
                    valid_idx += 1;
                }
                k += 1;
            }
        }
    }

    Ok(())
}

/// Classify how a tile was encoded and return reconstruction info.
/// This mirrors the logic in `encode_tile_inner` to determine whether the tile
/// ended up as constant, quantized, or raw binary.
#[allow(clippy::too_many_arguments)]
fn classify_recon(
    values: &[f64],
    quant: &[u32],
    num_valid: usize,
    z_min_f: f64,
    z_max_f: f64,
    max_z_error: f64,
    max_val_to_quantize: f64,
    z_max_depth: f64,
    _src_data_type: DataType,
    b_diff_enc: bool,
) -> TileReconInfo {
    if num_valid == 0 || (z_min_f == 0.0 && z_max_f == 0.0) {
        // Constant zero block
        return TileReconInfo::Constant(0.0);
    }

    let need_quantize = if max_z_error == 0.0 {
        z_max_f > z_min_f
    } else {
        let max_val = (z_max_f - z_min_f) / (2.0 * max_z_error);
        max_val <= max_val_to_quantize && (max_val + 0.5) as u32 != 0
    };

    if !need_quantize && z_min_f == z_max_f {
        // Constant block
        return TileReconInfo::Constant(z_min_f);
    }

    if !need_quantize {
        if !b_diff_enc {
            // Raw binary
            return TileReconInfo::RawBinary;
        }
        // Diff + raw binary is not allowed; this means non-diff won the size
        // comparison in encode_tile. This code path should not be reached since
        // the diff sentinel is always larger, but handle it safely.
        return TileReconInfo::RawBinary;
    }

    // Check if all values quantize to the same value (max_quant == 0)
    let max_quant = quant.iter().copied().max().unwrap_or(0);
    if max_quant == 0 {
        // All quantized to same -> constant block with z_min_f as offset
        return TileReconInfo::Constant(z_min_f);
    }

    // Check if raw binary was cheaper (only for non-diff mode).
    // We need to replicate the raw-binary fallback check from encode_tile_inner.
    if !b_diff_enc {
        // Estimate the encoded size: this mirrors the bitstuffer output size.
        // If the actual encoder fell back to raw binary, the quant values are
        // irrelevant and reconstruction should use original values.
        // We approximate by checking if the encoded bytes would exceed raw size.
        // However, we cannot know the exact encoded size here without re-encoding.
        // Instead, rely on the fact that if raw binary was chosen, the values
        // array contains the exact original values. In practice for lossy mode
        // with max_z_error > 0, quantization always beats raw for non-trivial
        // data ranges, so this fallback is rare.
        //
        // For correctness we conservatively return Quantized, which is correct
        // when quantization was used and harmless when raw was used (since raw
        // stores exact values, reconstruction is exact either way for depth 0,
        // and raw binary cannot be used with diff encoding).
        let _ = values;
    }

    let inv_scale = 2.0 * max_z_error;
    TileReconInfo::Quantized {
        offset: z_min_f,
        inv_scale,
        z_max: z_max_depth,
    }
}

/// Encode the payload for a single tile block, appending to `buf`.
/// When `b_diff_enc` is true, bit 2 of the compression flag is set and the
/// offset data type is forced to Int for integer source types.
/// `quant_scratch` is a reusable scratch buffer for quantized values.
#[allow(clippy::too_many_arguments)]
fn encode_tile_inner<T: LercDataType>(
    buf: &mut Vec<u8>,
    values: &[f64],
    quant_scratch: &mut Vec<u32>,
    num_valid: usize,
    z_min_f: f64,
    z_max_f: f64,
    max_z_error: f64,
    max_val_to_quantize: f64,
    integrity: u8,
    src_data_type: DataType,
    b_diff_enc: bool,
) {
    let diff_flag: u8 = if b_diff_enc { tile_flags::DIFF_ENCODING } else { 0 };

    if num_valid == 0 || (z_min_f == 0.0 && z_max_f == 0.0) {
        buf.push(TileCompressionMode::ConstZero as u8 | diff_flag | integrity);
        return;
    }

    // Check if we need quantization
    let need_quantize = if max_z_error == 0.0 {
        z_max_f > z_min_f
    } else {
        let max_val = (z_max_f - z_min_f) / (2.0 * max_z_error);
        max_val <= max_val_to_quantize && (max_val + 0.5) as u32 != 0
    };

    if !need_quantize && z_min_f == z_max_f {
        // Constant block (all same value / all same diff)
        let z_min_t = T::from_f64(z_min_f);
        let (dt_reduced, tc) = if b_diff_enc && src_data_type.is_integer() {
            tiles::reduce_data_type(z_min_f as i32, DataType::Int)
        } else {
            tiles::reduce_data_type(z_min_t, src_data_type)
        };
        let bits67 = (tc as u8) << tile_flags::TYPE_REDUCTION_SHIFT;
        buf.push(TileCompressionMode::ConstOffset as u8 | bits67 | diff_flag | integrity);
        tiles::write_variable_data_type(buf, z_min_f, dt_reduced);
        return;
    }

    if !need_quantize {
        if !b_diff_enc {
            buf.push(TileCompressionMode::RawBinary as u8 | integrity);
            for val in values {
                let t = T::from_f64(*val);
                t.extend_le_bytes(buf);
            }
            return;
        }
        // Raw binary is not allowed with diff enc per the decoder.
        // Append an impossibly-large sentinel so the non-diff path wins
        // the size comparison. We use raw-size + 1 so it always loses.
        let sentinel_len = 1 + num_valid * T::BYTES + 1;
        buf.resize(buf.len() + sentinel_len, 0u8);
        return;
    }

    // Quantize and bit-stuff
    let z_min_t = T::from_f64(z_min_f);
    let (dt_reduced, tc) = if b_diff_enc && src_data_type.is_integer() {
        tiles::reduce_data_type(z_min_f as i32, DataType::Int)
    } else {
        tiles::reduce_data_type(z_min_t, src_data_type)
    };
    let bits67 = (tc as u8) << tile_flags::TYPE_REDUCTION_SHIFT;

    // Quantize values (reusing scratch buffer)
    quant_scratch.clear();
    if T::is_integer() && max_z_error == 0.5 {
        // Lossless integer
        for val in values {
            quant_scratch.push((*val - z_min_f) as u32);
        }
    } else {
        let scale = 1.0 / (2.0 * max_z_error);
        for val in values {
            quant_scratch.push(((*val - z_min_f) * scale + 0.5) as u32);
        }
    }

    let max_quant = quant_scratch.iter().copied().max().unwrap_or(0);

    if max_quant == 0 {
        // All values quantize to same -> constant block
        buf.push(TileCompressionMode::ConstOffset as u8 | bits67 | diff_flag | integrity);
        tiles::write_variable_data_type(buf, z_min_f, dt_reduced);
        return;
    }

    // Try LUT vs simple encoding
    let encoded_start = buf.len();
    buf.push(TileCompressionMode::BitStuffed as u8 | bits67 | diff_flag | integrity);
    tiles::write_variable_data_type(buf, z_min_f, dt_reduced);

    // Use fast path for small tiles with small values (common for u8 data)
    if num_valid <= 256 && max_quant < 256 {
        crate::bitstuffer::encode_small_tile_into(buf, quant_scratch, max_quant);
    } else if let Some(lut_info) = crate::bitstuffer::should_use_lut(quant_scratch, max_quant) {
        crate::bitstuffer::encode_lut_into(buf, &lut_info, num_valid as u32);
    } else {
        crate::bitstuffer::encode_simple_into(buf, quant_scratch, max_quant);
    }

    // Check if raw binary would be smaller (only for non-diff mode)
    if !b_diff_enc {
        let raw_size = 1 + num_valid * T::BYTES;
        let encoded_len = buf.len() - encoded_start;
        if encoded_len >= raw_size {
            // Raw binary fallback: replace the encoded data
            buf.truncate(encoded_start);
            buf.push(TileCompressionMode::RawBinary as u8 | integrity);
            for val in values {
                let t = T::from_f64(*val);
                t.extend_le_bytes(buf);
            }
        }
    }
}


