use alloc::vec;
use alloc::vec::Vec;

use crate::bitmask::BitMask;
use crate::error::Result;
use crate::fpl;
use crate::header::{self, HeaderInfo};
use crate::huffman::HuffmanCodec;
use crate::rle;
use crate::tiles;
use crate::types::{DataType, LercDataType};
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
    for i in 1..n_bits {
        val >>= 1;
        counts[i] += (val & 1) as i32;
    }
}

/// Count set bits per bit-plane in a signed XOR difference value.
/// Equivalent to C++ `Lerc2::AddIntToCounts`.
#[inline]
fn add_int_to_counts(counts: &mut [i32], mut val: i32, n_bits: usize) {
    counts[0] += val & 1;
    for i in 1..n_bits {
        val >>= 1;
        counts[i] += val & 1;
    }
}

/// Try bit-plane compression for integer data. Analyzes XOR differences between
/// neighboring pixels to find bit planes that look like random noise (1-bit
/// frequency close to 0.5). Returns `Some(new_max_z_error)` if noisy bit planes
/// were found, or `None` if the data doesn't qualify.
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

    // Compute per-depth min/max
    let mut z_min_vec = vec![f64::MAX; n_depth];
    let mut z_max_vec = vec![f64::MIN; n_depth];
    let mut overall_min = f64::MAX;
    let mut overall_max = f64::MIN;

    for i in 0..height {
        for j in 0..width {
            let k = i * width + j;
            if mask.is_valid(k) {
                for m in 0..n_depth {
                    let val = data[k * n_depth + m].to_f64();
                    if val < z_min_vec[m] {
                        z_min_vec[m] = val;
                    }
                    if val > z_max_vec[m] {
                        z_max_vec[m] = val;
                    }
                    if val < overall_min {
                        overall_min = val;
                    }
                    if val > overall_max {
                        overall_max = val;
                    }
                }
            }
        }
    }

    if num_valid == 0 {
        overall_min = 0.0;
        overall_max = 0.0;
    }

    let mut hd = HeaderInfo {
        version: 6,
        checksum: 0,
        n_rows: height as i32,
        n_cols: width as i32,
        n_depth: n_depth as i32,
        num_valid_pixel: num_valid as i32,
        micro_block_size: 8, // will be updated after block size selection
        blob_size: 0,        // patched later
        data_type: T::DATA_TYPE,
        n_blobs_more,
        pass_no_data_values: false,
        is_int: false,
        max_z_error,
        z_min: overall_min,
        z_max: overall_max,
        no_data_val: 0.0,
        no_data_val_orig: 0.0,
    };

    // Select the best micro block size by trying both 8 and 16.
    // The block size only affects the tiling path, not Huffman or FPL.
    let best_block_size =
        select_block_size::<T>(data, mask, &hd, &z_min_vec, &z_max_vec)?;
    hd.micro_block_size = best_block_size;

    let mut blob = header::write_header(&hd);

    // Write mask
    let num_pixels = width * height;
    if num_valid == 0 || num_valid == num_pixels {
        // No mask data needed
        blob.extend_from_slice(&0i32.to_le_bytes());
    } else {
        let mask_compressed = rle::compress(mask.as_bytes());
        let mask_size = mask_compressed.len() as i32;
        blob.extend_from_slice(&mask_size.to_le_bytes());
        blob.extend_from_slice(&mask_compressed);
    }

    if num_valid == 0 {
        header::finalize_blob(&mut blob);
        return Ok(blob);
    }

    // Const image
    if overall_min == overall_max {
        header::finalize_blob(&mut blob);
        return Ok(blob);
    }

    // Write per-depth min/max ranges (always for v6)
    for m in 0..n_depth {
        write_typed_value::<T>(&mut blob, z_min_vec[m]);
    }
    for m in 0..n_depth {
        write_typed_value::<T>(&mut blob, z_max_vec[m]);
    }

    // Check if all depths are constant
    if z_min_vec == z_max_vec {
        header::finalize_blob(&mut blob);
        return Ok(blob);
    }

    // One sweep flag = 0 (we use tiling)
    blob.push(0u8);

    // Check if Huffman-eligible types
    let try_huffman_int = matches!(T::DATA_TYPE, DataType::Char | DataType::Byte)
        && max_z_error == 0.5;
    let try_huffman_flt =
        matches!(T::DATA_TYPE, DataType::Float | DataType::Double) && max_z_error == 0.0;

    if try_huffman_int {
        // Try Huffman encoding for 8-bit integer types
        if let Some(huffman_blob) = try_encode_huffman_int(data, mask, &hd) {
            // Huffman is beneficial; compare against tiling size
            let mut tiling_blob = Vec::new();
            encode_tiles(&mut tiling_blob, data, mask, &hd, &z_min_vec, &z_max_vec)?;

            if huffman_blob.len() < tiling_blob.len() {
                blob.extend_from_slice(&huffman_blob);
                header::finalize_blob(&mut blob);
                return Ok(blob);
            }
        }
        // Huffman not beneficial or failed; fall through to tiling
        // Write IEM_Tiling = 0
        blob.push(0u8);
    } else if try_huffman_flt {
        // For lossless float/double, use FPL encoding
        let is_double = T::DATA_TYPE == DataType::Double;
        let fpl_data = fpl::encode_huffman_flt(data, is_double, width, height, n_depth)?;
        blob.push(3u8); // IEM_DeltaDeltaHuffman
        blob.extend_from_slice(&fpl_data);
        header::finalize_blob(&mut blob);
        return Ok(blob);
    }

    // Write tiles
    encode_tiles(&mut blob, data, mask, &hd, &z_min_vec, &z_max_vec)?;

    header::finalize_blob(&mut blob);
    Ok(blob)
}

/// Try encoding tiles with block sizes 8 and 16, return whichever is smaller.
fn select_block_size<T: LercDataType>(
    data: &[T],
    mask: &BitMask,
    hd: &HeaderInfo,
    z_min_vec: &[f64],
    z_max_vec: &[f64],
) -> Result<i32> {
    // Encode with block size 8
    let mut hd8 = hd.clone();
    hd8.micro_block_size = 8;
    let mut buf8 = Vec::new();
    encode_tiles(&mut buf8, data, mask, &hd8, z_min_vec, z_max_vec)?;

    // Encode with block size 16
    let mut hd16 = hd.clone();
    hd16.micro_block_size = 16;
    let mut buf16 = Vec::new();
    encode_tiles(&mut buf16, data, mask, &hd16, z_min_vec, z_max_vec)?;

    if buf16.len() < buf8.len() {
        Ok(16)
    } else {
        Ok(8)
    }
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

/// MSB-first bit push (for Huffman data encoding).
fn push_value(buf: &mut [u8], bit_pos: &mut i32, value: u32, len: i32) {
    let uint_idx = (*bit_pos / 32) as usize;
    let local_bit = *bit_pos & 31;

    if 32 - local_bit >= len {
        if local_bit == 0 {
            buf[uint_idx * 4..uint_idx * 4 + 4].fill(0);
        }
        let mut temp = u32::from_le_bytes(
            buf[uint_idx * 4..uint_idx * 4 + 4].try_into().unwrap(),
        );
        temp |= value << (32 - local_bit - len);
        buf[uint_idx * 4..uint_idx * 4 + 4].copy_from_slice(&temp.to_le_bytes());
        *bit_pos += len;
    } else {
        let overflow = local_bit + len - 32;
        let mut temp = u32::from_le_bytes(
            buf[uint_idx * 4..uint_idx * 4 + 4].try_into().unwrap(),
        );
        temp |= value >> overflow;
        buf[uint_idx * 4..uint_idx * 4 + 4].copy_from_slice(&temp.to_le_bytes());

        let next_idx = uint_idx + 1;
        let temp2 = value << (32 - overflow);
        buf[next_idx * 4..next_idx * 4 + 4].copy_from_slice(&temp2.to_le_bytes());

        *bit_pos += len;
    }
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

    // Try direct Huffman (IEM_Huffman, v4+)
    let mut direct_codec = HuffmanCodec::new();
    let direct_ok = direct_codec.compute_codes(&histo);
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
        HuffMode::Delta => buf.push(1u8), // IEM_DeltaHuffman
        HuffMode::Direct => buf.push(2u8), // IEM_Huffman
    }

    // Write code table
    let code_table_bytes = codec.write_code_table(6).ok()?;
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
    let num_uints_data = ((total_bits + 31) / 32) as usize;
    let num_uints_total = num_uints_data + 1; // +1 for decode read-ahead padding
    let mut encoded = vec![0u8; num_uints_total * 4];
    let mut bit_pos = 0i32;

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
                            push_value(&mut encoded, &mut bit_pos, code, len as i32);
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
                            push_value(&mut encoded, &mut bit_pos, code, len as i32);
                        }
                    }
                }
            }
        }
    }

    // Compute final size with padding: (bitPos > 0 ? 1 : 0) + 1 extra uint32s
    let num_uints_final = if bit_pos > 0 {
        (bit_pos as usize / 32) + 1 + 1
    } else {
        (bit_pos as usize / 32) + 1
    };
    encoded.truncate(num_uints_final * 4);

    buf.extend_from_slice(&encoded);
    Some(buf)
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

    let num_tiles_vert = (n_rows + mb_size - 1) / mb_size;
    let num_tiles_hori = (n_cols + mb_size - 1) / mb_size;

    let max_val_to_quantize: f64 = match header.data_type {
        DataType::Char | DataType::Byte | DataType::Short | DataType::UShort => {
            ((1u32 << 15) - 1) as f64
        }
        _ => ((1u32 << 30) - 1) as f64,
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
                )?;
            }
        }
    }

    Ok(())
}

/// Encode a single tile block (one depth slice of one micro-block).
/// When `try_diff` is true (depth > 0 with nDepth > 1), the encoder tries
/// diff (delta) encoding relative to the previous depth slice and picks
/// whichever representation is smaller.
fn encode_tile<T: LercDataType>(
    blob: &mut Vec<u8>,
    data: &[T],
    mask: &BitMask,
    header: &HeaderInfo,
    _z_max_vec: &[f64],
    max_z_error: f64,
    max_val_to_quantize: f64,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    i_depth: usize,
    try_diff: bool,
) -> Result<()> {
    let n_cols = header.n_cols as usize;
    let n_depth = header.n_depth as usize;

    // Collect valid pixels for this depth slice
    let mut valid_data: Vec<f64> = Vec::new();
    let mut z_min_f = f64::MAX;
    let mut z_max_f = f64::MIN;

    for i in i0..i1 {
        let mut k = i * n_cols + j0;
        for _j in j0..j1 {
            if mask.is_valid(k) {
                let val = data[k * n_depth + i_depth].to_f64();
                valid_data.push(val);
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

    let num_valid = valid_data.len();

    // Integrity check bits
    let pattern: u8 = 14; // v5+
    let integrity = ((j0 as u8 >> 3) & pattern) << 2;

    // Compute diff values if applicable
    let diff_data: Option<Vec<f64>> = if try_diff {
        let mut diffs = Vec::with_capacity(num_valid);
        let mut overflow = false;
        let mut idx = 0;
        for i in i0..i1 {
            let mut k = i * n_cols + j0;
            for _j in j0..j1 {
                if mask.is_valid(k) {
                    let cur = data[k * n_depth + i_depth].to_f64();
                    let prev = data[k * n_depth + i_depth - 1].to_f64();
                    let diff = cur - prev;
                    // Check for integer overflow: for integer types, the diff
                    // must be representable as i32
                    if T::is_integer()
                        && (diff < i32::MIN as f64 || diff > i32::MAX as f64)
                    {
                        overflow = true;
                        break;
                    }
                    diffs.push(diff);
                    idx += 1;
                }
                k += 1;
            }
            if overflow {
                break;
            }
        }
        if overflow || idx != num_valid {
            None
        } else {
            Some(diffs)
        }
    } else {
        None
    };

    // Encode without diff
    let non_diff_encoded = encode_tile_inner::<T>(
        &valid_data,
        num_valid,
        z_min_f,
        z_max_f,
        max_z_error,
        max_val_to_quantize,
        integrity,
        header.data_type,
        false,
    );

    // Encode with diff (if available)
    let diff_encoded = if let Some(ref diffs) = diff_data {
        let mut diff_min = f64::MAX;
        let mut diff_max = f64::MIN;
        for &d in diffs.iter() {
            if d < diff_min {
                diff_min = d;
            }
            if d > diff_max {
                diff_max = d;
            }
        }
        Some(encode_tile_inner::<T>(
            diffs,
            num_valid,
            diff_min,
            diff_max,
            max_z_error,
            max_val_to_quantize,
            integrity,
            header.data_type,
            true,
        ))
    } else {
        None
    };

    // Pick the smaller encoding
    match diff_encoded {
        Some(ref diff_buf) if diff_buf.len() < non_diff_encoded.len() => {
            blob.extend_from_slice(diff_buf);
        }
        _ => {
            blob.extend_from_slice(&non_diff_encoded);
        }
    }

    Ok(())
}

/// Encode the payload for a single tile block. Returns the complete block bytes
/// (header byte + offset + bitstuffed data). When `b_diff_enc` is true, bit 2
/// of the compression flag is set and the offset data type is forced to Int
/// for integer source types.
fn encode_tile_inner<T: LercDataType>(
    values: &[f64],
    num_valid: usize,
    z_min_f: f64,
    z_max_f: f64,
    max_z_error: f64,
    max_val_to_quantize: f64,
    integrity: u8,
    src_data_type: DataType,
    b_diff_enc: bool,
) -> Vec<u8> {
    let diff_flag: u8 = if b_diff_enc { 4 } else { 0 };

    if num_valid == 0 || (z_min_f == 0.0 && z_max_f == 0.0) {
        // Constant 0 block (for diff enc this means all diffs are zero)
        return vec![2 | diff_flag | integrity];
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
        let bits67 = (tc as u8) << 6;
        let mut buf = vec![3 | bits67 | diff_flag | integrity];
        tiles::write_variable_data_type(&mut buf, z_min_f, dt_reduced);
        return buf;
    }

    if !need_quantize {
        if !b_diff_enc {
            let mut buf = vec![0 | integrity];
            for val in values {
                let t = T::from_f64(*val);
                write_typed_value_raw::<T>(&mut buf, t);
            }
            return buf;
        }
        // Raw binary is not allowed with diff enc per the decoder.
        // Return an impossibly-large sentinel so the non-diff path wins
        // the size comparison. We use raw-size + 1 so it always loses.
        return vec![0u8; 1 + num_valid * T::BYTES + 1];
    }

    // Quantize and bit-stuff
    let z_min_t = T::from_f64(z_min_f);
    let (dt_reduced, tc) = if b_diff_enc && src_data_type.is_integer() {
        tiles::reduce_data_type(z_min_f as i32, DataType::Int)
    } else {
        tiles::reduce_data_type(z_min_t, src_data_type)
    };
    let bits67 = (tc as u8) << 6;

    // Quantize values
    let mut quant_vec: Vec<u32> = Vec::with_capacity(num_valid);
    if T::is_integer() && max_z_error == 0.5 {
        // Lossless integer
        for val in values {
            quant_vec.push((*val - z_min_f) as u32);
        }
    } else {
        let scale = 1.0 / (2.0 * max_z_error);
        for val in values {
            quant_vec.push(((*val - z_min_f) * scale + 0.5) as u32);
        }
    }

    let max_quant = quant_vec.iter().copied().max().unwrap_or(0);

    if max_quant == 0 {
        // All values quantize to same -> constant block
        let mut buf = vec![3 | bits67 | diff_flag | integrity];
        tiles::write_variable_data_type(&mut buf, z_min_f, dt_reduced);
        return buf;
    }

    // Try LUT vs simple encoding
    let encoded = if let Some(sorted) = crate::bitstuffer::should_use_lut(&quant_vec) {
        let lut_bits67 = bits67;
        let mut tile_buf = vec![1 | lut_bits67 | diff_flag | integrity];
        tiles::write_variable_data_type(&mut tile_buf, z_min_f, dt_reduced);
        let stuffed = crate::bitstuffer::encode_lut(&sorted);
        tile_buf.extend_from_slice(&stuffed);
        tile_buf
    } else {
        let mut tile_buf = vec![1 | bits67 | diff_flag | integrity];
        tiles::write_variable_data_type(&mut tile_buf, z_min_f, dt_reduced);
        let stuffed = crate::bitstuffer::encode_simple(&quant_vec);
        tile_buf.extend_from_slice(&stuffed);
        tile_buf
    };

    // Check if raw binary would be smaller (only for non-diff mode)
    if !b_diff_enc {
        let raw_size = 1 + num_valid * T::BYTES;
        if encoded.len() >= raw_size {
            // Raw binary fallback
            let mut buf = vec![0 | integrity];
            for val in values {
                let t = T::from_f64(*val);
                write_typed_value_raw::<T>(&mut buf, t);
            }
            return buf;
        }
    }

    encoded
}

fn write_typed_value<T: LercDataType>(blob: &mut Vec<u8>, val: f64) {
    let t = T::from_f64(val);
    write_typed_value_raw::<T>(blob, t);
}

fn write_typed_value_raw<T: LercDataType>(blob: &mut Vec<u8>, val: T) {
    match T::DATA_TYPE {
        DataType::Char => blob.push(T::from_f64(val.to_f64()).to_f64() as i8 as u8),
        DataType::Byte => blob.push(val.to_f64() as u8),
        DataType::Short => {
            blob.extend_from_slice(&(val.to_f64() as i16).to_le_bytes())
        }
        DataType::UShort => {
            blob.extend_from_slice(&(val.to_f64() as u16).to_le_bytes())
        }
        DataType::Int => blob.extend_from_slice(&(val.to_f64() as i32).to_le_bytes()),
        DataType::UInt => {
            blob.extend_from_slice(&(val.to_f64() as u32).to_le_bytes())
        }
        DataType::Float => {
            blob.extend_from_slice(&(val.to_f64() as f32).to_le_bytes())
        }
        DataType::Double => blob.extend_from_slice(&val.to_f64().to_le_bytes()),
    }
}
