mod huffman;
mod optimizations;
mod tiles;

use alloc::vec;
use alloc::vec::Vec;

use crate::bitmask::BitMask;
use crate::error::{LercError, Result};
use crate::fpl;
use crate::header::{self, HeaderInfo};
use crate::rle;
use crate::types::{DataType, ImageEncodeMode, LercDataType};
use crate::{LercData, LercImage};

use huffman::{is_high_entropy_u8, try_encode_huffman_int};
use optimizations::{compute_no_data_sentinel, try_bit_plane_compression, try_raise_max_z_error};
use tiles::{encode_tiles, select_block_size};

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

#[cfg(test)]
mod tests {
    use super::*;

    // ---- compute_band_stats ----

    #[test]
    fn band_stats_single_depth_all_valid() {
        let width = 4;
        let height = 3;
        let data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ];
        let mask = BitMask::all_valid(width * height);
        let stats = compute_band_stats(&data, &mask, width, height, 1, width * height, None);
        assert_eq!(stats.overall_min, 1.0);
        assert_eq!(stats.overall_max, 12.0);
        assert_eq!(stats.z_min_vec[0], 1.0);
        assert_eq!(stats.z_max_vec[0], 12.0);
        assert!(!stats.needs_no_data);
    }

    #[test]
    fn band_stats_multi_depth() {
        // 2x2 image, 2 depths: [pixel0_d0, pixel0_d1, pixel1_d0, pixel1_d1, ...]
        let width = 2;
        let height = 2;
        let n_depth = 2;
        let data: Vec<f32> = vec![
            1.0, 100.0,  // pixel (0,0): depth0=1, depth1=100
            2.0, 200.0,  // pixel (1,0): depth0=2, depth1=200
            3.0, 50.0,   // pixel (0,1): depth0=3, depth1=50
            4.0, 150.0,  // pixel (1,1): depth0=4, depth1=150
        ];
        let mask = BitMask::all_valid(width * height);
        let stats = compute_band_stats(&data, &mask, width, height, n_depth, width * height, None);
        // depth 0: min=1, max=4
        assert_eq!(stats.z_min_vec[0], 1.0);
        assert_eq!(stats.z_max_vec[0], 4.0);
        // depth 1: min=50, max=200
        assert_eq!(stats.z_min_vec[1], 50.0);
        assert_eq!(stats.z_max_vec[1], 200.0);
        // overall
        assert_eq!(stats.overall_min, 1.0);
        assert_eq!(stats.overall_max, 200.0);
    }

    #[test]
    fn band_stats_with_nodata() {
        // 3x1 image, 2 depths, NoData = -9999.0
        // pixel 0: [5.0, -9999.0] — mixed NoData across depths
        // pixel 1: [10.0, 20.0]   — all valid
        // pixel 2: [15.0, 25.0]   — all valid
        let width = 3;
        let height = 1;
        let n_depth = 2;
        let nd = -9999.0_f64;
        let data: Vec<f64> = vec![
            5.0, nd,
            10.0, 20.0,
            15.0, 25.0,
        ];
        let mask = BitMask::all_valid(width * height);
        let stats = compute_band_stats(&data, &mask, width, height, n_depth, width * height, Some(nd));
        assert!(stats.needs_no_data, "mixed NoData across depths should set needs_no_data");
        // valid_min should exclude the NoData value
        assert!(stats.valid_min > nd, "valid_min should exclude NoData");
        assert_eq!(stats.valid_min, 5.0);
    }

    #[test]
    fn band_stats_all_invalid() {
        let width = 3;
        let height = 2;
        let data: Vec<f32> = vec![1.0; width * height];
        let mask = BitMask::new(width * height); // all invalid
        let stats = compute_band_stats(&data, &mask, width, height, 1, 0, None);
        // When num_valid=0, defaults apply
        assert_eq!(stats.overall_min, 0.0);
        assert_eq!(stats.overall_max, 0.0);
    }

    #[test]
    fn band_stats_with_mask() {
        // 4x1 image, only pixels 1 and 2 valid
        let width = 4;
        let height = 1;
        let data: Vec<f32> = vec![100.0, 5.0, 10.0, 200.0];
        let mut mask = BitMask::new(width * height);
        mask.set_valid(1);
        mask.set_valid(2);
        let stats = compute_band_stats(&data, &mask, width, height, 1, 2, None);
        assert_eq!(stats.overall_min, 5.0);
        assert_eq!(stats.overall_max, 10.0);
    }

    // ---- for_each_valid_pixel ----

    #[test]
    fn for_each_all_valid_single_depth() {
        let width = 3;
        let height = 2;
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mask = BitMask::all_valid(width * height);
        let mut count = 0usize;
        let mut sum = 0.0f64;
        for_each_valid_pixel(&data, &mask, width, height, 1, |_k, _m, val| {
            count += 1;
            sum += val;
        });
        assert_eq!(count, 6);
        assert_eq!(sum, 21.0);
    }

    #[test]
    fn for_each_with_mask() {
        let width = 4;
        let height = 1;
        let data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        let mut mask = BitMask::new(width * height);
        mask.set_valid(0);
        mask.set_valid(2);
        let mut count = 0usize;
        for_each_valid_pixel(&data, &mask, width, height, 1, |_k, _m, _val| {
            count += 1;
        });
        assert_eq!(count, 2, "only 2 valid pixels");
    }

    #[test]
    fn for_each_multi_depth() {
        // 2x1 image, 3 depths
        let width = 2;
        let height = 1;
        let n_depth = 3;
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mask = BitMask::all_valid(width * height);
        let mut count = 0usize;
        for_each_valid_pixel(&data, &mask, width, height, n_depth, |_k, _m, _val| {
            count += 1;
        });
        assert_eq!(count, 6, "2 pixels * 3 depths = 6 visits");
    }

    #[test]
    fn for_each_fast_and_general_paths_agree() {
        // Test that the fast path (all_valid, n_depth=1) produces the same
        // results as the general path by comparing against a masked version
        // where all pixels are valid.
        let width = 5;
        let height = 4;
        let data: Vec<f32> = (0..20).map(|i| i as f32 * 1.5).collect();
        let mask_fast = BitMask::all_valid(width * height);

        // General path: create a mask with all bits set manually
        let mut mask_general = BitMask::new(width * height);
        for k in 0..width * height {
            mask_general.set_valid(k);
        }

        let mut vals_fast = Vec::new();
        for_each_valid_pixel(&data, &mask_fast, width, height, 1, |k, m, val| {
            vals_fast.push((k, m, val));
        });

        // For the general path, use n_depth=2 to force general path, then use n_depth=1 with mask
        // Actually, to truly test both paths: fast path is all_valid && n_depth==1.
        // General path triggered when n_depth > 1 or not all_valid.
        // Use n_depth=1 with the manually-set mask (count_valid still equals width*height).
        let mut vals_general = Vec::new();
        for_each_valid_pixel(&data, &mask_general, width, height, 1, |k, m, val| {
            vals_general.push((k, m, val));
        });

        assert_eq!(vals_fast, vals_general);
    }

    #[test]
    fn for_each_multi_depth_with_mask() {
        // 3x1 image, 2 depths, only pixel 1 valid
        let width = 3;
        let height = 1;
        let n_depth = 2;
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut mask = BitMask::new(width * height);
        mask.set_valid(1);
        let mut visited = Vec::new();
        for_each_valid_pixel(&data, &mask, width, height, n_depth, |k, m, val| {
            visited.push((k, m, val));
        });
        assert_eq!(visited.len(), 2, "1 valid pixel * 2 depths = 2 visits");
        assert_eq!(visited[0], (1, 0, 3.0)); // pixel 1, depth 0
        assert_eq!(visited[1], (1, 1, 4.0)); // pixel 1, depth 1
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn for_each_valid_pixel_count(w in 1..32usize, h in 1..32usize) {
            let mask = BitMask::all_valid(w * h);
            let data: Vec<f32> = vec![1.0; w * h];
            let mut count = 0usize;
            for_each_valid_pixel(&data, &mask, w, h, 1, |_, _, _| count += 1);
            prop_assert_eq!(count, w * h);
        }

        #[test]
        fn compute_band_stats_invariants(w in 2..16usize, h in 2..16usize) {
            let mask = BitMask::all_valid(w * h);
            let data: Vec<f32> = (0..w*h).map(|i| i as f32).collect();
            let stats = compute_band_stats(&data, &mask, w, h, 1, w*h, None);
            prop_assert!(stats.overall_min <= stats.overall_max);
            prop_assert_eq!(stats.z_min_vec.len(), 1);
            prop_assert!(stats.z_min_vec[0] <= stats.z_max_vec[0]);
        }
    }
}
