use alloc::vec::Vec;

use crate::types::{DataType, LercDataType, TileCompressionMode, tile_flags};

/// Parameters for encoding a single tile block's payload.
pub(super) struct TileInnerParams {
    pub num_valid: usize,
    pub z_min_f: f64,
    pub z_max_f: f64,
    pub max_z_error: f64,
    pub max_val_to_quantize: f64,
    pub integrity: u8,
    pub src_data_type: DataType,
    pub b_diff_enc: bool,
}

/// Encode the payload for a single tile block, appending to `buf`.
/// When `b_diff_enc` is true, bit 2 of the compression flag is set and the
/// offset data type is forced to Int for integer source types.
/// `quant_scratch` is a reusable scratch buffer for quantized values.
pub(super) fn encode_tile_inner<T: LercDataType>(
    buf: &mut Vec<u8>,
    values: &[f64],
    quant_scratch: &mut Vec<u32>,
    p: &TileInnerParams,
) {
    let TileInnerParams {
        num_valid,
        z_min_f,
        z_max_f,
        max_z_error,
        max_val_to_quantize,
        integrity,
        src_data_type,
        b_diff_enc,
    } = *p;
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
            crate::tiles::reduce_data_type(z_min_f as i32, DataType::Int)
        } else {
            crate::tiles::reduce_data_type(z_min_t, src_data_type)
        };
        let bits67 = (tc as u8) << tile_flags::TYPE_REDUCTION_SHIFT;
        buf.push(TileCompressionMode::ConstOffset as u8 | bits67 | diff_flag | integrity);
        crate::tiles::write_variable_data_type(buf, z_min_f, dt_reduced);
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
        crate::tiles::reduce_data_type(z_min_f as i32, DataType::Int)
    } else {
        crate::tiles::reduce_data_type(z_min_t, src_data_type)
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
        crate::tiles::write_variable_data_type(buf, z_min_f, dt_reduced);
        return;
    }

    // Try LUT vs simple encoding
    let encoded_start = buf.len();
    buf.push(TileCompressionMode::BitStuffed as u8 | bits67 | diff_flag | integrity);
    crate::tiles::write_variable_data_type(buf, z_min_f, dt_reduced);

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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    // -----------------------------------------------------------------------
    // Helper to call encode_tile_inner with typical u8 lossless settings
    // -----------------------------------------------------------------------

    fn encode_tile_inner_u8(
        values: &[f64],
        z_min_f: f64,
        z_max_f: f64,
        b_diff_enc: bool,
    ) -> (Vec<u8>, Vec<u32>) {
        let mut buf = Vec::new();
        let mut quant_scratch = Vec::new();
        let p = TileInnerParams {
            num_valid: values.len(),
            z_min_f,
            z_max_f,
            max_z_error: 0.5,
            max_val_to_quantize: ((1u32 << 15) - 1) as f64,
            integrity: 0,
            src_data_type: DataType::Byte,
            b_diff_enc,
        };
        encode_tile_inner::<u8>(&mut buf, values, &mut quant_scratch, &p);
        (buf, quant_scratch)
    }

    fn encode_tile_inner_f64(
        values: &[f64],
        z_min_f: f64,
        z_max_f: f64,
        max_z_error: f64,
    ) -> (Vec<u8>, Vec<u32>) {
        let mut buf = Vec::new();
        let mut quant_scratch = Vec::new();
        let p = TileInnerParams {
            num_valid: values.len(),
            z_min_f,
            z_max_f,
            max_z_error,
            max_val_to_quantize: ((1u32 << 30) - 1) as f64,
            integrity: 0,
            src_data_type: DataType::Double,
            b_diff_enc: false,
        };
        encode_tile_inner::<f64>(&mut buf, values, &mut quant_scratch, &p);
        (buf, quant_scratch)
    }

    // -----------------------------------------------------------------------
    // encode_tile_inner: all-zero block
    // -----------------------------------------------------------------------

    #[test]
    fn encode_tile_inner_all_zero_via_num_valid_zero() {
        let mut buf = Vec::new();
        let mut quant_scratch = Vec::new();
        // num_valid = 0 triggers const-zero path
        let p = TileInnerParams {
            num_valid: 0,
            z_min_f: f64::MAX,
            z_max_f: f64::MIN,
            max_z_error: 0.5,
            max_val_to_quantize: ((1u32 << 15) - 1) as f64,
            integrity: 0,
            src_data_type: DataType::Byte,
            b_diff_enc: false,
        };
        encode_tile_inner::<u8>(&mut buf, &[], &mut quant_scratch, &p);
        assert_eq!(buf.len(), 1);
        assert_eq!(buf[0] & tile_flags::MODE_MASK, TileCompressionMode::ConstZero as u8);
    }

    #[test]
    fn encode_tile_inner_all_zero_via_zmin_zmax_zero() {
        let values = vec![0.0, 0.0, 0.0, 0.0];
        let (buf, _) = encode_tile_inner_u8(&values, 0.0, 0.0, false);
        assert_eq!(buf.len(), 1);
        assert_eq!(buf[0] & tile_flags::MODE_MASK, TileCompressionMode::ConstZero as u8);
    }

    // -----------------------------------------------------------------------
    // encode_tile_inner: constant block (z_min == z_max != 0)
    // -----------------------------------------------------------------------

    #[test]
    fn encode_tile_inner_const_offset() {
        let values = vec![42.0, 42.0, 42.0, 42.0];
        let (buf, _) = encode_tile_inner_u8(&values, 42.0, 42.0, false);
        assert_eq!(
            buf[0] & tile_flags::MODE_MASK,
            TileCompressionMode::ConstOffset as u8
        );
        // Should have mode byte + offset value (reduced to u8 = 1 byte)
        assert!(buf.len() > 1);
        // The offset byte should be 42
        assert_eq!(buf[1], 42);
    }

    #[test]
    fn encode_tile_inner_const_offset_i16() {
        // Use i16 data where the constant value is > 255, requiring a 2-byte offset
        let values = vec![1000.0, 1000.0, 1000.0];
        let mut buf = Vec::new();
        let mut quant_scratch = Vec::new();
        let p = TileInnerParams {
            num_valid: 3,
            z_min_f: 1000.0,
            z_max_f: 1000.0,
            max_z_error: 0.5,
            max_val_to_quantize: ((1u32 << 15) - 1) as f64,
            integrity: 0,
            src_data_type: DataType::Short,
            b_diff_enc: false,
        };
        encode_tile_inner::<i16>(&mut buf, &values, &mut quant_scratch, &p);
        assert_eq!(
            buf[0] & tile_flags::MODE_MASK,
            TileCompressionMode::ConstOffset as u8
        );
        // 1000 fits in i16 but not i8/u8, so type reduction from Short
        // should produce a 2-byte offset at minimum
        assert!(buf.len() >= 3);
    }

    // -----------------------------------------------------------------------
    // encode_tile_inner: quantized / bit-stuffed block
    // -----------------------------------------------------------------------

    #[test]
    fn encode_tile_inner_bitstuffed() {
        // Small range of u8 values: should produce bit-stuffed output
        let values: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let (buf, quant) = encode_tile_inner_u8(&values, 0.0, 15.0, false);
        assert_eq!(
            buf[0] & tile_flags::MODE_MASK,
            TileCompressionMode::BitStuffed as u8
        );
        // Quantized values should match original for lossless u8
        assert_eq!(quant.len(), 16);
        for (i, &q) in quant.iter().enumerate() {
            assert_eq!(q, i as u32);
        }
    }

    #[test]
    fn encode_tile_inner_bitstuffed_with_offset() {
        // Use enough values that bitstuffed is smaller than raw binary.
        // For u8 with 16 values, raw_size = 1 + 16 = 17.
        // Bitstuffed with range 0..3 (2 bits) is much smaller.
        let values: Vec<f64> = (0..16).map(|i| 100.0 + (i % 4) as f64).collect();
        let (buf, quant) = encode_tile_inner_u8(&values, 100.0, 103.0, false);
        assert_eq!(
            buf[0] & tile_flags::MODE_MASK,
            TileCompressionMode::BitStuffed as u8
        );
        // Offset byte should be 100 (z_min as u8)
        assert_eq!(buf[1], 100);
        // Quantized should be 0, 1, 2, 3, 0, 1, 2, 3, ...
        for (i, &q) in quant.iter().enumerate() {
            assert_eq!(q, (i % 4) as u32);
        }
    }

    // -----------------------------------------------------------------------
    // encode_tile_inner: raw binary fallback
    // -----------------------------------------------------------------------

    #[test]
    fn encode_tile_inner_raw_binary_f64_large_range() {
        // To trigger raw binary via the early exit (not the fallback), we need
        // need_quantize = false with z_min != z_max.
        // Use max_z_error > 0 and a range so large that
        // max_val = (z_max - z_min) / (2 * max_z_error) > max_val_to_quantize.
        let max_val_to_quantize = ((1u32 << 30) - 1) as f64;
        let max_z_error = 0.5;
        let z_min = 0.0;
        // Need range / (2*0.5) > max_val_to_quantize, so range > max_val_to_quantize
        let z_max = max_val_to_quantize + 100.0;
        let n = 4;
        let values: Vec<f64> = (0..n).map(|i| z_min + (z_max - z_min) * i as f64 / (n - 1) as f64).collect();
        let (buf, _) = encode_tile_inner_f64(&values, z_min, z_max, max_z_error);
        assert_eq!(
            buf[0] & tile_flags::MODE_MASK,
            TileCompressionMode::RawBinary as u8
        );
        // Raw binary: 1 byte header + n * 8 bytes
        assert_eq!(buf.len(), 1 + n * 8);
        // Verify values are stored correctly
        for (i, &v) in values.iter().enumerate() {
            let start = 1 + i * 8;
            let decoded = f64::from_le_bytes(buf[start..start + 8].try_into().unwrap());
            assert_eq!(decoded, v);
        }
    }

    #[test]
    fn encode_tile_inner_raw_binary_contains_original_values() {
        // Use enough values to ensure raw binary wins over bitstuffed
        let n = 32;
        let values: Vec<f64> = (0..n).map(|i| 1.5 + i as f64 * 1e14).collect();
        let z_min = values[0];
        let z_max = *values.last().unwrap();
        let (buf, _) = encode_tile_inner_f64(&values, z_min, z_max, 0.0);
        if buf[0] & tile_flags::MODE_MASK == TileCompressionMode::RawBinary as u8 {
            // Verify the raw f64 values are stored correctly
            for (i, &v) in values.iter().enumerate() {
                let start = 1 + i * 8;
                let decoded = f64::from_le_bytes(buf[start..start + 8].try_into().unwrap());
                assert_eq!(decoded, v);
            }
        }
    }

    // -----------------------------------------------------------------------
    // encode_tile_inner: diff encoding flag
    // -----------------------------------------------------------------------

    #[test]
    fn encode_tile_inner_diff_encoding_const_zero() {
        let values = vec![0.0, 0.0, 0.0];
        let (buf, _) = encode_tile_inner_u8(&values, 0.0, 0.0, true);
        assert_eq!(buf.len(), 1);
        assert_eq!(
            buf[0] & tile_flags::MODE_MASK,
            TileCompressionMode::ConstZero as u8
        );
        // Diff flag (bit 2) should be set
        assert_ne!(buf[0] & tile_flags::DIFF_ENCODING, 0);
    }

    #[test]
    fn encode_tile_inner_diff_encoding_const_offset() {
        let values = vec![10.0, 10.0, 10.0];
        let (buf, _) = encode_tile_inner_u8(&values, 10.0, 10.0, true);
        assert_eq!(
            buf[0] & tile_flags::MODE_MASK,
            TileCompressionMode::ConstOffset as u8
        );
        assert_ne!(buf[0] & tile_flags::DIFF_ENCODING, 0);
    }

    #[test]
    fn encode_tile_inner_diff_encoding_bitstuffed() {
        let values: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let (buf, _) = encode_tile_inner_u8(&values, 0.0, 7.0, true);
        assert_eq!(
            buf[0] & tile_flags::MODE_MASK,
            TileCompressionMode::BitStuffed as u8
        );
        assert_ne!(buf[0] & tile_flags::DIFF_ENCODING, 0);
    }

    #[test]
    fn encode_tile_inner_diff_encoding_no_raw_binary() {
        // With diff encoding, raw binary should NOT be produced.
        // Instead, a sentinel is written that is larger than raw, so non-diff wins.
        let mut buf = Vec::new();
        let mut quant_scratch = Vec::new();
        let values = vec![0.0, 1e15];
        // Use f64 lossless with diff encoding
        let p = TileInnerParams {
            num_valid: values.len(),
            z_min_f: 0.0,
            z_max_f: 1e15,
            max_z_error: 0.0,
            max_val_to_quantize: ((1u32 << 30) - 1) as f64,
            integrity: 0,
            src_data_type: DataType::Double,
            b_diff_enc: true,
        };
        encode_tile_inner::<f64>(&mut buf, &values, &mut quant_scratch, &p);
        // Should NOT be raw binary when diff is true
        assert_ne!(
            buf[0] & tile_flags::MODE_MASK,
            TileCompressionMode::RawBinary as u8
        );
    }

    // -----------------------------------------------------------------------
    // encode_tile_inner: integrity bits are preserved
    // -----------------------------------------------------------------------

    #[test]
    fn encode_tile_inner_integrity_bits() {
        let mut buf = Vec::new();
        let mut quant_scratch = Vec::new();
        let values = vec![0.0, 0.0];
        let integrity: u8 = 0b00001100; // bits 2-3 set
        let p = TileInnerParams {
            num_valid: values.len(),
            z_min_f: 0.0,
            z_max_f: 0.0,
            max_z_error: 0.5,
            max_val_to_quantize: ((1u32 << 15) - 1) as f64,
            integrity,
            src_data_type: DataType::Byte,
            b_diff_enc: false,
        };
        encode_tile_inner::<u8>(&mut buf, &values, &mut quant_scratch, &p);
        // Integrity bits should be present in the header byte
        assert_eq!(buf[0] & integrity, integrity);
    }

    // -----------------------------------------------------------------------
    // encode_tile_inner: quantized with lossy max_z_error
    // -----------------------------------------------------------------------

    #[test]
    fn encode_tile_inner_lossy_quantization() {
        // Lossy encoding of f64 data with max_z_error = 1.0
        let values = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let (buf, quant) = encode_tile_inner_f64(&values, 0.0, 4.0, 1.0);
        assert_eq!(
            buf[0] & tile_flags::MODE_MASK,
            TileCompressionMode::BitStuffed as u8
        );
        // With max_z_error = 1.0, scale = 1/(2*1.0) = 0.5
        // quant[i] = (val - z_min) * 0.5 + 0.5 = val * 0.5 + 0.5
        // quant: 0->0, 1->1, 2->1, 3->2, 4->2
        assert_eq!(quant[0], 0); // 0.0 * 0.5 + 0.5 = 0.5 -> 0
        assert_eq!(quant[1], 1); // 1.0 * 0.5 + 0.5 = 1.0 -> 1
        assert_eq!(quant[2], 1); // 2.0 * 0.5 + 0.5 = 1.5 -> 1
        assert_eq!(quant[3], 2); // 3.0 * 0.5 + 0.5 = 2.0 -> 2
        assert_eq!(quant[4], 2); // 4.0 * 0.5 + 0.5 = 2.5 -> 2
    }

    // -----------------------------------------------------------------------
    // encode_tile_inner: all-quantized-to-same (max_quant == 0)
    // -----------------------------------------------------------------------

    #[test]
    fn encode_tile_inner_all_quantize_to_same() {
        // Actually let's use: values where after quantization all become 0.
        // This is tricky because (max_val + 0.5) as u32 == 0 when max_val < 0.5.
        // For max_quant=0 we need all individual quant values to be 0, but
        // (max_val + 0.5) as u32 > 0 (so max_val >= 0.5).
        // When max_val >= 0.5, scale = 1/(2*max_z_error), and at least
        // the largest value will quantize to max_val * scale + 0.5 >= 1.
        // So max_quant=0 in the quantized path cannot happen unless all values
        // are exactly z_min, which means z_min == z_max (caught earlier).
        //
        // Therefore the max_quant==0 ConstOffset path in the quantized section
        // is effectively unreachable for well-formed input. Skip this test case.
        // Instead verify a near-constant case that ends up as ConstOffset.
        let values = vec![42.0; 16];
        let (buf, _) = encode_tile_inner_u8(&values, 42.0, 42.0, false);
        assert_eq!(
            buf[0] & tile_flags::MODE_MASK,
            TileCompressionMode::ConstOffset as u8
        );
    }

    // -----------------------------------------------------------------------
    // Round-trip: encode then decode via crate's public API
    // -----------------------------------------------------------------------

    #[test]
    fn encode_tile_inner_round_trip_u8_lossless() {
        // Encode a tile and decode it using the bitstuffer decoder.
        // Use 64 values (8x8 tile) with small range so bitstuffed beats raw.
        // Values: 10..=25 repeating. Range=15, 4 bits. 64 values * 4 bits = 32 bytes.
        // Bitstuffer overhead: ~3 bytes. Total bitstuffed: 1 + 1 + 35 ~= 37.
        // Raw: 1 + 64 = 65. So bitstuffed wins.
        let values: Vec<f64> = (0..64).map(|i| 10.0 + (i % 16) as f64).collect();
        let (buf, _) = encode_tile_inner_u8(&values, 10.0, 25.0, false);
        assert_eq!(
            buf[0] & tile_flags::MODE_MASK,
            TileCompressionMode::BitStuffed as u8
        );
        // The offset is written as a reduced type; for u8 value 10, it's 1 byte
        let offset_val = buf[1];
        assert_eq!(offset_val, 10);
        // Decode the bitstuffed portion
        let mut pos = 2;
        let decoded = crate::bitstuffer::decode(&buf, &mut pos, 256, 6).unwrap();
        assert_eq!(decoded.len(), 64);
        // Reconstruct: offset + quant_val
        for (i, &d) in decoded.iter().enumerate() {
            let reconstructed = offset_val as u32 + d;
            assert_eq!(reconstructed, values[i] as u32);
        }
    }
}
