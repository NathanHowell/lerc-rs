/// Compute Fletcher32 checksum matching the LERC C++ implementation.
///
/// The C++ code processes byte pairs as big-endian 16-bit words:
///   word = (byte\[i\] << 8) | byte\[i+1\]
/// with a batch size of 359 words before reduction.
pub fn fletcher32(data: &[u8]) -> u32 {
    let mut sum1: u32 = 0xffff;
    let mut sum2: u32 = 0xffff;

    let mut i = 0;
    let len = data.len();

    while i < len.saturating_sub(1) {
        let batch_end = (i + 359 * 2).min(len.saturating_sub(1));
        while i < batch_end {
            sum1 += (data[i] as u32) << 8;
            sum1 += data[i + 1] as u32;
            sum2 += sum1;
            i += 2;
        }
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }

    if i < len {
        sum1 += (data[i] as u32) << 8;
        sum2 += sum1;
    }

    sum1 = (sum1 & 0xffff) + (sum1 >> 16);
    sum2 = (sum2 & 0xffff) + (sum2 >> 16);

    (sum2 << 16) | sum1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        // Empty input should return initial state reduced
        let result = fletcher32(&[]);
        assert_eq!(result, 0xffff_ffff);
    }

    #[test]
    fn single_byte() {
        let result = fletcher32(&[0x01]);
        // sum1 = 0xffff + (0x01 << 8) = 0x100ff, reduced = 0x0100
        // sum2 = 0xffff + 0x100ff = 0x200fe, reduced = 0x0100
        // Wait, let me trace more carefully:
        // sum1 = 0xffff, sum2 = 0xffff
        // odd byte: sum1 += (0x01 << 8) = 0xffff + 0x100 = 0x100ff
        //           sum2 += sum1 => sum2 = 0xffff + 0x100ff = 0x200fe
        // reduce sum1: (0x100ff & 0xffff) + (0x100ff >> 16) = 0x00ff + 1 = 0x0100
        // reduce sum2: (0x200fe & 0xffff) + (0x200fe >> 16) = 0x00fe + 2 = 0x0100
        assert_eq!(result, 0x0100_0100);
    }

    #[test]
    fn two_bytes() {
        let result = fletcher32(&[0xAB, 0xCD]);
        // sum1 = 0xFFFF + 0xAB00 + 0xCD = 0x1ABCC -> reduced 0xABCD
        // sum2 = 0xFFFF + 0x1ABCC = 0x2ABCB -> reduced 0xABCD
        assert_eq!(result, 0xABCD_ABCD);
    }
}
