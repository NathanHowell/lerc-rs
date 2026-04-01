use bitvec::prelude::*;

/// Validity bitmask matching the C++ BitMask layout.
///
/// Newtype over `BitVec<u8, Msb0>` which stores bits MSB-first within each byte,
/// matching the C++ convention: bit k is at byte `k >> 3`, position `(1 << 7) >> (k & 7)`.
#[derive(Debug, Clone)]
pub struct BitMask(pub BitVec<u8, Msb0>);

impl BitMask {
    pub fn new(num_pixels: usize) -> Self {
        Self(bitvec![u8, Msb0; 0; num_pixels])
    }

    pub fn all_valid(num_pixels: usize) -> Self {
        Self(bitvec![u8, Msb0; 1; num_pixels])
    }

    pub fn from_bytes(data: Vec<u8>, num_pixels: usize) -> Self {
        let mut bits = BitVec::<u8, Msb0>::from_vec(data);
        bits.truncate(num_pixels);
        Self(bits)
    }

    #[inline]
    pub fn is_valid(&self, k: usize) -> bool {
        self.0[k]
    }

    #[inline]
    pub fn set_valid(&mut self, k: usize) {
        self.0.set(k, true);
    }

    #[inline]
    pub fn set_invalid(&mut self, k: usize) {
        self.0.set(k, false);
    }

    pub fn count_valid(&self) -> usize {
        self.0.count_ones()
    }

    pub fn num_pixels(&self) -> usize {
        self.0.len()
    }

    pub fn num_bytes(&self) -> usize {
        self.0.as_raw_slice().len()
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.0.as_raw_slice()
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.0.as_raw_mut_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_all_invalid() {
        let mask = BitMask::new(16);
        assert_eq!(mask.count_valid(), 0);
        for i in 0..16 {
            assert!(!mask.is_valid(i), "pixel {i} should be invalid");
        }
    }

    #[test]
    fn all_valid_creates_full_mask() {
        let mask = BitMask::all_valid(16);
        assert_eq!(mask.count_valid(), 16);
        for i in 0..16 {
            assert!(mask.is_valid(i), "pixel {i} should be valid");
        }
    }

    #[test]
    fn all_valid_non_byte_aligned() {
        // 13 pixels: not a multiple of 8
        let mask = BitMask::all_valid(13);
        assert_eq!(mask.count_valid(), 13);
        assert_eq!(mask.num_pixels(), 13);
        for i in 0..13 {
            assert!(mask.is_valid(i), "pixel {i} should be valid");
        }
    }

    #[test]
    fn set_valid_then_is_valid() {
        let mut mask = BitMask::new(16);
        mask.set_valid(5);
        assert!(mask.is_valid(5));
        assert_eq!(mask.count_valid(), 1);
        // Other pixels still invalid
        assert!(!mask.is_valid(0));
        assert!(!mask.is_valid(4));
        assert!(!mask.is_valid(6));
    }

    #[test]
    fn set_invalid_after_set_valid() {
        let mut mask = BitMask::new(16);
        mask.set_valid(7);
        assert!(mask.is_valid(7));
        mask.set_invalid(7);
        assert!(!mask.is_valid(7));
        assert_eq!(mask.count_valid(), 0);
    }

    #[test]
    fn from_bytes_msb_first_bit_ordering() {
        // C++ bit ordering: within each byte, MSB is pixel 0, next bit is pixel 1, etc.
        // byte 0x80 = 0b1000_0000 means pixel 0 valid, pixels 1-7 invalid
        let data = vec![0x80];
        let mask = BitMask::from_bytes(data, 8);
        assert!(mask.is_valid(0), "0x80 should make pixel 0 valid");
        for i in 1..8 {
            assert!(!mask.is_valid(i), "pixel {i} should be invalid");
        }
    }

    #[test]
    fn from_bytes_second_bit() {
        // 0x40 = 0b0100_0000 means pixel 1 valid
        let data = vec![0x40];
        let mask = BitMask::from_bytes(data, 8);
        assert!(!mask.is_valid(0));
        assert!(mask.is_valid(1), "0x40 should make pixel 1 valid");
        for i in 2..8 {
            assert!(!mask.is_valid(i), "pixel {i} should be invalid");
        }
    }

    #[test]
    fn from_bytes_multiple_bits() {
        // 0xC0 = 0b1100_0000 means pixels 0 and 1 valid
        // 0x01 = 0b0000_0001 means pixel 15 valid (bit 7 of byte 1)
        let data = vec![0xC0, 0x01];
        let mask = BitMask::from_bytes(data, 16);
        assert!(mask.is_valid(0));
        assert!(mask.is_valid(1));
        for i in 2..15 {
            assert!(!mask.is_valid(i), "pixel {i} should be invalid");
        }
        assert!(mask.is_valid(15));
        assert_eq!(mask.count_valid(), 3);
    }

    #[test]
    fn as_bytes_round_trip() {
        let original_data = vec![0xA5, 0x3C]; // arbitrary pattern
        let mask = BitMask::from_bytes(original_data.clone(), 16);
        let bytes = mask.as_bytes();
        assert_eq!(bytes, &original_data[..]);

        // Round-trip through from_bytes again
        let mask2 = BitMask::from_bytes(bytes.to_vec(), 16);
        for i in 0..16 {
            assert_eq!(mask.is_valid(i), mask2.is_valid(i), "mismatch at pixel {i}");
        }
    }

    #[test]
    fn as_bytes_round_trip_non_aligned() {
        // 10 pixels: 2 bytes needed, but only 10 bits used
        let mut mask = BitMask::new(10);
        mask.set_valid(0);
        mask.set_valid(3);
        mask.set_valid(9);

        let bytes = mask.as_bytes().to_vec();
        let mask2 = BitMask::from_bytes(bytes, 10);
        for i in 0..10 {
            assert_eq!(mask.is_valid(i), mask2.is_valid(i), "mismatch at pixel {i}");
        }
    }

    #[test]
    fn num_pixels_and_num_bytes_consistency() {
        // Exact multiple of 8
        let mask = BitMask::new(16);
        assert_eq!(mask.num_pixels(), 16);
        assert_eq!(mask.num_bytes(), 2); // ceil(16/8) = 2

        // Not a multiple of 8
        let mask = BitMask::new(13);
        assert_eq!(mask.num_pixels(), 13);
        assert_eq!(mask.num_bytes(), 2); // ceil(13/8) = 2

        let mask = BitMask::new(1);
        assert_eq!(mask.num_pixels(), 1);
        assert_eq!(mask.num_bytes(), 1);

        let mask = BitMask::new(8);
        assert_eq!(mask.num_pixels(), 8);
        assert_eq!(mask.num_bytes(), 1);

        let mask = BitMask::new(9);
        assert_eq!(mask.num_pixels(), 9);
        assert_eq!(mask.num_bytes(), 2);
    }

    #[test]
    fn num_bytes_matches_as_bytes_len() {
        for n in [1, 7, 8, 9, 15, 16, 17, 100, 256] {
            let mask = BitMask::all_valid(n);
            assert_eq!(
                mask.num_bytes(),
                mask.as_bytes().len(),
                "mismatch for {n} pixels"
            );
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_set_valid_is_valid(n in 1..1000usize, k in 0..999usize) {
                let n = n.max(1);
                let k = k % n;
                let mut mask = BitMask::new(n);
                mask.set_valid(k);
                prop_assert!(mask.is_valid(k));
            }

            #[test]
            fn prop_from_bytes_round_trip(n in 1..200usize) {
                let mask = BitMask::all_valid(n);
                let bytes = mask.as_bytes().to_vec();
                let restored = BitMask::from_bytes(bytes, n);
                for i in 0..n {
                    prop_assert_eq!(mask.is_valid(i), restored.is_valid(i));
                }
            }
        }
    }
}
