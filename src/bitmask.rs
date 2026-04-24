use alloc::vec::Vec;

use bitvec::prelude::*;

/// Validity bitmask matching the C++ BitMask layout.
///
/// Two canonical representations:
///
/// - [`BitMask::AllValid`] — every pixel is valid. No heap allocation; all
///   queries ([`is_valid`](Self::is_valid), [`count_valid`](Self::count_valid),
///   [`is_all_valid`](Self::is_all_valid)) are O(1). This is the canonical form
///   the decoder produces when the blob header reports full validity.
/// - [`BitMask::Explicit`] — per-pixel bits stored MSB-first within each byte,
///   matching the C++ convention: bit `k` is at byte `k >> 3`, position
///   `(1 << 7) >> (k & 7)`.
///
/// Mutating an [`AllValid`](Self::AllValid) mask via [`set_invalid`](Self::set_invalid)
/// transitions it to [`Explicit`](Self::Explicit) in place (one allocation).
#[derive(Debug, Clone)]
pub enum BitMask {
    /// Every pixel is valid. The `usize` is the pixel count.
    AllValid(usize),
    /// Explicit per-pixel validity bits.
    Explicit(BitVec<u8, Msb0>),
}

impl BitMask {
    /// Create a new mask with every pixel marked invalid.
    pub fn new(num_pixels: usize) -> Self {
        Self::Explicit(bitvec![u8, Msb0; 0; num_pixels])
    }

    /// Create a new mask with every pixel marked valid — O(1), no allocation.
    pub fn all_valid(num_pixels: usize) -> Self {
        Self::AllValid(num_pixels)
    }

    /// Create a mask from raw MSB-first bytes, truncated to `num_pixels` bits.
    pub fn from_bytes(data: Vec<u8>, num_pixels: usize) -> Self {
        let mut bits = BitVec::<u8, Msb0>::from_vec(data);
        bits.truncate(num_pixels);
        Self::Explicit(bits)
    }

    /// Returns `true` if pixel `k` is valid. O(1).
    #[inline]
    pub fn is_valid(&self, k: usize) -> bool {
        match self {
            Self::AllValid(_) => true,
            Self::Explicit(bits) => bits[k],
        }
    }

    /// Mark pixel `k` as valid.
    ///
    /// No-op for [`AllValid`](Self::AllValid).
    #[inline]
    pub fn set_valid(&mut self, k: usize) {
        if let Self::Explicit(bits) = self {
            bits.set(k, true);
        }
    }

    /// Mark pixel `k` as invalid.
    ///
    /// If the mask is [`AllValid`](Self::AllValid), this materializes it into
    /// [`Explicit`](Self::Explicit) first (one allocation of `(num_pixels + 7) / 8`
    /// bytes), then clears bit `k`.
    #[inline]
    pub fn set_invalid(&mut self, k: usize) {
        if let Self::AllValid(n) = *self {
            *self = Self::Explicit(bitvec![u8, Msb0; 1; n]);
        }
        if let Self::Explicit(bits) = self {
            bits.set(k, false);
        }
    }

    /// Returns the number of valid pixels. O(1) for `AllValid`; popcount for `Explicit`.
    pub fn count_valid(&self) -> usize {
        match self {
            Self::AllValid(n) => *n,
            Self::Explicit(bits) => bits.count_ones(),
        }
    }

    /// Returns `true` if every pixel is valid.
    ///
    /// O(1) for [`AllValid`](Self::AllValid); O(n) popcount fallback for
    /// [`Explicit`](Self::Explicit) (an explicit mask may still have every bit set).
    #[inline]
    pub fn is_all_valid(&self) -> bool {
        match self {
            Self::AllValid(_) => true,
            Self::Explicit(bits) => bits.count_ones() == bits.len(),
        }
    }

    /// Total pixel count. O(1).
    pub fn num_pixels(&self) -> usize {
        match self {
            Self::AllValid(n) => *n,
            Self::Explicit(bits) => bits.len(),
        }
    }

    /// Number of bytes in the raw byte representation (`ceil(num_pixels / 8)`).
    pub fn num_bytes(&self) -> usize {
        match self {
            Self::AllValid(n) => n.div_ceil(8),
            Self::Explicit(bits) => bits.as_raw_slice().len(),
        }
    }

    /// Borrow the underlying byte storage as a slice.
    ///
    /// Returns `None` for [`AllValid`](Self::AllValid) — no bytes are stored.
    /// Callers that need bytes for serialization should first branch on
    /// [`is_all_valid`](Self::is_all_valid), since the LERC blob format omits the
    /// mask section entirely for all-valid bands.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::AllValid(_) => None,
            Self::Explicit(bits) => Some(bits.as_raw_slice()),
        }
    }

    /// Mutably borrow the underlying byte storage.
    ///
    /// Returns `None` for [`AllValid`](Self::AllValid); convert via a mutation
    /// like [`set_invalid`](Self::set_invalid) first if you need mutable bytes.
    pub fn as_bytes_mut(&mut self) -> Option<&mut [u8]> {
        match self {
            Self::AllValid(_) => None,
            Self::Explicit(bits) => Some(bits.as_raw_mut_slice()),
        }
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
    fn all_valid_is_o1() {
        let mask = BitMask::all_valid(1_000_000);
        assert!(matches!(mask, BitMask::AllValid(1_000_000)));
        assert!(mask.is_all_valid());
        assert_eq!(mask.count_valid(), 1_000_000);
        assert_eq!(mask.num_pixels(), 1_000_000);
        for i in [0, 1, 999, 1_000, 999_999] {
            assert!(mask.is_valid(i));
        }
    }

    #[test]
    fn all_valid_non_byte_aligned_count() {
        let mask = BitMask::all_valid(13);
        assert_eq!(mask.count_valid(), 13);
        assert_eq!(mask.num_pixels(), 13);
        assert_eq!(mask.num_bytes(), 2); // ceil(13 / 8)
        for i in 0..13 {
            assert!(mask.is_valid(i));
        }
    }

    #[test]
    fn set_valid_on_all_valid_is_noop() {
        let mut mask = BitMask::all_valid(16);
        mask.set_valid(5);
        assert!(matches!(mask, BitMask::AllValid(16)));
        assert!(mask.is_all_valid());
    }

    #[test]
    fn set_invalid_materializes_all_valid() {
        let mut mask = BitMask::all_valid(16);
        mask.set_invalid(7);
        assert!(matches!(mask, BitMask::Explicit(_)));
        assert!(!mask.is_valid(7));
        for i in 0..16 {
            if i != 7 {
                assert!(mask.is_valid(i), "pixel {i} should still be valid");
            }
        }
        assert_eq!(mask.count_valid(), 15);
        assert!(!mask.is_all_valid());
    }

    #[test]
    fn set_valid_then_is_valid() {
        let mut mask = BitMask::new(16);
        mask.set_valid(5);
        assert!(mask.is_valid(5));
        assert_eq!(mask.count_valid(), 1);
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
        // 0x80 = 0b1000_0000 means pixel 0 valid, pixels 1-7 invalid
        let mask = BitMask::from_bytes(vec![0x80], 8);
        assert!(mask.is_valid(0));
        for i in 1..8 {
            assert!(!mask.is_valid(i));
        }
    }

    #[test]
    fn from_bytes_all_ones_not_autoconverted() {
        // An Explicit mask of all-1s is a valid Explicit BitMask; we don't
        // auto-canonicalize here (decoder canonicalizes via header info instead).
        let mask = BitMask::from_bytes(vec![0xFF; 2], 16);
        assert!(matches!(mask, BitMask::Explicit(_)));
        // But is_all_valid still detects it via popcount fallback.
        assert!(mask.is_all_valid());
    }

    #[test]
    fn from_bytes_multiple_bits() {
        let mask = BitMask::from_bytes(vec![0xC0, 0x01], 16);
        assert!(mask.is_valid(0));
        assert!(mask.is_valid(1));
        for i in 2..15 {
            assert!(!mask.is_valid(i));
        }
        assert!(mask.is_valid(15));
        assert_eq!(mask.count_valid(), 3);
    }

    #[test]
    fn as_bytes_returns_none_for_all_valid() {
        let mask = BitMask::all_valid(16);
        assert!(mask.as_bytes().is_none());
    }

    #[test]
    fn as_bytes_round_trip() {
        let original_data = vec![0xA5, 0x3C];
        let mask = BitMask::from_bytes(original_data.clone(), 16);
        let bytes = mask.as_bytes().unwrap();
        assert_eq!(bytes, &original_data[..]);

        let mask2 = BitMask::from_bytes(bytes.to_vec(), 16);
        for i in 0..16 {
            assert_eq!(mask.is_valid(i), mask2.is_valid(i));
        }
    }

    #[test]
    fn as_bytes_round_trip_non_aligned() {
        let mut mask = BitMask::new(10);
        mask.set_valid(0);
        mask.set_valid(3);
        mask.set_valid(9);

        let bytes = mask.as_bytes().unwrap().to_vec();
        let mask2 = BitMask::from_bytes(bytes, 10);
        for i in 0..10 {
            assert_eq!(mask.is_valid(i), mask2.is_valid(i));
        }
    }

    #[test]
    fn num_pixels_and_num_bytes_consistency() {
        let mask = BitMask::new(16);
        assert_eq!(mask.num_pixels(), 16);
        assert_eq!(mask.num_bytes(), 2);

        let mask = BitMask::new(13);
        assert_eq!(mask.num_pixels(), 13);
        assert_eq!(mask.num_bytes(), 2);

        let mask = BitMask::new(1);
        assert_eq!(mask.num_pixels(), 1);
        assert_eq!(mask.num_bytes(), 1);

        let mask = BitMask::new(8);
        assert_eq!(mask.num_pixels(), 8);
        assert_eq!(mask.num_bytes(), 1);

        let mask = BitMask::new(9);
        assert_eq!(mask.num_pixels(), 9);
        assert_eq!(mask.num_bytes(), 2);

        // AllValid reports the same byte count even though nothing is stored.
        let mask = BitMask::all_valid(13);
        assert_eq!(mask.num_bytes(), 2);
    }

    #[test]
    fn is_all_valid_fast_path() {
        assert!(BitMask::all_valid(100).is_all_valid());
    }

    #[test]
    fn is_all_valid_false_after_materialization() {
        let mut mask = BitMask::all_valid(16);
        assert!(mask.is_all_valid());
        mask.set_invalid(0);
        assert!(!mask.is_all_valid());
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
                let mask = BitMask::from_bytes(vec![0xFF; n.div_ceil(8)], n);
                let bytes = mask.as_bytes().unwrap().to_vec();
                let restored = BitMask::from_bytes(bytes, n);
                for i in 0..n {
                    prop_assert_eq!(mask.is_valid(i), restored.is_valid(i));
                }
            }

            #[test]
            fn prop_all_valid_is_valid_everywhere(n in 1..1000usize) {
                let mask = BitMask::all_valid(n);
                for i in 0..n {
                    prop_assert!(mask.is_valid(i));
                }
            }
        }
    }
}
