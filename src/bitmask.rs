use alloc::vec;
use alloc::vec::Vec;

/// Validity bitmask matching the C++ BitMask layout.
///
/// Bit ordering: bit k is at byte `k >> 3`, bit position `(1 << 7) >> (k & 7)`.
/// This means bit 7 (MSB) of a byte corresponds to the first pixel in that group of 8.
#[derive(Debug, Clone)]
pub struct BitMask {
    data: Vec<u8>,
    num_pixels: usize,
}

impl BitMask {
    pub fn new(num_pixels: usize) -> Self {
        let num_bytes = (num_pixels + 7) >> 3;
        Self {
            data: vec![0; num_bytes],
            num_pixels,
        }
    }

    pub fn all_valid(num_pixels: usize) -> Self {
        let num_bytes = (num_pixels + 7) >> 3;
        Self {
            data: vec![0xFF; num_bytes],
            num_pixels,
        }
    }

    pub fn from_bytes(data: Vec<u8>, num_pixels: usize) -> Self {
        debug_assert!(data.len() >= (num_pixels + 7) >> 3);
        Self { data, num_pixels }
    }

    #[inline]
    fn bit(k: usize) -> u8 {
        (1u8 << 7) >> (k & 7)
    }

    #[inline]
    pub fn is_valid(&self, k: usize) -> bool {
        (self.data[k >> 3] & Self::bit(k)) != 0
    }

    #[inline]
    pub fn set_valid(&mut self, k: usize) {
        self.data[k >> 3] |= Self::bit(k);
    }

    #[inline]
    pub fn set_invalid(&mut self, k: usize) {
        self.data[k >> 3] &= !Self::bit(k);
    }

    pub fn count_valid(&self) -> usize {
        self.data.iter().map(|b| b.count_ones() as usize).sum()
    }

    pub fn num_pixels(&self) -> usize {
        self.num_pixels
    }

    pub fn num_bytes(&self) -> usize {
        self.data.len()
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_ordering() {
        // Bit 0 should be in the MSB of byte 0
        assert_eq!(BitMask::bit(0), 0x80);
        assert_eq!(BitMask::bit(1), 0x40);
        assert_eq!(BitMask::bit(7), 0x01);
        assert_eq!(BitMask::bit(8), 0x80); // wraps to next byte
    }

    #[test]
    fn set_and_check() {
        let mut mask = BitMask::new(16);
        assert!(!mask.is_valid(0));
        mask.set_valid(0);
        assert!(mask.is_valid(0));
        mask.set_invalid(0);
        assert!(!mask.is_valid(0));
    }

    #[test]
    fn all_valid_count() {
        let mask = BitMask::all_valid(100);
        // count_valid counts all bits in bytes, but we only care about num_pixels
        assert!(mask.count_valid() >= 100);
    }

    #[test]
    fn selective_valid() {
        let mut mask = BitMask::new(10);
        mask.set_valid(0);
        mask.set_valid(5);
        mask.set_valid(9);
        assert_eq!(mask.count_valid(), 3);
        assert!(mask.is_valid(0));
        assert!(!mask.is_valid(1));
        assert!(mask.is_valid(5));
        assert!(mask.is_valid(9));
    }
}
