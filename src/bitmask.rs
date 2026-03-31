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
