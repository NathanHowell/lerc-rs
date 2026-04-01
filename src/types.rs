#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum DataType {
    #[default]
    Char = 0,
    Byte = 1,
    Short = 2,
    UShort = 3,
    Int = 4,
    UInt = 5,
    Float = 6,
    Double = 7,
}

impl DataType {
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::Char),
            1 => Some(Self::Byte),
            2 => Some(Self::Short),
            3 => Some(Self::UShort),
            4 => Some(Self::Int),
            5 => Some(Self::UInt),
            6 => Some(Self::Float),
            7 => Some(Self::Double),
            _ => None,
        }
    }

    pub fn size(self) -> usize {
        match self {
            Self::Char | Self::Byte => 1,
            Self::Short | Self::UShort => 2,
            Self::Int | Self::UInt | Self::Float => 4,
            Self::Double => 8,
        }
    }

    pub fn is_integer(self) -> bool {
        !matches!(self, Self::Float | Self::Double)
    }

    pub fn is_signed(self) -> bool {
        matches!(self, Self::Char | Self::Short | Self::Int)
    }
}

pub(crate) mod sealed {
    pub trait Sealed {}
}

pub trait LercDataType: sealed::Sealed + Copy + PartialOrd + Default + core::fmt::Debug {
    const DATA_TYPE: DataType;
    const BYTES: usize;

    fn to_f64(self) -> f64;
    fn from_f64(v: f64) -> Self;
    fn to_bits_u64(self) -> u64;
    fn from_bits_u64(v: u64) -> Self;
    fn is_integer() -> bool;

    /// Wrap a `Vec<Self>` into the corresponding `LercData` variant.
    fn into_lerc_data(v: alloc::vec::Vec<Self>) -> super::LercData;

    /// Try to borrow the pixel slice from a `LercData` if the variant matches.
    fn try_ref_lerc_data(data: &super::LercData) -> Option<&[Self]>;

    /// Try to unwrap the pixel vector from a `LercData` if the variant matches.
    fn try_from_lerc_data(data: super::LercData) -> core::result::Result<alloc::vec::Vec<Self>, super::LercData>;

    /// Read a value from a little-endian byte slice. The slice must be at least `BYTES` long.
    fn from_le_slice(s: &[u8]) -> Self;
}

macro_rules! impl_lerc_data_type {
    ($ty:ty, $dt:expr, $is_int:expr, $variant:ident, $n:literal) => {
        impl sealed::Sealed for $ty {}
        impl LercDataType for $ty {
            const DATA_TYPE: DataType = $dt;
            const BYTES: usize = $n;

            #[inline]
            fn to_f64(self) -> f64 {
                self as f64
            }

            #[inline]
            fn from_f64(v: f64) -> Self {
                v as Self
            }

            #[inline]
            fn to_bits_u64(self) -> u64 {
                self as u64
            }

            #[inline]
            fn from_bits_u64(v: u64) -> Self {
                v as Self
            }

            #[inline]
            fn is_integer() -> bool {
                $is_int
            }

            fn into_lerc_data(v: alloc::vec::Vec<Self>) -> super::LercData {
                super::LercData::$variant(v)
            }

            fn try_ref_lerc_data(data: &super::LercData) -> Option<&[Self]> {
                match data {
                    super::LercData::$variant(v) => Some(v),
                    _ => None,
                }
            }

            fn try_from_lerc_data(data: super::LercData) -> core::result::Result<alloc::vec::Vec<Self>, super::LercData> {
                match data {
                    super::LercData::$variant(v) => Ok(v),
                    other => Err(other),
                }
            }

            #[inline]
            fn from_le_slice(s: &[u8]) -> Self {
                let mut buf = [0u8; $n];
                buf.copy_from_slice(&s[..$n]);
                <$ty>::from_le_bytes(buf)
            }
        }
    };
}

impl_lerc_data_type!(i8, DataType::Char, true, I8, 1);
impl_lerc_data_type!(u8, DataType::Byte, true, U8, 1);
impl_lerc_data_type!(i16, DataType::Short, true, I16, 2);
impl_lerc_data_type!(u16, DataType::UShort, true, U16, 2);
impl_lerc_data_type!(i32, DataType::Int, true, I32, 4);
impl_lerc_data_type!(u32, DataType::UInt, true, U32, 4);


impl sealed::Sealed for f32 {}
impl LercDataType for f32 {
    const DATA_TYPE: DataType = DataType::Float;
    const BYTES: usize = 4;

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as Self
    }

    #[inline]
    fn to_bits_u64(self) -> u64 {
        self.to_bits() as u64
    }

    #[inline]
    fn from_bits_u64(v: u64) -> Self {
        Self::from_bits(v as u32)
    }

    #[inline]
    fn is_integer() -> bool {
        false
    }

    fn into_lerc_data(v: alloc::vec::Vec<Self>) -> super::LercData {
        super::LercData::F32(v)
    }

    fn try_ref_lerc_data(data: &super::LercData) -> Option<&[Self]> {
        match data {
            super::LercData::F32(v) => Some(v),
            _ => None,
        }
    }

    fn try_from_lerc_data(data: super::LercData) -> core::result::Result<alloc::vec::Vec<Self>, super::LercData> {
        match data {
            super::LercData::F32(v) => Ok(v),
            other => Err(other),
        }
    }

    #[inline]
    fn from_le_slice(s: &[u8]) -> Self {
        Self::from_bits(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
}

impl sealed::Sealed for f64 {}
impl LercDataType for f64 {
    const DATA_TYPE: DataType = DataType::Double;
    const BYTES: usize = 8;

    #[inline]
    fn to_f64(self) -> f64 {
        self
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }

    #[inline]
    fn to_bits_u64(self) -> u64 {
        self.to_bits()
    }

    #[inline]
    fn from_bits_u64(v: u64) -> Self {
        Self::from_bits(v)
    }

    #[inline]
    fn is_integer() -> bool {
        false
    }

    fn into_lerc_data(v: alloc::vec::Vec<Self>) -> super::LercData {
        super::LercData::F64(v)
    }

    fn try_ref_lerc_data(data: &super::LercData) -> Option<&[Self]> {
        match data {
            super::LercData::F64(v) => Some(v),
            _ => None,
        }
    }

    fn try_from_lerc_data(data: super::LercData) -> core::result::Result<alloc::vec::Vec<Self>, super::LercData> {
        match data {
            super::LercData::F64(v) => Ok(v),
            other => Err(other),
        }
    }

    #[inline]
    fn from_le_slice(s: &[u8]) -> Self {
        Self::from_bits(u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]))
    }
}

/// Image-level encoding mode (C++ `Lerc2::ImageEncodeMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum ImageEncodeMode {
    Tiling = 0,
    DeltaHuffman = 1,
    Huffman = 2,
    DeltaDeltaHuffman = 3,
}

impl TryFrom<u8> for ImageEncodeMode {
    type Error = crate::error::LercError;
    fn try_from(v: u8) -> core::result::Result<Self, Self::Error> {
        match v {
            0 => Ok(Self::Tiling),
            1 => Ok(Self::DeltaHuffman),
            2 => Ok(Self::Huffman),
            3 => Ok(Self::DeltaDeltaHuffman),
            _ => Err(crate::error::LercError::UnsupportedEncoding(v)),
        }
    }
}

/// Tile compression mode — the 2-bit value stored in bits 0-1 of the tile header byte.
///
/// C++ uses `BlockEncodeMode` for values 0-2 and separate logic for 2-3.
/// We unify all four wire values into one enum since they're mutually exclusive.
/// Note: "BitStuffLut" is not a separate wire value — LUT vs simple is determined
/// from the bit-stuffed payload, and both use wire value 1 (BitStuffed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum TileCompressionMode {
    RawBinary = 0,
    BitStuffed = 1,
    ConstZero = 2,
    ConstOffset = 3,
}

/// Compression flag bits within a tile header byte.
pub(crate) mod tile_flags {
    /// Bits 0-1: tile compression mode.
    pub const MODE_MASK: u8 = 0x03;
    /// Bit 2: diff encoding relative to previous depth slice.
    pub const DIFF_ENCODING: u8 = 0x04;
    /// Bits 6-7: type reduction code for the offset value.
    pub const TYPE_REDUCTION_SHIFT: u8 = 6;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify enum values match the C++ reference (Lerc2.h).
    ///
    /// C++: enum ImageEncodeMode { IEM_Tiling = 0, IEM_DeltaHuffman, IEM_Huffman, IEM_DeltaDeltaHuffman };
    /// C++: enum BlockEncodeMode { BEM_RawBinary = 0, BEM_BitStuffSimple, BEM_BitStuffLUT };
    /// C++: enum DataType { DT_Char = 0, DT_Byte, DT_Short, DT_UShort, DT_Int, DT_UInt, DT_Float, DT_Double };
    #[test]
    fn image_encode_mode_matches_cpp() {
        assert_eq!(ImageEncodeMode::Tiling as u8, 0);
        assert_eq!(ImageEncodeMode::DeltaHuffman as u8, 1);
        assert_eq!(ImageEncodeMode::Huffman as u8, 2);
        assert_eq!(ImageEncodeMode::DeltaDeltaHuffman as u8, 3);
    }

    #[test]
    fn tile_compression_mode_matches_cpp() {
        // C++ BlockEncodeMode: BEM_RawBinary=0, BEM_BitStuffSimple=1, BEM_BitStuffLUT=2
        // C++ also uses 2=const-zero and 3=const-offset in the same 2-bit field
        assert_eq!(TileCompressionMode::RawBinary as u8, 0);
        assert_eq!(TileCompressionMode::BitStuffed as u8, 1);
        assert_eq!(TileCompressionMode::ConstZero as u8, 2);
        assert_eq!(TileCompressionMode::ConstOffset as u8, 3);
    }

    #[test]
    fn data_type_matches_cpp() {
        assert_eq!(DataType::Char as i32, 0);
        assert_eq!(DataType::Byte as i32, 1);
        assert_eq!(DataType::Short as i32, 2);
        assert_eq!(DataType::UShort as i32, 3);
        assert_eq!(DataType::Int as i32, 4);
        assert_eq!(DataType::UInt as i32, 5);
        assert_eq!(DataType::Float as i32, 6);
        assert_eq!(DataType::Double as i32, 7);
    }

    #[test]
    fn tile_flags_matches_cpp() {
        assert_eq!(tile_flags::MODE_MASK, 0x03);
        assert_eq!(tile_flags::DIFF_ENCODING, 0x04);
        assert_eq!(tile_flags::TYPE_REDUCTION_SHIFT, 6);
    }

    #[test]
    fn image_encode_mode_try_from_round_trip() {
        for v in 0..=3u8 {
            let mode = ImageEncodeMode::try_from(v).unwrap();
            assert_eq!(mode as u8, v);
        }
        assert!(ImageEncodeMode::try_from(4).is_err());
        assert!(ImageEncodeMode::try_from(255).is_err());
    }
}
