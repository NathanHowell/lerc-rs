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
}

macro_rules! impl_lerc_data_type {
    ($ty:ty, $dt:expr, $is_int:expr) => {
        impl sealed::Sealed for $ty {}
        impl LercDataType for $ty {
            const DATA_TYPE: DataType = $dt;
            const BYTES: usize = core::mem::size_of::<$ty>();

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
        }
    };
}

impl_lerc_data_type!(i8, DataType::Char, true);
impl_lerc_data_type!(u8, DataType::Byte, true);
impl_lerc_data_type!(i16, DataType::Short, true);
impl_lerc_data_type!(u16, DataType::UShort, true);
impl_lerc_data_type!(i32, DataType::Int, true);
impl_lerc_data_type!(u32, DataType::UInt, true);

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
}
