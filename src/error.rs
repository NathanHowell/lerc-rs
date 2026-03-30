use alloc::string::String;
use core::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LercError {
    InvalidMagic,
    UnsupportedVersion(i32),
    ChecksumMismatch { expected: u32, computed: u32 },
    BufferTooSmall { needed: usize, available: usize },
    InvalidData(String),
    InvalidDataType(i32),
    UnsupportedEncoding(u8),
    IntegrityCheckFailed,
    TypeMismatch {
        expected: crate::types::DataType,
        actual: crate::types::DataType,
    },
    OutputBufferTooSmall {
        needed: usize,
        available: usize,
    },
}

impl fmt::Display for LercError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMagic => write!(f, "invalid LERC magic bytes"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported LERC version: {v}"),
            Self::ChecksumMismatch { expected, computed } => {
                write!(
                    f,
                    "checksum mismatch: expected {expected:#010x}, computed {computed:#010x}"
                )
            }
            Self::BufferTooSmall { needed, available } => {
                write!(f, "buffer too small: need {needed} bytes, have {available}")
            }
            Self::InvalidData(msg) => write!(f, "invalid data: {msg}"),
            Self::InvalidDataType(dt) => write!(f, "invalid data type: {dt}"),
            Self::UnsupportedEncoding(enc) => write!(f, "unsupported encoding: {enc}"),
            Self::IntegrityCheckFailed => write!(f, "block integrity check failed"),
            Self::TypeMismatch { expected, actual } => {
                write!(f, "type mismatch: expected {expected:?}, actual {actual:?}")
            }
            Self::OutputBufferTooSmall { needed, available } => {
                write!(
                    f,
                    "output buffer too small: need {needed} elements, have {available}"
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LercError {}

pub type Result<T> = core::result::Result<T, LercError>;
