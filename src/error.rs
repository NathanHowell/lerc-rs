use alloc::string::String;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LercError {
    #[error("invalid LERC magic bytes")]
    InvalidMagic,
    #[error("unsupported LERC version: {0}")]
    UnsupportedVersion(i32),
    #[error("checksum mismatch: expected {expected:#010x}, computed {computed:#010x}")]
    ChecksumMismatch { expected: u32, computed: u32 },
    #[error("buffer too small: need {needed} bytes, have {available}")]
    BufferTooSmall { needed: usize, available: usize },
    #[error("invalid data: {0}")]
    InvalidData(String),
    #[error("invalid data type: {0}")]
    InvalidDataType(i32),
    #[error("unsupported encoding: {0}")]
    UnsupportedEncoding(u8),
    #[error("block integrity check failed")]
    IntegrityCheckFailed,
    #[error("type mismatch: expected {expected:?}, actual {actual:?}")]
    TypeMismatch {
        expected: crate::types::DataType,
        actual: crate::types::DataType,
    },
    #[error("output buffer too small: need {needed} elements, have {available}")]
    OutputBufferTooSmall { needed: usize, available: usize },
}

pub type Result<T> = core::result::Result<T, LercError>;
