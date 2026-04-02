use alloc::string::String;

/// Errors that can occur during LERC encoding or decoding.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LercError {
    /// The data does not start with valid LERC magic bytes.
    #[error("invalid LERC magic bytes")]
    InvalidMagic,
    /// The LERC version in the header is not supported.
    #[error("unsupported LERC version: {0}")]
    UnsupportedVersion(i32),
    /// The computed checksum does not match the expected value.
    #[error("checksum mismatch: expected {expected:#010x}, computed {computed:#010x}")]
    ChecksumMismatch {
        /// Expected checksum from the header.
        expected: u32,
        /// Checksum computed from the data.
        computed: u32,
    },
    /// The input buffer is too small to read the required bytes.
    #[error("buffer too small: need {needed} bytes, have {available}")]
    BufferTooSmall {
        /// Minimum number of bytes required.
        needed: usize,
        /// Actual number of bytes available.
        available: usize,
    },
    /// The data is structurally invalid or internally inconsistent.
    #[error("invalid data: {0}")]
    InvalidData(String),
    /// The data type code in the header is not recognized.
    #[error("invalid data type: {0}")]
    InvalidDataType(i32),
    /// The encoding mode is not supported by this implementation.
    #[error("unsupported encoding: {0}")]
    UnsupportedEncoding(u8),
    /// A block-level integrity check failed during decoding.
    #[error("block integrity check failed")]
    IntegrityCheckFailed,
    /// The requested pixel type does not match the blob's data type.
    #[error("type mismatch: expected {expected:?}, actual {actual:?}")]
    TypeMismatch {
        /// The data type requested by the caller.
        expected: crate::types::DataType,
        /// The actual data type found in the blob.
        actual: crate::types::DataType,
    },
    /// The output buffer is too small to hold the decoded pixel data.
    #[error("output buffer too small: need {needed} elements, have {available}")]
    OutputBufferTooSmall {
        /// Minimum number of elements required.
        needed: usize,
        /// Actual number of elements available.
        available: usize,
    },
}

/// A specialized `Result` type for LERC operations.
pub type Result<T> = core::result::Result<T, LercError>;
