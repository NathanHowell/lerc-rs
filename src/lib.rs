#![cfg_attr(not(feature = "std"), no_std)]
#![allow(
    clippy::too_many_arguments,
    clippy::manual_div_ceil,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::manual_range_contains,
    clippy::ptr_arg,
    clippy::needless_range_loop,
    clippy::manual_memcpy,
    clippy::identity_op
)]

extern crate alloc;

pub mod error;
pub mod types;

pub(crate) mod bitstuffer;
pub mod bitmask;
#[allow(dead_code)]
pub(crate) mod checksum;
#[allow(dead_code)]
pub(crate) mod header;
#[allow(dead_code)]
pub(crate) mod huffman;
pub(crate) mod rle;

pub(crate) mod decode;
pub(crate) mod encode;
#[allow(dead_code)]
pub(crate) mod fpl;
pub(crate) mod lerc1;
#[allow(dead_code)]
pub(crate) mod tiles;

pub use error::{LercError, Result};
pub use types::{DataType, LercDataType};

use alloc::vec;
use alloc::vec::Vec;

use bitmask::BitMask;

#[derive(Debug, Clone, Default)]
pub struct LercInfo {
    pub version: i32,
    pub width: u32,
    pub height: u32,
    pub n_depth: u32,
    pub n_bands: u32,
    pub data_type: DataType,
    pub num_valid_pixels: u32,
    pub max_z_error: f64,
    pub z_min: f64,
    pub z_max: f64,
    pub blob_size: u32,
    /// The original NoData value, if the blob uses NoData encoding (v6+, nDepth > 1).
    pub no_data_value: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct LercImage {
    pub width: u32,
    pub height: u32,
    pub n_depth: u32,
    pub n_bands: u32,
    pub data_type: DataType,
    pub valid_masks: Vec<BitMask>,
    pub data: LercData,
    /// The original NoData value, if any. When set during encoding with nDepth > 1,
    /// pixels matching this value in invalid depth slices are encoded with a sentinel.
    /// On decode, the sentinel is remapped back to this value.
    pub no_data_value: Option<f64>,
}

impl Default for LercImage {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            n_depth: 1,
            n_bands: 1,
            data_type: DataType::Byte,
            valid_masks: Vec::new(),
            data: LercData::U8(Vec::new()),
            no_data_value: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum LercData {
    I8(Vec<i8>),
    U8(Vec<u8>),
    I16(Vec<i16>),
    U16(Vec<u16>),
    I32(Vec<i32>),
    U32(Vec<u32>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

pub fn decode_info(data: &[u8]) -> Result<LercInfo> {
    decode::decode_info(data)
}

pub fn decode(data: &[u8]) -> Result<LercImage> {
    decode::decode(data)
}

pub fn encode(image: &LercImage, max_z_error: f64) -> Result<Vec<u8>> {
    encode::encode(image, max_z_error)
}

// ---------------------------------------------------------------------------
// Typed convenience encode/decode helpers
// ---------------------------------------------------------------------------

/// Encode a single-band image with all pixels valid.
///
/// The pixel type `T` determines the LERC data type automatically via `LercDataType`.
/// Returns an error if `data.len() != width * height`.
pub fn encode_typed<T: LercDataType>(
    width: u32,
    height: u32,
    data: &[T],
    max_z_error: f64,
) -> Result<Vec<u8>> {
    let expected = (width as usize) * (height as usize);
    if data.len() != expected {
        return Err(LercError::InvalidData(alloc::format!(
            "data length {} does not match width*height {expected}",
            data.len()
        )));
    }
    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: T::DATA_TYPE,
        valid_masks: vec![BitMask::all_valid(expected)],
        data: T::into_lerc_data(data.to_vec()),
        no_data_value: None,
    };
    encode::encode(&image, max_z_error)
}

/// Encode a single-band image with a validity mask.
///
/// The pixel type `T` determines the LERC data type automatically via `LercDataType`.
/// Returns an error if `data.len() != width * height` or if the mask size does not match.
pub fn encode_typed_masked<T: LercDataType>(
    width: u32,
    height: u32,
    data: &[T],
    mask: &BitMask,
    max_z_error: f64,
) -> Result<Vec<u8>> {
    let expected = (width as usize) * (height as usize);
    if data.len() != expected {
        return Err(LercError::InvalidData(alloc::format!(
            "data length {} does not match width*height {expected}",
            data.len()
        )));
    }
    if mask.num_pixels() != expected {
        return Err(LercError::InvalidData(alloc::format!(
            "mask pixel count {} does not match width*height {expected}",
            mask.num_pixels()
        )));
    }
    let image = LercImage {
        width,
        height,
        n_depth: 1,
        n_bands: 1,
        data_type: T::DATA_TYPE,
        valid_masks: vec![mask.clone()],
        data: T::into_lerc_data(data.to_vec()),
        no_data_value: None,
    };
    encode::encode(&image, max_z_error)
}

/// Decode a LERC blob, returning typed pixel data, the validity mask, width, and height.
///
/// The pixel type `T` must match the blob's data type. Returns an error on mismatch.
pub fn decode_typed<T: LercDataType>(blob: &[u8]) -> Result<(Vec<T>, BitMask, u32, u32)> {
    let image = decode::decode(blob)?;
    let w = image.width;
    let h = image.height;
    let pixels = T::try_from_lerc_data(image.data).map_err(|_| {
        LercError::InvalidData(alloc::format!(
            "expected {:?} data but blob contains {:?}",
            T::DATA_TYPE,
            image.data_type
        ))
    })?;
    let mask = image
        .valid_masks
        .into_iter()
        .next()
        .unwrap_or_else(|| BitMask::all_valid((w as usize) * (h as usize)));
    Ok((pixels, mask, w, h))
}

// ---------------------------------------------------------------------------
// Typed accessor methods on LercImage
// ---------------------------------------------------------------------------

impl LercImage {
    /// Try to borrow the pixel data as `&[T]`.
    ///
    /// Returns `None` if the image's data type does not match `T`.
    pub fn as_typed<T: LercDataType>(&self) -> Option<&[T]> {
        T::try_ref_lerc_data(&self.data)
    }

    /// Return the validity mask for the first band, or `None` if no masks are present.
    pub fn mask(&self) -> Option<&BitMask> {
        self.valid_masks.first()
    }
}
