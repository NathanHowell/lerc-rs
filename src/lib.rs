#![cfg_attr(not(feature = "std"), no_std)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]

extern crate alloc;

pub mod error;
pub mod types;

pub mod bitmask;
pub(crate) mod bitstuffer;
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

/// Metadata returned from a decode-into operation (no owned pixel data).
#[derive(Debug, Clone)]
pub struct DecodeResult {
    pub width: u32,
    pub height: u32,
    pub n_depth: u32,
    pub n_bands: u32,
    pub data_type: DataType,
    pub valid_masks: Vec<BitMask>,
    pub no_data_value: Option<f64>,
}

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

/// Decode a single-band, single-depth LERC blob, returning typed pixel data,
/// the validity mask, width, and height.
///
/// The pixel type `T` must match the blob's data type. Returns an error on type
/// mismatch or if the blob contains multiple bands or depths (use [`decode`] for
/// multi-band/multi-depth blobs to get full shape and per-band masks).
pub fn decode_typed<T: LercDataType>(blob: &[u8]) -> Result<(Vec<T>, BitMask, u32, u32)> {
    let image = decode::decode(blob)?;
    if image.n_bands > 1 {
        return Err(LercError::InvalidData(alloc::format!(
            "decode_typed requires single-band data, got {} bands (use decode() instead)",
            image.n_bands
        )));
    }
    if image.n_depth > 1 {
        return Err(LercError::InvalidData(alloc::format!(
            "decode_typed requires single-depth data, got n_depth={} (use decode() instead)",
            image.n_depth
        )));
    }
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

    /// Get the pixel value at `(row, col)` for single-band, single-depth images.
    ///
    /// Returns `None` if the data type does not match `T`, if `n_bands > 1` or
    /// `n_depth > 1`, or if the coordinates are out of bounds.
    pub fn pixel<T: LercDataType>(&self, row: u32, col: u32) -> Option<T> {
        if self.n_bands != 1 || self.n_depth != 1 {
            return None;
        }
        if row >= self.height || col >= self.width {
            return None;
        }
        let data = self.as_typed::<T>()?;
        let idx = row as usize * self.width as usize + col as usize;
        Some(data[idx])
    }

    /// Iterate over valid pixels as `(row, col, value)` tuples.
    ///
    /// Only works for single-band, single-depth images. Returns `None` if the data
    /// type does not match `T` or if `n_bands > 1` or `n_depth > 1`.
    /// The iterator respects the validity mask, skipping invalid pixels.
    pub fn valid_pixels<'a, T: LercDataType + 'a>(
        &'a self,
    ) -> Option<impl Iterator<Item = (u32, u32, T)> + 'a> {
        if self.n_bands != 1 || self.n_depth != 1 {
            return None;
        }
        let data = self.as_typed::<T>()?;
        let width = self.width;
        let mask = self.valid_masks.first();
        Some(data.iter().enumerate().filter_map(move |(idx, &val)| {
            let is_valid = match mask {
                Some(m) => m.is_valid(idx),
                None => true,
            };
            if is_valid {
                let row = (idx / width as usize) as u32;
                let col = (idx % width as usize) as u32;
                Some((row, col, val))
            } else {
                None
            }
        }))
    }

    /// Get dimensions as `(width, height)`.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Total number of pixels (`width * height`).
    pub fn num_pixels(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Check if all pixels are valid (in the first band's mask).
    ///
    /// Returns `true` if there is no mask (all pixels are implicitly valid)
    /// or if every pixel in the mask is marked valid.
    pub fn all_valid(&self) -> bool {
        match self.valid_masks.first() {
            Some(m) => m.count_valid() == m.num_pixels(),
            None => true,
        }
    }

    /// Create a single-band, all-valid `LercImage` from a typed pixel vector
    /// and dimensions.
    ///
    /// Returns an error if `data.len() != width * height`.
    pub fn from_pixels<T: LercDataType>(width: u32, height: u32, data: Vec<T>) -> Result<Self> {
        let expected = width as usize * height as usize;
        if data.len() != expected {
            return Err(LercError::InvalidData(alloc::format!(
                "data length {} does not match width*height {expected}",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            n_depth: 1,
            n_bands: 1,
            data_type: T::DATA_TYPE,
            valid_masks: vec![BitMask::all_valid(expected)],
            data: T::into_lerc_data(data),
            no_data_value: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Zero-copy decode-into API
// ---------------------------------------------------------------------------

/// Decode a LERC blob into a pre-allocated buffer, returning metadata.
///
/// The type `T` must match the blob's data type (e.g., `f32` for `DataType::Float`).
/// The buffer must have at least `width * height * n_depth * n_bands` elements.
///
/// Returns `LercError::TypeMismatch` if `T` does not match the blob's data type.
/// Returns `LercError::OutputBufferTooSmall` if the buffer is too small.
pub fn decode_into<T: LercDataType>(data: &[u8], output: &mut [T]) -> Result<DecodeResult> {
    decode::decode_into(data, output)
}

/// Decode a LERC blob into a pre-allocated `i8` buffer.
pub fn decode_i8_into(data: &[u8], output: &mut [i8]) -> Result<DecodeResult> {
    decode::decode_into(data, output)
}

/// Decode a LERC blob into a pre-allocated `u8` buffer.
pub fn decode_u8_into(data: &[u8], output: &mut [u8]) -> Result<DecodeResult> {
    decode::decode_into(data, output)
}

/// Decode a LERC blob into a pre-allocated `i16` buffer.
pub fn decode_i16_into(data: &[u8], output: &mut [i16]) -> Result<DecodeResult> {
    decode::decode_into(data, output)
}

/// Decode a LERC blob into a pre-allocated `u16` buffer.
pub fn decode_u16_into(data: &[u8], output: &mut [u16]) -> Result<DecodeResult> {
    decode::decode_into(data, output)
}

/// Decode a LERC blob into a pre-allocated `i32` buffer.
pub fn decode_i32_into(data: &[u8], output: &mut [i32]) -> Result<DecodeResult> {
    decode::decode_into(data, output)
}

/// Decode a LERC blob into a pre-allocated `u32` buffer.
pub fn decode_u32_into(data: &[u8], output: &mut [u32]) -> Result<DecodeResult> {
    decode::decode_into(data, output)
}

/// Decode a LERC blob into a pre-allocated `f32` buffer.
pub fn decode_f32_into(data: &[u8], output: &mut [f32]) -> Result<DecodeResult> {
    decode::decode_into(data, output)
}

/// Decode a LERC blob into a pre-allocated `f64` buffer.
pub fn decode_f64_into(data: &[u8], output: &mut [f64]) -> Result<DecodeResult> {
    decode::decode_into(data, output)
}
