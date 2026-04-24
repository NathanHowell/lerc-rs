//! Pure Rust implementation of the LERC (Limited Error Raster Compression) format.
//!
//! Supports encoding and decoding of raster images with configurable lossy or lossless
//! compression. Compatible with ESRI's LERC2 format specification.

#![cfg_attr(not(feature = "std"), no_std)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]

extern crate alloc;

/// Error types for LERC encoding and decoding.
pub mod error;
/// Pixel data types and the `Sample` trait for type-safe encoding/decoding.
pub mod types;

/// Validity bitmask for tracking valid/invalid pixels.
pub mod bitmask;
pub(crate) mod bitstuffer;
/// Fletcher-32 checksum used by the LERC2 format.
#[allow(dead_code)]
pub mod checksum;
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
pub use types::{DataType, Sample};

use alloc::vec;
use alloc::vec::Vec;

use bitmask::BitMask;

/// Controls the precision/error tolerance for LERC encoding.
///
/// `Lossless` preserves exact values. `Tolerance(x)` allows decoded values
/// to differ from originals by at most +/-x.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Precision<T> {
    /// Lossless compression. Exact round-trip for all pixel values.
    #[default]
    Lossless,
    /// Lossy compression. Decoded values are within the given tolerance of originals.
    Tolerance(T),
}

/// Metadata returned from a decode-into operation (no owned pixel data).
#[derive(Debug, Clone)]
pub struct DecodeResult {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Number of values per pixel (depth slices).
    pub depth: u32,
    /// Number of bands in the image.
    pub bands: u32,
    /// Pixel data type of the decoded blob.
    pub data_type: DataType,
    /// Per-band validity masks indicating which pixels are valid.
    pub valid_masks: Vec<BitMask>,
    /// NoData sentinel value, if the blob uses NoData encoding.
    pub no_data_value: Option<f64>,
}

/// Header metadata extracted from a LERC blob without decoding pixel data.
#[derive(Debug, Clone, Default)]
pub struct LercInfo {
    /// LERC format version number.
    pub version: i32,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Number of values per pixel (depth slices).
    pub depth: u32,
    /// Number of bands in the image.
    pub bands: u32,
    /// Pixel data type stored in the blob.
    pub data_type: DataType,
    /// Number of valid (non-masked) pixels.
    pub valid_pixels: u32,
    /// Maximum error tolerance used during encoding.
    pub tolerance: f64,
    /// Minimum pixel value across all valid pixels.
    pub min_value: f64,
    /// Maximum pixel value across all valid pixels.
    pub max_value: f64,
    /// Total size of the LERC blob in bytes.
    pub blob_size: u32,
    /// The original NoData value, if the blob uses NoData encoding (v6+, depth > 1).
    pub no_data_value: Option<f64>,
}

/// A decoded raster image with pixel data and validity masks.
#[derive(Debug, Clone)]
pub struct Image {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Number of values per pixel (depth slices).
    pub depth: u32,
    /// Number of bands in the image.
    pub bands: u32,
    /// Pixel data type.
    pub data_type: DataType,
    /// Per-band validity masks indicating which pixels are valid.
    pub valid_masks: Vec<BitMask>,
    /// Pixel sample data stored as a typed vector.
    pub data: SampleData,
    /// The original NoData value, if any. When set during encoding with depth > 1,
    /// pixels matching this value in invalid depth slices are encoded with a sentinel.
    /// On decode, the sentinel is remapped back to this value.
    pub no_data_value: Option<f64>,
}

impl Default for Image {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            depth: 1,
            bands: 1,
            data_type: DataType::Byte,
            valid_masks: Vec::new(),
            data: SampleData::U8(Vec::new()),
            no_data_value: None,
        }
    }
}

/// Type-erased pixel data container, one variant per supported data type.
#[derive(Debug, Clone)]
pub enum SampleData {
    /// Signed 8-bit integer pixel data.
    I8(Vec<i8>),
    /// Unsigned 8-bit integer pixel data.
    U8(Vec<u8>),
    /// Signed 16-bit integer pixel data.
    I16(Vec<i16>),
    /// Unsigned 16-bit integer pixel data.
    U16(Vec<u16>),
    /// Signed 32-bit integer pixel data.
    I32(Vec<i32>),
    /// Unsigned 32-bit integer pixel data.
    U32(Vec<u32>),
    /// 32-bit floating-point pixel data.
    F32(Vec<f32>),
    /// 64-bit floating-point pixel data.
    F64(Vec<f64>),
}

/// Read header metadata from a LERC blob without decoding pixel data.
pub fn decode_info(data: &[u8]) -> Result<LercInfo> {
    decode::decode_info(data)
}

/// Decode a LERC blob, returning the image with pixel data and validity masks.
pub fn decode(data: &[u8]) -> Result<Image> {
    decode::decode(data)
}

/// Encode an image into a LERC blob with the given precision.
pub fn encode(image: &Image, precision: Precision<f64>) -> Result<Vec<u8>> {
    let max_z_error = match precision {
        Precision::Lossless => {
            if image.data_type.is_integer() {
                0.5
            } else {
                0.0
            }
        }
        Precision::Tolerance(val) => val,
    };
    encode::encode(image, max_z_error)
}

// ---------------------------------------------------------------------------
// Typed convenience encode/decode helpers
// ---------------------------------------------------------------------------

/// Encode a single-band image with all pixels valid.
///
/// The pixel type `T` determines the LERC data type automatically via `Sample`.
/// Returns an error if `data.len() != width * height`.
pub fn encode_slice<T: Sample>(
    width: u32,
    height: u32,
    data: &[T],
    precision: Precision<T>,
) -> Result<Vec<u8>> {
    let expected = (width as usize) * (height as usize);
    if data.len() != expected {
        return Err(LercError::InvalidData(alloc::format!(
            "data length {} does not match width*height {expected}",
            data.len()
        )));
    }
    let max_z_error: f64 = match precision {
        Precision::Lossless => {
            if T::is_integer() {
                0.5
            } else {
                0.0
            }
        }
        Precision::Tolerance(val) => val.to_f64(),
    };
    let image = Image {
        width,
        height,
        depth: 1,
        bands: 1,
        data_type: T::DATA_TYPE,
        valid_masks: vec![BitMask::all_valid(expected)],
        data: T::into_lerc_data(data.to_vec()),
        no_data_value: None,
    };
    encode::encode(&image, max_z_error)
}

/// Encode a single-band image with a validity mask.
///
/// The pixel type `T` determines the LERC data type automatically via `Sample`.
/// Returns an error if `data.len() != width * height` or if the mask size does not match.
pub fn encode_slice_masked<T: Sample>(
    width: u32,
    height: u32,
    data: &[T],
    mask: &BitMask,
    precision: Precision<T>,
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
    let max_z_error: f64 = match precision {
        Precision::Lossless => {
            if T::is_integer() {
                0.5
            } else {
                0.0
            }
        }
        Precision::Tolerance(val) => val.to_f64(),
    };
    let image = Image {
        width,
        height,
        depth: 1,
        bands: 1,
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
pub fn decode_slice<T: Sample>(blob: &[u8]) -> Result<(Vec<T>, BitMask, u32, u32)> {
    let image = decode::decode(blob)?;
    if image.bands > 1 {
        return Err(LercError::InvalidData(alloc::format!(
            "decode_slice requires single-band data, got {} bands (use decode() instead)",
            image.bands
        )));
    }
    if image.depth > 1 {
        return Err(LercError::InvalidData(alloc::format!(
            "decode_slice requires single-depth data, got depth={} (use decode() instead)",
            image.depth
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
// Typed accessor methods on Image
// ---------------------------------------------------------------------------

impl Image {
    /// Try to borrow the pixel data as `&[T]`.
    ///
    /// Returns `None` if the image's data type does not match `T`.
    pub fn as_typed<T: Sample>(&self) -> Option<&[T]> {
        T::try_ref_lerc_data(&self.data)
    }

    /// Return the validity mask for the first band, or `None` if no masks are present.
    pub fn mask(&self) -> Option<&BitMask> {
        self.valid_masks.first()
    }

    /// Get the pixel value at `(row, col)` for single-band, single-depth images.
    ///
    /// Returns `None` if the data type does not match `T`, if `bands > 1` or
    /// `depth > 1`, or if the coordinates are out of bounds.
    pub fn pixel<T: Sample>(&self, row: u32, col: u32) -> Option<T> {
        if self.bands != 1 || self.depth != 1 {
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
    /// type does not match `T` or if `bands > 1` or `depth > 1`.
    /// The iterator respects the validity mask, skipping invalid pixels.
    pub fn valid_pixels<'a, T: Sample + 'a>(
        &'a self,
    ) -> Option<impl Iterator<Item = (u32, u32, T)> + 'a> {
        if self.bands != 1 || self.depth != 1 {
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

    /// Check if all pixels in the first band are valid.
    ///
    /// Returns `true` if there is no mask (all pixels are implicitly valid),
    /// if the first band's mask is [`BitMask::AllValid`] (O(1)), or if an
    /// explicit mask happens to have every bit set (O(n) popcount fallback).
    pub fn all_valid(&self) -> bool {
        match self.valid_masks.first() {
            Some(m) => m.is_all_valid(),
            None => true,
        }
    }

    /// Create a single-band, all-valid `Image` from a typed pixel vector
    /// and dimensions.
    ///
    /// Returns an error if `data.len() != width * height`.
    pub fn from_pixels<T: Sample>(width: u32, height: u32, data: Vec<T>) -> Result<Self> {
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
            depth: 1,
            bands: 1,
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
pub fn decode_into<T: Sample>(data: &[u8], output: &mut [T]) -> Result<DecodeResult> {
    decode::decode_into(data, output)
}
