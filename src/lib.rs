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
#[allow(dead_code)]
pub(crate) mod tiles;

pub use error::{LercError, Result};
pub use types::DataType;

use alloc::vec::Vec;

use bitmask::BitMask;

#[derive(Debug, Clone)]
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
