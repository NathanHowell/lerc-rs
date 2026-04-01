use crate::checksum::fletcher32;
use crate::error::{LercError, Result};
use crate::types::DataType;

use alloc::vec::Vec;

const LERC2_MAGIC: &[u8; 6] = b"Lerc2 ";

const MIN_VERSION: i32 = 2;
const MAX_VERSION: i32 = 6;
const CURRENT_VERSION: i32 = 6;

#[derive(Debug, Clone)]
pub struct HeaderInfo {
    pub version: i32,
    pub checksum: u32,
    pub n_rows: i32,
    pub n_cols: i32,
    pub n_depth: i32,
    pub num_valid_pixel: i32,
    pub micro_block_size: i32,
    pub blob_size: i32,
    pub data_type: DataType,
    pub n_blobs_more: i32,
    pub pass_no_data_values: bool,
    pub is_int: bool,
    pub max_z_error: f64,
    pub z_min: f64,
    pub z_max: f64,
    pub no_data_val: f64,
    pub no_data_val_orig: f64,
}

impl Default for HeaderInfo {
    fn default() -> Self {
        Self {
            version: CURRENT_VERSION,
            checksum: 0,
            n_rows: 0,
            n_cols: 0,
            n_depth: 1,
            num_valid_pixel: 0,
            micro_block_size: 8,
            blob_size: 0,
            data_type: DataType::Byte,
            n_blobs_more: 0,
            pass_no_data_values: false,
            is_int: false,
            max_z_error: 0.0,
            z_min: 0.0,
            z_max: 0.0,
            no_data_val: 0.0,
            no_data_val_orig: 0.0,
        }
    }
}

impl HeaderInfo {
    pub fn header_size(version: i32) -> usize {
        let mut size = 6; // magic
        size += 4; // version
        if version >= 3 {
            size += 4; // checksum
        }
        size += 4 * 2; // nRows, nCols
        if version >= 4 {
            size += 4; // nDepth
        }
        size += 4 * 4; // numValidPixel, microBlockSize, blobSize, dataType
        if version >= 6 {
            size += 4; // nBlobsMore
            size += 4; // 4 flag bytes
        }
        size += 8 * 3; // maxZError, zMin, zMax
        if version >= 6 {
            size += 8 * 2; // noDataVal, noDataValOrig
        }
        size
    }
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.remaining() < n {
            return Err(LercError::BufferTooSmall {
                needed: self.pos + n,
                available: self.data.len(),
            });
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_i32(&mut self) -> Result<i32> {
        let bytes = self.read_bytes(4)?;
        Ok(i32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_u32(&mut self) -> Result<u32> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_f64(&mut self) -> Result<f64> {
        let bytes = self.read_bytes(8)?;
        Ok(f64::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_u8(&mut self) -> Result<u8> {
        let bytes = self.read_bytes(1)?;
        Ok(bytes[0])
    }
}

pub fn read_header(data: &[u8]) -> Result<(HeaderInfo, usize)> {
    let mut c = Cursor::new(data);

    // Magic
    let magic = c.read_bytes(6)?;
    if magic != LERC2_MAGIC {
        return Err(LercError::InvalidMagic);
    }

    let version = c.read_i32()?;
    if !(MIN_VERSION..=MAX_VERSION).contains(&version) {
        return Err(LercError::UnsupportedVersion(version));
    }

    let checksum = if version >= 3 { c.read_u32()? } else { 0 };

    let n_rows = c.read_i32()?;
    let n_cols = c.read_i32()?;

    let n_depth = if version >= 4 { c.read_i32()? } else { 1 };

    let num_valid_pixel = c.read_i32()?;
    let micro_block_size = c.read_i32()?;
    let blob_size = c.read_i32()?;
    let dt_raw = c.read_i32()?;
    let data_type = DataType::from_i32(dt_raw).ok_or(LercError::InvalidDataType(dt_raw))?;

    let (n_blobs_more, pass_no_data_values, is_int) = if version >= 6 {
        let nbm = c.read_i32()?;
        let pass_nd = c.read_u8()? != 0;
        let is_int = c.read_u8()? != 0;
        let _reserved3 = c.read_u8()?;
        let _reserved4 = c.read_u8()?;
        (nbm, pass_nd, is_int)
    } else {
        (0, false, false)
    };

    let max_z_error = c.read_f64()?;
    let z_min = c.read_f64()?;
    let z_max = c.read_f64()?;

    let (no_data_val, no_data_val_orig) = if version >= 6 {
        (c.read_f64()?, c.read_f64()?)
    } else {
        (0.0, 0.0)
    };

    let header = HeaderInfo {
        version,
        checksum,
        n_rows,
        n_cols,
        n_depth,
        num_valid_pixel,
        micro_block_size,
        blob_size,
        data_type,
        n_blobs_more,
        pass_no_data_values,
        is_int,
        max_z_error,
        z_min,
        z_max,
        no_data_val,
        no_data_val_orig,
    };

    Ok((header, c.pos))
}

pub fn verify_checksum(data: &[u8], header: &HeaderInfo) -> Result<()> {
    if header.version < 3 {
        return Ok(());
    }

    let blob_size = header.blob_size as usize;
    if data.len() < blob_size {
        return Err(LercError::BufferTooSmall {
            needed: blob_size,
            available: data.len(),
        });
    }

    // Checksum is computed over everything after the checksum field
    // Header layout: magic(6) + version(4) + checksum(4) = offset 14
    let checksum_data = &data[14..blob_size];
    let computed = fletcher32(checksum_data);

    if computed != header.checksum {
        return Err(LercError::ChecksumMismatch {
            expected: header.checksum,
            computed,
        });
    }

    Ok(())
}

pub fn write_header(header: &HeaderInfo) -> Vec<u8> {
    let size = HeaderInfo::header_size(header.version);
    let mut buf = Vec::with_capacity(size);

    buf.extend_from_slice(LERC2_MAGIC);
    buf.extend_from_slice(&header.version.to_le_bytes());

    if header.version >= 3 {
        buf.extend_from_slice(&header.checksum.to_le_bytes());
    }

    buf.extend_from_slice(&header.n_rows.to_le_bytes());
    buf.extend_from_slice(&header.n_cols.to_le_bytes());

    if header.version >= 4 {
        buf.extend_from_slice(&header.n_depth.to_le_bytes());
    }

    buf.extend_from_slice(&header.num_valid_pixel.to_le_bytes());
    buf.extend_from_slice(&header.micro_block_size.to_le_bytes());
    buf.extend_from_slice(&header.blob_size.to_le_bytes());
    buf.extend_from_slice(&(header.data_type as i32).to_le_bytes());

    if header.version >= 6 {
        buf.extend_from_slice(&header.n_blobs_more.to_le_bytes());
        buf.push(header.pass_no_data_values as u8);
        buf.push(header.is_int as u8);
        buf.push(0); // reserved3
        buf.push(0); // reserved4
    }

    buf.extend_from_slice(&header.max_z_error.to_le_bytes());
    buf.extend_from_slice(&header.z_min.to_le_bytes());
    buf.extend_from_slice(&header.z_max.to_le_bytes());

    if header.version >= 6 {
        buf.extend_from_slice(&header.no_data_val.to_le_bytes());
        buf.extend_from_slice(&header.no_data_val_orig.to_le_bytes());
    }

    debug_assert_eq!(buf.len(), size);
    buf
}

/// Patch the blob size and checksum into an already-written blob.
pub fn finalize_blob(blob: &mut [u8]) {
    let blob_size = blob.len() as i32;

    // Patch blobSize field.
    // Offset: magic(6) + version(4) + checksum(4) + nRows(4) + nCols(4) + nDepth(4) +
    //         numValidPixel(4) + microBlockSize(4) = 34
    // But this depends on version. For v6 the nDepth is present.
    // Let's read the version first.
    let version = i32::from_le_bytes(blob[6..10].try_into().unwrap());
    let blob_size_offset = if version >= 4 {
        // magic(6) + version(4) + checksum(4) + nRows(4) + nCols(4) + nDepth(4) +
        // numValidPixel(4) + microBlockSize(4) = 34
        if version >= 3 { 34 } else { 30 }
    } else {
        // magic(6) + version(4) [+ checksum(4)] + nRows(4) + nCols(4) +
        // numValidPixel(4) + microBlockSize(4)
        if version >= 3 { 30 } else { 26 }
    };

    blob[blob_size_offset..blob_size_offset + 4].copy_from_slice(&blob_size.to_le_bytes());

    // Patch checksum for v3+
    if version >= 3 {
        let checksum = fletcher32(&blob[14..]);
        blob[10..14].copy_from_slice(&checksum.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_size_v2() {
        // magic(6) + version(4) + nRows(4) + nCols(4) + numValid(4) + mbs(4) + blobSize(4) + dt(4) + 3*f64(24) = 58
        assert_eq!(HeaderInfo::header_size(2), 58);
    }

    #[test]
    fn header_size_v3() {
        assert_eq!(HeaderInfo::header_size(3), 62);
    }

    #[test]
    fn header_size_v4() {
        // v3 + nDepth(4) = 66
        assert_eq!(HeaderInfo::header_size(4), 66);
    }

    #[test]
    fn header_size_v6() {
        // v4(66) + nBlobsMore(4) + 4 flag bytes + 2*f64(16) = 90
        assert_eq!(HeaderInfo::header_size(6), 90);
    }

    #[test]
    fn round_trip_header() {
        let header = HeaderInfo {
            version: CURRENT_VERSION,
            checksum: 0,
            n_rows: 100,
            n_cols: 200,
            n_depth: 3,
            num_valid_pixel: 20000,
            micro_block_size: 8,
            blob_size: 1234,
            data_type: DataType::Float,
            n_blobs_more: 2,
            pass_no_data_values: false,
            is_int: false,
            max_z_error: 0.01,
            z_min: -100.5,
            z_max: 3000.7,
            no_data_val: -9999.0,
            no_data_val_orig: -9999.0,
        };

        let buf = write_header(&header);
        assert_eq!(buf.len(), HeaderInfo::header_size(CURRENT_VERSION));

        let (parsed, consumed) = read_header(&buf).unwrap();
        assert_eq!(consumed, buf.len());
        assert_eq!(parsed.version, header.version);
        assert_eq!(parsed.n_rows, header.n_rows);
        assert_eq!(parsed.n_cols, header.n_cols);
        assert_eq!(parsed.n_depth, header.n_depth);
        assert_eq!(parsed.num_valid_pixel, header.num_valid_pixel);
        assert_eq!(parsed.micro_block_size, header.micro_block_size);
        assert_eq!(parsed.blob_size, header.blob_size);
        assert_eq!(parsed.data_type, header.data_type);
        assert_eq!(parsed.n_blobs_more, header.n_blobs_more);
        assert_eq!(parsed.pass_no_data_values, header.pass_no_data_values);
        assert_eq!(parsed.is_int, header.is_int);
        assert_eq!(parsed.max_z_error, header.max_z_error);
        assert_eq!(parsed.z_min, header.z_min);
        assert_eq!(parsed.z_max, header.z_max);
        assert_eq!(parsed.no_data_val, header.no_data_val);
        assert_eq!(parsed.no_data_val_orig, header.no_data_val_orig);
    }
}
