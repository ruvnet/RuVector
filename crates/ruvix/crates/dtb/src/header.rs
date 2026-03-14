//! FDT header parsing.

use crate::{read_be_u32, DtbError, DtbResult, FDT_MAGIC, FDT_MIN_VERSION, FDT_VERSION};

/// Size of the FDT header in bytes.
pub const FDT_HEADER_SIZE: usize = 40;

/// FDT header structure.
///
/// All fields are stored in big-endian format in the blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FdtHeader {
    /// Magic number (should be 0xd00dfeed)
    pub magic: u32,
    /// Total size of the DTB blob
    pub total_size: u32,
    /// Offset to the structure block
    pub off_dt_struct: u32,
    /// Offset to the strings block
    pub off_dt_strings: u32,
    /// Offset to memory reservation block
    pub off_mem_rsvmap: u32,
    /// FDT version
    pub version: u32,
    /// Last compatible version
    pub last_comp_version: u32,
    /// Physical ID of the boot CPU
    pub boot_cpuid_phys: u32,
    /// Size of the strings block
    pub size_dt_strings: u32,
    /// Size of the structure block
    pub size_dt_struct: u32,
}

impl FdtHeader {
    /// Parse an FDT header from a byte slice.
    ///
    /// # Errors
    ///
    /// Returns `DtbError` if:
    /// - The blob is too small for a header
    /// - The magic number is invalid
    /// - The version is not supported
    /// - Any offset is invalid
    pub fn parse(blob: &[u8]) -> DtbResult<Self> {
        // Check minimum size
        if blob.len() < FDT_HEADER_SIZE {
            return Err(DtbError::blob_too_small(blob.len(), FDT_HEADER_SIZE));
        }

        // Read magic number
        let magic = read_be_u32(blob, 0).ok_or(DtbError::UnexpectedEnd)?;
        if magic != FDT_MAGIC {
            return Err(DtbError::invalid_magic(magic));
        }

        // Read remaining header fields
        let total_size = read_be_u32(blob, 4).ok_or(DtbError::UnexpectedEnd)?;
        let off_dt_struct = read_be_u32(blob, 8).ok_or(DtbError::UnexpectedEnd)?;
        let off_dt_strings = read_be_u32(blob, 12).ok_or(DtbError::UnexpectedEnd)?;
        let off_mem_rsvmap = read_be_u32(blob, 16).ok_or(DtbError::UnexpectedEnd)?;
        let version = read_be_u32(blob, 20).ok_or(DtbError::UnexpectedEnd)?;
        let last_comp_version = read_be_u32(blob, 24).ok_or(DtbError::UnexpectedEnd)?;
        let boot_cpuid_phys = read_be_u32(blob, 28).ok_or(DtbError::UnexpectedEnd)?;
        let size_dt_strings = read_be_u32(blob, 32).ok_or(DtbError::UnexpectedEnd)?;
        let size_dt_struct = read_be_u32(blob, 36).ok_or(DtbError::UnexpectedEnd)?;

        let header = Self {
            magic,
            total_size,
            off_dt_struct,
            off_dt_strings,
            off_mem_rsvmap,
            version,
            last_comp_version,
            boot_cpuid_phys,
            size_dt_strings,
            size_dt_struct,
        };

        // Validate the header
        header.validate(blob.len())?;

        Ok(header)
    }

    /// Validate the header against the blob size.
    ///
    /// # Errors
    ///
    /// Returns `DtbError` if any validation fails.
    pub fn validate(&self, blob_size: usize) -> DtbResult<()> {
        // Check total size
        if (self.total_size as usize) > blob_size {
            return Err(DtbError::invalid_total_size(self.total_size, blob_size));
        }

        // Check version
        if self.version < FDT_MIN_VERSION || self.version > FDT_VERSION + 10 {
            return Err(DtbError::unsupported_version(self.version));
        }

        // Check structure block offset
        if self.off_dt_struct as usize >= blob_size {
            return Err(DtbError::InvalidStructureOffset {
                offset: self.off_dt_struct,
            });
        }

        // Check strings block offset
        if self.off_dt_strings as usize >= blob_size {
            return Err(DtbError::InvalidStringsOffset {
                offset: self.off_dt_strings,
            });
        }

        // Check memory reservation offset
        if self.off_mem_rsvmap as usize >= blob_size {
            return Err(DtbError::InvalidReservationOffset {
                offset: self.off_mem_rsvmap,
            });
        }

        // Check structure block bounds
        let struct_end = self
            .off_dt_struct
            .checked_add(self.size_dt_struct)
            .ok_or(DtbError::InvalidStructureOffset {
                offset: self.off_dt_struct,
            })?;
        if struct_end as usize > blob_size {
            return Err(DtbError::InvalidStructureOffset {
                offset: self.off_dt_struct,
            });
        }

        // Check strings block bounds
        let strings_end = self
            .off_dt_strings
            .checked_add(self.size_dt_strings)
            .ok_or(DtbError::InvalidStringsOffset {
                offset: self.off_dt_strings,
            })?;
        if strings_end as usize > blob_size {
            return Err(DtbError::InvalidStringsOffset {
                offset: self.off_dt_strings,
            });
        }

        Ok(())
    }

    /// Get the structure block slice from a blob.
    #[must_use]
    pub fn structure_block<'a>(&self, blob: &'a [u8]) -> &'a [u8] {
        let start = self.off_dt_struct as usize;
        let end = start + self.size_dt_struct as usize;
        &blob[start..end]
    }

    /// Get the strings block slice from a blob.
    #[must_use]
    pub fn strings_block<'a>(&self, blob: &'a [u8]) -> &'a [u8] {
        let start = self.off_dt_strings as usize;
        let end = start + self.size_dt_strings as usize;
        &blob[start..end]
    }

    /// Get a string from the strings block by offset.
    pub fn get_string<'a>(&self, blob: &'a [u8], offset: u32) -> DtbResult<&'a str> {
        let strings = self.strings_block(blob);
        let start = offset as usize;

        if start >= strings.len() {
            return Err(DtbError::InvalidPropertyNameOffset { offset });
        }

        // Find null terminator
        let end = strings[start..]
            .iter()
            .position(|&b| b == 0)
            .ok_or(DtbError::InvalidString)?;

        // Convert to str
        core::str::from_utf8(&strings[start..start + end]).map_err(|_| DtbError::InvalidString)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_valid_header() -> [u8; 40] {
        let mut header = [0u8; 40];

        // Magic (0xd00dfeed)
        header[0..4].copy_from_slice(&0xd00d_feed_u32.to_be_bytes());
        // Total size (256 bytes)
        header[4..8].copy_from_slice(&256_u32.to_be_bytes());
        // Structure offset (40)
        header[8..12].copy_from_slice(&40_u32.to_be_bytes());
        // Strings offset (200)
        header[12..16].copy_from_slice(&200_u32.to_be_bytes());
        // Memory reservation offset (40)
        header[16..20].copy_from_slice(&40_u32.to_be_bytes());
        // Version (17)
        header[20..24].copy_from_slice(&17_u32.to_be_bytes());
        // Last compatible version (16)
        header[24..28].copy_from_slice(&16_u32.to_be_bytes());
        // Boot CPU ID (0)
        header[28..32].copy_from_slice(&0_u32.to_be_bytes());
        // Strings size (56)
        header[32..36].copy_from_slice(&56_u32.to_be_bytes());
        // Structure size (160)
        header[36..40].copy_from_slice(&160_u32.to_be_bytes());

        header
    }

    #[test]
    fn test_parse_valid_header() {
        let mut blob = [0u8; 256];
        blob[..40].copy_from_slice(&create_valid_header());

        let header = FdtHeader::parse(&blob).unwrap();

        assert_eq!(header.magic, FDT_MAGIC);
        assert_eq!(header.total_size, 256);
        assert_eq!(header.off_dt_struct, 40);
        assert_eq!(header.off_dt_strings, 200);
        assert_eq!(header.version, 17);
    }

    #[test]
    fn test_parse_too_small() {
        let blob = [0u8; 20];
        let result = FdtHeader::parse(&blob);
        assert!(matches!(result, Err(DtbError::BlobTooSmall { .. })));
    }

    #[test]
    fn test_parse_invalid_magic() {
        let mut blob = [0u8; 256];
        blob[..40].copy_from_slice(&create_valid_header());
        blob[0..4].copy_from_slice(&0xDEAD_BEEF_u32.to_be_bytes());

        let result = FdtHeader::parse(&blob);
        assert!(matches!(result, Err(DtbError::InvalidMagic { .. })));
    }

    #[test]
    fn test_parse_invalid_version() {
        let mut blob = [0u8; 256];
        blob[..40].copy_from_slice(&create_valid_header());
        blob[20..24].copy_from_slice(&1_u32.to_be_bytes()); // Version 1 not supported

        let result = FdtHeader::parse(&blob);
        assert!(matches!(result, Err(DtbError::UnsupportedVersion { .. })));
    }

    #[test]
    fn test_parse_invalid_total_size() {
        let mut blob = [0u8; 256];
        blob[..40].copy_from_slice(&create_valid_header());
        blob[4..8].copy_from_slice(&1000_u32.to_be_bytes()); // Larger than blob

        let result = FdtHeader::parse(&blob);
        assert!(matches!(result, Err(DtbError::InvalidTotalSize { .. })));
    }

    #[test]
    fn test_structure_block() {
        let mut blob = [0u8; 256];
        blob[..40].copy_from_slice(&create_valid_header());
        blob[40] = 0xAB;
        blob[199] = 0xCD;

        let header = FdtHeader::parse(&blob).unwrap();
        let structure = header.structure_block(&blob);

        assert_eq!(structure.len(), 160);
        assert_eq!(structure[0], 0xAB);
    }

    #[test]
    fn test_strings_block() {
        let mut blob = [0u8; 256];
        blob[..40].copy_from_slice(&create_valid_header());
        blob[200] = b'h';
        blob[201] = b'i';
        blob[202] = 0;

        let header = FdtHeader::parse(&blob).unwrap();
        let strings = header.strings_block(&blob);

        assert_eq!(strings.len(), 56);
        assert_eq!(strings[0], b'h');
    }

    #[test]
    fn test_get_string() {
        let mut blob = [0u8; 256];
        blob[..40].copy_from_slice(&create_valid_header());
        blob[200..206].copy_from_slice(b"hello\0");

        let header = FdtHeader::parse(&blob).unwrap();
        let s = header.get_string(&blob, 0).unwrap();

        assert_eq!(s, "hello");
    }

    #[test]
    fn test_get_string_invalid_offset() {
        let mut blob = [0u8; 256];
        blob[..40].copy_from_slice(&create_valid_header());

        let header = FdtHeader::parse(&blob).unwrap();
        let result = header.get_string(&blob, 100); // Beyond strings block

        assert!(matches!(
            result,
            Err(DtbError::InvalidPropertyNameOffset { .. })
        ));
    }
}
