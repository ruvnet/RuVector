//! DTB error types and result alias.

use core::fmt;

/// Result type alias for DTB operations.
pub type DtbResult<T> = Result<T, DtbError>;

/// Error type for DTB parsing and access operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DtbError {
    /// Invalid FDT magic number
    InvalidMagic {
        /// The magic number found
        found: u32,
    },
    /// Unsupported FDT version
    UnsupportedVersion {
        /// The version found
        version: u32,
    },
    /// Blob is too small to contain header
    BlobTooSmall {
        /// Actual size
        size: usize,
        /// Minimum required size
        min_size: usize,
    },
    /// Header indicates size larger than blob
    InvalidTotalSize {
        /// Size from header
        header_size: u32,
        /// Actual blob size
        blob_size: usize,
    },
    /// Structure block offset is invalid
    InvalidStructureOffset {
        /// The invalid offset
        offset: u32,
    },
    /// Strings block offset is invalid
    InvalidStringsOffset {
        /// The invalid offset
        offset: u32,
    },
    /// Memory reservation block offset is invalid
    InvalidReservationOffset {
        /// The invalid offset
        offset: u32,
    },
    /// Unexpected token encountered
    UnexpectedToken {
        /// Expected token
        expected: u32,
        /// Found token
        found: u32,
    },
    /// Invalid token value
    InvalidToken {
        /// The invalid token value
        value: u32,
    },
    /// Property name offset is out of bounds
    InvalidPropertyNameOffset {
        /// The invalid offset
        offset: u32,
    },
    /// Property value length exceeds available data
    InvalidPropertyLength {
        /// Property length from header
        length: u32,
        /// Available bytes
        available: usize,
    },
    /// Node path not found
    NodeNotFound,
    /// Property not found
    PropertyNotFound,
    /// Invalid property value format
    InvalidPropertyFormat,
    /// Path is too long
    PathTooLong {
        /// Path length
        length: usize,
    },
    /// Tree depth exceeds maximum
    TreeTooDeep {
        /// Current depth
        depth: usize,
    },
    /// Unexpected end of structure block
    UnexpectedEnd,
    /// Invalid node name (missing null terminator)
    InvalidNodeName,
    /// Invalid string (missing null terminator)
    InvalidString,
    /// Integer overflow in offset calculation (CVE-003 protection)
    IntegerOverflow,
}

impl DtbError {
    /// Create an invalid magic error
    #[must_use]
    pub const fn invalid_magic(found: u32) -> Self {
        Self::InvalidMagic { found }
    }

    /// Create an unsupported version error
    #[must_use]
    pub const fn unsupported_version(version: u32) -> Self {
        Self::UnsupportedVersion { version }
    }

    /// Create a blob too small error
    #[must_use]
    pub const fn blob_too_small(size: usize, min_size: usize) -> Self {
        Self::BlobTooSmall { size, min_size }
    }

    /// Create an invalid total size error
    #[must_use]
    pub const fn invalid_total_size(header_size: u32, blob_size: usize) -> Self {
        Self::InvalidTotalSize {
            header_size,
            blob_size,
        }
    }

    /// Check if this is a "not found" error
    #[must_use]
    pub const fn is_not_found(&self) -> bool {
        matches!(self, Self::NodeNotFound | Self::PropertyNotFound)
    }

    /// Check if this is a validation error
    #[must_use]
    pub const fn is_validation_error(&self) -> bool {
        matches!(
            self,
            Self::InvalidMagic { .. }
                | Self::UnsupportedVersion { .. }
                | Self::BlobTooSmall { .. }
                | Self::InvalidTotalSize { .. }
        )
    }

    /// Check if this is a parse error
    #[must_use]
    pub const fn is_parse_error(&self) -> bool {
        matches!(
            self,
            Self::UnexpectedToken { .. }
                | Self::InvalidToken { .. }
                | Self::InvalidPropertyNameOffset { .. }
                | Self::InvalidPropertyLength { .. }
                | Self::UnexpectedEnd
                | Self::InvalidNodeName
                | Self::InvalidString
        )
    }
}

impl fmt::Display for DtbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMagic { found } => {
                write!(f, "Invalid FDT magic: 0x{found:08x}, expected 0xd00dfeed")
            }
            Self::UnsupportedVersion { version } => {
                write!(f, "Unsupported FDT version: {version}")
            }
            Self::BlobTooSmall { size, min_size } => {
                write!(f, "DTB blob too small: {size} bytes, need at least {min_size}")
            }
            Self::InvalidTotalSize {
                header_size,
                blob_size,
            } => {
                write!(
                    f,
                    "Invalid total size: header says {header_size}, blob is {blob_size} bytes"
                )
            }
            Self::InvalidStructureOffset { offset } => {
                write!(f, "Invalid structure block offset: 0x{offset:x}")
            }
            Self::InvalidStringsOffset { offset } => {
                write!(f, "Invalid strings block offset: 0x{offset:x}")
            }
            Self::InvalidReservationOffset { offset } => {
                write!(f, "Invalid memory reservation offset: 0x{offset:x}")
            }
            Self::UnexpectedToken { expected, found } => {
                write!(
                    f,
                    "Unexpected token: expected 0x{expected:x}, found 0x{found:x}"
                )
            }
            Self::InvalidToken { value } => {
                write!(f, "Invalid token value: 0x{value:x}")
            }
            Self::InvalidPropertyNameOffset { offset } => {
                write!(f, "Invalid property name offset: 0x{offset:x}")
            }
            Self::InvalidPropertyLength { length, available } => {
                write!(
                    f,
                    "Invalid property length: {length} bytes, only {available} available"
                )
            }
            Self::NodeNotFound => write!(f, "Node not found"),
            Self::PropertyNotFound => write!(f, "Property not found"),
            Self::InvalidPropertyFormat => write!(f, "Invalid property format"),
            Self::PathTooLong { length } => {
                write!(f, "Path too long: {length} bytes")
            }
            Self::TreeTooDeep { depth } => {
                write!(f, "Tree too deep: depth {depth}")
            }
            Self::UnexpectedEnd => write!(f, "Unexpected end of structure block"),
            Self::InvalidNodeName => write!(f, "Invalid node name (missing null terminator)"),
            Self::InvalidString => write!(f, "Invalid string (missing null terminator)"),
            Self::IntegerOverflow => write!(f, "Integer overflow in offset calculation"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = DtbError::invalid_magic(0x12345678);
        if let DtbError::InvalidMagic { found } = err {
            assert_eq!(found, 0x12345678);
        } else {
            panic!("Wrong error type");
        }
    }

    #[test]
    fn test_is_not_found() {
        assert!(DtbError::NodeNotFound.is_not_found());
        assert!(DtbError::PropertyNotFound.is_not_found());
        assert!(!DtbError::InvalidMagic { found: 0 }.is_not_found());
    }

    #[test]
    fn test_is_validation_error() {
        assert!(DtbError::InvalidMagic { found: 0 }.is_validation_error());
        assert!(DtbError::UnsupportedVersion { version: 0 }.is_validation_error());
        assert!(!DtbError::NodeNotFound.is_validation_error());
    }

    #[test]
    fn test_is_parse_error() {
        assert!(DtbError::UnexpectedEnd.is_parse_error());
        assert!(DtbError::InvalidNodeName.is_parse_error());
        assert!(!DtbError::NodeNotFound.is_parse_error());
    }

    #[test]
    fn test_error_display() {
        extern crate std;
        let err = DtbError::invalid_magic(0xDEADBEEF);
        let msg = std::format!("{}", err);
        assert!(msg.contains("deadbeef"));
    }
}
