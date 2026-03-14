//! # RuVix Device Tree Blob Parser
//!
//! This crate provides a zero-copy parser for Flattened Device Tree (FDT) blobs.
//! It is designed for the RuVix Cognition Kernel to discover hardware configuration
//! at boot time.
//!
//! ## Design Principles
//!
//! - **No std dependency** - `#![no_std]` only
//! - **Zero-copy parsing** - References directly into the FDT blob
//! - **No allocations** - Works without heap allocation
//! - **Safe parsing** - Validates all data before access
//!
//! ## FDT Structure
//!
//! A Flattened Device Tree consists of:
//! - Header with magic number, version, and offsets
//! - Memory reservation block
//! - Structure block (nodes and properties)
//! - Strings block (property names)
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvix_dtb::DeviceTree;
//!
//! fn parse_dtb(blob: &[u8]) -> Result<(), ruvix_dtb::DtbError> {
//!     let dt = DeviceTree::parse(blob)?;
//!
//!     // Find the memory node
//!     if let Some(memory) = dt.find_node("/memory") {
//!         // Get the reg property (address and size)
//!         if let Some(reg) = dt.get_reg(memory)? {
//!             for (addr, size) in reg {
//!                 println!("Memory region: 0x{:x} size: 0x{:x}", addr, size);
//!             }
//!         }
//!     }
//!
//!     // Find all UART nodes
//!     for node in dt.find_compatible("ns16550a") {
//!         let addr = dt.get_u64(node, "reg")?;
//!         println!("UART at 0x{:x}", addr);
//!     }
//!
//!     Ok(())
//! }
//! ```

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

mod error;
mod header;
mod node;
mod parser;
mod property;

pub use error::{DtbError, DtbResult};
pub use header::FdtHeader;
pub use node::{Node, NodeIter, ParsedReg};
pub use parser::DeviceTree;
pub use property::{Property, PropertyValue, RegEntry, RegIter};

/// FDT magic number (big-endian: 0xd00dfeed)
pub const FDT_MAGIC: u32 = 0xd00d_feed;

/// FDT version we support
pub const FDT_VERSION: u32 = 17;

/// Minimum FDT version we support
pub const FDT_MIN_VERSION: u32 = 16;

/// FDT structure block tokens
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum FdtToken {
    /// Begin node token (FDT_BEGIN_NODE)
    BeginNode = 0x0000_0001,
    /// End node token (FDT_END_NODE)
    EndNode = 0x0000_0002,
    /// Property token (FDT_PROP)
    Prop = 0x0000_0003,
    /// NOP token (FDT_NOP)
    Nop = 0x0000_0004,
    /// End of structure block (FDT_END)
    End = 0x0000_0009,
}

impl FdtToken {
    /// Parse a token from a u32 value
    #[must_use]
    pub const fn from_u32(value: u32) -> Option<Self> {
        match value {
            0x0000_0001 => Some(Self::BeginNode),
            0x0000_0002 => Some(Self::EndNode),
            0x0000_0003 => Some(Self::Prop),
            0x0000_0004 => Some(Self::Nop),
            0x0000_0009 => Some(Self::End),
            _ => None,
        }
    }
}

/// DTB crate version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Maximum supported tree depth (for iteration without recursion)
pub const MAX_TREE_DEPTH: usize = 32;

/// Maximum property name length
pub const MAX_PROPERTY_NAME_LEN: usize = 64;

/// Maximum path length
pub const MAX_PATH_LEN: usize = 256;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::{DeviceTree, DtbError, DtbResult, Node, Property, RegEntry};
}

/// Helper function to read a big-endian u32 from a byte slice
#[inline]
#[must_use]
pub const fn read_be_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    if offset + 4 > bytes.len() {
        return None;
    }
    Some(
        ((bytes[offset] as u32) << 24)
            | ((bytes[offset + 1] as u32) << 16)
            | ((bytes[offset + 2] as u32) << 8)
            | (bytes[offset + 3] as u32),
    )
}

/// Helper function to read a big-endian u64 from a byte slice
#[inline]
#[must_use]
pub const fn read_be_u64(bytes: &[u8], offset: usize) -> Option<u64> {
    if offset + 8 > bytes.len() {
        return None;
    }
    let high = match read_be_u32(bytes, offset) {
        Some(v) => v as u64,
        None => return None,
    };
    let low = match read_be_u32(bytes, offset + 4) {
        Some(v) => v as u64,
        None => return None,
    };
    Some((high << 32) | low)
}

/// Align a value up to 4-byte boundary
#[inline]
#[must_use]
pub const fn align4(value: usize) -> usize {
    (value + 3) & !3
}

/// Find the length of a null-terminated string in a byte slice
#[must_use]
pub fn strlen(bytes: &[u8], offset: usize) -> Option<usize> {
    let slice = bytes.get(offset..)?;
    slice.iter().position(|&b| b == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fdt_token_parsing() {
        assert_eq!(FdtToken::from_u32(1), Some(FdtToken::BeginNode));
        assert_eq!(FdtToken::from_u32(2), Some(FdtToken::EndNode));
        assert_eq!(FdtToken::from_u32(3), Some(FdtToken::Prop));
        assert_eq!(FdtToken::from_u32(4), Some(FdtToken::Nop));
        assert_eq!(FdtToken::from_u32(9), Some(FdtToken::End));
        assert_eq!(FdtToken::from_u32(0), None);
        assert_eq!(FdtToken::from_u32(5), None);
    }

    #[test]
    fn test_read_be_u32() {
        let bytes = [0xDE, 0xAD, 0xBE, 0xEF, 0x12, 0x34];

        assert_eq!(read_be_u32(&bytes, 0), Some(0xDEAD_BEEF));
        assert_eq!(read_be_u32(&bytes, 2), Some(0xBEEF_1234));
        assert_eq!(read_be_u32(&bytes, 3), None);
    }

    #[test]
    fn test_read_be_u64() {
        let bytes = [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF];

        assert_eq!(read_be_u64(&bytes, 0), Some(0x0123_4567_89AB_CDEF));
        assert_eq!(read_be_u64(&bytes, 1), None);
    }

    #[test]
    fn test_align4() {
        assert_eq!(align4(0), 0);
        assert_eq!(align4(1), 4);
        assert_eq!(align4(2), 4);
        assert_eq!(align4(3), 4);
        assert_eq!(align4(4), 4);
        assert_eq!(align4(5), 8);
    }

    #[test]
    fn test_strlen() {
        let bytes = [b'h', b'e', b'l', b'l', b'o', 0, b'x'];

        assert_eq!(strlen(&bytes, 0), Some(5));
        assert_eq!(strlen(&bytes, 2), Some(3));
        assert_eq!(strlen(&bytes, 5), Some(0));
        assert_eq!(strlen(&bytes, 6), None); // No null terminator
        assert_eq!(strlen(&bytes, 100), None); // Out of bounds
    }
}
