//! # Device Tree Blob (DTB) Parsing
//!
//! This module provides basic DTB header parsing for early boot.
//! The firmware passes the DTB address in register x0.
//!
//! ## DTB Header Format
//!
//! The DTB header is defined by the DeviceTree specification:
//!
//! ```text
//! Offset  Size  Description
//! 0x00    4     magic (0xD00DFEED)
//! 0x04    4     totalsize
//! 0x08    4     off_dt_struct
//! 0x0C    4     off_dt_strings
//! 0x10    4     off_mem_rsvmap
//! 0x14    4     version
//! 0x18    4     last_comp_version
//! 0x1C    4     boot_cpuid_phys
//! 0x20    4     size_dt_strings
//! 0x24    4     size_dt_struct
//! ```
//!
//! ## Example
//!
//! ```rust,no_run
//! use ruvix_rpi_boot::dtb::{parse_dtb_header, DtbInfo};
//!
//! let dtb_addr = 0x100000; // Passed in x0
//! if let Some(info) = unsafe { parse_dtb_header(dtb_addr) } {
//!     println!("DTB size: {} bytes", info.total_size);
//!     println!("DTB version: {}", info.version);
//! }
//! ```

/// DTB magic number (big-endian: 0xD00DFEED).
pub const DTB_MAGIC: u32 = 0xD00DFEED;

/// Minimum valid DTB size.
pub const MIN_DTB_SIZE: usize = 40;

/// Maximum valid DTB size (16 MB should be plenty).
pub const MAX_DTB_SIZE: usize = 16 * 1024 * 1024;

/// DTB header structure (as stored in memory, big-endian).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DtbHeader {
    /// Magic number (0xD00DFEED).
    pub magic: u32,
    /// Total size of the DTB.
    pub totalsize: u32,
    /// Offset to structure block.
    pub off_dt_struct: u32,
    /// Offset to strings block.
    pub off_dt_strings: u32,
    /// Offset to memory reservation block.
    pub off_mem_rsvmap: u32,
    /// Version of the DTB format.
    pub version: u32,
    /// Last compatible version.
    pub last_comp_version: u32,
    /// Boot CPU physical ID.
    pub boot_cpuid_phys: u32,
    /// Size of strings block.
    pub size_dt_strings: u32,
    /// Size of structure block.
    pub size_dt_struct: u32,
}

impl DtbHeader {
    /// Validate the DTB header.
    pub fn is_valid(&self) -> bool {
        // Check magic (needs byte swap from big-endian)
        if swap32(self.magic) != DTB_MAGIC {
            return false;
        }

        let total_size = swap32(self.totalsize) as usize;

        // Check size bounds
        if total_size < MIN_DTB_SIZE || total_size > MAX_DTB_SIZE {
            return false;
        }

        // Check version (should be >= 16 for modern DTBs)
        let version = swap32(self.version);
        if version < 1 || version > 100 {
            return false;
        }

        true
    }

    /// Get the total size in native byte order.
    pub fn total_size(&self) -> u32 {
        swap32(self.totalsize)
    }

    /// Get the version in native byte order.
    pub fn version(&self) -> u32 {
        swap32(self.version)
    }

    /// Get the boot CPU ID.
    pub fn boot_cpu(&self) -> u32 {
        swap32(self.boot_cpuid_phys)
    }

    /// Get the structure block offset.
    pub fn struct_offset(&self) -> u32 {
        swap32(self.off_dt_struct)
    }

    /// Get the strings block offset.
    pub fn strings_offset(&self) -> u32 {
        swap32(self.off_dt_strings)
    }
}

/// Parsed DTB information.
#[derive(Debug, Clone, Copy)]
pub struct DtbInfo {
    /// Physical address of the DTB.
    pub address: usize,
    /// Total size in bytes.
    pub total_size: usize,
    /// DTB format version.
    pub version: u32,
    /// Boot CPU physical ID.
    pub boot_cpu: u32,
    /// Offset to structure block.
    pub struct_offset: usize,
    /// Offset to strings block.
    pub strings_offset: usize,
}

/// Byte swap for 32-bit value (big-endian to little-endian).
#[inline]
const fn swap32(value: u32) -> u32 {
    ((value & 0xFF00_0000) >> 24)
        | ((value & 0x00FF_0000) >> 8)
        | ((value & 0x0000_FF00) << 8)
        | ((value & 0x0000_00FF) << 24)
}

/// Parse the DTB header at the given address.
///
/// # Safety
///
/// The address must be a valid pointer to a DTB in memory.
///
/// # Returns
///
/// `Some(DtbInfo)` if the DTB is valid, `None` otherwise.
pub unsafe fn parse_dtb_header(address: usize) -> Option<DtbInfo> {
    if address == 0 {
        return None;
    }

    // Read the header
    let header_ptr = address as *const DtbHeader;
    let header = core::ptr::read_volatile(header_ptr);

    // Validate
    if !header.is_valid() {
        return None;
    }

    Some(DtbInfo {
        address,
        total_size: header.total_size() as usize,
        version: header.version(),
        boot_cpu: header.boot_cpu(),
        struct_offset: header.struct_offset() as usize,
        strings_offset: header.strings_offset() as usize,
    })
}

/// Check if a DTB exists at the given address.
///
/// # Safety
///
/// The address must be readable.
pub unsafe fn is_valid_dtb(address: usize) -> bool {
    if address == 0 {
        return false;
    }

    let magic_ptr = address as *const u32;
    let magic = core::ptr::read_volatile(magic_ptr);

    swap32(magic) == DTB_MAGIC
}

/// FDT (Flattened Device Tree) token types.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FdtToken {
    /// Begin node token.
    BeginNode = 0x0000_0001,
    /// End node token.
    EndNode = 0x0000_0002,
    /// Property token.
    Prop = 0x0000_0003,
    /// NOP token.
    Nop = 0x0000_0004,
    /// End of tree.
    End = 0x0000_0009,
}

impl FdtToken {
    /// Parse a token from a big-endian value.
    pub fn from_be(value: u32) -> Option<Self> {
        match swap32(value) {
            1 => Some(FdtToken::BeginNode),
            2 => Some(FdtToken::EndNode),
            3 => Some(FdtToken::Prop),
            4 => Some(FdtToken::Nop),
            9 => Some(FdtToken::End),
            _ => None,
        }
    }
}

/// Memory reservation entry (from mem_rsvmap).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MemoryReservation {
    /// Start address (big-endian).
    pub address: u64,
    /// Size in bytes (big-endian).
    pub size: u64,
}

impl MemoryReservation {
    /// Check if this is the terminating entry (both zero).
    pub fn is_end(&self) -> bool {
        self.address == 0 && self.size == 0
    }

    /// Get address in native byte order.
    pub fn get_address(&self) -> u64 {
        swap64(self.address)
    }

    /// Get size in native byte order.
    pub fn get_size(&self) -> u64 {
        swap64(self.size)
    }
}

/// Byte swap for 64-bit value.
#[inline]
const fn swap64(value: u64) -> u64 {
    ((value & 0xFF00_0000_0000_0000) >> 56)
        | ((value & 0x00FF_0000_0000_0000) >> 40)
        | ((value & 0x0000_FF00_0000_0000) >> 24)
        | ((value & 0x0000_00FF_0000_0000) >> 8)
        | ((value & 0x0000_0000_FF00_0000) << 8)
        | ((value & 0x0000_0000_00FF_0000) << 24)
        | ((value & 0x0000_0000_0000_FF00) << 40)
        | ((value & 0x0000_0000_0000_00FF) << 56)
}

/// Find a property in the DTB structure block.
///
/// This is a simplified search that looks for properties in the root node.
///
/// # Safety
///
/// The DTB must be valid and the address must be correct.
pub unsafe fn find_property<'a>(dtb_info: &'a DtbInfo, name: &str) -> Option<&'a [u8]> {
    let struct_base = dtb_info.address + dtb_info.struct_offset;
    let strings_base = dtb_info.address + dtb_info.strings_offset;

    let mut offset = 0;
    let max_offset = dtb_info.total_size - dtb_info.struct_offset;

    while offset < max_offset {
        let token_ptr = (struct_base + offset) as *const u32;
        let token_value = core::ptr::read_volatile(token_ptr);

        match FdtToken::from_be(token_value) {
            Some(FdtToken::Prop) => {
                // Property format: token(4) + len(4) + nameoff(4) + data + padding
                let len_ptr = (struct_base + offset + 4) as *const u32;
                let nameoff_ptr = (struct_base + offset + 8) as *const u32;

                let len = swap32(core::ptr::read_volatile(len_ptr)) as usize;
                let nameoff = swap32(core::ptr::read_volatile(nameoff_ptr)) as usize;

                // Get property name from strings block
                let name_ptr = (strings_base + nameoff) as *const u8;
                let prop_name = read_cstring(name_ptr);

                if prop_name == name.as_bytes() {
                    let data_ptr = (struct_base + offset + 12) as *const u8;
                    return Some(core::slice::from_raw_parts(data_ptr, len));
                }

                // Skip to next token (aligned to 4 bytes)
                offset += 12 + ((len + 3) & !3);
            }
            Some(FdtToken::BeginNode) => {
                // Skip node name (null-terminated, aligned)
                offset += 4;
                let name_ptr = (struct_base + offset) as *const u8;
                let name_len = strlen(name_ptr);
                offset += (name_len + 4) & !3;
            }
            Some(FdtToken::EndNode) | Some(FdtToken::Nop) => {
                offset += 4;
            }
            Some(FdtToken::End) | None => {
                break;
            }
        }
    }

    None
}

/// Read a null-terminated string.
unsafe fn read_cstring(ptr: *const u8) -> &'static [u8] {
    let len = strlen(ptr);
    core::slice::from_raw_parts(ptr, len)
}

/// Get length of null-terminated string.
unsafe fn strlen(ptr: *const u8) -> usize {
    let mut len = 0;
    while core::ptr::read_volatile(ptr.add(len)) != 0 {
        len += 1;
        if len > 256 {
            break; // Safety limit
        }
    }
    len
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swap32() {
        // swap32 reverses byte order: 0xD00DFEED -> 0xEDFE0DD0
        assert_eq!(swap32(DTB_MAGIC), 0xEDFE0DD0);
        assert_eq!(swap32(0x12345678), 0x78563412);
    }

    #[test]
    fn test_swap64() {
        assert_eq!(swap64(0x0102030405060708), 0x0807060504030201);
    }

    #[test]
    fn test_fdt_token() {
        assert_eq!(FdtToken::from_be(swap32(1)), Some(FdtToken::BeginNode));
        assert_eq!(FdtToken::from_be(swap32(2)), Some(FdtToken::EndNode));
        assert_eq!(FdtToken::from_be(swap32(3)), Some(FdtToken::Prop));
        assert_eq!(FdtToken::from_be(swap32(9)), Some(FdtToken::End));
        assert_eq!(FdtToken::from_be(swap32(99)), None);
    }
}
