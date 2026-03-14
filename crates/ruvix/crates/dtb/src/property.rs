//! FDT property parsing.

use crate::{read_be_u32, read_be_u64, DtbError, DtbResult};

/// A property in the device tree.
#[derive(Debug, Clone, Copy)]
pub struct Property<'a> {
    /// Property name
    name: &'a str,
    /// Property value (raw bytes)
    value: &'a [u8],
}

impl<'a> Property<'a> {
    /// Create a new property.
    #[must_use]
    pub const fn new(name: &'a str, value: &'a [u8]) -> Self {
        Self { name, value }
    }

    /// Get the property name.
    #[must_use]
    pub const fn name(&self) -> &str {
        self.name
    }

    /// Get the raw property value.
    #[must_use]
    pub const fn value(&self) -> &[u8] {
        self.value
    }

    /// Get the value length.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.value.len()
    }

    /// Check if the value is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.value.is_empty()
    }

    /// Check if this is an empty (boolean-style) property.
    #[must_use]
    pub const fn is_boolean(&self) -> bool {
        self.value.is_empty()
    }

    /// Get the value as a u32.
    ///
    /// # Errors
    ///
    /// Returns `DtbError::InvalidPropertyFormat` if the value is not 4 bytes.
    pub fn as_u32(&self) -> DtbResult<u32> {
        if self.value.len() != 4 {
            return Err(DtbError::InvalidPropertyFormat);
        }
        read_be_u32(self.value, 0).ok_or(DtbError::InvalidPropertyFormat)
    }

    /// Get the value as a u64.
    ///
    /// # Errors
    ///
    /// Returns `DtbError::InvalidPropertyFormat` if the value is not 8 bytes.
    pub fn as_u64(&self) -> DtbResult<u64> {
        if self.value.len() != 8 {
            return Err(DtbError::InvalidPropertyFormat);
        }
        read_be_u64(self.value, 0).ok_or(DtbError::InvalidPropertyFormat)
    }

    /// Get the value as a string.
    ///
    /// # Errors
    ///
    /// Returns `DtbError::InvalidPropertyFormat` if the value is not valid UTF-8
    /// or doesn't have a null terminator.
    pub fn as_str(&self) -> DtbResult<&'a str> {
        if self.value.is_empty() {
            return Err(DtbError::InvalidPropertyFormat);
        }

        // Check for null terminator
        if self.value[self.value.len() - 1] != 0 {
            return Err(DtbError::InvalidPropertyFormat);
        }

        // Convert to string (without null terminator)
        core::str::from_utf8(&self.value[..self.value.len() - 1])
            .map_err(|_| DtbError::InvalidPropertyFormat)
    }

    /// Get the value as a string list.
    ///
    /// Returns an iterator over null-terminated strings.
    #[must_use]
    pub fn as_stringlist(&self) -> StringListIter<'a> {
        StringListIter::new(self.value)
    }

    /// Parse as a property value enum.
    #[must_use]
    pub fn parse(&self) -> PropertyValue<'a> {
        if self.value.is_empty() {
            return PropertyValue::Empty;
        }

        if self.value.len() == 4 {
            if let Some(v) = read_be_u32(self.value, 0) {
                return PropertyValue::U32(v);
            }
        }

        if self.value.len() == 8 {
            if let Some(v) = read_be_u64(self.value, 0) {
                return PropertyValue::U64(v);
            }
        }

        // Check if it's a string (ends with null, printable)
        if self.value[self.value.len() - 1] == 0 {
            if let Ok(s) = self.as_str() {
                if s.chars().all(|c| c.is_ascii_graphic() || c == ' ') {
                    return PropertyValue::String(s);
                }
            }
        }

        PropertyValue::Bytes(self.value)
    }
}

/// Parsed property value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PropertyValue<'a> {
    /// Empty (boolean-style) property
    Empty,
    /// 32-bit unsigned integer
    U32(u32),
    /// 64-bit unsigned integer
    U64(u64),
    /// Null-terminated string
    String(&'a str),
    /// Raw bytes
    Bytes(&'a [u8]),
}

/// Iterator over strings in a stringlist property.
#[derive(Debug, Clone)]
pub struct StringListIter<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> StringListIter<'a> {
    /// Create a new string list iterator.
    #[must_use]
    pub const fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }
}

impl<'a> Iterator for StringListIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.data.len() {
            return None;
        }

        // Find next null terminator
        let remaining = &self.data[self.offset..];
        let end = remaining.iter().position(|&b| b == 0)?;

        let s = core::str::from_utf8(&remaining[..end]).ok()?;
        self.offset += end + 1;

        Some(s)
    }
}

/// A reg property entry (address + size pair).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegEntry {
    /// Base address
    pub address: u64,
    /// Region size
    pub size: u64,
}

impl RegEntry {
    /// Create a new reg entry.
    #[must_use]
    pub const fn new(address: u64, size: u64) -> Self {
        Self { address, size }
    }

    /// Get the end address (exclusive).
    #[must_use]
    pub const fn end(&self) -> u64 {
        self.address + self.size
    }

    /// Check if an address is within this region.
    #[must_use]
    pub const fn contains(&self, addr: u64) -> bool {
        addr >= self.address && addr < self.end()
    }
}

/// Iterator over reg entries.
#[derive(Debug, Clone)]
pub struct RegIter<'a> {
    data: &'a [u8],
    offset: usize,
    address_cells: u32,
    size_cells: u32,
}

impl<'a> RegIter<'a> {
    /// Create a new reg iterator.
    #[must_use]
    pub const fn new(data: &'a [u8], address_cells: u32, size_cells: u32) -> Self {
        Self {
            data,
            offset: 0,
            address_cells,
            size_cells,
        }
    }

    /// Calculate the entry size in bytes.
    const fn entry_size(&self) -> usize {
        ((self.address_cells + self.size_cells) * 4) as usize
    }
}

impl Iterator for RegIter<'_> {
    type Item = RegEntry;

    fn next(&mut self) -> Option<Self::Item> {
        let entry_size = self.entry_size();
        if self.offset + entry_size > self.data.len() {
            return None;
        }

        // Read address (1 or 2 cells)
        let address = if self.address_cells == 2 {
            read_be_u64(self.data, self.offset)?
        } else {
            u64::from(read_be_u32(self.data, self.offset)?)
        };

        let addr_offset = (self.address_cells * 4) as usize;

        // Read size (1 or 2 cells)
        let size = if self.size_cells == 2 {
            read_be_u64(self.data, self.offset + addr_offset)?
        } else if self.size_cells == 1 {
            u64::from(read_be_u32(self.data, self.offset + addr_offset)?)
        } else {
            0 // size_cells == 0 means no size
        };

        self.offset += entry_size;

        Some(RegEntry::new(address, size))
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use std::vec;
    use std::vec::Vec;

    use super::*;

    #[test]
    fn test_property_empty() {
        let prop = Property::new("test", &[]);
        assert!(prop.is_empty());
        assert!(prop.is_boolean());
        assert_eq!(prop.parse(), PropertyValue::Empty);
    }

    #[test]
    fn test_property_u32() {
        let value = 0x1234_5678_u32.to_be_bytes();
        let prop = Property::new("test", &value);

        assert_eq!(prop.as_u32().unwrap(), 0x1234_5678);
        assert_eq!(prop.parse(), PropertyValue::U32(0x1234_5678));
    }

    #[test]
    fn test_property_u64() {
        let value = 0x1234_5678_9ABC_DEF0_u64.to_be_bytes();
        let prop = Property::new("test", &value);

        assert_eq!(prop.as_u64().unwrap(), 0x1234_5678_9ABC_DEF0);
        assert_eq!(prop.parse(), PropertyValue::U64(0x1234_5678_9ABC_DEF0));
    }

    #[test]
    fn test_property_string() {
        let value = b"hello\0";
        let prop = Property::new("test", value);

        assert_eq!(prop.as_str().unwrap(), "hello");
        assert_eq!(prop.parse(), PropertyValue::String("hello"));
    }

    #[test]
    fn test_property_stringlist() {
        let value = b"one\0two\0three\0";
        let prop = Property::new("compatible", value);

        let strings: Vec<&str> = prop.as_stringlist().collect();
        assert_eq!(strings, vec!["one", "two", "three"]);
    }

    #[test]
    fn test_property_invalid_format() {
        let value = [1, 2, 3]; // Not 4 or 8 bytes
        let prop = Property::new("test", &value);

        assert!(prop.as_u32().is_err());
        assert!(prop.as_u64().is_err());
        assert_eq!(prop.parse(), PropertyValue::Bytes(&[1, 2, 3]));
    }

    #[test]
    fn test_reg_entry() {
        let entry = RegEntry::new(0x8000_0000, 0x1000_0000);

        assert_eq!(entry.address, 0x8000_0000);
        assert_eq!(entry.size, 0x1000_0000);
        assert_eq!(entry.end(), 0x9000_0000);
        assert!(entry.contains(0x8000_0000));
        assert!(entry.contains(0x8FFF_FFFF));
        assert!(!entry.contains(0x9000_0000));
    }

    #[test]
    fn test_reg_iter_32bit() {
        // Two entries: (0x1000, 0x100), (0x2000, 0x200)
        let data = [
            0x00, 0x00, 0x10, 0x00, // addr: 0x1000
            0x00, 0x00, 0x01, 0x00, // size: 0x100
            0x00, 0x00, 0x20, 0x00, // addr: 0x2000
            0x00, 0x00, 0x02, 0x00, // size: 0x200
        ];

        let entries: Vec<RegEntry> = RegIter::new(&data, 1, 1).collect();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0], RegEntry::new(0x1000, 0x100));
        assert_eq!(entries[1], RegEntry::new(0x2000, 0x200));
    }

    #[test]
    fn test_reg_iter_64bit() {
        // One entry with 64-bit address and size
        let data = [
            0x00, 0x00, 0x00, 0x00, // addr high
            0x80, 0x00, 0x00, 0x00, // addr low: 0x80000000
            0x00, 0x00, 0x00, 0x00, // size high
            0x10, 0x00, 0x00, 0x00, // size low: 0x10000000
        ];

        let entries: Vec<RegEntry> = RegIter::new(&data, 2, 2).collect();

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], RegEntry::new(0x8000_0000, 0x1000_0000));
    }

    #[cfg(feature = "alloc")]
    extern crate alloc;
    #[cfg(feature = "alloc")]
    use alloc::vec::Vec;
}
