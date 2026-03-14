//! Block device abstraction for RuVix filesystems.
//!
//! This module provides the `BlockDevice` trait which abstracts over
//! different storage backends (memory, disk, etc.) for filesystem implementations.

use crate::error::{FsError, FsResult};

/// Block device trait for storage abstraction.
///
/// This trait provides a hardware-independent interface for reading and writing
/// fixed-size blocks of data. It is used by filesystem implementations to
/// access the underlying storage medium.
///
/// # Block Addressing
///
/// Blocks are addressed using logical block addresses (LBAs) starting from 0.
/// The block size is determined by the implementation (typically 512 or 4096 bytes).
pub trait BlockDevice {
    /// Read a block from the device into the buffer.
    ///
    /// # Arguments
    ///
    /// * `lba` - Logical block address to read from
    /// * `buf` - Buffer to read into (must be exactly `block_size()` bytes)
    ///
    /// # Errors
    ///
    /// Returns `FsError::BlockDeviceError` on read failure or invalid LBA.
    /// Returns `FsError::InvalidArgument` if buffer size doesn't match block size.
    fn read_block(&self, lba: u64, buf: &mut [u8]) -> FsResult<()>;

    /// Write a block to the device from the buffer.
    ///
    /// # Arguments
    ///
    /// * `lba` - Logical block address to write to
    /// * `buf` - Buffer to write from (must be exactly `block_size()` bytes)
    ///
    /// # Errors
    ///
    /// Returns `FsError::BlockDeviceError` on write failure or invalid LBA.
    /// Returns `FsError::InvalidArgument` if buffer size doesn't match block size.
    /// Returns `FsError::ReadOnly` if the device is read-only.
    fn write_block(&self, lba: u64, buf: &[u8]) -> FsResult<()>;

    /// Returns the block size in bytes.
    ///
    /// Common values are 512 (traditional disk sectors) or 4096 (modern SSDs).
    fn block_size(&self) -> usize;

    /// Returns the total number of blocks on the device.
    fn block_count(&self) -> u64;

    /// Returns whether the device is read-only.
    fn is_read_only(&self) -> bool {
        false
    }

    /// Sync any cached writes to the device.
    ///
    /// This should be called periodically to ensure data durability.
    fn sync(&self) -> FsResult<()> {
        Ok(())
    }

    /// Read multiple contiguous blocks.
    ///
    /// # Arguments
    ///
    /// * `start_lba` - Starting logical block address
    /// * `buf` - Buffer to read into (must be exactly `count * block_size()` bytes)
    /// * `count` - Number of blocks to read
    ///
    /// Default implementation reads blocks one at a time.
    fn read_blocks(&self, start_lba: u64, buf: &mut [u8], count: usize) -> FsResult<()> {
        let block_size = self.block_size();
        if buf.len() != count * block_size {
            return Err(FsError::InvalidArgument);
        }

        for i in 0..count {
            let offset = i * block_size;
            self.read_block(start_lba + i as u64, &mut buf[offset..offset + block_size])?;
        }
        Ok(())
    }

    /// Write multiple contiguous blocks.
    ///
    /// # Arguments
    ///
    /// * `start_lba` - Starting logical block address
    /// * `buf` - Buffer to write from (must be exactly `count * block_size()` bytes)
    /// * `count` - Number of blocks to write
    ///
    /// Default implementation writes blocks one at a time.
    fn write_blocks(&self, start_lba: u64, buf: &[u8], count: usize) -> FsResult<()> {
        let block_size = self.block_size();
        if buf.len() != count * block_size {
            return Err(FsError::InvalidArgument);
        }

        for i in 0..count {
            let offset = i * block_size;
            self.write_block(start_lba + i as u64, &buf[offset..offset + block_size])?;
        }
        Ok(())
    }
}

/// A null block device that always returns zeros on read and discards writes.
///
/// Useful for testing or as a placeholder.
#[derive(Debug, Clone)]
pub struct NullBlockDevice {
    block_size: usize,
    block_count: u64,
    read_only: bool,
}

impl NullBlockDevice {
    /// Create a new null block device.
    ///
    /// # Arguments
    ///
    /// * `block_size` - Size of each block in bytes
    /// * `block_count` - Total number of blocks
    #[must_use]
    pub const fn new(block_size: usize, block_count: u64) -> Self {
        Self {
            block_size,
            block_count,
            read_only: false,
        }
    }

    /// Create a read-only null block device.
    #[must_use]
    pub const fn new_read_only(block_size: usize, block_count: u64) -> Self {
        Self {
            block_size,
            block_count,
            read_only: true,
        }
    }
}

impl Default for NullBlockDevice {
    fn default() -> Self {
        Self::new(512, 2048) // 1MB default
    }
}

impl BlockDevice for NullBlockDevice {
    fn read_block(&self, lba: u64, buf: &mut [u8]) -> FsResult<()> {
        if lba >= self.block_count {
            return Err(FsError::BlockDeviceError);
        }
        if buf.len() != self.block_size {
            return Err(FsError::InvalidArgument);
        }
        buf.fill(0);
        Ok(())
    }

    fn write_block(&self, lba: u64, buf: &[u8]) -> FsResult<()> {
        if self.read_only {
            return Err(FsError::ReadOnly);
        }
        if lba >= self.block_count {
            return Err(FsError::BlockDeviceError);
        }
        if buf.len() != self.block_size {
            return Err(FsError::InvalidArgument);
        }
        // Null device discards writes
        Ok(())
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    fn block_count(&self) -> u64 {
        self.block_count
    }

    fn is_read_only(&self) -> bool {
        self.read_only
    }
}

/// A memory-backed block device for testing and RamFS.
///
/// This block device stores all data in a contiguous memory buffer,
/// making it suitable for testing and in-memory filesystems.
#[cfg(feature = "alloc")]
#[derive(Debug)]
pub struct MemoryBlockDevice {
    data: alloc::vec::Vec<u8>,
    block_size: usize,
    read_only: bool,
}

#[cfg(feature = "alloc")]
impl MemoryBlockDevice {
    /// Create a new memory block device with the specified size.
    ///
    /// # Arguments
    ///
    /// * `block_size` - Size of each block in bytes
    /// * `block_count` - Total number of blocks
    ///
    /// # Panics
    ///
    /// Panics if the total size exceeds available memory.
    #[must_use]
    pub fn new(block_size: usize, block_count: u64) -> Self {
        let total_size = block_size * block_count as usize;
        Self {
            data: alloc::vec![0u8; total_size],
            block_size,
            read_only: false,
        }
    }

    /// Create a memory block device from existing data.
    ///
    /// # Arguments
    ///
    /// * `data` - Existing data buffer
    /// * `block_size` - Size of each block in bytes
    ///
    /// # Panics
    ///
    /// Panics if data length is not a multiple of block size.
    #[must_use]
    pub fn from_data(data: alloc::vec::Vec<u8>, block_size: usize) -> Self {
        assert!(
            data.len() % block_size == 0,
            "Data length must be a multiple of block size"
        );
        Self {
            data,
            block_size,
            read_only: false,
        }
    }

    /// Create a read-only memory block device from existing data.
    #[must_use]
    pub fn from_data_read_only(data: alloc::vec::Vec<u8>, block_size: usize) -> Self {
        assert!(
            data.len() % block_size == 0,
            "Data length must be a multiple of block size"
        );
        Self {
            data,
            block_size,
            read_only: true,
        }
    }

    /// Get a reference to the underlying data.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get a mutable reference to the underlying data.
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Set whether the device is read-only.
    pub fn set_read_only(&mut self, read_only: bool) {
        self.read_only = read_only;
    }
}

#[cfg(feature = "alloc")]
impl BlockDevice for MemoryBlockDevice {
    fn read_block(&self, lba: u64, buf: &mut [u8]) -> FsResult<()> {
        if buf.len() != self.block_size {
            return Err(FsError::InvalidArgument);
        }

        let offset = lba as usize * self.block_size;
        if offset + self.block_size > self.data.len() {
            return Err(FsError::BlockDeviceError);
        }

        buf.copy_from_slice(&self.data[offset..offset + self.block_size]);
        Ok(())
    }

    fn write_block(&self, _lba: u64, _buf: &[u8]) -> FsResult<()> {
        if self.read_only {
            return Err(FsError::ReadOnly);
        }
        // Note: This requires interior mutability for a proper implementation.
        // For simplicity, we return ReadOnly in the trait implementation.
        // Use MemoryBlockDeviceMut for writable operations.
        Err(FsError::ReadOnly)
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    fn block_count(&self) -> u64 {
        (self.data.len() / self.block_size) as u64
    }

    fn is_read_only(&self) -> bool {
        self.read_only
    }
}

/// A mutable memory-backed block device using RefCell for interior mutability.
#[cfg(feature = "alloc")]
#[derive(Debug)]
pub struct MemoryBlockDeviceMut {
    data: core::cell::RefCell<alloc::vec::Vec<u8>>,
    block_size: usize,
    read_only: bool,
}

#[cfg(feature = "alloc")]
impl MemoryBlockDeviceMut {
    /// Create a new mutable memory block device.
    #[must_use]
    pub fn new(block_size: usize, block_count: u64) -> Self {
        let total_size = block_size * block_count as usize;
        Self {
            data: core::cell::RefCell::new(alloc::vec![0u8; total_size]),
            block_size,
            read_only: false,
        }
    }

    /// Create from existing data.
    #[must_use]
    pub fn from_data(data: alloc::vec::Vec<u8>, block_size: usize) -> Self {
        assert!(
            data.len() % block_size == 0,
            "Data length must be a multiple of block size"
        );
        Self {
            data: core::cell::RefCell::new(data),
            block_size,
            read_only: false,
        }
    }

    /// Set read-only mode.
    pub fn set_read_only(&mut self, read_only: bool) {
        self.read_only = read_only;
    }

    /// Get a copy of the underlying data.
    #[must_use]
    pub fn data(&self) -> alloc::vec::Vec<u8> {
        self.data.borrow().clone()
    }
}

#[cfg(feature = "alloc")]
impl BlockDevice for MemoryBlockDeviceMut {
    fn read_block(&self, lba: u64, buf: &mut [u8]) -> FsResult<()> {
        if buf.len() != self.block_size {
            return Err(FsError::InvalidArgument);
        }

        let data = self.data.borrow();
        let offset = lba as usize * self.block_size;
        if offset + self.block_size > data.len() {
            return Err(FsError::BlockDeviceError);
        }

        buf.copy_from_slice(&data[offset..offset + self.block_size]);
        Ok(())
    }

    fn write_block(&self, lba: u64, buf: &[u8]) -> FsResult<()> {
        if self.read_only {
            return Err(FsError::ReadOnly);
        }
        if buf.len() != self.block_size {
            return Err(FsError::InvalidArgument);
        }

        let mut data = self.data.borrow_mut();
        let offset = lba as usize * self.block_size;
        if offset + self.block_size > data.len() {
            return Err(FsError::BlockDeviceError);
        }

        data[offset..offset + self.block_size].copy_from_slice(buf);
        Ok(())
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    fn block_count(&self) -> u64 {
        (self.data.borrow().len() / self.block_size) as u64
    }

    fn is_read_only(&self) -> bool {
        self.read_only
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_device_read() {
        let dev = NullBlockDevice::new(512, 100);
        let mut buf = [0xFFu8; 512];
        assert!(dev.read_block(0, &mut buf).is_ok());
        assert!(buf.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_null_device_write() {
        let dev = NullBlockDevice::new(512, 100);
        let buf = [0xAAu8; 512];
        assert!(dev.write_block(0, &buf).is_ok());
    }

    #[test]
    fn test_null_device_read_only() {
        let dev = NullBlockDevice::new_read_only(512, 100);
        let buf = [0xAAu8; 512];
        assert_eq!(dev.write_block(0, &buf), Err(FsError::ReadOnly));
    }

    #[test]
    fn test_null_device_invalid_lba() {
        let dev = NullBlockDevice::new(512, 100);
        let mut buf = [0u8; 512];
        assert_eq!(dev.read_block(100, &mut buf), Err(FsError::BlockDeviceError));
        assert_eq!(dev.read_block(101, &mut buf), Err(FsError::BlockDeviceError));
    }

    #[test]
    fn test_null_device_invalid_buffer_size() {
        let dev = NullBlockDevice::new(512, 100);
        let mut small_buf = [0u8; 256];
        let mut large_buf = [0u8; 1024];
        assert_eq!(dev.read_block(0, &mut small_buf), Err(FsError::InvalidArgument));
        assert_eq!(dev.read_block(0, &mut large_buf), Err(FsError::InvalidArgument));
    }

    #[test]
    fn test_null_device_properties() {
        let dev = NullBlockDevice::new(4096, 1000);
        assert_eq!(dev.block_size(), 4096);
        assert_eq!(dev.block_count(), 1000);
        assert!(!dev.is_read_only());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_memory_device_read_write() {
        let dev = MemoryBlockDeviceMut::new(512, 10);

        // Write some data
        let write_buf = [0xABu8; 512];
        assert!(dev.write_block(5, &write_buf).is_ok());

        // Read it back
        let mut read_buf = [0u8; 512];
        assert!(dev.read_block(5, &mut read_buf).is_ok());
        assert_eq!(read_buf, write_buf);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_memory_device_read_blocks() {
        let dev = MemoryBlockDeviceMut::new(512, 10);

        // Write to blocks 2 and 3
        let buf2 = [0x11u8; 512];
        let buf3 = [0x22u8; 512];
        dev.write_block(2, &buf2).unwrap();
        dev.write_block(3, &buf3).unwrap();

        // Read both blocks at once
        let mut read_buf = [0u8; 1024];
        assert!(dev.read_blocks(2, &mut read_buf, 2).is_ok());
        assert_eq!(&read_buf[..512], &buf2);
        assert_eq!(&read_buf[512..], &buf3);
    }

    #[test]
    fn test_default_null_device() {
        let dev = NullBlockDevice::default();
        assert_eq!(dev.block_size(), 512);
        assert_eq!(dev.block_count(), 2048);
    }
}
