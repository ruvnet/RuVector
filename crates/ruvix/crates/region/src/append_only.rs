//! Append-only region implementation.
//!
//! Contents can only be appended, never overwritten or truncated.
//! A monotonic write cursor tracks the append position.
//!
//! # Design (from ADR-087 Section 4)
//!
//! - Only append, never overwrite or truncated
//! - Monotonic write cursor tracks append position
//! - Max size enforcement
//!
//! Ideal for: witness logs, event streams, time-series vectors.

use crate::backing::MemoryBacking;
use crate::Result;
use ruvix_types::KernelError;

/// An append-only memory region.
///
/// Data can only be appended, never modified or truncated. The write cursor
/// monotonically increases until max_size is reached.
pub struct AppendOnlyRegion<B: MemoryBacking> {
    /// Memory backing store.
    backing: B,
    /// Pointer to the data area.
    data_ptr: *mut u8,
    /// Maximum size in bytes.
    max_size: usize,
    /// Actual allocated size (may be >= max_size due to alignment).
    allocated_size: usize,
    /// Current write cursor position.
    write_cursor: usize,
    /// Region handle for capability checking.
    handle: ruvix_types::RegionHandle,
}

impl<B: MemoryBacking> AppendOnlyRegion<B> {
    /// Creates a new append-only region.
    ///
    /// # Arguments
    ///
    /// * `backing` - Memory backing store
    /// * `max_size` - Maximum size in bytes
    /// * `handle` - Region handle for this region
    ///
    /// # Errors
    ///
    /// Returns `OutOfMemory` if the backing cannot allocate sufficient memory.
    pub fn new(mut backing: B, max_size: usize, handle: ruvix_types::RegionHandle) -> Result<Self> {
        if max_size == 0 {
            return Err(KernelError::InvalidArgument);
        }

        let (data_ptr, allocated_size) = backing.allocate(max_size)?;

        Ok(Self {
            backing,
            data_ptr,
            max_size,
            allocated_size,
            write_cursor: 0,
            handle,
        })
    }

    /// Appends data to the region.
    ///
    /// Returns the offset at which the data was written.
    ///
    /// # Errors
    ///
    /// Returns `RegionFull` if there is not enough space remaining.
    pub fn append(&mut self, data: &[u8]) -> Result<usize> {
        if data.is_empty() {
            return Ok(self.write_cursor);
        }

        let new_cursor = self
            .write_cursor
            .checked_add(data.len())
            .ok_or(KernelError::RegionFull)?;

        if new_cursor > self.max_size {
            return Err(KernelError::RegionFull);
        }

        let offset = self.write_cursor;

        // SAFETY: We've verified bounds above
        unsafe {
            core::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.data_ptr.add(offset),
                data.len(),
            );
        }

        self.write_cursor = new_cursor;
        Ok(offset)
    }

    /// Reads data from the region at the specified offset.
    ///
    /// # Arguments
    ///
    /// * `offset` - Byte offset to start reading from
    /// * `buf` - Buffer to read into
    ///
    /// # Returns
    ///
    /// The number of bytes actually read.
    ///
    /// # Errors
    ///
    /// Returns `InvalidArgument` if offset is beyond the write cursor.
    pub fn read(&self, offset: usize, buf: &mut [u8]) -> Result<usize> {
        if offset >= self.write_cursor {
            return Err(KernelError::InvalidArgument);
        }

        let available = self.write_cursor - offset;
        let to_read = buf.len().min(available);

        // SAFETY: We've verified bounds above
        unsafe {
            core::ptr::copy_nonoverlapping(
                self.data_ptr.add(offset),
                buf.as_mut_ptr(),
                to_read,
            );
        }

        Ok(to_read)
    }

    /// Reads all data from the region.
    ///
    /// # Arguments
    ///
    /// * `buf` - Buffer to read into (must be at least `len()` bytes)
    ///
    /// # Returns
    ///
    /// The number of bytes read.
    pub fn read_all(&self, buf: &mut [u8]) -> Result<usize> {
        self.read(0, buf)
    }

    /// Returns the current write cursor position (amount of data written).
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.write_cursor
    }

    /// Returns true if the region is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.write_cursor == 0
    }

    /// Returns the maximum size in bytes.
    #[inline]
    #[must_use]
    pub const fn max_size(&self) -> usize {
        self.max_size
    }

    /// Returns the remaining space in bytes.
    #[inline]
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.max_size - self.write_cursor
    }

    /// Returns true if the region is full.
    #[inline]
    #[must_use]
    pub const fn is_full(&self) -> bool {
        self.write_cursor >= self.max_size
    }

    /// Returns the fill percentage (0.0 to 1.0).
    #[inline]
    #[must_use]
    pub fn fill_ratio(&self) -> f32 {
        if self.max_size == 0 {
            1.0
        } else {
            self.write_cursor as f32 / self.max_size as f32
        }
    }

    /// Returns the region handle.
    #[inline]
    #[must_use]
    pub const fn handle(&self) -> ruvix_types::RegionHandle {
        self.handle
    }

    /// Returns a slice of the data written so far.
    ///
    /// # Safety
    ///
    /// The returned slice is valid as long as no more appends are made.
    /// In practice, since this is append-only, existing data is never modified.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: We maintain the invariant that write_cursor <= allocated_size
        unsafe { core::slice::from_raw_parts(self.data_ptr, self.write_cursor) }
    }

    /// Appends a u64 value (little-endian).
    pub fn append_u64(&mut self, value: u64) -> Result<usize> {
        self.append(&value.to_le_bytes())
    }

    /// Appends a u32 value (little-endian).
    pub fn append_u32(&mut self, value: u32) -> Result<usize> {
        self.append(&value.to_le_bytes())
    }

    /// Reads a u64 value at the specified offset (little-endian).
    pub fn read_u64(&self, offset: usize) -> Result<u64> {
        let mut buf = [0u8; 8];
        let read = self.read(offset, &mut buf)?;
        if read < 8 {
            return Err(KernelError::BufferTooSmall);
        }
        Ok(u64::from_le_bytes(buf))
    }

    /// Reads a u32 value at the specified offset (little-endian).
    pub fn read_u32(&self, offset: usize) -> Result<u32> {
        let mut buf = [0u8; 4];
        let read = self.read(offset, &mut buf)?;
        if read < 4 {
            return Err(KernelError::BufferTooSmall);
        }
        Ok(u32::from_le_bytes(buf))
    }
}

// SAFETY: AppendOnlyRegion can be sent between threads if its backing can
unsafe impl<B: MemoryBacking + Send> Send for AppendOnlyRegion<B> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backing::StaticBacking;

    #[test]
    fn test_append_only_basic() {
        let backing = StaticBacking::<1024>::new();
        let handle = ruvix_types::RegionHandle::new(1, 0);
        let mut region = AppendOnlyRegion::new(backing, 256, handle).unwrap();

        assert!(region.is_empty());
        assert_eq!(region.max_size(), 256);
        assert_eq!(region.remaining(), 256);

        let data = b"Hello, RuVix!";
        let offset = region.append(data).unwrap();
        assert_eq!(offset, 0);
        assert_eq!(region.len(), data.len());
        assert!(!region.is_empty());
    }

    #[test]
    fn test_append_only_read() {
        let backing = StaticBacking::<1024>::new();
        let handle = ruvix_types::RegionHandle::new(1, 0);
        let mut region = AppendOnlyRegion::new(backing, 256, handle).unwrap();

        let data = b"Hello, RuVix!";
        region.append(data).unwrap();

        let mut buf = [0u8; 32];
        let read = region.read(0, &mut buf).unwrap();
        assert_eq!(read, data.len());
        assert_eq!(&buf[..data.len()], data);
    }

    #[test]
    fn test_append_only_multiple_appends() {
        let backing = StaticBacking::<1024>::new();
        let handle = ruvix_types::RegionHandle::new(1, 0);
        let mut region = AppendOnlyRegion::new(backing, 256, handle).unwrap();

        let data1 = b"First";
        let data2 = b"Second";
        let data3 = b"Third";

        let off1 = region.append(data1).unwrap();
        let off2 = region.append(data2).unwrap();
        let off3 = region.append(data3).unwrap();

        assert_eq!(off1, 0);
        assert_eq!(off2, data1.len());
        assert_eq!(off3, data1.len() + data2.len());

        // Read all
        let slice = region.as_slice();
        assert_eq!(slice, b"FirstSecondThird");
    }

    #[test]
    fn test_append_only_full() {
        let backing = StaticBacking::<64>::new();
        let handle = ruvix_types::RegionHandle::new(1, 0);
        let mut region = AppendOnlyRegion::new(backing, 32, handle).unwrap();

        // Fill the region
        let data = [0xABu8; 32];
        region.append(&data).unwrap();
        assert!(region.is_full());

        // Should fail to append more
        assert!(region.append(&[0x00]).is_err());
    }

    #[test]
    fn test_append_only_fill_ratio() {
        let backing = StaticBacking::<256>::new();
        let handle = ruvix_types::RegionHandle::new(1, 0);
        let mut region = AppendOnlyRegion::new(backing, 100, handle).unwrap();

        assert!((region.fill_ratio() - 0.0).abs() < 0.01);

        region.append(&[0u8; 50]).unwrap();
        assert!((region.fill_ratio() - 0.5).abs() < 0.01);

        region.append(&[0u8; 50]).unwrap();
        assert!((region.fill_ratio() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_append_only_u64() {
        let backing = StaticBacking::<256>::new();
        let handle = ruvix_types::RegionHandle::new(1, 0);
        let mut region = AppendOnlyRegion::new(backing, 100, handle).unwrap();

        region.append_u64(0x123456789ABCDEF0).unwrap();
        region.append_u32(0xDEADBEEF).unwrap();

        assert_eq!(region.read_u64(0).unwrap(), 0x123456789ABCDEF0);
        assert_eq!(region.read_u32(8).unwrap(), 0xDEADBEEF);
    }

    #[test]
    fn test_append_only_read_beyond_cursor() {
        let backing = StaticBacking::<256>::new();
        let handle = ruvix_types::RegionHandle::new(1, 0);
        let mut region = AppendOnlyRegion::new(backing, 100, handle).unwrap();

        region.append(&[0u8; 10]).unwrap();

        let mut buf = [0u8; 10];
        // Reading at cursor position should fail
        assert!(region.read(10, &mut buf).is_err());
        // Reading beyond cursor should fail
        assert!(region.read(20, &mut buf).is_err());
    }
}
