//! Immutable region implementation.
//!
//! Contents are set once at creation and never modified.
//! The kernel may deduplicate identical immutable regions.
//!
//! # Design (from ADR-087 Section 4)
//!
//! - Set once at creation, never modified
//! - Deduplicatable by content hash
//!
//! Ideal for: RVF component code, trained model weights, lookup tables.

use crate::backing::MemoryBacking;
use crate::Result;
use ruvix_types::KernelError;

/// An immutable memory region.
///
/// Data is set once at creation and cannot be modified afterwards.
/// This enables safe sharing between tasks and potential deduplication.
pub struct ImmutableRegion<B: MemoryBacking> {
    /// Memory backing store.
    _backing: B,
    /// Pointer to the data area.
    data_ptr: *const u8,
    /// Size of the data in bytes.
    size: usize,
    /// Region handle for capability checking.
    handle: ruvix_types::RegionHandle,
    /// Content hash for deduplication (SHA-256).
    content_hash: [u8; 32],
}

impl<B: MemoryBacking> ImmutableRegion<B> {
    /// Creates a new immutable region with the given data.
    ///
    /// The data is copied into the region and cannot be modified afterwards.
    ///
    /// # Arguments
    ///
    /// * `backing` - Memory backing store
    /// * `data` - Initial data to store
    /// * `handle` - Region handle for this region
    ///
    /// # Errors
    ///
    /// Returns `OutOfMemory` if the backing cannot allocate sufficient memory.
    pub fn new(mut backing: B, data: &[u8], handle: ruvix_types::RegionHandle) -> Result<Self> {
        let size = data.len();
        if size == 0 {
            // Empty region
            return Ok(Self {
                _backing: backing,
                data_ptr: core::ptr::null(),
                size: 0,
                handle,
                content_hash: [0; 32],
            });
        }

        let (data_ptr, _) = backing.allocate(size)?;

        // Copy data into the region
        // SAFETY: We just allocated this memory
        unsafe {
            core::ptr::copy_nonoverlapping(data.as_ptr(), data_ptr, size);
        }

        // Compute content hash (simple FNV-1a hash for now, could use SHA-256)
        let content_hash = compute_hash(data);

        Ok(Self {
            _backing: backing,
            data_ptr,
            size,
            handle,
            content_hash,
        })
    }

    /// Returns the data as a slice.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        if self.data_ptr.is_null() || self.size == 0 {
            &[]
        } else {
            // SAFETY: We maintain the invariant that data_ptr points to valid memory
            unsafe { core::slice::from_raw_parts(self.data_ptr, self.size) }
        }
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
    /// Returns `InvalidArgument` if offset is beyond the data size.
    pub fn read(&self, offset: usize, buf: &mut [u8]) -> Result<usize> {
        if offset >= self.size {
            return Err(KernelError::InvalidArgument);
        }

        let available = self.size - offset;
        let to_read = buf.len().min(available);

        // SAFETY: We've verified bounds above
        unsafe {
            core::ptr::copy_nonoverlapping(self.data_ptr.add(offset), buf.as_mut_ptr(), to_read);
        }

        Ok(to_read)
    }

    /// Returns the size of the data in bytes.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.size
    }

    /// Returns true if the region is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Returns the region handle.
    #[inline]
    #[must_use]
    pub const fn handle(&self) -> ruvix_types::RegionHandle {
        self.handle
    }

    /// Returns the content hash for deduplication.
    #[inline]
    #[must_use]
    pub const fn content_hash(&self) -> &[u8; 32] {
        &self.content_hash
    }

    /// Checks if this region has the same content as another.
    ///
    /// Uses content hash for fast comparison.
    #[inline]
    #[must_use]
    pub fn content_equals(&self, other: &Self) -> bool {
        self.content_hash == other.content_hash
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

    /// Gets a byte at the specified offset.
    #[inline]
    pub fn get(&self, offset: usize) -> Option<u8> {
        if offset < self.size {
            // SAFETY: We've verified bounds
            Some(unsafe { *self.data_ptr.add(offset) })
        } else {
            None
        }
    }
}

// SAFETY: ImmutableRegion is safe to share between threads since it's immutable
unsafe impl<B: MemoryBacking + Send> Send for ImmutableRegion<B> {}
unsafe impl<B: MemoryBacking + Sync> Sync for ImmutableRegion<B> {}

/// Computes a simple content hash (FNV-1a based).
///
/// This is a placeholder - in production we'd use SHA-256 or similar.
fn compute_hash(data: &[u8]) -> [u8; 32] {
    // FNV-1a hash extended to 256 bits (4x 64-bit)
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut h1 = FNV_OFFSET;
    let mut h2 = FNV_OFFSET.wrapping_mul(2);
    let mut h3 = FNV_OFFSET.wrapping_mul(3);
    let mut h4 = FNV_OFFSET.wrapping_mul(5);

    for (i, &byte) in data.iter().enumerate() {
        match i % 4 {
            0 => {
                h1 ^= byte as u64;
                h1 = h1.wrapping_mul(FNV_PRIME);
            }
            1 => {
                h2 ^= byte as u64;
                h2 = h2.wrapping_mul(FNV_PRIME);
            }
            2 => {
                h3 ^= byte as u64;
                h3 = h3.wrapping_mul(FNV_PRIME);
            }
            _ => {
                h4 ^= byte as u64;
                h4 = h4.wrapping_mul(FNV_PRIME);
            }
        }
    }

    let mut result = [0u8; 32];
    result[0..8].copy_from_slice(&h1.to_le_bytes());
    result[8..16].copy_from_slice(&h2.to_le_bytes());
    result[16..24].copy_from_slice(&h3.to_le_bytes());
    result[24..32].copy_from_slice(&h4.to_le_bytes());
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backing::StaticBacking;

    #[test]
    fn test_immutable_basic() {
        let backing = StaticBacking::<1024>::new();
        let handle = ruvix_types::RegionHandle::new(1, 0);
        let data = b"Hello, RuVix!";
        let region = ImmutableRegion::new(backing, data, handle).unwrap();

        assert_eq!(region.len(), data.len());
        assert!(!region.is_empty());
        assert_eq!(region.as_slice(), data);
    }

    #[test]
    fn test_immutable_read() {
        let backing = StaticBacking::<1024>::new();
        let handle = ruvix_types::RegionHandle::new(1, 0);
        let data = b"Hello, RuVix!";
        let region = ImmutableRegion::new(backing, data, handle).unwrap();

        let mut buf = [0u8; 32];
        let read = region.read(0, &mut buf).unwrap();
        assert_eq!(read, data.len());
        assert_eq!(&buf[..data.len()], data);

        // Read from offset
        let read = region.read(7, &mut buf).unwrap();
        assert_eq!(read, 6); // "RuVix!"
        assert_eq!(&buf[..6], b"RuVix!");
    }

    #[test]
    fn test_immutable_empty() {
        let backing = StaticBacking::<1024>::new();
        let handle = ruvix_types::RegionHandle::new(1, 0);
        let region = ImmutableRegion::new(backing, &[], handle).unwrap();

        assert!(region.is_empty());
        assert_eq!(region.len(), 0);
        assert_eq!(region.as_slice(), &[]);
    }

    #[test]
    fn test_immutable_content_hash() {
        let backing1 = StaticBacking::<1024>::new();
        let backing2 = StaticBacking::<1024>::new();
        let backing3 = StaticBacking::<1024>::new();
        let handle = ruvix_types::RegionHandle::new(1, 0);

        let data = b"Hello, RuVix!";
        let region1 = ImmutableRegion::new(backing1, data, handle).unwrap();
        let region2 = ImmutableRegion::new(backing2, data, handle).unwrap();
        let region3 = ImmutableRegion::new(backing3, b"Different", handle).unwrap();

        // Same content should have same hash
        assert!(region1.content_equals(&region2));

        // Different content should have different hash
        assert!(!region1.content_equals(&region3));
    }

    #[test]
    fn test_immutable_read_beyond() {
        let backing = StaticBacking::<1024>::new();
        let handle = ruvix_types::RegionHandle::new(1, 0);
        let data = b"Hello";
        let region = ImmutableRegion::new(backing, data, handle).unwrap();

        let mut buf = [0u8; 10];
        // Reading at size should fail
        assert!(region.read(5, &mut buf).is_err());
        // Reading beyond size should fail
        assert!(region.read(10, &mut buf).is_err());
    }

    #[test]
    fn test_immutable_get() {
        let backing = StaticBacking::<1024>::new();
        let handle = ruvix_types::RegionHandle::new(1, 0);
        let data = b"Hello";
        let region = ImmutableRegion::new(backing, data, handle).unwrap();

        assert_eq!(region.get(0), Some(b'H'));
        assert_eq!(region.get(4), Some(b'o'));
        assert_eq!(region.get(5), None);
    }

    #[test]
    fn test_immutable_u64() {
        let backing = StaticBacking::<1024>::new();
        let handle = ruvix_types::RegionHandle::new(1, 0);

        let mut data = [0u8; 16];
        data[0..8].copy_from_slice(&0x123456789ABCDEF0u64.to_le_bytes());
        data[8..12].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());

        let region = ImmutableRegion::new(backing, &data, handle).unwrap();

        assert_eq!(region.read_u64(0).unwrap(), 0x123456789ABCDEF0);
        assert_eq!(region.read_u32(8).unwrap(), 0xDEADBEEF);
    }
}
