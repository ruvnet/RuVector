//! DMA buffer management for cache-coherent memory transfers.

use crate::{DmaError, DmaResult, DMA_BUFFER_ALIGNMENT};

/// Flags controlling DMA buffer behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DmaBufferFlags {
    /// Buffer is cache-coherent (no explicit cache operations needed).
    pub cache_coherent: bool,
    /// Buffer is mapped for reading from device.
    pub readable: bool,
    /// Buffer is mapped for writing to device.
    pub writable: bool,
    /// Buffer memory is physically contiguous.
    pub contiguous: bool,
    /// Buffer should be zero-initialized.
    pub zero_init: bool,
}

impl DmaBufferFlags {
    /// Create flags for a read-only buffer.
    #[must_use]
    pub const fn read_only() -> Self {
        Self {
            cache_coherent: true,
            readable: true,
            writable: false,
            contiguous: true,
            zero_init: false,
        }
    }

    /// Create flags for a write-only buffer.
    #[must_use]
    pub const fn write_only() -> Self {
        Self {
            cache_coherent: true,
            readable: false,
            writable: true,
            contiguous: true,
            zero_init: false,
        }
    }

    /// Create flags for a read-write buffer.
    #[must_use]
    pub const fn read_write() -> Self {
        Self {
            cache_coherent: true,
            readable: true,
            writable: true,
            contiguous: true,
            zero_init: false,
        }
    }

    /// Create flags with zero initialization.
    #[must_use]
    pub const fn zeroed() -> Self {
        Self {
            cache_coherent: true,
            readable: true,
            writable: true,
            contiguous: true,
            zero_init: true,
        }
    }
}

/// A cache-coherent DMA buffer with physical address tracking.
///
/// This structure represents a memory region suitable for DMA transfers.
/// It tracks both the physical address (for hardware) and provides
/// methods for cache management.
#[derive(Debug, Clone)]
pub struct DmaBuffer {
    /// Physical address of the buffer (for DMA hardware).
    physical_addr: u64,
    /// Virtual address of the buffer (for CPU access).
    virtual_addr: u64,
    /// Size of the buffer in bytes.
    size: usize,
    /// Alignment of the buffer.
    alignment: usize,
    /// Buffer flags.
    flags: DmaBufferFlags,
    /// Generation counter for tracking invalidations.
    generation: u32,
}

impl DmaBuffer {
    /// Create a new DMA buffer descriptor.
    ///
    /// # Arguments
    ///
    /// * `physical_addr` - Physical address of the buffer
    /// * `virtual_addr` - Virtual address of the buffer
    /// * `size` - Size in bytes
    /// * `flags` - Buffer flags
    ///
    /// # Errors
    ///
    /// Returns `DmaError` if:
    /// - Size is zero
    /// - Physical address is not properly aligned
    pub fn new(
        physical_addr: u64,
        virtual_addr: u64,
        size: usize,
        flags: DmaBufferFlags,
    ) -> DmaResult<Self> {
        Self::with_alignment(physical_addr, virtual_addr, size, DMA_BUFFER_ALIGNMENT, flags)
    }

    /// Create a new DMA buffer with custom alignment.
    ///
    /// # Errors
    ///
    /// Returns `DmaError` if:
    /// - Size is zero
    /// - Alignment is not a power of two
    /// - Physical address is not properly aligned
    pub fn with_alignment(
        physical_addr: u64,
        virtual_addr: u64,
        size: usize,
        alignment: usize,
        flags: DmaBufferFlags,
    ) -> DmaResult<Self> {
        // Validate size
        if size == 0 {
            return Err(DmaError::config_error());
        }

        // Validate alignment is power of two
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(DmaError::config_error());
        }

        // Check physical address alignment
        if physical_addr % (alignment as u64) != 0 {
            return Err(DmaError::alignment_error(physical_addr));
        }

        Ok(Self {
            physical_addr,
            virtual_addr,
            size,
            alignment,
            flags,
            generation: 0,
        })
    }

    /// Get the physical address of the buffer.
    #[must_use]
    pub const fn physical_addr(&self) -> u64 {
        self.physical_addr
    }

    /// Get the virtual address of the buffer.
    #[must_use]
    pub const fn virtual_addr(&self) -> u64 {
        self.virtual_addr
    }

    /// Get the size of the buffer in bytes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.size
    }

    /// Check if the buffer is empty (zero size).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get the alignment of the buffer.
    #[must_use]
    pub const fn alignment(&self) -> usize {
        self.alignment
    }

    /// Get the buffer flags.
    #[must_use]
    pub const fn flags(&self) -> DmaBufferFlags {
        self.flags
    }

    /// Check if the buffer is cache-coherent.
    #[must_use]
    pub const fn is_cache_coherent(&self) -> bool {
        self.flags.cache_coherent
    }

    /// Check if the buffer is readable.
    #[must_use]
    pub const fn is_readable(&self) -> bool {
        self.flags.readable
    }

    /// Check if the buffer is writable.
    #[must_use]
    pub const fn is_writable(&self) -> bool {
        self.flags.writable
    }

    /// Check if the buffer is physically contiguous.
    #[must_use]
    pub const fn is_contiguous(&self) -> bool {
        self.flags.contiguous
    }

    /// Get the generation counter.
    #[must_use]
    pub const fn generation(&self) -> u32 {
        self.generation
    }

    /// Calculate the end address of the buffer.
    #[must_use]
    pub const fn end_addr(&self) -> u64 {
        self.physical_addr + self.size as u64
    }

    /// Check if an address is within this buffer.
    #[must_use]
    pub const fn contains(&self, addr: u64) -> bool {
        addr >= self.physical_addr && addr < self.end_addr()
    }

    /// Check if another buffer overlaps with this one.
    #[must_use]
    pub const fn overlaps(&self, other: &DmaBuffer) -> bool {
        self.physical_addr < other.end_addr() && other.physical_addr < self.end_addr()
    }

    /// Create a sub-buffer from this buffer.
    ///
    /// # Arguments
    ///
    /// * `offset` - Offset from the start of this buffer
    /// * `size` - Size of the sub-buffer
    ///
    /// # Errors
    ///
    /// Returns `DmaError` if the sub-buffer would extend past the end.
    pub fn slice(&self, offset: usize, size: usize) -> DmaResult<Self> {
        if offset + size > self.size {
            return Err(DmaError::config_error());
        }

        Ok(Self {
            physical_addr: self.physical_addr + offset as u64,
            virtual_addr: self.virtual_addr + offset as u64,
            size,
            alignment: self.alignment,
            flags: self.flags,
            generation: self.generation,
        })
    }

    /// Invalidate cache for this buffer (before reading from device).
    ///
    /// This marks the buffer as requiring cache invalidation. The actual
    /// cache operation is performed by the DMA controller implementation.
    pub fn invalidate_cache(&mut self) {
        if !self.flags.cache_coherent {
            self.generation = self.generation.wrapping_add(1);
        }
    }

    /// Clean cache for this buffer (before writing to device).
    ///
    /// This marks the buffer as requiring cache cleaning. The actual
    /// cache operation is performed by the DMA controller implementation.
    pub fn clean_cache(&mut self) {
        if !self.flags.cache_coherent {
            self.generation = self.generation.wrapping_add(1);
        }
    }

    /// Sync buffer for device access.
    ///
    /// Call this before starting a DMA transfer to the device.
    pub fn sync_for_device(&mut self) {
        self.clean_cache();
    }

    /// Sync buffer for CPU access.
    ///
    /// Call this after a DMA transfer from the device completes.
    pub fn sync_for_cpu(&mut self) {
        self.invalidate_cache();
    }
}

/// Iterator over chunks of a DMA buffer.
#[derive(Debug)]
pub struct DmaBufferChunks<'a> {
    buffer: &'a DmaBuffer,
    offset: usize,
    chunk_size: usize,
}

impl<'a> DmaBufferChunks<'a> {
    /// Create a new chunk iterator.
    pub fn new(buffer: &'a DmaBuffer, chunk_size: usize) -> Self {
        Self {
            buffer,
            offset: 0,
            chunk_size,
        }
    }
}

impl Iterator for DmaBufferChunks<'_> {
    type Item = (u64, usize); // (physical_addr, size)

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.buffer.size {
            return None;
        }

        let remaining = self.buffer.size - self.offset;
        let size = remaining.min(self.chunk_size);
        let addr = self.buffer.physical_addr + self.offset as u64;

        self.offset += size;
        Some((addr, size))
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use std::vec::Vec;

    use super::*;

    #[test]
    fn test_buffer_flags() {
        let ro = DmaBufferFlags::read_only();
        assert!(ro.readable);
        assert!(!ro.writable);

        let wo = DmaBufferFlags::write_only();
        assert!(!wo.readable);
        assert!(wo.writable);

        let rw = DmaBufferFlags::read_write();
        assert!(rw.readable);
        assert!(rw.writable);

        let zeroed = DmaBufferFlags::zeroed();
        assert!(zeroed.zero_init);
    }

    #[test]
    fn test_buffer_creation() {
        let buf = DmaBuffer::new(0x1000, 0xFFFF_0000_1000, 4096, DmaBufferFlags::read_write());
        assert!(buf.is_ok());

        let buf = buf.unwrap();
        assert_eq!(buf.physical_addr(), 0x1000);
        assert_eq!(buf.virtual_addr(), 0xFFFF_0000_1000);
        assert_eq!(buf.len(), 4096);
    }

    #[test]
    fn test_buffer_alignment_error() {
        let buf = DmaBuffer::new(
            0x1001, // Not aligned to 64
            0xFFFF_0000_1001,
            4096,
            DmaBufferFlags::read_write(),
        );
        assert!(buf.is_err());
    }

    #[test]
    fn test_buffer_zero_size_error() {
        let buf = DmaBuffer::new(0x1000, 0xFFFF_0000_1000, 0, DmaBufferFlags::read_write());
        assert!(buf.is_err());
    }

    #[test]
    fn test_buffer_contains() {
        let buf =
            DmaBuffer::new(0x1000, 0xFFFF_0000_1000, 4096, DmaBufferFlags::read_write()).unwrap();

        assert!(buf.contains(0x1000));
        assert!(buf.contains(0x1FFF));
        assert!(!buf.contains(0x2000));
        assert!(!buf.contains(0x0FFF));
    }

    #[test]
    fn test_buffer_overlaps() {
        let buf1 =
            DmaBuffer::new(0x1000, 0xFFFF_0000_1000, 4096, DmaBufferFlags::read_write()).unwrap();
        let buf2 =
            DmaBuffer::new(0x1800, 0xFFFF_0000_1800, 4096, DmaBufferFlags::read_write()).unwrap();
        let buf3 =
            DmaBuffer::new(0x3000, 0xFFFF_0000_3000, 4096, DmaBufferFlags::read_write()).unwrap();

        assert!(buf1.overlaps(&buf2));
        assert!(buf2.overlaps(&buf1));
        assert!(!buf1.overlaps(&buf3));
    }

    #[test]
    fn test_buffer_slice() {
        let buf =
            DmaBuffer::new(0x1000, 0xFFFF_0000_1000, 4096, DmaBufferFlags::read_write()).unwrap();

        let sub = buf.slice(1024, 2048).unwrap();
        assert_eq!(sub.physical_addr(), 0x1400);
        assert_eq!(sub.len(), 2048);

        // Out of bounds
        assert!(buf.slice(2048, 4096).is_err());
    }

    #[test]
    fn test_buffer_chunks() {
        let buf =
            DmaBuffer::new(0x1000, 0xFFFF_0000_1000, 4096, DmaBufferFlags::read_write()).unwrap();

        let chunks: Vec<_> = DmaBufferChunks::new(&buf, 1024).collect();
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0], (0x1000, 1024));
        assert_eq!(chunks[1], (0x1400, 1024));
        assert_eq!(chunks[2], (0x1800, 1024));
        assert_eq!(chunks[3], (0x1C00, 1024));
    }

    #[test]
    fn test_buffer_cache_operations() {
        let mut buf =
            DmaBuffer::new(0x1000, 0xFFFF_0000_1000, 4096, DmaBufferFlags::read_write()).unwrap();

        let gen0 = buf.generation();
        buf.sync_for_device();
        buf.sync_for_cpu();

        // For non-coherent buffers, generation should change
        // But our test buffer is coherent, so generation stays same
        assert_eq!(buf.generation(), gen0);
    }
}
