//! Memory backing implementations.
//!
//! This module provides the low-level memory allocation primitives used by
//! regions. On Linux, we use mmap for efficient memory mapping. On other
//! platforms or when mmap is not available, we fall back to standard allocation.

use crate::Result;
use ruvix_types::KernelError;

/// A backing store for region memory.
///
/// This trait abstracts over different memory allocation strategies:
/// - mmap on Linux
/// - Standard heap allocation via alloc
/// - Direct physical memory (for bare metal)
pub trait MemoryBacking {
    /// Allocate `size` bytes of memory.
    ///
    /// Returns a pointer to the allocated memory and its actual size.
    /// The actual size may be larger than requested due to alignment.
    fn allocate(&mut self, size: usize) -> Result<(*mut u8, usize)>;

    /// Deallocate memory previously allocated by this backing.
    ///
    /// # Safety
    ///
    /// The pointer must have been returned by a previous call to `allocate`
    /// on this same backing instance.
    unsafe fn deallocate(&mut self, ptr: *mut u8, size: usize) -> Result<()>;

    /// Returns the total capacity of this backing store.
    fn capacity(&self) -> usize;

    /// Returns the amount of memory currently allocated.
    fn allocated(&self) -> usize;
}

/// A simple heap-based backing store.
///
/// Uses the global allocator. Suitable for testing and when mmap is not available.
#[cfg(feature = "std")]
pub struct HeapBacking {
    allocated: usize,
    max_size: usize,
}

#[cfg(feature = "std")]
impl HeapBacking {
    /// Creates a new heap backing with the specified maximum size.
    pub fn new(max_size: usize) -> Self {
        Self {
            allocated: 0,
            max_size,
        }
    }
}

#[cfg(feature = "std")]
impl MemoryBacking for HeapBacking {
    fn allocate(&mut self, size: usize) -> Result<(*mut u8, usize)> {
        if self.allocated + size > self.max_size {
            return Err(KernelError::OutOfMemory);
        }

        // Align to 8 bytes
        let aligned_size = (size + 7) & !7;

        let layout = std::alloc::Layout::from_size_align(aligned_size, 8)
            .map_err(|_| KernelError::InvalidArgument)?;

        // SAFETY: We've verified the layout is valid
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };

        if ptr.is_null() {
            return Err(KernelError::OutOfMemory);
        }

        self.allocated += aligned_size;
        Ok((ptr, aligned_size))
    }

    unsafe fn deallocate(&mut self, ptr: *mut u8, size: usize) -> Result<()> {
        let aligned_size = (size + 7) & !7;

        let layout = std::alloc::Layout::from_size_align(aligned_size, 8)
            .map_err(|_| KernelError::InvalidArgument)?;

        // SAFETY: Caller guarantees ptr was allocated by this backing
        unsafe {
            std::alloc::dealloc(ptr, layout);
        }

        self.allocated = self.allocated.saturating_sub(aligned_size);
        Ok(())
    }

    fn capacity(&self) -> usize {
        self.max_size
    }

    fn allocated(&self) -> usize {
        self.allocated
    }
}

/// mmap-based backing store for Linux.
///
/// Uses anonymous mmap for efficient memory mapping without file backing.
#[cfg(all(unix, feature = "mmap"))]
pub struct MmapBacking {
    allocated: usize,
    max_size: usize,
}

#[cfg(all(unix, feature = "mmap"))]
impl MmapBacking {
    /// Creates a new mmap backing with the specified maximum size.
    pub fn new(max_size: usize) -> Self {
        Self {
            allocated: 0,
            max_size,
        }
    }
}

#[cfg(all(unix, feature = "mmap"))]
impl MemoryBacking for MmapBacking {
    fn allocate(&mut self, size: usize) -> Result<(*mut u8, usize)> {
        use libc::{mmap, MAP_ANON, MAP_PRIVATE, PROT_READ, PROT_WRITE};

        if self.allocated + size > self.max_size {
            return Err(KernelError::OutOfMemory);
        }

        // Align to page size (typically 4096)
        let page_size = 4096usize;
        let aligned_size = (size + page_size - 1) & !(page_size - 1);

        // SAFETY: We're calling mmap with valid parameters
        let ptr = unsafe {
            mmap(
                core::ptr::null_mut(),
                aligned_size,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANON,
                -1,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(KernelError::OutOfMemory);
        }

        self.allocated += aligned_size;
        Ok((ptr as *mut u8, aligned_size))
    }

    unsafe fn deallocate(&mut self, ptr: *mut u8, size: usize) -> Result<()> {
        use libc::munmap;

        let page_size = 4096usize;
        let aligned_size = (size + page_size - 1) & !(page_size - 1);

        // SAFETY: Caller guarantees ptr was allocated by this backing
        let result = unsafe { munmap(ptr as *mut libc::c_void, aligned_size) };

        if result != 0 {
            return Err(KernelError::InternalError);
        }

        self.allocated = self.allocated.saturating_sub(aligned_size);
        Ok(())
    }

    fn capacity(&self) -> usize {
        self.max_size
    }

    fn allocated(&self) -> usize {
        self.allocated
    }
}

/// A static backing store using a fixed buffer.
///
/// Useful for no_std environments where dynamic allocation is not available.
pub struct StaticBacking<const N: usize> {
    buffer: [u8; N],
    next_free: usize,
}

impl<const N: usize> StaticBacking<N> {
    /// Creates a new static backing store.
    pub const fn new() -> Self {
        Self {
            buffer: [0u8; N],
            next_free: 0,
        }
    }

    /// Returns the buffer as a slice.
    pub fn as_slice(&self) -> &[u8] {
        &self.buffer[..self.next_free]
    }

    /// Returns the buffer as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.buffer[..self.next_free]
    }
}

impl<const N: usize> Default for StaticBacking<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> MemoryBacking for StaticBacking<N> {
    fn allocate(&mut self, size: usize) -> Result<(*mut u8, usize)> {
        // Align to 8 bytes
        let aligned_offset = (self.next_free + 7) & !7;
        let aligned_size = (size + 7) & !7;

        if aligned_offset + aligned_size > N {
            return Err(KernelError::OutOfMemory);
        }

        let ptr = self.buffer.as_mut_ptr();
        // SAFETY: We've verified bounds
        let result_ptr = unsafe { ptr.add(aligned_offset) };

        // Zero the memory
        // SAFETY: We've verified bounds
        unsafe {
            core::ptr::write_bytes(result_ptr, 0, aligned_size);
        }

        self.next_free = aligned_offset + aligned_size;
        Ok((result_ptr, aligned_size))
    }

    unsafe fn deallocate(&mut self, _ptr: *mut u8, _size: usize) -> Result<()> {
        // Static backing doesn't support deallocation - this is a bump allocator
        // In a real kernel, we'd track allocations more carefully
        Ok(())
    }

    fn capacity(&self) -> usize {
        N
    }

    fn allocated(&self) -> usize {
        self.next_free
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "std")]
    #[test]
    fn test_heap_backing() {
        let mut backing = HeapBacking::new(1024);

        let (ptr, size) = backing.allocate(100).unwrap();
        assert!(!ptr.is_null());
        assert!(size >= 100);
        assert!(backing.allocated() >= 100);

        unsafe {
            backing.deallocate(ptr, size).unwrap();
        }
    }

    #[test]
    fn test_static_backing() {
        let mut backing = StaticBacking::<1024>::new();

        let (ptr1, size1) = backing.allocate(100).unwrap();
        assert!(!ptr1.is_null());
        assert!(size1 >= 100);

        let (ptr2, size2) = backing.allocate(200).unwrap();
        assert!(!ptr2.is_null());
        assert!(size2 >= 200);
        assert!(ptr2 > ptr1);

        // Should fail when full
        let result = backing.allocate(800);
        assert!(result.is_err());
    }
}
