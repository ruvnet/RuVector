//! Slab allocator for fixed-size slot allocation.
//!
//! The slab allocator provides O(1) allocation and deallocation of fixed-size
//! slots from a pre-allocated region. It uses a free list for efficient reuse
//! and generation counters for use-after-free prevention.
//!
//! # Design (from ADR-087 Section 4)
//!
//! - Fixed-size slots allocated from a free list
//! - Slots can be freed and reused
//! - No fragmentation by construction
//! - Generation counters for use-after-free prevention
//!
//! Ideal for: task control blocks, capability tables, queue ring buffers.

use crate::backing::MemoryBacking;
use crate::Result;
use ruvix_types::KernelError;

/// Index used internally for free list management.
const FREE_LIST_END: u32 = u32::MAX;

/// Metadata for a single slot in the slab.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct SlotMeta {
    /// Generation counter. Incremented on each free.
    generation: u32,
    /// Next free slot index, or FREE_LIST_END if allocated/end of list.
    next_free: u32,
}

impl SlotMeta {
    const fn new() -> Self {
        Self {
            generation: 0,
            next_free: FREE_LIST_END,
        }
    }
}

/// Handle to an allocated slot.
///
/// Contains the slot index and generation counter for validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct SlotHandle {
    /// Slot index within the slab.
    pub index: u32,
    /// Generation at allocation time.
    pub generation: u32,
}

impl SlotHandle {
    /// Creates a new slot handle.
    #[inline]
    #[must_use]
    pub const fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    /// Creates an invalid slot handle.
    #[inline]
    #[must_use]
    pub const fn invalid() -> Self {
        Self {
            index: FREE_LIST_END,
            generation: 0,
        }
    }

    /// Checks if this handle is invalid.
    #[inline]
    #[must_use]
    pub const fn is_invalid(&self) -> bool {
        self.index == FREE_LIST_END
    }
}

impl Default for SlotHandle {
    fn default() -> Self {
        Self::invalid()
    }
}

/// A slab allocator for fixed-size slot allocation.
///
/// Provides O(1) allocation and deallocation with generation-based
/// use-after-free detection.
pub struct SlabAllocator<B: MemoryBacking> {
    /// Memory backing store.
    backing: B,
    /// Pointer to the slot data area.
    data_ptr: *mut u8,
    /// Slot metadata (generation, next_free).
    meta: SlabMetaStorage,
    /// Size of each slot in bytes.
    slot_size: usize,
    /// Total number of slots.
    slot_count: usize,
    /// Head of the free list.
    free_head: u32,
    /// Number of currently allocated slots.
    allocated_count: usize,
}

/// Storage for slot metadata.
///
/// Uses a fixed-size array for no_std compatibility, with dynamic
/// allocation available when std is enabled.
enum SlabMetaStorage {
    /// Static storage for small slabs.
    Static([SlotMeta; 256]),
    /// Inline storage for up to 64 slots.
    Inline([SlotMeta; 64]),
    #[cfg(feature = "std")]
    /// Dynamic storage for large slabs.
    Dynamic(std::vec::Vec<SlotMeta>),
}

impl SlabMetaStorage {
    fn new(count: usize) -> Self {
        if count <= 64 {
            Self::Inline([SlotMeta::new(); 64])
        } else if count <= 256 {
            Self::Static([SlotMeta::new(); 256])
        } else {
            #[cfg(feature = "std")]
            {
                let mut v = std::vec::Vec::with_capacity(count);
                v.resize(count, SlotMeta::new());
                Self::Dynamic(v)
            }
            #[cfg(not(feature = "std"))]
            {
                // Fall back to static storage, limiting slots
                Self::Static([SlotMeta::new(); 256])
            }
        }
    }

    fn get(&self, index: usize) -> Option<&SlotMeta> {
        match self {
            Self::Static(arr) => arr.get(index),
            Self::Inline(arr) => arr.get(index),
            #[cfg(feature = "std")]
            Self::Dynamic(v) => v.get(index),
        }
    }

    fn get_mut(&mut self, index: usize) -> Option<&mut SlotMeta> {
        match self {
            Self::Static(arr) => arr.get_mut(index),
            Self::Inline(arr) => arr.get_mut(index),
            #[cfg(feature = "std")]
            Self::Dynamic(v) => v.get_mut(index),
        }
    }
}

impl<B: MemoryBacking> SlabAllocator<B> {
    /// Creates a new slab allocator.
    ///
    /// # Arguments
    ///
    /// * `backing` - Memory backing store
    /// * `slot_size` - Size of each slot in bytes (will be aligned to 8 bytes)
    /// * `slot_count` - Total number of slots
    ///
    /// # Errors
    ///
    /// Returns `OutOfMemory` if the backing cannot allocate sufficient memory.
    pub fn new(mut backing: B, slot_size: usize, slot_count: usize) -> Result<Self> {
        if slot_size == 0 || slot_count == 0 {
            return Err(KernelError::InvalidArgument);
        }

        // Align slot size to 8 bytes
        let aligned_slot_size = (slot_size + 7) & !7;
        let total_size = aligned_slot_size
            .checked_mul(slot_count)
            .ok_or(KernelError::InvalidArgument)?;

        // Allocate memory for slot data
        let (data_ptr, _) = backing.allocate(total_size)?;

        // Initialize metadata and free list
        let mut meta = SlabMetaStorage::new(slot_count);

        // Build initial free list: 0 -> 1 -> 2 -> ... -> n-1 -> END
        for i in 0..slot_count {
            if let Some(slot_meta) = meta.get_mut(i) {
                slot_meta.generation = 0;
                slot_meta.next_free = if i + 1 < slot_count {
                    (i + 1) as u32
                } else {
                    FREE_LIST_END
                };
            }
        }

        Ok(Self {
            backing,
            data_ptr,
            meta,
            slot_size: aligned_slot_size,
            slot_count,
            free_head: 0,
            allocated_count: 0,
        })
    }

    /// Allocates a slot from the slab.
    ///
    /// Returns a handle that must be used for all subsequent operations on the slot.
    ///
    /// # Errors
    ///
    /// Returns `SlabFull` if no slots are available.
    pub fn alloc(&mut self) -> Result<SlotHandle> {
        if self.free_head == FREE_LIST_END {
            return Err(KernelError::SlabFull);
        }

        let slot_index = self.free_head as usize;
        let meta = self.meta.get_mut(slot_index).ok_or(KernelError::InternalError)?;

        // Remove from free list
        self.free_head = meta.next_free;
        meta.next_free = FREE_LIST_END; // Mark as allocated

        self.allocated_count += 1;

        Ok(SlotHandle::new(slot_index as u32, meta.generation))
    }

    /// Frees a previously allocated slot.
    ///
    /// The slot's generation counter is incremented, invalidating any existing handles.
    ///
    /// # Errors
    ///
    /// Returns `InvalidSlot` if the handle is stale (generation mismatch).
    pub fn free(&mut self, handle: SlotHandle) -> Result<()> {
        self.validate_handle(handle)?;

        let slot_index = handle.index as usize;
        let meta = self.meta.get_mut(slot_index).ok_or(KernelError::InternalError)?;

        // Increment generation to invalidate existing handles
        meta.generation = meta.generation.wrapping_add(1);

        // Add to front of free list
        meta.next_free = self.free_head;
        self.free_head = handle.index;

        self.allocated_count = self.allocated_count.saturating_sub(1);

        // Zero the slot data for security
        self.zero_slot(slot_index);

        Ok(())
    }

    /// Returns a pointer to the slot data.
    ///
    /// # Errors
    ///
    /// Returns `InvalidSlot` if the handle is stale.
    ///
    /// # Safety
    ///
    /// The caller must ensure proper synchronization when accessing the data.
    pub fn slot_ptr(&self, handle: SlotHandle) -> Result<*mut u8> {
        self.validate_handle(handle)?;

        let offset = handle.index as usize * self.slot_size;
        // SAFETY: We've validated the handle and bounds
        Ok(unsafe { self.data_ptr.add(offset) })
    }

    /// Reads data from a slot.
    ///
    /// # Errors
    ///
    /// Returns `InvalidSlot` if the handle is stale.
    /// Returns `BufferTooSmall` if the buffer is smaller than slot_size.
    pub fn read(&self, handle: SlotHandle, buf: &mut [u8]) -> Result<usize> {
        self.validate_handle(handle)?;

        let to_read = buf.len().min(self.slot_size);
        let ptr = self.slot_ptr(handle)?;

        // SAFETY: We've validated the handle and bounds
        unsafe {
            core::ptr::copy_nonoverlapping(ptr, buf.as_mut_ptr(), to_read);
        }

        Ok(to_read)
    }

    /// Writes data to a slot.
    ///
    /// # Errors
    ///
    /// Returns `InvalidSlot` if the handle is stale.
    /// Returns `BufferTooSmall` if the data exceeds slot_size.
    pub fn write(&mut self, handle: SlotHandle, data: &[u8]) -> Result<usize> {
        self.validate_handle(handle)?;

        if data.len() > self.slot_size {
            return Err(KernelError::BufferTooSmall);
        }

        let ptr = self.slot_ptr(handle)?;

        // SAFETY: We've validated the handle and bounds
        unsafe {
            core::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }

        Ok(data.len())
    }

    /// Validates a slot handle.
    fn validate_handle(&self, handle: SlotHandle) -> Result<()> {
        if handle.is_invalid() {
            return Err(KernelError::InvalidSlot);
        }

        let slot_index = handle.index as usize;
        if slot_index >= self.slot_count {
            return Err(KernelError::InvalidSlot);
        }

        let meta = self.meta.get(slot_index).ok_or(KernelError::InternalError)?;

        // Check generation - if it doesn't match, the handle is stale
        if meta.generation != handle.generation {
            return Err(KernelError::InvalidSlot);
        }

        // Check if the slot is actually allocated (not in free list)
        if meta.next_free != FREE_LIST_END {
            return Err(KernelError::InvalidSlot);
        }

        Ok(())
    }

    /// Zeros the data in a slot.
    fn zero_slot(&mut self, slot_index: usize) {
        let offset = slot_index * self.slot_size;
        // SAFETY: slot_index is validated before this is called
        unsafe {
            core::ptr::write_bytes(self.data_ptr.add(offset), 0, self.slot_size);
        }
    }

    /// Returns the slot size in bytes.
    #[inline]
    #[must_use]
    pub const fn slot_size(&self) -> usize {
        self.slot_size
    }

    /// Returns the total number of slots.
    #[inline]
    #[must_use]
    pub const fn slot_count(&self) -> usize {
        self.slot_count
    }

    /// Returns the number of currently allocated slots.
    #[inline]
    #[must_use]
    pub const fn allocated_count(&self) -> usize {
        self.allocated_count
    }

    /// Returns the number of free slots.
    #[inline]
    #[must_use]
    pub const fn free_count(&self) -> usize {
        self.slot_count - self.allocated_count
    }

    /// Returns true if the slab is full.
    #[inline]
    #[must_use]
    pub const fn is_full(&self) -> bool {
        self.allocated_count == self.slot_count
    }

    /// Returns true if the slab is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.allocated_count == 0
    }
}

/// A slab region wrapping a slab allocator with region semantics.
pub struct SlabRegion<B: MemoryBacking> {
    /// The underlying slab allocator.
    allocator: SlabAllocator<B>,
    /// Region handle for capability checking.
    handle: ruvix_types::RegionHandle,
}

impl<B: MemoryBacking> SlabRegion<B> {
    /// Creates a new slab region.
    pub fn new(
        backing: B,
        slot_size: usize,
        slot_count: usize,
        handle: ruvix_types::RegionHandle,
    ) -> Result<Self> {
        let allocator = SlabAllocator::new(backing, slot_size, slot_count)?;
        Ok(Self { allocator, handle })
    }

    /// Returns the region handle.
    #[inline]
    #[must_use]
    pub const fn handle(&self) -> ruvix_types::RegionHandle {
        self.handle
    }

    /// Allocates a slot.
    #[inline]
    pub fn alloc(&mut self) -> Result<SlotHandle> {
        self.allocator.alloc()
    }

    /// Frees a slot.
    #[inline]
    pub fn free(&mut self, handle: SlotHandle) -> Result<()> {
        self.allocator.free(handle)
    }

    /// Reads from a slot.
    #[inline]
    pub fn read(&self, handle: SlotHandle, buf: &mut [u8]) -> Result<usize> {
        self.allocator.read(handle, buf)
    }

    /// Writes to a slot.
    #[inline]
    pub fn write(&mut self, handle: SlotHandle, data: &[u8]) -> Result<usize> {
        self.allocator.write(handle, data)
    }

    /// Returns the slot size.
    #[inline]
    #[must_use]
    pub const fn slot_size(&self) -> usize {
        self.allocator.slot_size()
    }

    /// Returns the slot count.
    #[inline]
    #[must_use]
    pub const fn slot_count(&self) -> usize {
        self.allocator.slot_count()
    }

    /// Returns the allocated count.
    #[inline]
    #[must_use]
    pub const fn allocated_count(&self) -> usize {
        self.allocator.allocated_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backing::StaticBacking;

    #[test]
    fn test_slab_alloc_free() {
        let backing = StaticBacking::<4096>::new();
        let mut slab = SlabAllocator::new(backing, 64, 16).unwrap();

        assert_eq!(slab.slot_count(), 16);
        assert_eq!(slab.allocated_count(), 0);
        assert_eq!(slab.free_count(), 16);

        // Allocate a slot
        let handle = slab.alloc().unwrap();
        assert!(!handle.is_invalid());
        assert_eq!(slab.allocated_count(), 1);

        // Free the slot
        slab.free(handle).unwrap();
        assert_eq!(slab.allocated_count(), 0);
    }

    #[test]
    fn test_slab_generation_counter() {
        let backing = StaticBacking::<4096>::new();
        let mut slab = SlabAllocator::new(backing, 64, 16).unwrap();

        let handle1 = slab.alloc().unwrap();
        slab.free(handle1).unwrap();

        // Old handle should now be invalid
        assert!(slab.free(handle1).is_err());

        // New allocation at same slot should have different generation
        let handle2 = slab.alloc().unwrap();
        assert_eq!(handle1.index, handle2.index);
        assert_ne!(handle1.generation, handle2.generation);
    }

    #[test]
    fn test_slab_read_write() {
        let backing = StaticBacking::<4096>::new();
        let mut slab = SlabAllocator::new(backing, 64, 16).unwrap();

        let handle = slab.alloc().unwrap();

        // Write data
        let data = b"Hello, RuVix!";
        slab.write(handle, data).unwrap();

        // Read it back
        let mut buf = [0u8; 64];
        let read = slab.read(handle, &mut buf).unwrap();
        assert_eq!(read, 64);
        assert_eq!(&buf[..data.len()], data);
    }

    #[test]
    fn test_slab_full() {
        let backing = StaticBacking::<512>::new();
        let mut slab = SlabAllocator::new(backing, 64, 4).unwrap();

        // Allocate all slots
        let mut handles = [SlotHandle::invalid(); 4];
        for handle in &mut handles {
            *handle = slab.alloc().unwrap();
        }

        assert!(slab.is_full());

        // Should fail to allocate more
        assert!(slab.alloc().is_err());

        // Free one and try again
        slab.free(handles[0]).unwrap();
        assert!(!slab.is_full());

        let new_handle = slab.alloc().unwrap();
        assert!(!new_handle.is_invalid());
    }

    #[test]
    fn test_slab_zero_on_free() {
        let backing = StaticBacking::<4096>::new();
        let mut slab = SlabAllocator::new(backing, 64, 16).unwrap();

        let handle = slab.alloc().unwrap();

        // Write data
        let data = [0xABu8; 64];
        slab.write(handle, &data).unwrap();

        // Free the slot (should zero it)
        slab.free(handle).unwrap();

        // Allocate same slot again
        let handle2 = slab.alloc().unwrap();

        // Should be zeroed
        let mut buf = [0xFFu8; 64];
        slab.read(handle2, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_slot_handle_default() {
        let handle = SlotHandle::default();
        assert!(handle.is_invalid());
    }
}
