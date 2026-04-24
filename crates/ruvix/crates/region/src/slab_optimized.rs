//! Optimized slab allocator with bitmap-based O(1) allocation (ADR-087).
//!
//! This module provides a high-performance slab allocator using:
//! - Bitmap tracking with CTZ (count trailing zeros) for O(1) free slot finding
//! - Cache-line aligned slots for optimal memory access
//! - Generation counters for use-after-free prevention
//!
//! Performance targets:
//! - Allocation: O(1) using CTZ hardware instruction
//! - Deallocation: O(1) single bit flip
//! - Slot access: <50ns with cache-line alignment

use crate::backing::MemoryBacking;
use crate::Result;
use ruvix_types::KernelError;

/// Cache line size for alignment.
const CACHE_LINE_SIZE: usize = 64;

/// Maximum slots per bitmap chunk (u64 = 64 bits).
const BITS_PER_CHUNK: usize = 64;

/// Optimized slot handle with packed index and generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct OptimizedSlotHandle {
    /// Combined index (lower 24 bits) and generation (upper 8 bits).
    /// Supports up to 16M slots with 256 generations.
    packed: u32,
}

impl OptimizedSlotHandle {
    /// Maximum slot index (24 bits = 16M).
    const INDEX_MASK: u32 = 0x00FF_FFFF;

    /// Generation shift amount.
    const GEN_SHIFT: u32 = 24;

    /// Creates a new slot handle.
    #[inline]
    #[must_use]
    pub const fn new(index: u32, generation: u8) -> Self {
        Self {
            packed: (index & Self::INDEX_MASK) | ((generation as u32) << Self::GEN_SHIFT),
        }
    }

    /// Creates an invalid slot handle.
    #[inline]
    #[must_use]
    pub const fn invalid() -> Self {
        Self { packed: u32::MAX }
    }

    /// Returns the slot index.
    #[inline]
    #[must_use]
    pub const fn index(&self) -> u32 {
        self.packed & Self::INDEX_MASK
    }

    /// Returns the generation counter.
    #[inline]
    #[must_use]
    pub const fn generation(&self) -> u8 {
        (self.packed >> Self::GEN_SHIFT) as u8
    }

    /// Returns true if this handle is invalid.
    #[inline]
    #[must_use]
    pub const fn is_invalid(&self) -> bool {
        self.packed == u32::MAX
    }
}

impl Default for OptimizedSlotHandle {
    fn default() -> Self {
        Self::invalid()
    }
}

/// Cache-line aligned slot metadata.
#[derive(Clone, Copy)]
#[repr(C, align(8))]
struct SlotGeneration {
    /// Generation counter (incremented on each free).
    generation: u8,
    /// Reserved for future use.
    _reserved: [u8; 7],
}

impl SlotGeneration {
    const fn new() -> Self {
        Self {
            generation: 0,
            _reserved: [0; 7],
        }
    }
}

/// Maximum bitmap chunks for 256-slot slab (4 x 64 bits = 256 slots).
const MAX_BITMAP_CHUNKS: usize = 4;

/// Maximum supported slots (256).
const MAX_SLOTS: usize = 256;

/// Optimized slab allocator with bitmap-based free slot tracking.
///
/// Uses CTZ (count trailing zeros) instruction for O(1) allocation.
/// Each slot is cache-line aligned for optimal access patterns.
pub struct OptimizedSlabAllocator<B: MemoryBacking, const N: usize = 256> {
    /// Memory backing store.
    backing: B,

    /// Pointer to the slot data area.
    data_ptr: *mut u8,

    /// Free slot bitmap (1 = free, 0 = allocated).
    /// Using u64 chunks for efficient CTZ operation.
    /// Fixed size supports up to 256 slots (4 * 64 bits).
    free_bitmap: [u64; MAX_BITMAP_CHUNKS],

    /// Generation counters for each slot.
    /// Fixed size array to avoid const generic arithmetic.
    generations: [SlotGeneration; MAX_SLOTS],

    /// Size of each slot in bytes (aligned to cache line).
    slot_size: usize,

    /// Number of slots in use.
    slot_count: usize,

    /// Number of currently allocated slots.
    allocated_count: usize,

    /// Hint for next free chunk (optimization for sequential allocation).
    free_hint: usize,
}

impl<B: MemoryBacking, const N: usize> OptimizedSlabAllocator<B, N> {
    /// Number of bitmap chunks needed for the given slot count.
    const BITMAP_CHUNKS: usize = if N <= 64 {
        1
    } else if N <= 128 {
        2
    } else if N <= 192 {
        3
    } else {
        4
    };

    /// Creates a new optimized slab allocator.
    ///
    /// # Arguments
    ///
    /// * `backing` - Memory backing store
    /// * `slot_size` - Size of each slot in bytes (will be aligned to cache line)
    ///
    /// # Panics
    ///
    /// Panics if N > 256 (maximum supported slots).
    ///
    /// # Errors
    ///
    /// Returns `OutOfMemory` if the backing cannot allocate sufficient memory.
    pub fn new(mut backing: B, slot_size: usize) -> Result<Self> {
        assert!(
            N <= MAX_SLOTS,
            "OptimizedSlabAllocator supports max 256 slots"
        );

        if slot_size == 0 {
            return Err(KernelError::InvalidArgument);
        }

        // Align slot size to cache line for optimal access
        let aligned_slot_size = (slot_size + CACHE_LINE_SIZE - 1) & !(CACHE_LINE_SIZE - 1);
        let total_size = aligned_slot_size
            .checked_mul(N)
            .ok_or(KernelError::InvalidArgument)?;

        // Allocate memory for slot data
        let (data_ptr, _) = backing.allocate(total_size)?;

        // Initialize free bitmap with all slots free (all bits set to 1)
        let mut free_bitmap = [0u64; MAX_BITMAP_CHUNKS];
        for i in 0..MAX_BITMAP_CHUNKS {
            let remaining = N.saturating_sub(i * BITS_PER_CHUNK);
            if remaining >= BITS_PER_CHUNK {
                free_bitmap[i] = u64::MAX; // All 64 slots free
            } else if remaining > 0 {
                free_bitmap[i] = (1u64 << remaining) - 1; // Partial chunk
            }
        }

        Ok(Self {
            backing,
            data_ptr,
            free_bitmap,
            generations: [SlotGeneration::new(); MAX_SLOTS],
            slot_size: aligned_slot_size,
            slot_count: N,
            allocated_count: 0,
            free_hint: 0,
        })
    }

    /// Allocates a slot using CTZ for O(1) free slot finding.
    ///
    /// # Errors
    ///
    /// Returns `SlabFull` if no slots are available.
    #[inline]
    pub fn alloc(&mut self) -> Result<OptimizedSlotHandle> {
        // Start from hint for better locality
        let start_chunk = self.free_hint;

        // First pass: search from hint to end
        for chunk_idx in start_chunk..Self::BITMAP_CHUNKS {
            if let Some(handle) = self.try_alloc_from_chunk(chunk_idx) {
                self.free_hint = chunk_idx;
                return Ok(handle);
            }
        }

        // Second pass: search from beginning to hint
        for chunk_idx in 0..start_chunk {
            if let Some(handle) = self.try_alloc_from_chunk(chunk_idx) {
                self.free_hint = chunk_idx;
                return Ok(handle);
            }
        }

        Err(KernelError::SlabFull)
    }

    /// Tries to allocate from a specific bitmap chunk.
    #[inline]
    fn try_alloc_from_chunk(&mut self, chunk_idx: usize) -> Option<OptimizedSlotHandle> {
        let chunk = self.free_bitmap[chunk_idx];
        if chunk == 0 {
            return None; // No free slots in this chunk
        }

        // CTZ finds first set bit (first free slot)
        let bit_pos = chunk.trailing_zeros() as usize;
        let slot_idx = chunk_idx * BITS_PER_CHUNK + bit_pos;

        if slot_idx >= self.slot_count {
            return None; // Beyond valid slot range
        }

        // Clear the bit (mark as allocated)
        self.free_bitmap[chunk_idx] &= !(1u64 << bit_pos);
        self.allocated_count += 1;

        let generation = self.generations[slot_idx].generation;
        Some(OptimizedSlotHandle::new(slot_idx as u32, generation))
    }

    /// Frees a previously allocated slot (O(1) operation).
    ///
    /// # Errors
    ///
    /// Returns `InvalidSlot` if the handle is stale or invalid.
    #[inline]
    pub fn free(&mut self, handle: OptimizedSlotHandle) -> Result<()> {
        self.validate_handle(handle)?;

        let slot_idx = handle.index() as usize;
        let chunk_idx = slot_idx / BITS_PER_CHUNK;
        let bit_pos = slot_idx % BITS_PER_CHUNK;

        // Increment generation to invalidate existing handles
        self.generations[slot_idx].generation =
            self.generations[slot_idx].generation.wrapping_add(1);

        // Set the bit (mark as free)
        self.free_bitmap[chunk_idx] |= 1u64 << bit_pos;
        self.allocated_count = self.allocated_count.saturating_sub(1);

        // Update hint if this slot is earlier
        if chunk_idx < self.free_hint {
            self.free_hint = chunk_idx;
        }

        // Zero the slot data for security
        self.zero_slot(slot_idx);

        Ok(())
    }

    /// Returns a pointer to the slot data (O(1) operation).
    ///
    /// # Errors
    ///
    /// Returns `InvalidSlot` if the handle is stale.
    #[inline]
    pub fn slot_ptr(&self, handle: OptimizedSlotHandle) -> Result<*mut u8> {
        self.validate_handle(handle)?;

        let offset = handle.index() as usize * self.slot_size;
        // SAFETY: We've validated the handle and bounds
        Ok(unsafe { self.data_ptr.add(offset) })
    }

    /// Reads data from a slot.
    #[inline]
    pub fn read(&self, handle: OptimizedSlotHandle, buf: &mut [u8]) -> Result<usize> {
        let ptr = self.slot_ptr(handle)?;
        let to_read = buf.len().min(self.slot_size);

        // SAFETY: We've validated the handle and bounds
        unsafe {
            core::ptr::copy_nonoverlapping(ptr, buf.as_mut_ptr(), to_read);
        }

        Ok(to_read)
    }

    /// Writes data to a slot.
    #[inline]
    pub fn write(&mut self, handle: OptimizedSlotHandle, data: &[u8]) -> Result<usize> {
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
    #[inline]
    fn validate_handle(&self, handle: OptimizedSlotHandle) -> Result<()> {
        if handle.is_invalid() {
            return Err(KernelError::InvalidSlot);
        }

        let slot_idx = handle.index() as usize;
        if slot_idx >= self.slot_count {
            return Err(KernelError::InvalidSlot);
        }

        // Check generation
        if self.generations[slot_idx].generation != handle.generation() {
            return Err(KernelError::InvalidSlot);
        }

        // Check if slot is allocated (bit should be 0)
        let chunk_idx = slot_idx / BITS_PER_CHUNK;
        let bit_pos = slot_idx % BITS_PER_CHUNK;
        if (self.free_bitmap[chunk_idx] & (1u64 << bit_pos)) != 0 {
            return Err(KernelError::InvalidSlot);
        }

        Ok(())
    }

    /// Zeros the data in a slot.
    #[inline]
    fn zero_slot(&mut self, slot_idx: usize) {
        let offset = slot_idx * self.slot_size;
        // SAFETY: slot_idx is validated before this is called
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
    pub fn slot_count(&self) -> usize {
        self.slot_count
    }

    /// Returns the number of currently allocated slots.
    #[inline]
    #[must_use]
    pub fn allocated_count(&self) -> usize {
        self.allocated_count
    }

    /// Returns the number of free slots.
    #[inline]
    #[must_use]
    pub fn free_count(&self) -> usize {
        self.slot_count - self.allocated_count
    }

    /// Returns true if the slab is full.
    #[inline]
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.allocated_count == self.slot_count
    }

    /// Returns true if the slab is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.allocated_count == 0
    }

    /// Returns the backing memory reference.
    #[inline]
    pub fn backing(&self) -> &B {
        &self.backing
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backing::StaticBacking;

    #[test]
    fn test_optimized_handle_packing() {
        let handle = OptimizedSlotHandle::new(1000, 42);
        assert_eq!(handle.index(), 1000);
        assert_eq!(handle.generation(), 42);
        assert!(!handle.is_invalid());

        let invalid = OptimizedSlotHandle::invalid();
        assert!(invalid.is_invalid());
    }

    #[test]
    fn test_optimized_slab_alloc_free() {
        let backing = StaticBacking::<16384>::new();
        let mut slab = OptimizedSlabAllocator::<_, 64>::new(backing, 64).unwrap();

        assert_eq!(slab.slot_count(), 64);
        assert_eq!(slab.allocated_count(), 0);
        assert_eq!(slab.free_count(), 64);

        // Allocate a slot
        let handle = slab.alloc().unwrap();
        assert!(!handle.is_invalid());
        assert_eq!(slab.allocated_count(), 1);

        // Free the slot
        slab.free(handle).unwrap();
        assert_eq!(slab.allocated_count(), 0);
    }

    #[test]
    fn test_optimized_slab_generation_counter() {
        let backing = StaticBacking::<16384>::new();
        let mut slab = OptimizedSlabAllocator::<_, 64>::new(backing, 64).unwrap();

        let handle1 = slab.alloc().unwrap();
        slab.free(handle1).unwrap();

        // Old handle should now be invalid
        assert!(slab.free(handle1).is_err());

        // New allocation at same slot should have different generation
        let handle2 = slab.alloc().unwrap();
        assert_eq!(handle1.index(), handle2.index());
        assert_ne!(handle1.generation(), handle2.generation());
    }

    #[test]
    fn test_optimized_slab_ctz_allocation() {
        let backing = StaticBacking::<32768>::new();
        let mut slab = OptimizedSlabAllocator::<_, 128>::new(backing, 64).unwrap();

        // Allocate first 64 slots
        let mut handles = [OptimizedSlotHandle::invalid(); 64];
        for (i, handle) in handles.iter_mut().enumerate() {
            *handle = slab.alloc().unwrap();
            assert_eq!(handle.index() as usize, i);
        }

        // Free slot 32
        slab.free(handles[32]).unwrap();

        // Next allocation should reuse slot 32 (CTZ finds it)
        let new_handle = slab.alloc().unwrap();
        assert_eq!(new_handle.index(), 32);
    }

    #[test]
    fn test_optimized_slab_full() {
        let backing = StaticBacking::<1024>::new();
        let mut slab = OptimizedSlabAllocator::<_, 8>::new(backing, 64).unwrap();

        // Allocate all slots
        for _ in 0..8 {
            slab.alloc().unwrap();
        }

        assert!(slab.is_full());

        // Should fail to allocate more
        assert!(slab.alloc().is_err());
    }

    #[test]
    fn test_optimized_slab_read_write() {
        let backing = StaticBacking::<16384>::new();
        let mut slab = OptimizedSlabAllocator::<_, 64>::new(backing, 64).unwrap();

        let handle = slab.alloc().unwrap();

        // Write data
        let data = b"Hello, RuVix Optimized!";
        slab.write(handle, data).unwrap();

        // Read it back
        let mut buf = [0u8; 64];
        let read = slab.read(handle, &mut buf).unwrap();
        assert_eq!(read, 64);
        assert_eq!(&buf[..data.len()], data);
    }

    #[test]
    fn test_optimized_slab_bitmap_chunks() {
        // Test allocation across multiple bitmap chunks (>64 slots)
        let backing = StaticBacking::<65536>::new();
        let mut slab = OptimizedSlabAllocator::<_, 200>::new(backing, 64).unwrap();

        // Allocate all 200 slots (spans 4 bitmap chunks)
        let mut handles = [OptimizedSlotHandle::invalid(); 200];
        for (i, handle) in handles.iter_mut().enumerate() {
            *handle = slab.alloc().unwrap();
            assert_eq!(handle.index() as usize, i);
        }

        assert!(slab.is_full());
        assert_eq!(slab.allocated_count(), 200);

        // Free slot 150 (in third chunk)
        slab.free(handles[150]).unwrap();
        assert_eq!(slab.allocated_count(), 199);

        // Next allocation should find slot 150
        let new_handle = slab.alloc().unwrap();
        assert_eq!(new_handle.index(), 150);
    }
}
