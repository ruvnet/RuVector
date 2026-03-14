//! Optimized capability structures for high-performance kernel operations.
//!
//! This module provides cache-optimized data structures following ADR-087
//! performance targets:
//! - Capability lookup: O(1), <50ns
//! - Cache-line aligned (64 bytes) for minimal cache misses
//! - Direct indexing for constant-time access
//! - Generation counters for stale handle detection

use crate::error::{CapError, CapResult};
use ruvix_types::{CapHandle, CapRights, Capability, ObjectType, TaskHandle};

/// Cache-line size in bytes.
const CACHE_LINE_SIZE: usize = 64;

/// Optimized capability slot aligned to cache line (64 bytes).
///
/// This structure is designed for O(1) lookup with minimal cache misses.
/// Each slot occupies exactly one cache line, ensuring that capability
/// lookups only touch a single cache line.
///
/// Layout (64 bytes total):
/// - capability: 32 bytes (object_id + type + rights + badge + epoch)
/// - generation: 4 bytes
/// - flags: 4 bytes
/// - owner: 8 bytes (TaskHandle)
/// - depth: 1 byte
/// - _reserved: 15 bytes (padding to 64 bytes)
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct OptimizedCapSlot {
    /// The capability data (valid if FLAG_VALID is set).
    pub capability: Capability,

    /// Generation counter for stale handle detection.
    /// Incremented each time the slot is reused.
    pub generation: u32,

    /// Slot flags.
    pub flags: u32,

    /// The task that owns this capability.
    pub owner: TaskHandle,

    /// Delegation depth (0 = root capability).
    pub depth: u8,

    /// Reserved for future use and padding.
    _reserved: [u8; 15],
}

impl OptimizedCapSlot {
    /// Flag indicating this slot is valid (in use).
    pub const FLAG_VALID: u32 = 1 << 0;

    /// Flag indicating this is a root capability (no parent).
    pub const FLAG_ROOT: u32 = 1 << 1;

    /// Flag indicating this capability is revoked but not yet cleaned up.
    pub const FLAG_REVOKED: u32 = 1 << 2;

    /// Creates a new empty (invalid) slot.
    #[inline]
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            capability: Capability::new(0, ObjectType::Task, CapRights::NONE, 0, 0),
            generation: 0,
            flags: 0,
            owner: TaskHandle::null(),
            depth: 0,
            _reserved: [0; 15],
        }
    }

    /// Creates a new valid slot with a root capability.
    #[inline]
    #[must_use]
    pub const fn new_root(capability: Capability, generation: u32, owner: TaskHandle) -> Self {
        Self {
            capability,
            generation,
            flags: Self::FLAG_VALID | Self::FLAG_ROOT,
            owner,
            depth: 0,
            _reserved: [0; 15],
        }
    }

    /// Creates a new valid slot with a derived capability.
    #[inline]
    #[must_use]
    pub const fn new_derived(
        capability: Capability,
        generation: u32,
        owner: TaskHandle,
        depth: u8,
    ) -> Self {
        Self {
            capability,
            generation,
            flags: Self::FLAG_VALID,
            owner,
            depth,
            _reserved: [0; 15],
        }
    }

    /// Returns true if this slot is valid.
    #[inline]
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        (self.flags & Self::FLAG_VALID) != 0
    }

    /// Returns true if this is a root capability.
    #[inline]
    #[must_use]
    pub const fn is_root(&self) -> bool {
        (self.flags & Self::FLAG_ROOT) != 0
    }

    /// Returns true if this capability has been revoked.
    #[inline]
    #[must_use]
    pub const fn is_revoked(&self) -> bool {
        (self.flags & Self::FLAG_REVOKED) != 0
    }

    /// Returns the handle for this slot at the given index.
    #[inline]
    #[must_use]
    pub const fn handle(&self, index: u32) -> CapHandle {
        CapHandle::new(index, self.generation)
    }

    /// Checks if the given handle matches this slot.
    #[inline]
    #[must_use]
    pub const fn matches_handle(&self, handle: CapHandle) -> bool {
        self.is_valid() && self.generation == handle.raw().generation
    }

    /// Invalidates this slot (marks as free for reuse).
    #[inline]
    pub fn invalidate(&mut self) {
        self.flags = 0;
        self.generation = self.generation.wrapping_add(1);
    }

    /// Marks this slot as revoked.
    #[inline]
    pub fn revoke(&mut self) {
        self.flags |= Self::FLAG_REVOKED;
        self.flags &= !Self::FLAG_VALID;
    }
}

impl Default for OptimizedCapSlot {
    fn default() -> Self {
        Self::empty()
    }
}

// Compile-time assertion that OptimizedCapSlot is exactly 64 bytes
const _: () = assert!(core::mem::size_of::<OptimizedCapSlot>() == CACHE_LINE_SIZE);

/// Maximum bitmap chunks for 256-slot table (4 x 64 bits = 256 slots).
const MAX_BITMAP_CHUNKS: usize = 4;

/// Optimized capability table with direct indexing.
///
/// Uses a flat array of cache-line aligned slots for O(1) lookup.
/// Slot lookup is a single array index operation with no branching
/// in the hot path.
pub struct OptimizedCapTable<const N: usize = 256> {
    /// Cache-line aligned slots array.
    slots: [OptimizedCapSlot; N],

    /// Number of currently valid entries.
    count: usize,

    /// Bitmap for fast free slot finding.
    /// Each bit represents a slot (1 = free, 0 = in use).
    /// Using u64 chunks for efficient CTZ (count trailing zeros) operation.
    /// Fixed size supports up to 256 slots (4 * 64 bits).
    free_bitmap: [u64; MAX_BITMAP_CHUNKS],
}

impl<const N: usize> OptimizedCapTable<N> {
    /// Number of u64 chunks needed for the bitmap.
    const BITMAP_CHUNKS: usize = if N <= 64 { 1 } else if N <= 128 { 2 } else if N <= 192 { 3 } else { 4 };

    /// Creates a new empty capability table.
    ///
    /// # Panics
    ///
    /// Panics if N > 256 (bitmap size limit).
    #[must_use]
    pub fn new() -> Self {
        assert!(N <= 256, "OptimizedCapTable supports max 256 slots");

        // Initialize with all slots free (all bits set to 1)
        let mut free_bitmap = [0u64; MAX_BITMAP_CHUNKS];
        for i in 0..MAX_BITMAP_CHUNKS {
            let remaining = N.saturating_sub(i * 64);
            if remaining >= 64 {
                free_bitmap[i] = u64::MAX;
            } else if remaining > 0 {
                free_bitmap[i] = (1u64 << remaining) - 1;
            }
        }

        Self {
            slots: [OptimizedCapSlot::empty(); N],
            count: 0,
            free_bitmap,
        }
    }

    /// Returns the capacity of the table.
    #[inline]
    #[must_use]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Returns the number of valid entries.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Returns true if the table is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns true if the table is full.
    #[inline]
    #[must_use]
    pub const fn is_full(&self) -> bool {
        self.count >= N
    }

    /// Looks up a capability by handle (O(1), <50ns target).
    ///
    /// This is the performance-critical path optimized for:
    /// - Single cache line access
    /// - Minimal branching
    /// - Direct index calculation
    #[inline]
    pub fn get(&self, handle: CapHandle) -> CapResult<&Capability> {
        let index = handle.raw().id as usize;

        // Bounds check (should be predictable branch)
        if index >= N {
            return Err(CapError::InvalidHandle);
        }

        // Direct slot access (single cache line)
        let slot = &self.slots[index];

        // Check validity and generation in one comparison if possible
        if !slot.is_valid() {
            return Err(CapError::InvalidHandle);
        }

        if slot.generation != handle.raw().generation {
            return Err(CapError::StaleHandle);
        }

        Ok(&slot.capability)
    }

    /// Looks up a capability slot by handle.
    #[inline]
    pub fn get_slot(&self, handle: CapHandle) -> CapResult<&OptimizedCapSlot> {
        let index = handle.raw().id as usize;

        if index >= N {
            return Err(CapError::InvalidHandle);
        }

        let slot = &self.slots[index];

        if !slot.is_valid() {
            return Err(CapError::InvalidHandle);
        }

        if slot.generation != handle.raw().generation {
            return Err(CapError::StaleHandle);
        }

        Ok(slot)
    }

    /// Allocates a slot for a root capability using CTZ for O(1) free slot finding.
    pub fn allocate_root(
        &mut self,
        capability: Capability,
        owner: TaskHandle,
    ) -> CapResult<CapHandle> {
        let index = self.find_free_slot_ctz()?;
        let slot = &mut self.slots[index];
        let generation = slot.generation;

        *slot = OptimizedCapSlot::new_root(capability, generation, owner);
        self.mark_slot_used(index);
        self.count += 1;

        Ok(CapHandle::new(index as u32, generation))
    }

    /// Allocates a slot for a derived capability.
    pub fn allocate_derived(
        &mut self,
        capability: Capability,
        owner: TaskHandle,
        depth: u8,
    ) -> CapResult<CapHandle> {
        let index = self.find_free_slot_ctz()?;
        let slot = &mut self.slots[index];
        let generation = slot.generation;

        *slot = OptimizedCapSlot::new_derived(capability, generation, owner, depth);
        self.mark_slot_used(index);
        self.count += 1;

        Ok(CapHandle::new(index as u32, generation))
    }

    /// Deallocates a capability slot.
    pub fn deallocate(&mut self, handle: CapHandle) -> CapResult<()> {
        let index = handle.raw().id as usize;

        if index >= N {
            return Err(CapError::InvalidHandle);
        }

        let slot = &mut self.slots[index];

        if !slot.is_valid() {
            return Err(CapError::InvalidHandle);
        }

        if slot.generation != handle.raw().generation {
            return Err(CapError::StaleHandle);
        }

        slot.invalidate();
        self.mark_slot_free(index);
        self.count -= 1;

        Ok(())
    }

    /// Finds a free slot using CTZ (count trailing zeros) instruction.
    ///
    /// This is O(1) amortized - we scan bitmap chunks and use CTZ
    /// to find the first set bit, which maps to hardware instruction.
    #[inline]
    fn find_free_slot_ctz(&self) -> CapResult<usize> {
        for (chunk_idx, &chunk) in self.free_bitmap.iter().enumerate() {
            if chunk != 0 {
                // CTZ finds first set bit (free slot)
                let bit_pos = chunk.trailing_zeros() as usize;
                let slot_idx = chunk_idx * 64 + bit_pos;

                if slot_idx < N {
                    return Ok(slot_idx);
                }
            }
        }

        Err(CapError::TableFull)
    }

    /// Marks a slot as used in the bitmap.
    #[inline]
    fn mark_slot_used(&mut self, index: usize) {
        let chunk_idx = index / 64;
        let bit_pos = index % 64;
        self.free_bitmap[chunk_idx] &= !(1u64 << bit_pos);
    }

    /// Marks a slot as free in the bitmap.
    #[inline]
    fn mark_slot_free(&mut self, index: usize) {
        let chunk_idx = index / 64;
        let bit_pos = index % 64;
        self.free_bitmap[chunk_idx] |= 1u64 << bit_pos;
    }

    /// Returns an iterator over all valid entries.
    pub fn iter(&self) -> impl Iterator<Item = (CapHandle, &OptimizedCapSlot)> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_valid())
            .map(|(i, s)| (s.handle(i as u32), s))
    }
}

impl<const N: usize> Default for OptimizedCapTable<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimized_cap_slot_size() {
        assert_eq!(core::mem::size_of::<OptimizedCapSlot>(), 64);
        assert_eq!(core::mem::align_of::<OptimizedCapSlot>(), 64);
    }

    #[test]
    fn test_optimized_table_allocate_root() {
        let mut table = OptimizedCapTable::<64>::new();
        let owner = TaskHandle::new(1, 0);
        let cap = Capability::new(100, ObjectType::VectorStore, CapRights::ALL, 0, 0);

        let handle = table.allocate_root(cap, owner).unwrap();
        assert_eq!(table.len(), 1);

        let retrieved = table.get(handle).unwrap();
        assert_eq!(retrieved.object_id, 100);
    }

    #[test]
    fn test_optimized_table_deallocate() {
        let mut table = OptimizedCapTable::<64>::new();
        let owner = TaskHandle::new(1, 0);
        let cap = Capability::new(100, ObjectType::Region, CapRights::READ, 0, 0);

        let handle = table.allocate_root(cap, owner).unwrap();
        assert_eq!(table.len(), 1);

        table.deallocate(handle).unwrap();
        assert_eq!(table.len(), 0);

        // Old handle should be invalid
        assert!(matches!(table.get(handle), Err(CapError::InvalidHandle)));
    }

    #[test]
    fn test_optimized_table_generation_counter() {
        let mut table = OptimizedCapTable::<64>::new();
        let owner = TaskHandle::new(1, 0);
        let cap = Capability::new(100, ObjectType::Queue, CapRights::WRITE, 0, 0);

        let handle1 = table.allocate_root(cap, owner).unwrap();
        table.deallocate(handle1).unwrap();

        // Allocate again in the same slot
        let handle2 = table.allocate_root(cap, owner).unwrap();

        // Same slot index but different generation
        assert_eq!(handle1.raw().id, handle2.raw().id);
        assert_ne!(handle1.raw().generation, handle2.raw().generation);

        // Old handle should be stale
        assert!(matches!(table.get(handle1), Err(CapError::StaleHandle)));
    }

    #[test]
    fn test_optimized_table_full() {
        let mut table = OptimizedCapTable::<4>::new();
        let owner = TaskHandle::new(1, 0);
        let cap = Capability::new(100, ObjectType::Timer, CapRights::READ, 0, 0);

        for _ in 0..4 {
            table.allocate_root(cap, owner).unwrap();
        }

        assert!(table.is_full());
        assert!(matches!(
            table.allocate_root(cap, owner),
            Err(CapError::TableFull)
        ));
    }

    #[test]
    fn test_optimized_table_ctz_allocation() {
        let mut table = OptimizedCapTable::<128>::new();
        let owner = TaskHandle::new(1, 0);
        let cap = Capability::new(100, ObjectType::Region, CapRights::READ, 0, 0);

        // Allocate first 64 slots using fixed-size array
        let mut handles = [CapHandle::null(); 64];
        for (i, handle) in handles.iter_mut().enumerate() {
            *handle = table.allocate_root(cap, owner).unwrap();
            assert_eq!(handle.raw().id as usize, i);
        }

        // Free slot 32
        table.deallocate(handles[32]).unwrap();

        // Next allocation should reuse slot 32 (CTZ finds it)
        let new_handle = table.allocate_root(cap, owner).unwrap();
        assert_eq!(new_handle.raw().id, 32);
    }

    #[test]
    fn test_slot_flags() {
        let owner = TaskHandle::new(1, 0);
        let cap = Capability::new(100, ObjectType::Region, CapRights::READ, 0, 0);

        let root_slot = OptimizedCapSlot::new_root(cap, 0, owner);
        assert!(root_slot.is_valid());
        assert!(root_slot.is_root());
        assert!(!root_slot.is_revoked());

        let derived_slot = OptimizedCapSlot::new_derived(cap, 0, owner, 1);
        assert!(derived_slot.is_valid());
        assert!(!derived_slot.is_root());
        assert!(!derived_slot.is_revoked());

        let mut slot = root_slot;
        slot.revoke();
        assert!(!slot.is_valid());
        assert!(slot.is_revoked());
    }
}
