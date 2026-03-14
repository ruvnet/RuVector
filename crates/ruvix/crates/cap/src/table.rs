//! Capability table implementation.
//!
//! Each task has a capability table that stores its held capabilities.
//! The table uses a slot-based design with generation counters for
//! stale handle detection.

use crate::error::{CapError, CapResult};
use crate::DEFAULT_CAP_TABLE_CAPACITY;
use ruvix_types::{CapHandle, CapRights, Capability, ObjectType, TaskHandle};

/// Entry in the capability table.
///
/// Each slot contains either a valid capability or is marked as free.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapTableEntry {
    /// The capability data (valid if `is_valid` is true).
    pub capability: Capability,

    /// Generation counter for stale handle detection.
    /// Incremented each time the slot is reused.
    pub generation: u32,

    /// Whether this entry is currently valid (in use).
    pub is_valid: bool,

    /// The task that owns this capability.
    pub owner: TaskHandle,

    /// Delegation depth (0 = root capability).
    pub depth: u8,

    /// Parent capability handle (for derivation tracking).
    /// Null handle if this is a root capability.
    pub parent: CapHandle,
}

impl CapTableEntry {
    /// Creates a new empty (invalid) entry.
    #[inline]
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            capability: Capability::new(0, ObjectType::Task, CapRights::NONE, 0, 0),
            generation: 0,
            is_valid: false,
            owner: TaskHandle::null(),
            depth: 0,
            parent: CapHandle::null(),
        }
    }

    /// Creates a new valid entry with a root capability.
    #[inline]
    #[must_use]
    pub const fn new_root(
        capability: Capability,
        generation: u32,
        owner: TaskHandle,
    ) -> Self {
        Self {
            capability,
            generation,
            is_valid: true,
            owner,
            depth: 0,
            parent: CapHandle::null(),
        }
    }

    /// Creates a new valid entry with a derived capability.
    #[inline]
    #[must_use]
    pub const fn new_derived(
        capability: Capability,
        generation: u32,
        owner: TaskHandle,
        depth: u8,
        parent: CapHandle,
    ) -> Self {
        Self {
            capability,
            generation,
            is_valid: true,
            owner,
            depth,
            parent: CapHandle::null(),
        }
    }

    /// Returns the handle for this entry at the given index.
    #[inline]
    #[must_use]
    pub const fn handle(&self, index: u32) -> CapHandle {
        CapHandle::new(index, self.generation)
    }

    /// Checks if the given handle matches this entry.
    #[inline]
    #[must_use]
    pub const fn matches_handle(&self, handle: CapHandle) -> bool {
        self.is_valid && self.generation == handle.raw().generation
    }

    /// Invalidates this entry (marks as free for reuse).
    #[inline]
    pub fn invalidate(&mut self) {
        self.is_valid = false;
        self.generation = self.generation.wrapping_add(1);
    }
}

impl Default for CapTableEntry {
    fn default() -> Self {
        Self::empty()
    }
}

/// The capability table for a task.
///
/// Uses a fixed-size array to avoid dynamic allocation in no_std environments.
/// Slots are reused with generation counters for stale handle detection.
pub struct CapabilityTable<const N: usize = DEFAULT_CAP_TABLE_CAPACITY> {
    /// The table entries.
    entries: [CapTableEntry; N],

    /// Number of currently valid entries.
    count: usize,

    /// Index of the first potentially free slot (optimization).
    free_hint: usize,
}

impl<const N: usize> CapabilityTable<N> {
    /// Creates a new empty capability table.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: [CapTableEntry::empty(); N],
            count: 0,
            free_hint: 0,
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

    /// Allocates a new slot for a root capability.
    ///
    /// Returns the handle to the new capability.
    pub fn allocate_root(
        &mut self,
        capability: Capability,
        owner: TaskHandle,
    ) -> CapResult<CapHandle> {
        let index = self.find_free_slot()?;
        let entry = &mut self.entries[index];
        let generation = entry.generation;

        *entry = CapTableEntry::new_root(capability, generation, owner);
        self.count += 1;

        Ok(CapHandle::new(index as u32, generation))
    }

    /// Allocates a new slot for a derived capability.
    ///
    /// Returns the handle to the new capability.
    pub fn allocate_derived(
        &mut self,
        capability: Capability,
        owner: TaskHandle,
        depth: u8,
        parent: CapHandle,
    ) -> CapResult<CapHandle> {
        let index = self.find_free_slot()?;
        let entry = &mut self.entries[index];
        let generation = entry.generation;

        *entry = CapTableEntry::new_derived(capability, generation, owner, depth, parent);
        self.count += 1;

        Ok(CapHandle::new(index as u32, generation))
    }

    /// Looks up a capability by handle.
    ///
    /// Returns the entry if the handle is valid.
    pub fn lookup(&self, handle: CapHandle) -> CapResult<&CapTableEntry> {
        let index = handle.raw().id as usize;
        if index >= N {
            return Err(CapError::InvalidHandle);
        }

        let entry = &self.entries[index];
        if !entry.is_valid {
            return Err(CapError::InvalidHandle);
        }
        if entry.generation != handle.raw().generation {
            return Err(CapError::StaleHandle);
        }

        Ok(entry)
    }

    /// Looks up a capability mutably by handle.
    pub fn lookup_mut(&mut self, handle: CapHandle) -> CapResult<&mut CapTableEntry> {
        let index = handle.raw().id as usize;
        if index >= N {
            return Err(CapError::InvalidHandle);
        }

        let entry = &mut self.entries[index];
        if !entry.is_valid {
            return Err(CapError::InvalidHandle);
        }
        if entry.generation != handle.raw().generation {
            return Err(CapError::StaleHandle);
        }

        Ok(entry)
    }

    /// Deallocates a capability slot.
    ///
    /// Increments the generation counter to invalidate existing handles.
    pub fn deallocate(&mut self, handle: CapHandle) -> CapResult<()> {
        let index = handle.raw().id as usize;
        if index >= N {
            return Err(CapError::InvalidHandle);
        }

        let entry = &mut self.entries[index];
        if !entry.is_valid {
            return Err(CapError::InvalidHandle);
        }
        if entry.generation != handle.raw().generation {
            return Err(CapError::StaleHandle);
        }

        entry.invalidate();
        self.count -= 1;

        // Update free hint if this slot is earlier
        if index < self.free_hint {
            self.free_hint = index;
        }

        Ok(())
    }

    /// Returns an iterator over all valid entries.
    pub fn iter(&self) -> impl Iterator<Item = (CapHandle, &CapTableEntry)> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.is_valid)
            .map(|(i, e)| (e.handle(i as u32), e))
    }

    /// Finds a free slot in the table.
    fn find_free_slot(&mut self) -> CapResult<usize> {
        // Start from the hint
        for i in self.free_hint..N {
            if !self.entries[i].is_valid {
                self.free_hint = i + 1;
                return Ok(i);
            }
        }

        // Wrap around from the beginning
        for i in 0..self.free_hint {
            if !self.entries[i].is_valid {
                self.free_hint = i + 1;
                return Ok(i);
            }
        }

        Err(CapError::TableFull)
    }
}

impl<const N: usize> Default for CapabilityTable<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cap_table_allocate_root() {
        let mut table = CapabilityTable::<64>::new();
        let owner = TaskHandle::new(1, 0);
        let cap = Capability::new(100, ObjectType::VectorStore, CapRights::ALL, 0, 0);

        let handle = table.allocate_root(cap, owner).unwrap();
        assert_eq!(table.len(), 1);

        let entry = table.lookup(handle).unwrap();
        assert_eq!(entry.capability.object_id, 100);
        assert_eq!(entry.depth, 0);
        assert!(entry.parent.is_null());
    }

    #[test]
    fn test_cap_table_deallocate() {
        let mut table = CapabilityTable::<64>::new();
        let owner = TaskHandle::new(1, 0);
        let cap = Capability::new(100, ObjectType::Region, CapRights::READ, 0, 0);

        let handle = table.allocate_root(cap, owner).unwrap();
        assert_eq!(table.len(), 1);

        table.deallocate(handle).unwrap();
        assert_eq!(table.len(), 0);

        // Old handle should be stale
        assert_eq!(table.lookup(handle), Err(CapError::InvalidHandle));
    }

    #[test]
    fn test_cap_table_generation_counter() {
        let mut table = CapabilityTable::<64>::new();
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
        assert_eq!(table.lookup(handle1), Err(CapError::StaleHandle));
    }

    #[test]
    fn test_cap_table_full() {
        let mut table = CapabilityTable::<4>::new();
        let owner = TaskHandle::new(1, 0);
        let cap = Capability::new(100, ObjectType::Timer, CapRights::READ, 0, 0);

        for _ in 0..4 {
            table.allocate_root(cap, owner).unwrap();
        }

        assert!(table.is_full());
        assert_eq!(table.allocate_root(cap, owner), Err(CapError::TableFull));
    }
}
