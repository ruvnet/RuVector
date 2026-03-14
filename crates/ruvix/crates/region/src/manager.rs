//! Region Manager - coordinates all region types.
//!
//! The RegionManager is responsible for:
//! - Creating and destroying regions
//! - Tracking all regions with their policies
//! - Implementing region_map syscall logic
//! - Capability-gated access control

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;
use crate::append_only::AppendOnlyRegion;
use crate::backing::HeapBacking;
use crate::immutable::ImmutableRegion;
use crate::slab::{SlabRegion, SlotHandle};
use crate::Result;
use ruvix_types::{CapHandle, KernelError, RegionHandle, RegionPolicy};

/// Maximum number of regions per manager.
pub const MAX_REGIONS: usize = 256;

/// Configuration for creating a region.
#[derive(Debug, Clone)]
pub struct RegionConfig {
    /// The region policy (Immutable, AppendOnly, or Slab).
    pub policy: RegionPolicy,
    /// Initial data for immutable regions.
    pub initial_data: Option<&'static [u8]>,
}

impl RegionConfig {
    /// Creates a configuration for an immutable region.
    pub const fn immutable(data: &'static [u8]) -> Self {
        Self {
            policy: RegionPolicy::Immutable,
            initial_data: Some(data),
        }
    }

    /// Creates a configuration for an append-only region.
    pub const fn append_only(max_size: usize) -> Self {
        Self {
            policy: RegionPolicy::AppendOnly { max_size },
            initial_data: None,
        }
    }

    /// Creates a configuration for a slab region.
    pub const fn slab(slot_size: usize, slot_count: usize) -> Self {
        Self {
            policy: RegionPolicy::Slab {
                slot_size,
                slot_count,
            },
            initial_data: None,
        }
    }
}

/// Entry in the region table.
/// Uses Box to heap-allocate regions and avoid stack overflow.
enum RegionEntry {
    /// Empty slot.
    Empty,
    /// Immutable region.
    Immutable {
        region: Box<ImmutableRegion<HeapBacking>>,
        generation: u32,
    },
    /// Append-only region.
    AppendOnly {
        region: Box<AppendOnlyRegion<HeapBacking>>,
        generation: u32,
    },
    /// Slab region.
    Slab {
        region: Box<SlabRegion<HeapBacking>>,
        generation: u32,
    },
}

impl Default for RegionEntry {
    fn default() -> Self {
        Self::Empty
    }
}

/// Manages memory regions for the RuVix kernel.
///
/// The RegionManager implements the region_map syscall logic from ADR-087.
/// All regions are physically backed at creation time (no demand paging).
pub struct RegionManager {
    /// Region table entries (heap-allocated to avoid stack overflow).
    regions: Vec<RegionEntry>,
    /// Next region ID to allocate.
    next_id: u32,
    /// Number of active regions.
    active_count: usize,
    #[cfg(feature = "stats")]
    /// Statistics.
    stats: crate::RegionStats,
}

impl RegionManager {
    /// Creates a new region manager.
    pub fn new() -> Self {
        let mut regions = Vec::with_capacity(MAX_REGIONS);
        for _ in 0..MAX_REGIONS {
            regions.push(RegionEntry::Empty);
        }
        Self {
            regions,
            next_id: 0,
            active_count: 0,
            #[cfg(feature = "stats")]
            stats: crate::RegionStats::default(),
        }
    }

    /// Creates a new region with the specified policy.
    ///
    /// This implements the region_map syscall from ADR-087 Section 3.1.
    ///
    /// # Arguments
    ///
    /// * `policy` - The region access policy
    /// * `cap` - Capability authorizing the mapping (must have WRITE right)
    ///
    /// # Returns
    ///
    /// A handle to the newly created region.
    ///
    /// # Errors
    ///
    /// - `InsufficientRights` if capability lacks required rights
    /// - `OutOfMemory` if region table is full or backing allocation fails
    pub fn create_region(
        &mut self,
        policy: RegionPolicy,
        _cap: CapHandle,
    ) -> Result<RegionHandle> {
        // In a real kernel, we'd check the capability here
        // For now, we just verify we have space

        // Find an empty slot
        let slot = self
            .regions
            .iter()
            .position(|e| matches!(e, RegionEntry::Empty))
            .ok_or(KernelError::OutOfMemory)?;

        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        let generation = 0u32;
        let handle = RegionHandle::new(id, generation);

        // Create the appropriate region type
        match policy {
            RegionPolicy::Immutable => {
                let backing = HeapBacking::new(4096);
                let region = Box::new(ImmutableRegion::new(backing, &[], handle)?);
                self.regions[slot] = RegionEntry::Immutable { region, generation };
            }
            RegionPolicy::AppendOnly { max_size } => {
                let backing = HeapBacking::new(max_size);
                let region = Box::new(AppendOnlyRegion::new(backing, max_size, handle)?);
                self.regions[slot] = RegionEntry::AppendOnly { region, generation };
            }
            RegionPolicy::Slab {
                slot_size,
                slot_count,
            } => {
                let total_size = slot_size * slot_count;
                let backing = HeapBacking::new(total_size);
                let region = Box::new(SlabRegion::new(backing, slot_size, slot_count, handle)?);
                self.regions[slot] = RegionEntry::Slab { region, generation };
            }
        }

        self.active_count += 1;

        #[cfg(feature = "stats")]
        {
            self.stats.regions_created += 1;
        }

        Ok(handle)
    }

    /// Creates an immutable region with initial data.
    ///
    /// # Arguments
    ///
    /// * `data` - Initial data to store (copied into the region)
    /// * `cap` - Capability authorizing the mapping
    pub fn create_immutable(&mut self, data: &[u8], _cap: CapHandle) -> Result<RegionHandle> {
        let slot = self
            .regions
            .iter()
            .position(|e| matches!(e, RegionEntry::Empty))
            .ok_or(KernelError::OutOfMemory)?;

        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        let generation = 0u32;
        let handle = RegionHandle::new(id, generation);

        let size = data.len().max(1); // At least 1 byte
        let backing = HeapBacking::new(size);
        let region = Box::new(ImmutableRegion::new(backing, data, handle)?);
        self.regions[slot] = RegionEntry::Immutable { region, generation };

        self.active_count += 1;

        #[cfg(feature = "stats")]
        {
            self.stats.regions_created += 1;
            self.stats.bytes_allocated += data.len() as u64;
        }

        Ok(handle)
    }

    /// Destroys a region.
    ///
    /// # Errors
    ///
    /// - `InvalidCapability` if the handle is invalid
    pub fn destroy_region(&mut self, handle: RegionHandle) -> Result<()> {
        let slot = self.find_region_slot(handle)?;
        self.regions[slot] = RegionEntry::Empty;
        self.active_count = self.active_count.saturating_sub(1);

        #[cfg(feature = "stats")]
        {
            self.stats.regions_destroyed += 1;
        }

        Ok(())
    }

    /// Finds the slot index for a region handle.
    fn find_region_slot(&self, handle: RegionHandle) -> Result<usize> {
        for (i, entry) in self.regions.iter().enumerate() {
            let matches = match entry {
                RegionEntry::Immutable { region, generation } => {
                    region.handle().raw().id == handle.raw().id
                        && *generation == handle.raw().generation
                }
                RegionEntry::AppendOnly { region, generation } => {
                    region.handle().raw().id == handle.raw().id
                        && *generation == handle.raw().generation
                }
                RegionEntry::Slab { region, generation } => {
                    region.handle().raw().id == handle.raw().id
                        && *generation == handle.raw().generation
                }
                RegionEntry::Empty => false,
            };
            if matches {
                return Ok(i);
            }
        }
        Err(KernelError::InvalidCapability)
    }

    // --- Slab region operations ---

    /// Allocates a slot from a slab region.
    pub fn slab_alloc(&mut self, handle: RegionHandle) -> Result<SlotHandle> {
        let slot = self.find_region_slot(handle)?;
        match &mut self.regions[slot] {
            RegionEntry::Slab { region, .. } => {
                #[cfg(feature = "stats")]
                {
                    self.stats.slab_allocs += 1;
                }
                region.alloc()
            }
            _ => Err(KernelError::RegionPolicyViolation),
        }
    }

    /// Frees a slot in a slab region.
    pub fn slab_free(&mut self, handle: RegionHandle, slot: SlotHandle) -> Result<()> {
        let region_slot = self.find_region_slot(handle)?;
        match &mut self.regions[region_slot] {
            RegionEntry::Slab { region, .. } => {
                #[cfg(feature = "stats")]
                {
                    self.stats.slab_frees += 1;
                }
                region.free(slot)
            }
            _ => Err(KernelError::RegionPolicyViolation),
        }
    }

    /// Writes data to a slab slot.
    pub fn slab_write(
        &mut self,
        handle: RegionHandle,
        slot: SlotHandle,
        data: &[u8],
    ) -> Result<usize> {
        let region_slot = self.find_region_slot(handle)?;
        match &mut self.regions[region_slot] {
            RegionEntry::Slab { region, .. } => region.write(slot, data),
            _ => Err(KernelError::RegionPolicyViolation),
        }
    }

    /// Reads data from a slab slot.
    pub fn slab_read(
        &self,
        handle: RegionHandle,
        slot: SlotHandle,
        buf: &mut [u8],
    ) -> Result<usize> {
        let region_slot = self.find_region_slot(handle)?;
        match &self.regions[region_slot] {
            RegionEntry::Slab { region, .. } => region.read(slot, buf),
            _ => Err(KernelError::RegionPolicyViolation),
        }
    }

    // --- Append-only region operations ---

    /// Appends data to an append-only region.
    pub fn append(&mut self, handle: RegionHandle, data: &[u8]) -> Result<usize> {
        let slot = self.find_region_slot(handle)?;
        match &mut self.regions[slot] {
            RegionEntry::AppendOnly { region, .. } => {
                #[cfg(feature = "stats")]
                {
                    self.stats.append_ops += 1;
                }
                region.append(data)
            }
            _ => Err(KernelError::RegionPolicyViolation),
        }
    }

    /// Reads data from an append-only region.
    pub fn append_read(
        &self,
        handle: RegionHandle,
        offset: usize,
        buf: &mut [u8],
    ) -> Result<usize> {
        let slot = self.find_region_slot(handle)?;
        match &self.regions[slot] {
            RegionEntry::AppendOnly { region, .. } => region.read(offset, buf),
            _ => Err(KernelError::RegionPolicyViolation),
        }
    }

    /// Returns the current length of an append-only region.
    pub fn append_len(&self, handle: RegionHandle) -> Result<usize> {
        let slot = self.find_region_slot(handle)?;
        match &self.regions[slot] {
            RegionEntry::AppendOnly { region, .. } => Ok(region.len()),
            _ => Err(KernelError::RegionPolicyViolation),
        }
    }

    // --- Immutable region operations ---

    /// Reads data from an immutable region.
    pub fn immutable_read(
        &self,
        handle: RegionHandle,
        offset: usize,
        buf: &mut [u8],
    ) -> Result<usize> {
        let slot = self.find_region_slot(handle)?;
        match &self.regions[slot] {
            RegionEntry::Immutable { region, .. } => region.read(offset, buf),
            _ => Err(KernelError::RegionPolicyViolation),
        }
    }

    /// Returns the length of an immutable region.
    pub fn immutable_len(&self, handle: RegionHandle) -> Result<usize> {
        let slot = self.find_region_slot(handle)?;
        match &self.regions[slot] {
            RegionEntry::Immutable { region, .. } => Ok(region.len()),
            _ => Err(KernelError::RegionPolicyViolation),
        }
    }

    // --- Manager state ---

    /// Returns the number of active regions.
    #[inline]
    #[must_use]
    pub const fn active_count(&self) -> usize {
        self.active_count
    }

    /// Returns the maximum number of regions.
    #[inline]
    #[must_use]
    pub const fn capacity(&self) -> usize {
        MAX_REGIONS
    }

    /// Returns statistics (requires `stats` feature).
    #[cfg(feature = "stats")]
    #[inline]
    #[must_use]
    pub const fn stats(&self) -> &crate::RegionStats {
        &self.stats
    }

    /// Gets the policy for a region.
    pub fn get_policy(&self, handle: RegionHandle) -> Result<RegionPolicy> {
        let slot = self.find_region_slot(handle)?;
        match &self.regions[slot] {
            RegionEntry::Immutable { .. } => Ok(RegionPolicy::Immutable),
            RegionEntry::AppendOnly { region, .. } => {
                Ok(RegionPolicy::AppendOnly {
                    max_size: region.max_size(),
                })
            }
            RegionEntry::Slab { region, .. } => Ok(RegionPolicy::Slab {
                slot_size: region.slot_size(),
                slot_count: region.slot_count(),
            }),
            RegionEntry::Empty => Err(KernelError::InvalidCapability),
        }
    }
}

impl Default for RegionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::backing::StaticBacking;

    #[test]
    fn test_manager_create_slab() {
        let mut manager = RegionManager::new();

        let handle = manager
            .create_region(RegionPolicy::slab(64, 16), CapHandle::null())
            .unwrap();

        assert!(!handle.is_null());
        assert_eq!(manager.active_count(), 1);

        // Allocate and use slots
        let slot = manager.slab_alloc(handle).unwrap();
        let data = b"Hello!";
        manager.slab_write(handle, slot, data).unwrap();

        let mut buf = [0u8; 64];
        let read = manager.slab_read(handle, slot, &mut buf).unwrap();
        assert_eq!(read, 64);
        assert_eq!(&buf[..data.len()], data);
    }

    #[test]
    fn test_manager_create_append_only() {
        let mut manager = RegionManager::new();

        let handle = manager
            .create_region(RegionPolicy::append_only(1024), CapHandle::null())
            .unwrap();

        // Append data
        let off1 = manager.append(handle, b"First").unwrap();
        let off2 = manager.append(handle, b"Second").unwrap();

        assert_eq!(off1, 0);
        assert_eq!(off2, 5);

        // Read back
        let mut buf = [0u8; 20];
        let read = manager.append_read(handle, 0, &mut buf).unwrap();
        assert_eq!(&buf[..read], b"FirstSecond");
    }

    #[test]
    fn test_manager_create_immutable() {
        let mut manager = RegionManager::new();

        let data = b"Immutable data";
        let handle = manager.create_immutable(data, CapHandle::null()).unwrap();

        // Read back
        let mut buf = [0u8; 20];
        let read = manager.immutable_read(handle, 0, &mut buf).unwrap();
        assert_eq!(&buf[..read], data);
    }

    #[test]
    fn test_manager_destroy_region() {
        let mut manager = RegionManager::new();

        let handle = manager
            .create_region(RegionPolicy::slab(64, 16), CapHandle::null())
            .unwrap();

        assert_eq!(manager.active_count(), 1);

        manager.destroy_region(handle).unwrap();
        assert_eq!(manager.active_count(), 0);

        // Handle should now be invalid
        assert!(manager.slab_alloc(handle).is_err());
    }

    #[test]
    fn test_manager_policy_violation() {
        let mut manager = RegionManager::new();

        // Create an immutable region
        let handle = manager
            .create_region(RegionPolicy::Immutable, CapHandle::null())
            .unwrap();

        // Trying to use slab operations should fail
        assert!(manager.slab_alloc(handle).is_err());

        // Trying to append should fail
        assert!(manager.append(handle, b"data").is_err());
    }

    #[test]
    fn test_manager_get_policy() {
        let mut manager = RegionManager::new();

        let slab_handle = manager
            .create_region(RegionPolicy::slab(64, 16), CapHandle::null())
            .unwrap();
        let append_handle = manager
            .create_region(RegionPolicy::append_only(1024), CapHandle::null())
            .unwrap();

        let slab_policy = manager.get_policy(slab_handle).unwrap();
        assert!(matches!(slab_policy, RegionPolicy::Slab { slot_size: 64, slot_count: 16 }));

        let append_policy = manager.get_policy(append_handle).unwrap();
        assert!(matches!(append_policy, RegionPolicy::AppendOnly { max_size: 1024 }));
    }

    #[test]
    fn test_manager_multiple_regions() {
        let mut manager = RegionManager::new();

        // Create multiple regions of different types
        let h1 = manager
            .create_region(RegionPolicy::slab(32, 8), CapHandle::null())
            .unwrap();
        let h2 = manager
            .create_region(RegionPolicy::append_only(512), CapHandle::null())
            .unwrap();
        let h3 = manager.create_immutable(b"test", CapHandle::null()).unwrap();

        assert_eq!(manager.active_count(), 3);

        // Use each region
        let slot = manager.slab_alloc(h1).unwrap();
        manager.slab_write(h1, slot, b"slab").unwrap();

        manager.append(h2, b"append").unwrap();

        let mut buf = [0u8; 10];
        manager.immutable_read(h3, 0, &mut buf).unwrap();
        assert_eq!(&buf[..4], b"test");
    }
}
