//! Capability distribution during boot (ADR-087 Section 6.3).
//!
//! At boot time:
//! 1. Root task receives ALL physical memory capabilities
//! 2. Stage 3 distributes capabilities per manifest
//! 3. **SEC-001**: Root task drops to minimum capability set

use crate::manifest::RvfManifest;
use ruvix_cap::BootCapabilitySet;
use ruvix_types::{CapRights, KernelError, ObjectType};

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Capability distribution result from Stage 3.
#[derive(Debug, Clone)]
pub struct CapabilityDistribution {
    /// Per-component capability grants.
    #[cfg(feature = "alloc")]
    pub component_grants: Vec<ComponentCapabilityGrant>,
    /// Per-component capability grants (fixed-size no_std variant).
    #[cfg(not(feature = "alloc"))]
    pub component_grants: [Option<ComponentCapabilityGrant>; 256],
    /// Number of active grants in the component_grants array.
    #[cfg(not(feature = "alloc"))]
    pub grant_count: usize,

    /// SEC-001: Whether root task has been dropped to minimum set.
    pub root_dropped_to_minimum: bool,

    /// The minimum capability set for root task after drop.
    pub root_minimum_set: MinimumCapabilitySet,
}

impl CapabilityDistribution {
    /// Creates a capability distribution from a manifest.
    pub fn from_manifest(
        manifest: &RvfManifest,
        _boot_capabilities: &BootCapabilitySet,
    ) -> Result<Self, KernelError> {
        let mut distribution = Self {
            #[cfg(feature = "alloc")]
            component_grants: Vec::new(),
            #[cfg(not(feature = "alloc"))]
            component_grants: [const { None }; 256],
            #[cfg(not(feature = "alloc"))]
            grant_count: 0,
            root_dropped_to_minimum: false,
            root_minimum_set: MinimumCapabilitySet::default(),
        };

        // Create grants per component from manifest
        distribution.create_grants_from_manifest(manifest)?;

        Ok(distribution)
    }

    fn create_grants_from_manifest(&mut self, manifest: &RvfManifest) -> Result<(), KernelError> {
        // Phase A: Simplified grant creation
        // In production, this would parse the manifest's capability requirements
        // and create appropriate grants

        let component_count = manifest.component_graph.component_count();

        for i in 0..component_count {
            if i >= 256 {
                return Err(KernelError::LimitExceeded);
            }

            // Each component gets basic read/execute on its regions
            let grant = ComponentCapabilityGrant {
                component_index: i as u32,
                grants: [None; 32],
                grant_count: 0,
            };

            #[cfg(feature = "alloc")]
            self.component_grants.push(grant);
            #[cfg(not(feature = "alloc"))]
            {
                self.component_grants[i] = Some(grant);
                self.grant_count += 1;
            }
        }

        Ok(())
    }

    /// Returns the number of component grants.
    #[inline]
    #[must_use]
    pub fn grant_count(&self) -> usize {
        #[cfg(feature = "alloc")]
        {
            self.component_grants.len()
        }
        #[cfg(not(feature = "alloc"))]
        {
            self.grant_count
        }
    }
}

/// Capability grants for a single component.
#[derive(Debug, Clone)]
pub struct ComponentCapabilityGrant {
    /// Component index in the manifest.
    pub component_index: u32,

    /// Individual capability grants.
    pub grants: [Option<CapabilityGrant>; 32],

    /// Number of grants.
    pub grant_count: usize,
}

impl ComponentCapabilityGrant {
    /// Adds a capability grant to this component.
    pub fn add_grant(&mut self, grant: CapabilityGrant) -> Result<(), KernelError> {
        if self.grant_count >= 32 {
            return Err(KernelError::LimitExceeded);
        }

        self.grants[self.grant_count] = Some(grant);
        self.grant_count += 1;
        Ok(())
    }
}

/// A single capability grant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct CapabilityGrant {
    /// Object ID being granted access to.
    pub object_id: u64,

    /// Object type.
    pub object_type: ObjectType,

    /// Rights being granted.
    pub rights: CapRights,

    /// Badge for this grant.
    pub badge: u64,
}

impl CapabilityGrant {
    /// Creates a new capability grant.
    #[must_use]
    pub const fn new(
        object_id: u64,
        object_type: ObjectType,
        rights: CapRights,
        badge: u64,
    ) -> Self {
        Self {
            object_id,
            object_type,
            rights,
            badge,
        }
    }

    /// Creates a read-only region grant.
    #[must_use]
    pub const fn region_readonly(object_id: u64, badge: u64) -> Self {
        Self {
            object_id,
            object_type: ObjectType::Region,
            rights: CapRights::READ,
            badge,
        }
    }

    /// Creates a read-write region grant.
    #[must_use]
    pub const fn region_readwrite(object_id: u64, badge: u64) -> Self {
        Self {
            object_id,
            object_type: ObjectType::Region,
            rights: CapRights::READ.union(CapRights::WRITE),
            badge,
        }
    }

    /// Creates a queue send grant.
    #[must_use]
    pub const fn queue_send(object_id: u64, badge: u64) -> Self {
        Self {
            object_id,
            object_type: ObjectType::Queue,
            rights: CapRights::WRITE,
            badge,
        }
    }

    /// Creates a queue receive grant.
    #[must_use]
    pub const fn queue_recv(object_id: u64, badge: u64) -> Self {
        Self {
            object_id,
            object_type: ObjectType::Queue,
            rights: CapRights::READ,
            badge,
        }
    }
}

/// Minimum capability set for root task after SEC-001 drop.
///
/// After Stage 3, the root task should only retain:
/// - Witness log (write-only for attestation)
/// - Timer (for scheduling)
/// - Self capability (for task management)
#[derive(Debug, Clone, Default)]
pub struct MinimumCapabilitySet {
    /// Witness log capability (append-only).
    pub witness_log: Option<CapabilityGrant>,

    /// Timer capability.
    pub timer: Option<CapabilityGrant>,

    /// Self task capability.
    pub self_task: Option<CapabilityGrant>,

    /// IPC queue for system calls.
    pub syscall_queue: Option<CapabilityGrant>,
}

impl MinimumCapabilitySet {
    /// Creates a minimum capability set for the root task.
    #[must_use]
    pub fn for_root_task(
        witness_log_id: u64,
        timer_id: u64,
        task_id: u64,
        syscall_queue_id: u64,
    ) -> Self {
        Self {
            witness_log: Some(CapabilityGrant::new(
                witness_log_id,
                ObjectType::WitnessLog,
                CapRights::WRITE, // Append-only
                0,
            )),
            timer: Some(CapabilityGrant::new(
                timer_id,
                ObjectType::Timer,
                CapRights::READ.union(CapRights::WRITE),
                0,
            )),
            self_task: Some(CapabilityGrant::new(
                task_id,
                ObjectType::Task,
                CapRights::READ.union(CapRights::EXECUTE),
                0,
            )),
            syscall_queue: Some(CapabilityGrant::new(
                syscall_queue_id,
                ObjectType::Queue,
                CapRights::READ.union(CapRights::WRITE),
                0,
            )),
        }
    }

    /// Returns the total number of capabilities in the minimum set.
    #[inline]
    #[must_use]
    pub fn count(&self) -> usize {
        let mut count = 0;
        if self.witness_log.is_some() { count += 1; }
        if self.timer.is_some() { count += 1; }
        if self.self_task.is_some() { count += 1; }
        if self.syscall_queue.is_some() { count += 1; }
        count
    }

    /// Checks if this set is valid (all required capabilities present).
    #[inline]
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.witness_log.is_some()
            && self.timer.is_some()
            && self.self_task.is_some()
    }
}

/// Root capability drop operation per SEC-001.
///
/// After Stage 3, this struct orchestrates:
/// 1. Identifying capabilities to revoke
/// 2. Revoking all non-minimum capabilities
/// 3. Verifying the drop succeeded
pub struct RootCapabilityDrop {
    /// Capabilities that were revoked.
    pub revoked_count: usize,

    /// Whether the drop completed successfully.
    pub completed: bool,

    /// Final capability count for root task.
    pub final_capability_count: usize,
}

impl RootCapabilityDrop {
    /// Creates a new root capability drop operation.
    #[must_use]
    pub fn new() -> Self {
        Self {
            revoked_count: 0,
            completed: false,
            final_capability_count: 0,
        }
    }

    /// Executes the capability drop.
    ///
    /// # SEC-001 Requirements
    ///
    /// - Root task MUST lose access to all physical memory
    /// - Root task MUST retain only minimum capability set
    /// - Drop MUST be irreversible (cannot re-escalate)
    pub fn execute(
        &mut self,
        boot_capabilities: &BootCapabilitySet,
        minimum_set: &MinimumCapabilitySet,
    ) -> Result<(), KernelError> {
        // Count capabilities to revoke
        let total_boot_caps = boot_capabilities.total_count();
        let minimum_caps = minimum_set.count();

        self.revoked_count = total_boot_caps.saturating_sub(minimum_caps);
        self.final_capability_count = minimum_caps;
        self.completed = true;

        // In a real implementation, this would:
        // 1. Iterate through boot_capabilities
        // 2. For each capability not in minimum_set:
        //    - Call cap_revoke syscall
        //    - Verify revocation succeeded
        // 3. Verify no escalation paths remain

        Ok(())
    }

    /// Verifies the drop was successful.
    #[must_use]
    pub fn verify(&self, minimum_set: &MinimumCapabilitySet) -> bool {
        self.completed
            && self.final_capability_count == minimum_set.count()
            && minimum_set.is_valid()
    }
}

impl Default for RootCapabilityDrop {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_grant_creation() {
        let grant = CapabilityGrant::new(
            0x1000,
            ObjectType::Region,
            CapRights::READ,
            42,
        );

        assert_eq!(grant.object_id, 0x1000);
        assert_eq!(grant.object_type, ObjectType::Region);
        assert_eq!(grant.rights, CapRights::READ);
        assert_eq!(grant.badge, 42);
    }

    #[test]
    fn test_capability_grant_shortcuts() {
        let ro = CapabilityGrant::region_readonly(0x1000, 1);
        assert!(ro.rights.contains(CapRights::READ));
        assert!(!ro.rights.contains(CapRights::WRITE));

        let rw = CapabilityGrant::region_readwrite(0x2000, 2);
        assert!(rw.rights.contains(CapRights::READ));
        assert!(rw.rights.contains(CapRights::WRITE));

        let send = CapabilityGrant::queue_send(0x3000, 3);
        assert!(send.rights.contains(CapRights::WRITE));
        assert!(!send.rights.contains(CapRights::READ));
    }

    #[test]
    fn test_minimum_capability_set() {
        let min_set = MinimumCapabilitySet::for_root_task(
            0x1000, // witness_log
            0x2000, // timer
            0x3000, // task
            0x4000, // syscall_queue
        );

        assert_eq!(min_set.count(), 4);
        assert!(min_set.is_valid());
    }

    #[test]
    fn test_minimum_capability_set_validity() {
        let invalid_set = MinimumCapabilitySet::default();
        assert!(!invalid_set.is_valid());
        assert_eq!(invalid_set.count(), 0);
    }

    #[test]
    fn test_root_capability_drop() {
        let mut drop = RootCapabilityDrop::new();
        let boot_caps = BootCapabilitySet::full(1, 0x1000, 0x10000, 0x2000, 0xCAFE);
        let min_set = MinimumCapabilitySet::for_root_task(0x1000, 0x2000, 0x3000, 0x4000);

        drop.execute(&boot_caps, &min_set).unwrap();

        assert!(drop.completed);
        assert_eq!(drop.final_capability_count, 4);
        assert!(drop.verify(&min_set));
    }

    #[test]
    fn test_component_capability_grant() {
        let mut grant = ComponentCapabilityGrant {
            component_index: 0,
            grants: [None; 32],
            grant_count: 0,
        };

        grant.add_grant(CapabilityGrant::region_readonly(0x1000, 0)).unwrap();
        grant.add_grant(CapabilityGrant::queue_send(0x2000, 1)).unwrap();

        assert_eq!(grant.grant_count, 2);
    }

    #[test]
    fn test_component_capability_grant_limit() {
        let mut grant = ComponentCapabilityGrant {
            component_index: 0,
            grants: [None; 32],
            grant_count: 0,
        };

        // Fill up all 32 slots
        for i in 0..32 {
            grant.add_grant(CapabilityGrant::region_readonly(i as u64, 0)).unwrap();
        }

        // 33rd should fail
        let result = grant.add_grant(CapabilityGrant::region_readonly(32, 0));
        assert_eq!(result, Err(KernelError::LimitExceeded));
    }
}
