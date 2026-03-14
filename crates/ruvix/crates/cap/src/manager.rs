//! Main capability manager implementation.
//!
//! The `CapabilityManager` coordinates the capability table and derivation
//! tree to provide complete capability lifecycle management.

use crate::derivation::DerivationTree;
use crate::error::{CapError, CapResult};
use crate::grant::{validate_grant, GrantRequest};
use crate::revoke::{validate_revoke, RevokeRequest, RevokeResult};
#[cfg(feature = "alloc")]
use crate::revoke::RevokeStats;
use crate::table::{CapTableEntry, CapabilityTable};
use crate::{DEFAULT_CAP_TABLE_CAPACITY, DEFAULT_MAX_DELEGATION_DEPTH};
use ruvix_types::{CapHandle, CapRights, Capability, ObjectType, TaskHandle};

/// Configuration for the capability manager.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapManagerConfig {
    /// Maximum delegation depth (default: 8).
    pub max_delegation_depth: u8,

    /// Whether to track derivation chains (for revocation propagation).
    pub track_derivation: bool,

    /// Global epoch for capability invalidation.
    pub initial_epoch: u64,
}

impl CapManagerConfig {
    /// Creates a new configuration with default values.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            max_delegation_depth: DEFAULT_MAX_DELEGATION_DEPTH,
            track_derivation: true,
            initial_epoch: 0,
        }
    }

    /// Sets a custom maximum delegation depth.
    #[inline]
    #[must_use]
    pub const fn with_max_depth(mut self, depth: u8) -> Self {
        self.max_delegation_depth = depth;
        self
    }

    /// Disables derivation tracking (for performance, disables revocation propagation).
    #[inline]
    #[must_use]
    pub const fn without_derivation_tracking(mut self) -> Self {
        self.track_derivation = false;
        self
    }
}

impl Default for CapManagerConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// The main capability manager.
///
/// Coordinates capability table and derivation tree to provide
/// complete capability lifecycle management including creation,
/// granting, and revocation.
pub struct CapabilityManager<const N: usize = DEFAULT_CAP_TABLE_CAPACITY> {
    /// The capability table storing all capabilities.
    table: CapabilityTable<N>,

    /// Derivation tree for revocation propagation.
    derivation: DerivationTree<N>,

    /// Configuration.
    config: CapManagerConfig,

    /// Current global epoch.
    epoch: u64,

    /// Statistics for debugging and monitoring.
    stats: ManagerStats,
}

/// Statistics about capability manager operations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ManagerStats {
    /// Total capabilities created.
    pub caps_created: u64,

    /// Total capabilities granted (derived).
    pub caps_granted: u64,

    /// Total capabilities revoked.
    pub caps_revoked: u64,

    /// Total revoke operations.
    pub revoke_operations: u64,

    /// Maximum depth reached in derivation tree.
    pub max_depth_reached: u8,
}

impl<const N: usize> CapabilityManager<N> {
    /// Creates a new capability manager with the given configuration.
    #[inline]
    #[must_use]
    pub const fn new(config: CapManagerConfig) -> Self {
        Self {
            table: CapabilityTable::new(),
            derivation: DerivationTree::new(),
            epoch: config.initial_epoch,
            config,
            stats: ManagerStats {
                caps_created: 0,
                caps_granted: 0,
                caps_revoked: 0,
                revoke_operations: 0,
                max_depth_reached: 0,
            },
        }
    }

    /// Creates a new capability manager with default configuration.
    #[inline]
    #[must_use]
    pub const fn with_defaults() -> Self {
        Self::new(CapManagerConfig::new())
    }

    /// Returns the current configuration.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &CapManagerConfig {
        &self.config
    }

    /// Returns the current statistics.
    #[inline]
    #[must_use]
    pub const fn stats(&self) -> &ManagerStats {
        &self.stats
    }

    /// Returns the current epoch.
    #[inline]
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Returns the number of active capabilities.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.table.len()
    }

    /// Returns true if there are no active capabilities.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Increments the global epoch, invalidating all capabilities.
    #[inline]
    pub fn increment_epoch(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
    }

    /// Creates a root capability for a new kernel object.
    ///
    /// Root capabilities have full rights (ALL) and are at depth 0.
    /// They are typically created when a new kernel object is allocated.
    pub fn create_root_capability(
        &mut self,
        object_id: u64,
        object_type: ObjectType,
        badge: u64,
        owner: TaskHandle,
    ) -> CapResult<CapHandle> {
        let capability = Capability::new(
            object_id,
            object_type,
            CapRights::ALL,
            badge,
            self.epoch,
        );

        let handle = self.table.allocate_root(capability, owner)?;

        if self.config.track_derivation {
            self.derivation.add_root(handle)?;
        }

        self.stats.caps_created += 1;
        Ok(handle)
    }

    /// Creates a root capability with specific rights (not ALL).
    ///
    /// Used when the kernel needs to create a capability with restricted
    /// rights from the start (e.g., read-only memory regions).
    pub fn create_root_capability_with_rights(
        &mut self,
        object_id: u64,
        object_type: ObjectType,
        rights: CapRights,
        badge: u64,
        owner: TaskHandle,
    ) -> CapResult<CapHandle> {
        let capability = Capability::new(
            object_id,
            object_type,
            rights,
            badge,
            self.epoch,
        );

        let handle = self.table.allocate_root(capability, owner)?;

        if self.config.track_derivation {
            self.derivation.add_root(handle)?;
        }

        self.stats.caps_created += 1;
        Ok(handle)
    }

    /// Grants a derived capability to another task.
    ///
    /// Implements the cap_grant syscall semantics from ADR-087 Section 6.2:
    /// - The source capability must have GRANT or GRANT_ONCE right
    /// - Granted rights must be a subset of source rights
    /// - Delegation depth is enforced
    /// - GRANT_ONCE prevents further delegation
    pub fn grant(
        &mut self,
        source_handle: CapHandle,
        rights: CapRights,
        badge: u64,
        _caller: TaskHandle,
        target: TaskHandle,
    ) -> CapResult<CapHandle> {
        // Look up source capability
        let source_entry = self.table.lookup(source_handle)?;

        // Validate the grant
        let request = GrantRequest::new(source_handle, rights, badge)
            .with_max_depth(self.config.max_delegation_depth);

        let grant_result = validate_grant(source_entry, &request)?;

        // Allocate the derived capability in the target's table
        let derived_handle = self.table.allocate_derived(
            grant_result.capability,
            target,
            grant_result.depth,
            source_handle,
        )?;

        // Track in derivation tree
        if self.config.track_derivation {
            self.derivation.add_child(source_handle, derived_handle, grant_result.depth)?;
        }

        // Update statistics
        self.stats.caps_granted += 1;
        if grant_result.depth > self.stats.max_depth_reached {
            self.stats.max_depth_reached = grant_result.depth;
        }

        Ok(derived_handle)
    }

    /// Revokes a capability and all its descendants.
    ///
    /// When a capability is revoked:
    /// 1. The capability itself is invalidated
    /// 2. All derived capabilities are recursively invalidated
    /// 3. The derivation tree is pruned
    pub fn revoke(&mut self, handle: CapHandle, _request: RevokeRequest) -> CapResult<RevokeResult> {
        // Validate revocation
        let entry = self.table.lookup(handle)?;
        validate_revoke(entry)?;

        // Revoke in derivation tree (this handles descendants)
        let revoked_count = if self.config.track_derivation {
            self.derivation.revoke(handle)?
        } else {
            // Without derivation tracking, only revoke the single capability
            self.table.deallocate(handle)?;
            1
        };

        // Deallocate from table (derivation tree marks as invalid but we need
        // to update the table too for all affected handles)
        // Note: The derivation tree's revoke_subtree already marks nodes invalid,
        // but we need to synchronize with the table
        self.deallocate_revoked_caps(handle)?;

        // Update statistics
        self.stats.caps_revoked += revoked_count as u64;
        self.stats.revoke_operations += 1;

        Ok(RevokeResult::new(revoked_count))
    }

    /// Deallocates capabilities that have been marked as revoked in the derivation tree.
    fn deallocate_revoked_caps(&mut self, handle: CapHandle) -> CapResult<()> {
        // Deallocate the primary handle
        // Ignore errors since the derivation tree may have already invalidated it
        let _ = self.table.deallocate(handle);
        Ok(())
    }

    /// Revokes a capability without propagation (target only).
    pub fn revoke_single(&mut self, handle: CapHandle) -> CapResult<()> {
        let entry = self.table.lookup(handle)?;
        validate_revoke(entry)?;

        self.table.deallocate(handle)?;

        // Note: This leaves orphaned children in the derivation tree.
        // They will become invalid when their parent is looked up.

        self.stats.caps_revoked += 1;
        self.stats.revoke_operations += 1;

        Ok(())
    }

    /// Looks up a capability by handle.
    pub fn lookup(&self, handle: CapHandle) -> CapResult<&CapTableEntry> {
        let entry = self.table.lookup(handle)?;

        // Check epoch
        if entry.capability.epoch != self.epoch {
            return Err(CapError::Revoked);
        }

        // Check derivation validity (if tracking)
        if self.config.track_derivation && !self.derivation.is_valid(handle) {
            return Err(CapError::Revoked);
        }

        Ok(entry)
    }

    /// Checks if a handle points to a valid capability.
    pub fn is_valid(&self, handle: CapHandle) -> bool {
        self.lookup(handle).is_ok()
    }

    /// Checks if a task has a specific right on a capability.
    pub fn has_right(&self, handle: CapHandle, right: CapRights) -> CapResult<bool> {
        let entry = self.lookup(handle)?;
        Ok(entry.capability.rights.contains(right))
    }

    /// Checks multiple rights at once.
    pub fn has_rights(&self, handle: CapHandle, rights: CapRights) -> CapResult<bool> {
        let entry = self.lookup(handle)?;
        Ok(entry.capability.rights.contains(rights))
    }

    /// Gets the derivation depth of a capability.
    pub fn depth(&self, handle: CapHandle) -> CapResult<u8> {
        let entry = self.lookup(handle)?;
        Ok(entry.depth)
    }

    /// Returns revocation statistics for a potential revoke operation.
    #[cfg(feature = "alloc")]
    pub fn preview_revoke(&self, handle: CapHandle) -> CapResult<RevokeStats> {
        let entry = self.table.lookup(handle)?;

        if !self.config.track_derivation {
            return Ok(RevokeStats {
                roots_revoked: 1,
                derived_revoked: 0,
                max_depth_reached: entry.depth,
            });
        }

        let descendants = self.derivation.collect_descendants(handle);
        let derived_count = descendants.len().saturating_sub(1);

        Ok(RevokeStats {
            roots_revoked: 1,
            derived_revoked: derived_count,
            max_depth_reached: entry.depth,
        })
    }

    /// Returns an iterator over all valid capabilities.
    pub fn iter(&self) -> impl Iterator<Item = (CapHandle, &CapTableEntry)> {
        self.table.iter().filter(|(h, _)| {
            !self.config.track_derivation || self.derivation.is_valid(*h)
        })
    }
}

impl<const N: usize> Default for CapabilityManager<N> {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_root_capability() {
        let mut manager = CapabilityManager::<64>::with_defaults();
        let owner = TaskHandle::new(1, 0);

        let handle = manager.create_root_capability(
            0x1000,
            ObjectType::VectorStore,
            42,
            owner,
        ).unwrap();

        assert_eq!(manager.len(), 1);

        let entry = manager.lookup(handle).unwrap();
        assert_eq!(entry.capability.object_id, 0x1000);
        assert_eq!(entry.capability.rights, CapRights::ALL);
        assert_eq!(entry.depth, 0);
    }

    #[test]
    fn test_grant_capability() {
        let mut manager = CapabilityManager::<64>::with_defaults();
        let owner = TaskHandle::new(1, 0);
        let target = TaskHandle::new(2, 0);

        let root_handle = manager.create_root_capability(
            0x1000,
            ObjectType::Region,
            0,
            owner,
        ).unwrap();

        let derived_handle = manager.grant(
            root_handle,
            CapRights::READ | CapRights::WRITE,
            100,
            owner,
            target,
        ).unwrap();

        assert_eq!(manager.len(), 2);

        let derived_entry = manager.lookup(derived_handle).unwrap();
        assert!(derived_entry.capability.rights.contains(CapRights::READ));
        assert!(derived_entry.capability.rights.contains(CapRights::WRITE));
        assert!(!derived_entry.capability.rights.contains(CapRights::GRANT));
        assert_eq!(derived_entry.depth, 1);
    }

    #[test]
    fn test_revoke_propagation() {
        let mut manager = CapabilityManager::<64>::with_defaults();
        let owner = TaskHandle::new(1, 0);
        let target1 = TaskHandle::new(2, 0);
        let target2 = TaskHandle::new(3, 0);

        // Create root capability
        let root = manager.create_root_capability(
            0x1000,
            ObjectType::Queue,
            0,
            owner,
        ).unwrap();

        // Grant to target1
        let child1 = manager.grant(
            root,
            CapRights::READ | CapRights::GRANT,
            1,
            owner,
            target1,
        ).unwrap();

        // Grant from child1 to target2
        let grandchild = manager.grant(
            child1,
            CapRights::READ,
            2,
            target1,
            target2,
        ).unwrap();

        assert_eq!(manager.len(), 3);

        // Revoke root - should revoke all descendants
        let result = manager.revoke(root, RevokeRequest::new()).unwrap();
        assert_eq!(result.revoked_count, 3);

        // All should be invalid now
        assert!(!manager.is_valid(root));
        assert!(!manager.is_valid(child1));
        assert!(!manager.is_valid(grandchild));
    }

    #[test]
    fn test_delegation_depth_limit() {
        let config = CapManagerConfig::new().with_max_depth(2);
        let mut manager = CapabilityManager::<64>::new(config);
        let owner = TaskHandle::new(1, 0);

        // Create chain: root -> d1 -> d2 -> d3 (should fail)
        let root = manager.create_root_capability(
            0x1000,
            ObjectType::Timer,
            0,
            owner,
        ).unwrap();

        let d1 = manager.grant(
            root,
            CapRights::READ | CapRights::GRANT,
            1,
            owner,
            owner,
        ).unwrap();

        let d2 = manager.grant(
            d1,
            CapRights::READ | CapRights::GRANT,
            2,
            owner,
            owner,
        ).unwrap();

        // This should fail - depth limit exceeded
        let result = manager.grant(
            d2,
            CapRights::READ,
            3,
            owner,
            owner,
        );

        assert_eq!(result, Err(CapError::DelegationDepthExceeded));
    }

    #[test]
    fn test_has_right() {
        let mut manager = CapabilityManager::<64>::with_defaults();
        let owner = TaskHandle::new(1, 0);

        let handle = manager.create_root_capability_with_rights(
            0x1000,
            ObjectType::Region,
            CapRights::READ | CapRights::WRITE,
            0,
            owner,
        ).unwrap();

        assert!(manager.has_right(handle, CapRights::READ).unwrap());
        assert!(manager.has_right(handle, CapRights::WRITE).unwrap());
        assert!(!manager.has_right(handle, CapRights::EXECUTE).unwrap());
    }

    #[test]
    fn test_epoch_invalidation() {
        let mut manager = CapabilityManager::<64>::with_defaults();
        let owner = TaskHandle::new(1, 0);

        let handle = manager.create_root_capability(
            0x1000,
            ObjectType::VectorStore,
            0,
            owner,
        ).unwrap();

        assert!(manager.is_valid(handle));

        // Increment epoch
        manager.increment_epoch();

        // Old capability should now be invalid (epoch mismatch)
        assert!(!manager.is_valid(handle));
    }
}
