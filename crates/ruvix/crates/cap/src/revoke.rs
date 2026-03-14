//! Capability revocation logic.
//!
//! Implements revocation with derivation chain propagation:
//! - Revoking a capability invalidates all derived capabilities
//! - Propagation follows the derivation tree structure
//! - Revocation requires the REVOKE right

use crate::error::{CapError, CapResult};
use crate::table::CapTableEntry;
use ruvix_types::CapRights;

/// Request to revoke a capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RevokeRequest {
    /// Whether to only revoke the target (not descendants).
    /// If false (default), all descendants are also revoked.
    pub target_only: bool,
}

impl RevokeRequest {
    /// Creates a new revoke request (revokes target and all descendants).
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { target_only: false }
    }

    /// Creates a revoke request for only the target capability.
    #[inline]
    #[must_use]
    pub const fn target_only() -> Self {
        Self { target_only: true }
    }
}

impl Default for RevokeRequest {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a successful revocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RevokeResult {
    /// Number of capabilities revoked.
    pub revoked_count: usize,

    /// Whether the derivation tree was pruned.
    pub tree_pruned: bool,
}

impl RevokeResult {
    /// Creates a new revoke result.
    #[inline]
    #[must_use]
    pub const fn new(revoked_count: usize) -> Self {
        Self {
            revoked_count,
            tree_pruned: true,
        }
    }
}

/// Validates that a revocation is allowed.
///
/// Checks:
/// 1. The capability exists and is valid
/// 2. The caller has REVOKE rights on the capability
///
/// Note: The actual revocation is performed by the CapabilityManager
/// which coordinates the table and derivation tree.
pub fn validate_revoke(entry: &CapTableEntry) -> CapResult<()> {
    if !entry.is_valid {
        return Err(CapError::InvalidHandle);
    }

    // Check for REVOKE right
    if !entry.capability.rights.contains(CapRights::REVOKE) {
        return Err(CapError::CannotRevoke);
    }

    Ok(())
}

/// Checks if a capability can be revoked.
#[inline]
#[must_use]
pub fn can_revoke(entry: &CapTableEntry) -> bool {
    entry.is_valid && entry.capability.rights.contains(CapRights::REVOKE)
}

/// Statistics about a revocation operation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RevokeStats {
    /// Number of root capabilities revoked.
    pub roots_revoked: usize,

    /// Number of derived capabilities revoked.
    pub derived_revoked: usize,

    /// Maximum depth reached during propagation.
    pub max_depth_reached: u8,
}

impl RevokeStats {
    /// Returns the total number of capabilities revoked.
    #[inline]
    #[must_use]
    pub const fn total(&self) -> usize {
        self.roots_revoked + self.derived_revoked
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvix_types::{Capability, ObjectType, TaskHandle};

    fn make_entry(rights: CapRights) -> CapTableEntry {
        let cap = Capability::new(100, ObjectType::Region, rights, 0, 0);
        CapTableEntry::new_root(cap, 0, TaskHandle::new(1, 0))
    }

    #[test]
    fn test_validate_revoke_success() {
        let entry = make_entry(CapRights::REVOKE);
        assert!(validate_revoke(&entry).is_ok());
    }

    #[test]
    fn test_validate_revoke_no_right() {
        let entry = make_entry(CapRights::READ | CapRights::WRITE);
        assert_eq!(validate_revoke(&entry), Err(CapError::CannotRevoke));
    }

    #[test]
    fn test_validate_revoke_invalid_entry() {
        let mut entry = make_entry(CapRights::REVOKE);
        entry.is_valid = false;
        assert_eq!(validate_revoke(&entry), Err(CapError::InvalidHandle));
    }

    #[test]
    fn test_can_revoke() {
        let with_revoke = make_entry(CapRights::REVOKE);
        let without_revoke = make_entry(CapRights::READ);

        assert!(can_revoke(&with_revoke));
        assert!(!can_revoke(&without_revoke));
    }

    #[test]
    fn test_revoke_stats() {
        let stats = RevokeStats {
            roots_revoked: 1,
            derived_revoked: 5,
            max_depth_reached: 3,
        };

        assert_eq!(stats.total(), 6);
    }
}
