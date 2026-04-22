//! Capability grant logic.
//!
//! Implements the cap_grant syscall semantics from ADR-087 Section 6.2:
//! - A task can only grant capabilities it holds
//! - Granted rights must be equal or fewer than held rights
//! - GRANT_ONCE prevents further delegation

use crate::error::{CapError, CapResult};
use crate::table::CapTableEntry;
use crate::DEFAULT_MAX_DELEGATION_DEPTH;
use ruvix_types::{CapHandle, CapRights, Capability};

/// Request to grant a capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GrantRequest {
    /// The capability handle being granted.
    pub source_handle: CapHandle,

    /// The rights to grant (must be subset of source rights).
    pub rights: CapRights,

    /// Badge for the new capability (for demultiplexing).
    pub badge: u64,

    /// Maximum delegation depth (default: 8).
    pub max_depth: u8,
}

impl GrantRequest {
    /// Creates a new grant request.
    #[inline]
    #[must_use]
    pub const fn new(source_handle: CapHandle, rights: CapRights, badge: u64) -> Self {
        Self {
            source_handle,
            rights,
            badge,
            max_depth: DEFAULT_MAX_DELEGATION_DEPTH,
        }
    }

    /// Sets a custom maximum delegation depth.
    #[inline]
    #[must_use]
    pub const fn with_max_depth(mut self, max_depth: u8) -> Self {
        self.max_depth = max_depth;
        self
    }
}

/// Result of a successful grant operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GrantResult {
    /// The new capability that was created.
    pub capability: Capability,

    /// The depth of the new capability in the derivation tree.
    pub depth: u8,
}

/// Validates and prepares a capability grant.
///
/// This function checks all grant preconditions:
/// 1. The source capability exists and is valid
/// 2. The source has GRANT or GRANT_ONCE rights
/// 3. The requested rights are a subset of source rights
/// 4. The delegation depth limit is not exceeded
///
/// Returns the derived capability and its depth on success.
pub fn validate_grant(
    source_entry: &CapTableEntry,
    request: &GrantRequest,
) -> CapResult<GrantResult> {
    // Check if source is valid
    if !source_entry.is_valid {
        return Err(CapError::InvalidHandle);
    }

    let source_cap = &source_entry.capability;
    let source_rights = source_cap.rights;

    // Check for GRANT or GRANT_ONCE right
    let has_grant = source_rights.contains(CapRights::GRANT);
    let has_grant_once = source_rights.contains(CapRights::GRANT_ONCE);

    if !has_grant && !has_grant_once {
        return Err(CapError::CannotGrant);
    }

    // Check delegation depth limit
    let new_depth = source_entry.depth.saturating_add(1);
    if new_depth > request.max_depth {
        return Err(CapError::DelegationDepthExceeded);
    }

    // Check that requested rights are a subset of source rights
    if !request.rights.is_subset_of(source_rights) {
        return Err(CapError::RightsEscalation);
    }

    // Compute effective rights for the derived capability
    let mut effective_rights = source_rights.intersection(request.rights);

    // GRANT_ONCE means derived cap cannot have GRANT or GRANT_ONCE
    if has_grant_once && !has_grant {
        effective_rights = effective_rights
            .difference(CapRights::GRANT)
            .difference(CapRights::GRANT_ONCE);
    }

    // Create the derived capability
    let derived = Capability::new(
        source_cap.object_id,
        source_cap.object_type,
        effective_rights,
        request.badge,
        source_cap.epoch,
    );

    Ok(GrantResult {
        capability: derived,
        depth: new_depth,
    })
}

/// Checks if a grant operation is allowed.
///
/// This is a lightweight check without creating the derived capability.
pub fn can_grant(source_entry: &CapTableEntry, requested_rights: CapRights, max_depth: u8) -> bool {
    if !source_entry.is_valid {
        return false;
    }

    let source_rights = source_entry.capability.rights;

    // Must have GRANT or GRANT_ONCE
    if !source_rights.contains(CapRights::GRANT) && !source_rights.contains(CapRights::GRANT_ONCE) {
        return false;
    }

    // Check depth
    if source_entry.depth >= max_depth {
        return false;
    }

    // Check rights are subset
    requested_rights.is_subset_of(source_rights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvix_types::{ObjectType, TaskHandle};

    fn make_entry(rights: CapRights, depth: u8) -> CapTableEntry {
        let cap = Capability::new(100, ObjectType::VectorStore, rights, 0, 0);
        let mut entry = CapTableEntry::new_root(cap, 0, TaskHandle::new(1, 0));
        entry.depth = depth;
        entry
    }

    #[test]
    fn test_validate_grant_success() {
        let entry = make_entry(CapRights::ALL, 0);
        let request = GrantRequest::new(CapHandle::new(0, 0), CapRights::READ, 42);

        let result = validate_grant(&entry, &request).unwrap();
        assert!(result.capability.rights.contains(CapRights::READ));
        assert!(!result.capability.rights.contains(CapRights::WRITE));
        assert_eq!(result.depth, 1);
    }

    #[test]
    fn test_validate_grant_no_grant_right() {
        let entry = make_entry(CapRights::READ | CapRights::WRITE, 0);
        let request = GrantRequest::new(CapHandle::new(0, 0), CapRights::READ, 0);

        assert_eq!(validate_grant(&entry, &request), Err(CapError::CannotGrant));
    }

    #[test]
    fn test_validate_grant_rights_escalation() {
        let entry = make_entry(CapRights::READ | CapRights::GRANT, 0);
        let request =
            GrantRequest::new(CapHandle::new(0, 0), CapRights::READ | CapRights::WRITE, 0);

        assert_eq!(
            validate_grant(&entry, &request),
            Err(CapError::RightsEscalation)
        );
    }

    #[test]
    fn test_validate_grant_depth_exceeded() {
        let entry = make_entry(CapRights::ALL, 8);
        let request = GrantRequest::new(CapHandle::new(0, 0), CapRights::READ, 0);

        assert_eq!(
            validate_grant(&entry, &request),
            Err(CapError::DelegationDepthExceeded)
        );
    }

    #[test]
    fn test_validate_grant_once() {
        // GRANT_ONCE without GRANT - derived cap should lose grant ability
        let entry = make_entry(
            CapRights::READ | CapRights::WRITE | CapRights::GRANT_ONCE,
            0,
        );
        // Request same rights as source (subset check passes)
        let request = GrantRequest::new(
            CapHandle::new(0, 0),
            CapRights::READ | CapRights::WRITE | CapRights::GRANT_ONCE,
            0,
        );

        let result = validate_grant(&entry, &request).unwrap();

        // Derived cap should not have GRANT or GRANT_ONCE (stripped by GRANT_ONCE semantics)
        assert!(!result.capability.rights.contains(CapRights::GRANT));
        assert!(!result.capability.rights.contains(CapRights::GRANT_ONCE));
        // But should retain data rights
        assert!(result.capability.rights.contains(CapRights::READ));
        assert!(result.capability.rights.contains(CapRights::WRITE));
    }

    #[test]
    fn test_can_grant() {
        let entry = make_entry(CapRights::READ | CapRights::GRANT, 0);

        assert!(can_grant(&entry, CapRights::READ, 8));
        assert!(!can_grant(&entry, CapRights::WRITE, 8)); // Rights escalation
    }
}
