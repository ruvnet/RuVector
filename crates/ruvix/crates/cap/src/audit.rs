//! Capability audit system for periodic security checks.
//!
//! The audit system monitors capability usage and derivation patterns
//! to detect potential security issues and policy violations.

use crate::error::CapResult;
use crate::table::CapTableEntry;
use crate::{AUDIT_DEPTH_WARNING_THRESHOLD, DEFAULT_MAX_DELEGATION_DEPTH};
use ruvix_types::{CapHandle, CapRights, ObjectType};

/// Result of a capability audit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuditResult {
    /// Total capabilities audited.
    pub total_audited: usize,

    /// Number of capabilities with deep derivation chains.
    pub deep_chains: usize,

    /// Number of capabilities with broad grant rights.
    pub broad_grants: usize,

    /// Number of orphaned capabilities (parent revoked).
    pub orphaned: usize,

    /// Number of capabilities nearing epoch overflow.
    pub epoch_warnings: usize,

    /// Maximum depth observed.
    pub max_depth_observed: u8,

    /// Whether the audit passed all checks.
    pub passed: bool,
}

impl AuditResult {
    /// Creates an empty audit result.
    #[inline]
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            total_audited: 0,
            deep_chains: 0,
            broad_grants: 0,
            orphaned: 0,
            epoch_warnings: 0,
            max_depth_observed: 0,
            passed: true,
        }
    }

    /// Returns true if any warnings were generated.
    #[inline]
    #[must_use]
    pub const fn has_warnings(&self) -> bool {
        self.deep_chains > 0
            || self.broad_grants > 0
            || self.orphaned > 0
            || self.epoch_warnings > 0
    }
}

impl Default for AuditResult {
    fn default() -> Self {
        Self::empty()
    }
}

/// A single audit entry describing one capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuditEntry {
    /// The capability handle.
    pub handle: CapHandle,

    /// Object type.
    pub object_type: ObjectType,

    /// Current rights.
    pub rights: CapRights,

    /// Derivation depth.
    pub depth: u8,

    /// Whether this capability has GRANT rights.
    pub can_grant: bool,

    /// Audit flags for this entry.
    pub flags: AuditFlags,
}

/// Flags indicating audit issues for a capability.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AuditFlags {
    /// Depth exceeds warning threshold.
    pub deep_chain: bool,

    /// Has broad grant rights that may be security concern.
    pub broad_grant: bool,

    /// Parent capability was revoked.
    pub orphaned: bool,

    /// Near epoch overflow (unlikely but checked).
    pub epoch_warning: bool,
}

impl AuditFlags {
    /// Returns true if any flag is set.
    #[inline]
    #[must_use]
    pub const fn any_set(&self) -> bool {
        self.deep_chain || self.broad_grant || self.orphaned || self.epoch_warning
    }
}

/// Configuration for the capability auditor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuditConfig {
    /// Depth threshold for warnings (default: 4).
    pub depth_warning_threshold: u8,

    /// Maximum delegation depth (default: 8).
    pub max_delegation_depth: u8,

    /// Whether to flag capabilities with GRANT rights.
    pub flag_grants: bool,

    /// Whether to check for orphaned capabilities.
    pub check_orphans: bool,

    /// Epoch threshold for warnings (within this many of overflow).
    pub epoch_warning_threshold: u64,
}

impl AuditConfig {
    /// Creates default audit configuration.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            depth_warning_threshold: AUDIT_DEPTH_WARNING_THRESHOLD,
            max_delegation_depth: DEFAULT_MAX_DELEGATION_DEPTH,
            flag_grants: true,
            check_orphans: true,
            epoch_warning_threshold: 1000,
        }
    }

    /// Sets a custom depth warning threshold.
    #[inline]
    #[must_use]
    pub const fn with_depth_threshold(mut self, threshold: u8) -> Self {
        self.depth_warning_threshold = threshold;
        self
    }

    /// Disables grant flagging.
    #[inline]
    #[must_use]
    pub const fn without_grant_flags(mut self) -> Self {
        self.flag_grants = false;
        self
    }
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// The capability auditor.
///
/// Performs periodic audits of the capability table to detect
/// potential security issues and policy violations.
pub struct CapabilityAuditor {
    /// Audit configuration.
    config: AuditConfig,

    /// Last audit result.
    last_result: AuditResult,

    /// Number of audits performed.
    audit_count: u64,
}

impl CapabilityAuditor {
    /// Creates a new auditor with default configuration.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            config: AuditConfig::new(),
            last_result: AuditResult::empty(),
            audit_count: 0,
        }
    }

    /// Creates a new auditor with custom configuration.
    #[inline]
    #[must_use]
    pub const fn with_config(config: AuditConfig) -> Self {
        Self {
            config,
            last_result: AuditResult::empty(),
            audit_count: 0,
        }
    }

    /// Returns the current configuration.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &AuditConfig {
        &self.config
    }

    /// Returns the last audit result.
    #[inline]
    #[must_use]
    pub const fn last_result(&self) -> &AuditResult {
        &self.last_result
    }

    /// Returns the number of audits performed.
    #[inline]
    #[must_use]
    pub const fn audit_count(&self) -> u64 {
        self.audit_count
    }

    /// Audits a single capability entry.
    pub fn audit_entry(
        &self,
        handle: CapHandle,
        entry: &CapTableEntry,
        current_epoch: u64,
    ) -> AuditEntry {
        let rights = entry.capability.rights;
        let depth = entry.depth;

        let mut flags = AuditFlags::default();

        // Check depth threshold
        if depth > self.config.depth_warning_threshold {
            flags.deep_chain = true;
        }

        // Check for broad grant rights
        if self.config.flag_grants {
            let has_grant = rights.contains(CapRights::GRANT);
            let has_grant_once = rights.contains(CapRights::GRANT_ONCE);
            let has_revoke = rights.contains(CapRights::REVOKE);

            // Flag if it has GRANT combined with other powerful rights
            if has_grant
                && (rights.contains(CapRights::WRITE) || rights.contains(CapRights::EXECUTE))
            {
                flags.broad_grant = true;
            }

            // GRANT + REVOKE is particularly concerning
            if has_grant && has_revoke {
                flags.broad_grant = true;
            }

            // GRANT_ONCE is safer, don't flag unless combined with REVOKE
            if has_grant_once && has_revoke && !has_grant {
                flags.broad_grant = true;
            }
        }

        // Check epoch proximity to overflow
        let epochs_remaining = u64::MAX - current_epoch;
        if epochs_remaining < self.config.epoch_warning_threshold {
            flags.epoch_warning = true;
        }

        AuditEntry {
            handle,
            object_type: entry.capability.object_type,
            rights,
            depth,
            can_grant: rights.contains(CapRights::GRANT) || rights.contains(CapRights::GRANT_ONCE),
            flags,
        }
    }

    /// Performs a full audit of an iterator of capabilities.
    ///
    /// This is the main audit entry point, typically called periodically
    /// or when security events occur.
    pub fn audit<'a, I>(&mut self, entries: I, current_epoch: u64) -> CapResult<AuditResult>
    where
        I: Iterator<Item = (CapHandle, &'a CapTableEntry)>,
    {
        let mut result = AuditResult::empty();

        for (handle, entry) in entries {
            if !entry.is_valid {
                continue;
            }

            let audit_entry = self.audit_entry(handle, entry, current_epoch);
            result.total_audited += 1;

            if audit_entry.depth > result.max_depth_observed {
                result.max_depth_observed = audit_entry.depth;
            }

            if audit_entry.flags.deep_chain {
                result.deep_chains += 1;
            }

            if audit_entry.flags.broad_grant {
                result.broad_grants += 1;
            }

            if audit_entry.flags.orphaned {
                result.orphaned += 1;
            }

            if audit_entry.flags.epoch_warning {
                result.epoch_warnings += 1;
            }
        }

        // Determine if audit passed (no critical issues)
        result.passed = result.epoch_warnings == 0;

        self.last_result = result;
        self.audit_count += 1;

        Ok(result)
    }

    /// Quick check if a capability needs attention.
    #[inline]
    #[must_use]
    pub fn needs_attention(&self, entry: &CapTableEntry) -> bool {
        entry.depth > self.config.depth_warning_threshold
            || (self.config.flag_grants
                && entry.capability.rights.contains(CapRights::GRANT)
                && entry.capability.rights.contains(CapRights::REVOKE))
    }
}

impl Default for CapabilityAuditor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvix_types::{Capability, TaskHandle};

    fn make_entry(rights: CapRights, depth: u8) -> CapTableEntry {
        let cap = Capability::new(100, ObjectType::VectorStore, rights, 0, 0);
        let mut entry = CapTableEntry::new_root(cap, 0, TaskHandle::new(1, 0));
        entry.depth = depth;
        entry
    }

    #[test]
    fn test_audit_entry_deep_chain() {
        let auditor = CapabilityAuditor::new();
        let entry = make_entry(CapRights::READ, 5);
        let handle = CapHandle::new(0, 0);

        let audit = auditor.audit_entry(handle, &entry, 0);

        assert!(audit.flags.deep_chain);
        assert_eq!(audit.depth, 5);
    }

    #[test]
    fn test_audit_entry_broad_grant() {
        let auditor = CapabilityAuditor::new();
        let entry = make_entry(CapRights::GRANT | CapRights::REVOKE, 0);
        let handle = CapHandle::new(0, 0);

        let audit = auditor.audit_entry(handle, &entry, 0);

        assert!(audit.flags.broad_grant);
        assert!(audit.can_grant);
    }

    #[test]
    fn test_audit_entry_safe() {
        let auditor = CapabilityAuditor::new();
        let entry = make_entry(CapRights::READ, 1);
        let handle = CapHandle::new(0, 0);

        let audit = auditor.audit_entry(handle, &entry, 0);

        assert!(!audit.flags.any_set());
        assert!(!audit.can_grant);
    }

    #[test]
    fn test_needs_attention() {
        let auditor = CapabilityAuditor::new();

        let deep_entry = make_entry(CapRights::READ, 5);
        assert!(auditor.needs_attention(&deep_entry));

        let dangerous_entry = make_entry(CapRights::GRANT | CapRights::REVOKE, 0);
        assert!(auditor.needs_attention(&dangerous_entry));

        let safe_entry = make_entry(CapRights::READ, 1);
        assert!(!auditor.needs_attention(&safe_entry));
    }

    #[test]
    fn test_audit_config() {
        let config = AuditConfig::new()
            .with_depth_threshold(6)
            .without_grant_flags();

        assert_eq!(config.depth_warning_threshold, 6);
        assert!(!config.flag_grants);
    }
}
