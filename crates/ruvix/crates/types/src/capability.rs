//! Capability types for kernel object access control.
//!
//! Every kernel object is accessed exclusively through capabilities.
//! A capability is an unforgeable kernel-managed token comprising an
//! object identifier, type, rights bitmap, badge, and epoch.

use crate::handle::Handle;
use crate::object::ObjectType;

/// Handle to a capability entry in the capability table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct CapHandle(pub Handle);

impl CapHandle {
    /// Creates a new capability handle.
    #[inline]
    #[must_use]
    pub const fn new(id: u32, generation: u32) -> Self {
        Self(Handle::new(id, generation))
    }

    /// Creates a null (invalid) capability handle.
    #[inline]
    #[must_use]
    pub const fn null() -> Self {
        Self(Handle::null())
    }

    /// Checks if this handle is null.
    #[inline]
    #[must_use]
    pub const fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Returns the raw handle.
    #[inline]
    #[must_use]
    pub const fn raw(&self) -> Handle {
        self.0
    }
}

impl Default for CapHandle {
    fn default() -> Self {
        Self::null()
    }
}

/// Capability rights bitmap.
///
/// Rights determine what operations are permitted on the capability's target object.
/// Rights are checked at syscall time and cannot be escalated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct CapRights(u32);

impl CapRights {
    /// Right to read from the object (e.g., `vector_get`, `queue_recv`).
    pub const READ: Self = Self(0b0000_0001);

    /// Right to write to the object (e.g., `queue_send`, region writes).
    pub const WRITE: Self = Self(0b0000_0010);

    /// Right to grant this capability to other tasks via `cap_grant`.
    pub const GRANT: Self = Self(0b0000_0100);

    /// Right to revoke capabilities derived from this one.
    pub const REVOKE: Self = Self(0b0000_1000);

    /// Right to execute (e.g., task entry point, RVF component).
    pub const EXECUTE: Self = Self(0b0001_0000);

    /// Right to generate proof tokens for this object.
    /// Required for `vector_put_proved` and `graph_apply_proved`.
    pub const PROVE: Self = Self(0b0010_0000);

    /// Non-transitive grant right (cannot be further delegated).
    /// See Section 20.2 of ADR-087.
    pub const GRANT_ONCE: Self = Self(0b0100_0000);

    /// No rights.
    pub const NONE: Self = Self(0);

    /// All rights.
    pub const ALL: Self = Self(0b0111_1111);

    /// Creates a new `CapRights` from a raw value.
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Returns the raw bits.
    #[inline]
    #[must_use]
    pub const fn bits(&self) -> u32 {
        self.0
    }

    /// Checks if no rights are set.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.0 == 0
    }

    /// Checks if all specified rights are set.
    #[inline]
    #[must_use]
    pub const fn contains(&self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Returns the union of two rights sets.
    #[inline]
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Returns the intersection of two rights sets.
    #[inline]
    #[must_use]
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Returns the difference (rights in self but not in other).
    #[inline]
    #[must_use]
    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// Checks if this rights set is a subset of another (can be derived from it).
    #[inline]
    #[must_use]
    pub const fn is_subset_of(&self, other: Self) -> bool {
        (self.0 & other.0) == self.0
    }
}

impl Default for CapRights {
    fn default() -> Self {
        Self::NONE
    }
}

impl core::ops::BitOr for CapRights {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

impl core::ops::BitAnd for CapRights {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        self.intersection(rhs)
    }
}

impl core::ops::BitOrAssign for CapRights {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl core::ops::BitAndAssign for CapRights {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

/// A capability is a kernel-managed, unforgeable access token.
///
/// Capabilities follow seL4's design principles:
/// - No syscall succeeds without an appropriate capability handle
/// - A task can only grant capabilities it holds, with equal or fewer rights
/// - Revoking a capability invalidates all derived capabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct Capability {
    /// Unique identifier for the kernel object.
    pub object_id: u64,

    /// The type of kernel object.
    pub object_type: ObjectType,

    /// Rights bitmap determining permitted operations.
    pub rights: CapRights,

    /// Caller-visible identifier for demultiplexing.
    /// Allows a task to distinguish between multiple capabilities
    /// to the same underlying object.
    pub badge: u64,

    /// Epoch counter for capability revocation.
    /// Invalidated if the object is destroyed or the capability is revoked.
    pub epoch: u64,
}

impl Capability {
    /// Creates a new capability.
    #[inline]
    #[must_use]
    pub const fn new(
        object_id: u64,
        object_type: ObjectType,
        rights: CapRights,
        badge: u64,
        epoch: u64,
    ) -> Self {
        Self {
            object_id,
            object_type,
            rights,
            badge,
            epoch,
        }
    }

    /// Checks if this capability has the specified rights.
    #[inline]
    #[must_use]
    pub const fn has_rights(&self, required: CapRights) -> bool {
        self.rights.contains(required)
    }

    /// Creates a derived capability with reduced rights.
    ///
    /// Returns `None` if the requested rights are not a subset of current rights,
    /// or if attempting to derive from a `GRANT_ONCE` capability.
    #[inline]
    #[must_use]
    pub fn derive(&self, new_rights: CapRights, new_badge: u64) -> Option<Self> {
        // Cannot derive if we don't have GRANT right
        if !self.has_rights(CapRights::GRANT) {
            return None;
        }

        // Cannot derive with more rights than we have
        if !new_rights.is_subset_of(self.rights) {
            return None;
        }

        // GRANT_ONCE means the derived capability cannot have GRANT
        let final_rights = if self.rights.contains(CapRights::GRANT_ONCE) {
            new_rights
                .difference(CapRights::GRANT)
                .difference(CapRights::GRANT_ONCE)
        } else {
            new_rights
        };

        Some(Self {
            object_id: self.object_id,
            object_type: self.object_type,
            rights: final_rights,
            badge: new_badge,
            epoch: self.epoch,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cap_rights_operations() {
        let rw = CapRights::READ | CapRights::WRITE;
        assert!(rw.contains(CapRights::READ));
        assert!(rw.contains(CapRights::WRITE));
        assert!(!rw.contains(CapRights::GRANT));
    }

    #[test]
    fn test_cap_rights_subset() {
        let read = CapRights::READ;
        let read_write = CapRights::READ | CapRights::WRITE;
        assert!(read.is_subset_of(read_write));
        assert!(!read_write.is_subset_of(read));
    }

    #[test]
    fn test_capability_derive() {
        let cap = Capability::new(
            1,
            ObjectType::Region,
            CapRights::READ | CapRights::WRITE | CapRights::GRANT,
            0,
            1,
        );

        // Can derive with fewer rights
        let derived = cap.derive(CapRights::READ, 42).unwrap();
        assert_eq!(derived.rights, CapRights::READ);
        assert_eq!(derived.badge, 42);
        assert_eq!(derived.object_id, cap.object_id);
    }

    #[test]
    fn test_capability_derive_fails_without_grant() {
        let cap = Capability::new(1, ObjectType::Region, CapRights::READ, 0, 1);
        assert!(cap.derive(CapRights::READ, 0).is_none());
    }

    #[test]
    fn test_capability_derive_fails_with_more_rights() {
        let cap = Capability::new(
            1,
            ObjectType::Region,
            CapRights::READ | CapRights::GRANT,
            0,
            1,
        );
        assert!(cap.derive(CapRights::READ | CapRights::WRITE, 0).is_none());
    }

    #[test]
    fn test_grant_once() {
        let cap = Capability::new(
            1,
            ObjectType::Region,
            CapRights::READ | CapRights::GRANT | CapRights::GRANT_ONCE,
            0,
            1,
        );

        // Derived capability should not have GRANT
        let derived = cap.derive(CapRights::READ | CapRights::GRANT, 0).unwrap();
        assert!(!derived.rights.contains(CapRights::GRANT));
    }
}
