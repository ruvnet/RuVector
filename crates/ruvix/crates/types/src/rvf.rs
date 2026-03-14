//! RVF (RuVector Format) types for boot and component mounting.
//!
//! RuVix boots from a single signed RVF file and mounts additional
//! RVF packages as components in the namespace.

use crate::handle::Handle;

/// Handle to a mounted RVF package.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct RvfMountHandle(pub Handle);

impl RvfMountHandle {
    /// Creates a new RVF mount handle.
    #[inline]
    #[must_use]
    pub const fn new(id: u32, generation: u32) -> Self {
        Self(Handle::new(id, generation))
    }

    /// Creates a null (invalid) RVF mount handle.
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

impl Default for RvfMountHandle {
    fn default() -> Self {
        Self::null()
    }
}

/// Identifier for an RVF component within a mounted package.
///
/// Each RVF package can contain multiple components (WASM modules
/// with WIT interfaces). Components are identified by index within
/// the package's component graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct RvfComponentId {
    /// The mount handle of the containing RVF package.
    pub mount: RvfMountHandle,

    /// Component index within the package (0-based).
    pub component_index: u32,
}

impl RvfComponentId {
    /// Creates a new component ID.
    #[inline]
    #[must_use]
    pub const fn new(mount: RvfMountHandle, component_index: u32) -> Self {
        Self {
            mount,
            component_index,
        }
    }

    /// Creates a component ID for the root component (index 0).
    #[inline]
    #[must_use]
    pub const fn root(mount: RvfMountHandle) -> Self {
        Self {
            mount,
            component_index: 0,
        }
    }
}

/// RVF package verification status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RvfVerifyStatus {
    /// Package signature is valid (ML-DSA-65 verified).
    SignatureValid = 0,

    /// Signature verification failed.
    SignatureInvalid = 1,

    /// Package manifest is malformed.
    ManifestInvalid = 2,

    /// Required component is missing.
    ComponentMissing = 3,

    /// Proof policy cannot be satisfied.
    ProofPolicyInvalid = 4,

    /// Package requires capabilities not available.
    CapabilitiesInsufficient = 5,
}

impl RvfVerifyStatus {
    /// Returns true if the package is valid for mounting.
    #[inline]
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        matches!(self, Self::SignatureValid)
    }

    /// Returns the status as a string.
    #[inline]
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::SignatureValid => "Signature valid",
            Self::SignatureInvalid => "Signature invalid",
            Self::ManifestInvalid => "Manifest invalid",
            Self::ComponentMissing => "Component missing",
            Self::ProofPolicyInvalid => "Proof policy invalid",
            Self::CapabilitiesInsufficient => "Capabilities insufficient",
        }
    }
}

/// RVF WIT (WASM Interface Types) type identifier.
///
/// Used for queue message schema validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct WitTypeId(pub u32);

impl WitTypeId {
    /// No schema (raw bytes).
    pub const NONE: Self = Self(0);

    /// Creates a new WIT type ID.
    #[inline]
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Returns true if this is the NONE type (no validation).
    #[inline]
    #[must_use]
    pub const fn is_none(&self) -> bool {
        self.0 == 0
    }
}

impl Default for WitTypeId {
    fn default() -> Self {
        Self::NONE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rvf_mount_handle() {
        let h = RvfMountHandle::new(1, 2);
        assert!(!h.is_null());
    }

    #[test]
    fn test_rvf_component_id() {
        let mount = RvfMountHandle::new(1, 0);
        let component = RvfComponentId::new(mount, 3);
        assert_eq!(component.component_index, 3);
    }

    #[test]
    fn test_rvf_verify_status() {
        assert!(RvfVerifyStatus::SignatureValid.is_valid());
        assert!(!RvfVerifyStatus::SignatureInvalid.is_valid());
    }
}
