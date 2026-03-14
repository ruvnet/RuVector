//! Capability-specific error types.

use ruvix_types::KernelError;

/// Errors that can occur during capability operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapError {
    /// The capability handle is invalid or does not exist.
    InvalidHandle,

    /// The capability has been revoked (epoch mismatch).
    Revoked,

    /// Insufficient rights to perform the operation.
    InsufficientRights,

    /// Cannot grant more rights than currently held.
    RightsEscalation,

    /// Maximum delegation depth would be exceeded.
    DelegationDepthExceeded,

    /// The capability table is full.
    TableFull,

    /// The target task cannot receive capabilities.
    InvalidTarget,

    /// The capability cannot be granted (no GRANT or GRANT_ONCE right).
    CannotGrant,

    /// The capability cannot be revoked (no REVOKE right).
    CannotRevoke,

    /// Stale handle (generation mismatch).
    StaleHandle,

    /// The object ID does not match.
    ObjectMismatch,

    /// Internal error in the capability manager.
    InternalError,
}

impl CapError {
    /// Returns a human-readable description of the error.
    #[inline]
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::InvalidHandle => "Invalid capability handle",
            Self::Revoked => "Capability has been revoked",
            Self::InsufficientRights => "Insufficient capability rights",
            Self::RightsEscalation => "Cannot grant more rights than held",
            Self::DelegationDepthExceeded => "Maximum delegation depth exceeded",
            Self::TableFull => "Capability table is full",
            Self::InvalidTarget => "Invalid target task",
            Self::CannotGrant => "Capability cannot be granted",
            Self::CannotRevoke => "Capability cannot be revoked",
            Self::StaleHandle => "Stale handle (generation mismatch)",
            Self::ObjectMismatch => "Object ID mismatch",
            Self::InternalError => "Internal capability manager error",
        }
    }

    /// Converts to a kernel error code.
    #[inline]
    #[must_use]
    pub const fn to_kernel_error(&self) -> KernelError {
        match self {
            Self::InvalidHandle => KernelError::InvalidCapability,
            Self::Revoked => KernelError::CapabilityRevoked,
            Self::InsufficientRights => KernelError::InsufficientRights,
            Self::RightsEscalation => KernelError::InsufficientRights,
            Self::DelegationDepthExceeded => KernelError::DelegationDepthExceeded,
            Self::TableFull => KernelError::LimitExceeded,
            Self::InvalidTarget => KernelError::NotFound,
            Self::CannotGrant => KernelError::CannotGrant,
            Self::CannotRevoke => KernelError::NotPermitted,
            Self::StaleHandle => KernelError::StaleHandle,
            Self::ObjectMismatch => KernelError::InvalidCapability,
            Self::InternalError => KernelError::InternalError,
        }
    }
}

/// Result type for capability operations.
pub type CapResult<T> = Result<T, CapError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cap_error_to_kernel_error() {
        assert_eq!(
            CapError::InvalidHandle.to_kernel_error(),
            KernelError::InvalidCapability
        );
        assert_eq!(
            CapError::DelegationDepthExceeded.to_kernel_error(),
            KernelError::DelegationDepthExceeded
        );
    }
}
