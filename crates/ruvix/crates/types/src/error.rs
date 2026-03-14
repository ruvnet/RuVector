//! Kernel error types.
//!
//! All syscalls return `Result<T, KernelError>` to indicate success or failure.
//! Errors are designed to be informative without leaking sensitive information.

/// Kernel error codes.
///
/// Every syscall can return these errors. Error codes are stable across
/// kernel versions (part of the ABI contract).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum KernelError {
    /// The specified capability handle is invalid or does not exist.
    InvalidCapability = 1,

    /// The capability does not have the required rights for this operation.
    InsufficientRights = 2,

    /// The capability has been revoked (epoch mismatch).
    CapabilityRevoked = 3,

    /// The proof token is invalid (wrong hash, expired, or nonce reused).
    ProofRejected = 4,

    /// Out of memory when trying to allocate a region.
    OutOfMemory = 5,

    /// The queue is full and cannot accept more messages.
    QueueFull = 6,

    /// The queue is empty (timeout expired during receive).
    QueueEmpty = 7,

    /// The operation timed out.
    Timeout = 8,

    /// The RVF package signature verification failed.
    InvalidSignature = 9,

    /// The RVF manifest is malformed or references invalid components.
    InvalidManifest = 10,

    /// The object type does not support this operation.
    InvalidObjectType = 11,

    /// The operation would exceed system limits (e.g., max tasks, max regions).
    LimitExceeded = 12,

    /// The resource already exists (e.g., duplicate mount point).
    AlreadyExists = 13,

    /// The resource was not found.
    NotFound = 14,

    /// The operation is not permitted in the current state.
    NotPermitted = 15,

    /// The operation would violate coherence constraints.
    CoherenceViolation = 16,

    /// Maximum delegation depth exceeded.
    DelegationDepthExceeded = 17,

    /// The capability cannot be granted (no GRANT or GRANT_ONCE right).
    CannotGrant = 18,

    /// The handle generation does not match (stale handle detection).
    StaleHandle = 19,

    /// The slot handle is invalid or stale (slab regions).
    InvalidSlot = 20,

    /// The slab region has no free slots available.
    SlabFull = 21,

    /// The append-only region has reached its maximum size.
    RegionFull = 22,

    /// Operation violates the region's policy (e.g., write to immutable).
    RegionPolicyViolation = 23,

    /// The provided buffer is too small for the requested operation.
    BufferTooSmall = 24,

    /// An invalid argument was provided.
    InvalidArgument = 25,

    /// The message exceeds the queue's maximum message size.
    MessageTooLarge = 26,

    /// The descriptor references a region with an invalid policy for zero-copy.
    /// Only Immutable and AppendOnly regions allow descriptors (TOCTOU protection).
    InvalidDescriptorRegion = 27,

    /// Internal kernel error (should not happen in normal operation).
    InternalError = 255,
}

impl KernelError {
    /// Returns a human-readable description of the error.
    #[inline]
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::InvalidCapability => "Invalid capability handle",
            Self::InsufficientRights => "Insufficient capability rights",
            Self::CapabilityRevoked => "Capability has been revoked",
            Self::ProofRejected => "Proof token rejected",
            Self::OutOfMemory => "Out of memory",
            Self::QueueFull => "Queue is full",
            Self::QueueEmpty => "Queue is empty",
            Self::Timeout => "Operation timed out",
            Self::InvalidSignature => "Invalid RVF signature",
            Self::InvalidManifest => "Invalid RVF manifest",
            Self::InvalidObjectType => "Invalid object type for operation",
            Self::LimitExceeded => "System limit exceeded",
            Self::AlreadyExists => "Resource already exists",
            Self::NotFound => "Resource not found",
            Self::NotPermitted => "Operation not permitted",
            Self::CoherenceViolation => "Coherence constraint violated",
            Self::DelegationDepthExceeded => "Maximum delegation depth exceeded",
            Self::CannotGrant => "Capability cannot be granted",
            Self::StaleHandle => "Stale handle (generation mismatch)",
            Self::InvalidSlot => "Invalid or stale slot handle",
            Self::SlabFull => "Slab region has no free slots",
            Self::RegionFull => "Region has reached maximum size",
            Self::RegionPolicyViolation => "Operation violates region policy",
            Self::BufferTooSmall => "Buffer too small for operation",
            Self::InvalidArgument => "Invalid argument provided",
            Self::MessageTooLarge => "Message exceeds maximum size",
            Self::InvalidDescriptorRegion => "Descriptor region policy invalid for zero-copy",
            Self::InternalError => "Internal kernel error",
        }
    }

    /// Converts from a raw u16 error code.
    #[inline]
    #[must_use]
    pub const fn from_u16(code: u16) -> Option<Self> {
        match code {
            1 => Some(Self::InvalidCapability),
            2 => Some(Self::InsufficientRights),
            3 => Some(Self::CapabilityRevoked),
            4 => Some(Self::ProofRejected),
            5 => Some(Self::OutOfMemory),
            6 => Some(Self::QueueFull),
            7 => Some(Self::QueueEmpty),
            8 => Some(Self::Timeout),
            9 => Some(Self::InvalidSignature),
            10 => Some(Self::InvalidManifest),
            11 => Some(Self::InvalidObjectType),
            12 => Some(Self::LimitExceeded),
            13 => Some(Self::AlreadyExists),
            14 => Some(Self::NotFound),
            15 => Some(Self::NotPermitted),
            16 => Some(Self::CoherenceViolation),
            17 => Some(Self::DelegationDepthExceeded),
            18 => Some(Self::CannotGrant),
            19 => Some(Self::StaleHandle),
            20 => Some(Self::InvalidSlot),
            21 => Some(Self::SlabFull),
            22 => Some(Self::RegionFull),
            23 => Some(Self::RegionPolicyViolation),
            24 => Some(Self::BufferTooSmall),
            25 => Some(Self::InvalidArgument),
            26 => Some(Self::MessageTooLarge),
            27 => Some(Self::InvalidDescriptorRegion),
            255 => Some(Self::InternalError),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_roundtrip() {
        let errors = [
            KernelError::InvalidCapability,
            KernelError::InsufficientRights,
            KernelError::ProofRejected,
            KernelError::OutOfMemory,
            KernelError::InternalError,
        ];

        for err in errors {
            let code = err as u16;
            let parsed = KernelError::from_u16(code).unwrap();
            assert_eq!(err, parsed);
        }
    }

    #[test]
    fn test_error_as_str() {
        assert_eq!(
            KernelError::DelegationDepthExceeded.as_str(),
            "Maximum delegation depth exceeded"
        );
    }
}
