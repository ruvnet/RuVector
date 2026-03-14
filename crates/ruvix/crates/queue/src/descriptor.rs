//! Zero-copy message descriptors.
//!
//! When sender and receiver share a region, messages can be passed by reference
//! using descriptors instead of copying data. This module provides the descriptor
//! type and validation logic.

use ruvix_types::{KernelError, MsgPriority, RegionHandle, RegionPolicy};

use crate::Result;

/// A zero-copy message descriptor.
///
/// Instead of copying message data into the ring buffer, a descriptor
/// references data in a shared region. The receiver reads directly from
/// the shared region using the offset and length.
///
/// # TOCTOU Protection
///
/// Only Immutable or AppendOnly regions can use descriptors (ADR-087 Section 20.5).
/// This prevents time-of-check-to-time-of-use attacks where a sender modifies
/// shared data after the receiver reads the descriptor but before processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct MessageDescriptor {
    /// Handle to the shared region containing the data.
    pub region: RegionHandle,
    /// Byte offset within the region.
    pub offset: u64,
    /// Length in bytes.
    pub length: u32,
    /// Reserved padding (ensures 8-byte alignment).
    _padding: u32,
}

impl MessageDescriptor {
    /// Size of a descriptor in bytes.
    pub const SIZE: usize = core::mem::size_of::<Self>();

    /// Create a new message descriptor.
    #[inline]
    pub const fn new(region: RegionHandle, offset: u64, length: u32) -> Self {
        Self {
            region,
            offset,
            length,
            _padding: 0,
        }
    }

    /// Check if this descriptor is valid (non-null region, non-zero length).
    #[inline]
    pub fn is_valid(&self) -> bool {
        !self.region.is_null() && self.length > 0
    }

    /// Convert to bytes for storage in ring buffer.
    #[inline]
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        // SAFETY: MessageDescriptor is repr(C) with no padding
        unsafe { core::mem::transmute(*self) }
    }

    /// Create from bytes read from ring buffer.
    #[inline]
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < Self::SIZE {
            return None;
        }

        let mut arr = [0u8; Self::SIZE];
        arr.copy_from_slice(&bytes[..Self::SIZE]);

        // SAFETY: MessageDescriptor is repr(C) with defined layout
        Some(unsafe { core::mem::transmute(arr) })
    }
}

/// Validator for message descriptors.
///
/// Ensures that descriptors only reference regions with policies that
/// are safe for zero-copy message passing (Immutable or AppendOnly).
pub struct DescriptorValidator {
    /// Whether to allow Immutable regions.
    allow_immutable: bool,
    /// Whether to allow AppendOnly regions.
    allow_append_only: bool,
}

impl DescriptorValidator {
    /// Create a new validator with default settings (Immutable and AppendOnly allowed).
    pub const fn new() -> Self {
        Self {
            allow_immutable: true,
            allow_append_only: true,
        }
    }

    /// Create a validator that only allows Immutable regions.
    pub const fn immutable_only() -> Self {
        Self {
            allow_immutable: true,
            allow_append_only: false,
        }
    }

    /// Validate that a region policy is safe for descriptor-based messaging.
    ///
    /// # Errors
    ///
    /// Returns `InvalidDescriptorRegion` if the policy doesn't allow descriptors.
    pub fn validate_policy(&self, policy: &RegionPolicy) -> Result<()> {
        match policy {
            RegionPolicy::Immutable if self.allow_immutable => Ok(()),
            RegionPolicy::AppendOnly { .. } if self.allow_append_only => Ok(()),
            RegionPolicy::Slab { .. } => {
                // Slab regions allow overwriting, which creates TOCTOU vulnerabilities
                Err(KernelError::InvalidDescriptorRegion)
            }
            _ => Err(KernelError::InvalidDescriptorRegion),
        }
    }

    /// Validate a descriptor against region bounds.
    ///
    /// # Arguments
    ///
    /// * `descriptor` - The descriptor to validate
    /// * `region_size` - The total size of the region
    ///
    /// # Errors
    ///
    /// Returns `InvalidParameter` if the descriptor references memory outside the region.
    pub fn validate_bounds(&self, descriptor: &MessageDescriptor, region_size: usize) -> Result<()> {
        let end = descriptor
            .offset
            .checked_add(descriptor.length as u64)
            .ok_or(KernelError::InvalidArgument)?;

        if end > region_size as u64 {
            return Err(KernelError::InvalidArgument);
        }

        Ok(())
    }

    /// Full validation of a descriptor.
    ///
    /// Checks that:
    /// 1. The descriptor is valid (non-null region, non-zero length)
    /// 2. The region policy allows descriptors
    /// 3. The offset + length is within region bounds
    pub fn validate(
        &self,
        descriptor: &MessageDescriptor,
        policy: &RegionPolicy,
        region_size: usize,
    ) -> Result<()> {
        if !descriptor.is_valid() {
            return Err(KernelError::InvalidArgument);
        }

        self.validate_policy(policy)?;
        self.validate_bounds(descriptor, region_size)?;

        Ok(())
    }
}

impl Default for DescriptorValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// A descriptor with attached priority for queue operations.
#[derive(Debug, Clone, Copy)]
pub struct PrioritizedDescriptor {
    /// The underlying descriptor.
    pub descriptor: MessageDescriptor,
    /// Message priority.
    pub priority: MsgPriority,
}

impl PrioritizedDescriptor {
    /// Create a new prioritized descriptor.
    #[inline]
    pub const fn new(descriptor: MessageDescriptor, priority: MsgPriority) -> Self {
        Self {
            descriptor,
            priority,
        }
    }

    /// Create with default (normal) priority.
    #[inline]
    pub const fn with_normal_priority(descriptor: MessageDescriptor) -> Self {
        Self {
            descriptor,
            priority: MsgPriority::Normal,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvix_types::Handle;

    fn test_region() -> RegionHandle {
        RegionHandle(Handle::new(1, 0))
    }

    #[test]
    fn test_descriptor_size() {
        // Ensure predictable size for ring buffer calculations
        assert_eq!(MessageDescriptor::SIZE, 24);
    }

    #[test]
    fn test_descriptor_roundtrip() {
        let desc = MessageDescriptor::new(test_region(), 100, 256);
        let bytes = desc.to_bytes();
        let recovered = MessageDescriptor::from_bytes(&bytes).unwrap();

        assert_eq!(desc.region, recovered.region);
        assert_eq!(desc.offset, recovered.offset);
        assert_eq!(desc.length, recovered.length);
    }

    #[test]
    fn test_descriptor_validation_null() {
        let desc = MessageDescriptor::new(RegionHandle::null(), 0, 100);
        assert!(!desc.is_valid());
    }

    #[test]
    fn test_descriptor_validation_zero_length() {
        let desc = MessageDescriptor::new(test_region(), 0, 0);
        assert!(!desc.is_valid());
    }

    #[test]
    fn test_validator_immutable_ok() {
        let validator = DescriptorValidator::new();
        let result = validator.validate_policy(&RegionPolicy::Immutable);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validator_append_only_ok() {
        let validator = DescriptorValidator::new();
        let result = validator.validate_policy(&RegionPolicy::AppendOnly { max_size: 1024 });
        assert!(result.is_ok());
    }

    #[test]
    fn test_validator_slab_rejected() {
        let validator = DescriptorValidator::new();
        let result = validator.validate_policy(&RegionPolicy::Slab {
            slot_size: 64,
            slot_count: 16,
        });
        assert!(matches!(result, Err(KernelError::InvalidDescriptorRegion)));
    }

    #[test]
    fn test_validator_bounds() {
        let validator = DescriptorValidator::new();
        let desc = MessageDescriptor::new(test_region(), 100, 256);

        // Within bounds
        assert!(validator.validate_bounds(&desc, 500).is_ok());

        // Exactly at boundary
        assert!(validator.validate_bounds(&desc, 356).is_ok());

        // Out of bounds
        assert!(validator.validate_bounds(&desc, 355).is_err());
    }

    #[test]
    fn test_validator_bounds_overflow() {
        let validator = DescriptorValidator::new();
        let desc = MessageDescriptor::new(test_region(), u64::MAX - 10, 100);

        // This would overflow, should be detected
        assert!(validator.validate_bounds(&desc, 1000).is_err());
    }

    #[test]
    fn test_full_validation() {
        let validator = DescriptorValidator::new();
        let desc = MessageDescriptor::new(test_region(), 100, 256);

        // Valid descriptor with immutable region
        assert!(validator
            .validate(&desc, &RegionPolicy::Immutable, 500)
            .is_ok());

        // Invalid descriptor with slab region
        assert!(validator
            .validate(
                &desc,
                &RegionPolicy::Slab {
                    slot_size: 64,
                    slot_count: 16
                },
                500
            )
            .is_err());
    }

    #[test]
    fn test_immutable_only_validator() {
        let validator = DescriptorValidator::immutable_only();

        // Immutable should be allowed
        assert!(validator.validate_policy(&RegionPolicy::Immutable).is_ok());

        // AppendOnly should be rejected
        assert!(validator
            .validate_policy(&RegionPolicy::AppendOnly { max_size: 1024 })
            .is_err());
    }
}
