//! Physical memory allocator errors.
//!
//! This module defines error types specific to physical memory allocation.

use core::fmt;

use ruvix_types::KernelError;

/// Physical memory allocator errors.
///
/// These errors are returned by the buddy allocator when operations fail.
/// They can be converted to `KernelError` for use in syscall contexts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhysMemError {
    /// No memory available for the requested allocation.
    OutOfMemory,

    /// The requested allocation size is too large.
    ///
    /// The maximum allocation size is 512 pages (2MB).
    AllocationTooLarge,

    /// The requested allocation size is zero.
    ZeroAllocation,

    /// The address is not page-aligned.
    UnalignedAddress,

    /// The address is outside the managed memory range.
    AddressOutOfRange,

    /// The block at the specified address is not allocated.
    ///
    /// This can happen when trying to free memory that was never allocated
    /// or has already been freed (double-free detection).
    NotAllocated,

    /// The free operation has an incorrect size.
    ///
    /// The size passed to `free_pages` must match the original allocation size.
    SizeMismatch,

    /// The allocator has not been initialized.
    NotInitialized,

    /// Invalid order specified (exceeds `MAX_ORDER`).
    InvalidOrder,

    /// Internal allocator corruption detected.
    ///
    /// This indicates a bug in the allocator or memory corruption.
    InternalCorruption,
}

impl PhysMemError {
    /// Returns a human-readable description of the error.
    #[inline]
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::OutOfMemory => "Out of physical memory",
            Self::AllocationTooLarge => "Allocation size exceeds maximum block size",
            Self::ZeroAllocation => "Cannot allocate zero pages",
            Self::UnalignedAddress => "Address is not page-aligned",
            Self::AddressOutOfRange => "Address is outside managed memory range",
            Self::NotAllocated => "Block is not allocated (possible double-free)",
            Self::SizeMismatch => "Free size does not match allocation size",
            Self::NotInitialized => "Allocator has not been initialized",
            Self::InvalidOrder => "Invalid block order specified",
            Self::InternalCorruption => "Internal allocator corruption detected",
        }
    }

    /// Converts this error to a `KernelError`.
    ///
    /// This allows physical memory errors to be propagated through the
    /// kernel's standard error handling mechanism.
    #[inline]
    #[must_use]
    pub const fn to_kernel_error(self) -> KernelError {
        match self {
            Self::OutOfMemory | Self::AllocationTooLarge => KernelError::OutOfMemory,
            Self::ZeroAllocation | Self::InvalidOrder => KernelError::InvalidArgument,
            Self::UnalignedAddress | Self::AddressOutOfRange => KernelError::InvalidArgument,
            Self::NotAllocated | Self::SizeMismatch => KernelError::InvalidArgument,
            Self::NotInitialized => KernelError::NotPermitted,
            Self::InternalCorruption => KernelError::InternalError,
        }
    }
}

impl fmt::Display for PhysMemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl From<PhysMemError> for KernelError {
    #[inline]
    fn from(err: PhysMemError) -> Self {
        err.to_kernel_error()
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PhysMemError {}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::format;
    use super::*;

    #[test]
    fn test_error_as_str() {
        assert_eq!(PhysMemError::OutOfMemory.as_str(), "Out of physical memory");
        assert_eq!(
            PhysMemError::AllocationTooLarge.as_str(),
            "Allocation size exceeds maximum block size"
        );
        assert_eq!(
            PhysMemError::ZeroAllocation.as_str(),
            "Cannot allocate zero pages"
        );
        assert_eq!(
            PhysMemError::UnalignedAddress.as_str(),
            "Address is not page-aligned"
        );
    }

    #[test]
    fn test_to_kernel_error() {
        assert_eq!(
            PhysMemError::OutOfMemory.to_kernel_error(),
            KernelError::OutOfMemory
        );
        assert_eq!(
            PhysMemError::AllocationTooLarge.to_kernel_error(),
            KernelError::OutOfMemory
        );
        assert_eq!(
            PhysMemError::ZeroAllocation.to_kernel_error(),
            KernelError::InvalidArgument
        );
        assert_eq!(
            PhysMemError::InternalCorruption.to_kernel_error(),
            KernelError::InternalError
        );
    }

    #[test]
    fn test_into_kernel_error() {
        let err: KernelError = PhysMemError::OutOfMemory.into();
        assert_eq!(err, KernelError::OutOfMemory);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", PhysMemError::OutOfMemory), "Out of physical memory");
    }

    #[test]
    fn test_debug() {
        assert_eq!(format!("{:?}", PhysMemError::OutOfMemory), "OutOfMemory");
    }
}
