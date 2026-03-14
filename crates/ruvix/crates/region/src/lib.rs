//! Memory Region Management for RuVix Cognition Kernel.
//!
//! This crate implements the Region primitive from ADR-087. A region is a
//! contiguous, capability-protected memory object with one of three policies:
//!
//! - **Immutable**: Set once at creation, never modified, deduplicatable
//! - **AppendOnly**: Only append, never overwrite, with max_size and write cursor
//! - **Slab**: Fixed-size slots from free list, no fragmentation
//!
//! # Architecture
//!
//! RuVix does not implement demand paging. All regions are physically backed
//! at `region_map` time. This eliminates page faults, swap, and copy-on-write
//! complexity.
//!
//! # Features
//!
//! - `std`: Enable standard library support (default)
//! - `mmap`: Use mmap for backing memory on Linux
//! - `stats`: Enable statistics collection
//!
//! # Example
//!
//! ```rust,ignore
//! use ruvix_region::{RegionManager, RegionConfig};
//! use ruvix_types::{RegionPolicy, CapHandle};
//!
//! let mut manager = RegionManager::new();
//!
//! // Create a slab region for fixed-size allocations
//! let slab_handle = manager.create_region(
//!     RegionPolicy::slab(64, 1024), // 64-byte slots, 1024 slots
//!     CapHandle::null(), // Root capability
//! )?;
//!
//! // Allocate a slot
//! let slot = manager.slab_alloc(slab_handle)?;
//!
//! // Write to the slot
//! manager.slab_write(slab_handle, slot, &data)?;
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod append_only;
pub mod backing;
pub mod immutable;
pub mod manager;
pub mod slab;
pub mod slab_optimized;

// Re-exports
pub use append_only::AppendOnlyRegion;
pub use immutable::ImmutableRegion;
pub use manager::{RegionConfig, RegionManager};
pub use slab::{SlabAllocator, SlabRegion};
pub use slab_optimized::{OptimizedSlabAllocator, OptimizedSlotHandle};

/// Result type for region operations.
pub type Result<T> = core::result::Result<T, ruvix_types::KernelError>;

/// Statistics for region operations.
#[cfg(feature = "stats")]
#[derive(Debug, Clone, Default)]
pub struct RegionStats {
    /// Total regions created.
    pub regions_created: u64,
    /// Total regions destroyed.
    pub regions_destroyed: u64,
    /// Total bytes allocated.
    pub bytes_allocated: u64,
    /// Total bytes freed.
    pub bytes_freed: u64,
    /// Slab allocations performed.
    pub slab_allocs: u64,
    /// Slab deallocations performed.
    pub slab_frees: u64,
    /// Append operations performed.
    pub append_ops: u64,
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_module_compiles() {
        // Basic compilation test
        assert!(true);
    }
}
