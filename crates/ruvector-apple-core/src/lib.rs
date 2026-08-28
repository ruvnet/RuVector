//! Portable, bounded vector search primitives for native Apple applications.
//!
//! The Rust API is safe and dependency-free. The C ABI uses opaque handles,
//! fixed-width types, explicit ownership transfer, and panic containment so it
//! can be consumed from Swift, Objective-C, and other native runtimes.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

mod ffi;
mod index;
mod snapshot;

pub use ffi::{
    ruvector_apple_core_abi_version, ruvector_apple_core_bytes_data,
    ruvector_apple_core_bytes_free, ruvector_apple_core_index_create,
    ruvector_apple_core_index_destroy, ruvector_apple_core_index_info,
    ruvector_apple_core_index_remove, ruvector_apple_core_index_restore,
    ruvector_apple_core_index_search, ruvector_apple_core_index_snapshot,
    ruvector_apple_core_index_upsert, ruvector_apple_core_results_data,
    ruvector_apple_core_results_free, RvAppleIndex, RvIndexInfo, RvOwnedBytes, RvOwnedResults,
    RvSearchResult, RvStatus,
};
pub use index::{
    DistanceMetric, ExactVectorIndex, IndexConfig, IndexError, SearchHit, MAX_CAPACITY,
    MAX_DIMENSIONS, MAX_SEARCH_RESULTS, MAX_TOTAL_VALUES,
};
pub use snapshot::MAX_SNAPSHOT_BYTES;

/// Current version of the native C ABI.
pub const RVECTOR_APPLE_CORE_ABI_VERSION: u32 = 1;
