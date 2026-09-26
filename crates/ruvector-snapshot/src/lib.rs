//! Snapshot and restore functionality for rUvector collections
//!
//! This crate provides backup and restore capabilities for vector collections,
//! including compression, checksums, and multiple storage backends.

mod error;
mod manager;
mod snapshot;
mod storage;

#[cfg(feature = "cdc")]
mod cdc_storage;

pub use error::{Result, SnapshotError};
pub use manager::SnapshotManager;
pub use snapshot::{
    CollectionConfig, DistanceMetric, HnswConfig, Snapshot, SnapshotData, SnapshotMetadata,
    VectorRecord,
};
pub use storage::{LocalStorage, SnapshotStorage};

#[cfg(feature = "cdc")]
pub use cdc_storage::CdcLocalStorage;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify all public exports are accessible
        let _: Option<SnapshotError> = None;
        let _: Option<SnapshotManager> = None;
        let _: Option<Snapshot> = None;
    }
}
