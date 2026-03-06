//! Index structures for efficient vector search

pub mod flat;
#[cfg(feature = "hnsw")]
pub mod hnsw;

use crate::error::Result;
use crate::types::{QuantumVector, SearchResult, VectorId};

/// Trait for vector index implementations
pub trait VectorIndex: Send + Sync {
    /// Add a vector to the index
    fn add(&mut self, id: VectorId, vector: QuantumVector) -> Result<()>;

    /// Add multiple vectors in batch
    fn add_batch(&mut self, entries: Vec<(VectorId, QuantumVector)>) -> Result<()> {
        for (id, vector) in entries {
            self.add(id, vector)?;
        }
        Ok(())
    }

    /// Search for k nearest neighbors
    fn search(&self, query: &QuantumVector, k: usize) -> Result<Vec<SearchResult>>;

    /// Remove a vector from the index
    fn remove(&mut self, id: &VectorId) -> Result<bool>;

    /// Get the number of vectors in the index
    fn len(&self) -> usize;

    /// Check if the index is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Dump the index to a byte buffer for O(1) fast persistence
    fn dump(&self) -> Result<Option<Vec<u8>>> {
        Ok(None)
    }
}
