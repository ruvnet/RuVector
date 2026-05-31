//! Memory entry and store types.

use serde::{Deserialize, Serialize};

/// Opaque identifier for a memory entry.
pub type MemId = u64;

/// A single item in an agent's long-term memory store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique identifier.
    pub id: MemId,
    /// Dense embedding vector.
    pub vector: Vec<f32>,
    /// Unix timestamp (seconds).  Newer = higher value.
    pub timestamp: u64,
    /// Relevance score assigned by the agent (0.0 … 1.0).
    pub importance: f32,
    /// Semantic cluster label used for recall evaluation (synthetic data only).
    pub cluster_id: Option<u64>,
}

impl MemoryEntry {
    /// Create a new entry without a cluster label.
    pub fn new(id: MemId, vector: Vec<f32>, timestamp: u64, importance: f32) -> Self {
        Self {
            id,
            vector,
            timestamp,
            importance,
            cluster_id: None,
        }
    }

    /// Attach a semantic cluster label (used during synthetic dataset generation).
    pub fn with_cluster(mut self, cluster_id: u64) -> Self {
        self.cluster_id = Some(cluster_id);
        self
    }

    /// Number of dimensions in the embedding vector.
    pub fn dim(&self) -> usize {
        self.vector.len()
    }
}

/// An ordered collection of memory entries.
#[derive(Debug, Clone, Default)]
pub struct MemoryStore {
    /// All stored entries.
    pub entries: Vec<MemoryEntry>,
}

impl MemoryStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an entry.
    pub fn push(&mut self, entry: MemoryEntry) {
        self.entries.push(entry);
    }

    /// Number of stored entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if the store contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Approximate heap memory consumed by all entries (bytes).
    pub fn memory_bytes(&self) -> usize {
        self.entries
            .iter()
            .map(|e| std::mem::size_of::<MemoryEntry>() + e.vector.len() * size_of::<f32>())
            .sum()
    }
}
