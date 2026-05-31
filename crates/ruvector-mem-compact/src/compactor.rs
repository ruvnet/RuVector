//! Core trait and data structures for memory compaction.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// A single entry in the agent's vector memory store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Stable identifier.
    pub id: u64,
    /// Embedding vector.
    pub vector: Vec<f32>,
    /// Logical timestamp (monotonically increasing, lower = older).
    pub age_tick: u64,
    /// How many times this entry was accessed since insertion.
    pub access_count: u64,
}

/// The in-memory store holding all agent memory vectors.
pub struct MemoryStore {
    pub entries: Vec<MemoryEntry>,
    pub dims: usize,
}

impl MemoryStore {
    pub fn new(dims: usize) -> Self {
        Self {
            entries: Vec::new(),
            dims,
        }
    }

    pub fn push(&mut self, entry: MemoryEntry) {
        self.entries.push(entry);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Keep only entries whose ids are in `kept_ids`.
    pub fn apply(&self, kept_ids: &[u64]) -> MemoryStore {
        let id_set: std::collections::HashSet<u64> = kept_ids.iter().copied().collect();
        let entries = self
            .entries
            .iter()
            .filter(|e| id_set.contains(&e.id))
            .cloned()
            .collect();
        MemoryStore {
            entries,
            dims: self.dims,
        }
    }
}

/// Output of a compaction pass.
#[derive(Debug)]
pub struct CompactionResult {
    /// IDs of entries that survive compaction.
    pub kept_ids: Vec<u64>,
    /// Fraction of the original store that was kept (0..1).
    pub kept_ratio: f32,
    /// Wall-clock time spent compacting.
    pub duration: Duration,
}

/// Trait implemented by every compaction strategy.
pub trait MemoryCompactor {
    /// Compact `store` so that approximately `target_ratio * n` entries remain.
    ///
    /// `target_ratio` is in `(0, 1]`.
    fn compact(&self, store: &MemoryStore, target_ratio: f32) -> CompactionResult;

    /// Human-readable strategy name for reporting.
    fn name(&self) -> &'static str;
}

/// L2-normalise a vector in place.  Returns the original norm.
pub fn l2_normalize(v: &mut Vec<f32>) -> f32 {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    norm
}

/// Cosine similarity between two **already-normalised** vectors.
#[inline]
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Brute-force nearest-neighbour query.  Returns `(index, similarity)` pairs
/// sorted descending.
pub fn knn_brute(query: &[f32], store: &MemoryStore, k: usize) -> Vec<(usize, f32)> {
    let mut sims: Vec<(usize, f32)> = store
        .entries
        .iter()
        .enumerate()
        .map(|(i, e)| (i, cosine_sim(query, &e.vector)))
        .collect();
    sims.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    sims.truncate(k);
    sims
}
