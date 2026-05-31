//! Compaction strategy trait and three concrete implementations.
//!
//! All strategies share the same interface: given a `MemoryStore` and a
//! `budget` (maximum number of entries in the result), return a compacted
//! `MemoryStore`.
//!
//! | Strategy            | Mechanism                                          |
//! |---------------------|----------------------------------------------------|
//! | `AgeEviction`       | Keeps the `budget` most recent entries             |
//! | `ImportanceEviction`| Keeps the `budget` highest-importance entries      |
//! | `GraphCutCompaction`| Cluster → one centroid per cluster → trim if needed|

use crate::graph::{centroid, cluster_by_similarity};
use crate::memory::{MemId, MemoryEntry, MemoryStore};

// ── Trait ────────────────────────────────────────────────────────────────────

/// A strategy for reducing a memory store to at most `budget` entries.
pub trait CompactionStrategy {
    /// Human-readable name used in benchmark output.
    fn name(&self) -> &'static str;

    /// Compact `store` to at most `budget` entries.
    fn compact(&self, store: &MemoryStore, budget: usize) -> MemoryStore;
}

// ── Age eviction ─────────────────────────────────────────────────────────────

/// Keep the `budget` most-recently timestamped entries.
///
/// Cheap O(n log n) sort.  Reflects a common cache-eviction policy.  Loses
/// historical semantic context proportional to how much time passes.
pub struct AgeEviction;

impl CompactionStrategy for AgeEviction {
    fn name(&self) -> &'static str {
        "AgeEviction"
    }

    fn compact(&self, store: &MemoryStore, budget: usize) -> MemoryStore {
        let mut entries = store.entries.clone();
        entries.sort_unstable_by(|a, b| b.timestamp.cmp(&a.timestamp));
        entries.truncate(budget);
        MemoryStore { entries }
    }
}

// ── Importance eviction ──────────────────────────────────────────────────────

/// Keep the `budget` entries with the highest importance score.
///
/// Cheap O(n log n).  Depends on accurate importance labels; with random
/// importance the result is equivalent to random sampling.
pub struct ImportanceEviction;

impl CompactionStrategy for ImportanceEviction {
    fn name(&self) -> &'static str {
        "ImportanceEviction"
    }

    fn compact(&self, store: &MemoryStore, budget: usize) -> MemoryStore {
        let mut entries = store.entries.clone();
        entries.sort_unstable_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        entries.truncate(budget);
        MemoryStore { entries }
    }
}

// ── Graph-cut compaction ─────────────────────────────────────────────────────

/// Cosine-similarity graph clustering followed by centroid synthesis.
///
/// Algorithm:
/// 1. Build pairwise similarity graph over all stored vectors.
/// 2. Union-Find: merge pairs with similarity ≥ `similarity_threshold` into
///    connected components (semantic clusters).
/// 3. Replace each cluster with its arithmetic centroid vector.  Inherit the
///    maximum importance and most-recent timestamp from cluster members so that
///    the representative retains the cluster's "best" metadata.
/// 4. If the number of cluster representatives still exceeds `budget`, trim by
///    descending importance (same rule as `ImportanceEviction`).
///
/// Time: O(n² · d) for the pairwise pass (n = corpus size, d = dimension),
/// then O(n α(n)) for union-find.  Suitable for agent memory stores up to
/// ~10 K entries; beyond that, switch to a k-NN graph approximation.
pub struct GraphCutCompaction {
    /// Cosine similarity threshold for merging two entries into the same
    /// cluster.  Higher = fewer, tighter clusters.  0.70 works well for
    /// typical text embeddings with within-cluster variation ≤ 0.15 noise.
    pub similarity_threshold: f32,
}

impl Default for GraphCutCompaction {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.70,
        }
    }
}

impl GraphCutCompaction {
    /// Create with a custom threshold.
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            similarity_threshold: threshold,
        }
    }
}

impl CompactionStrategy for GraphCutCompaction {
    fn name(&self) -> &'static str {
        "GraphCutCompaction"
    }

    fn compact(&self, store: &MemoryStore, budget: usize) -> MemoryStore {
        let n = store.entries.len();
        if n == 0 || budget >= n {
            return store.clone();
        }

        // Step 1: collect all vectors for the similarity pass.
        let vectors: Vec<Vec<f32>> = store.entries.iter().map(|e| e.vector.clone()).collect();

        // Step 2: cluster by cosine similarity.
        let clusters = cluster_by_similarity(&vectors, self.similarity_threshold);

        // Step 3: synthesise one representative per cluster.
        let mut next_id: MemId = store.entries.iter().map(|e| e.id).max().unwrap_or(0) + 1;

        let mut representatives: Vec<MemoryEntry> = clusters
            .iter()
            .map(|members| {
                let vecs: Vec<&Vec<f32>> =
                    members.iter().map(|&i| &store.entries[i].vector).collect();
                let rep_vec = centroid(&vecs);

                let max_importance = members
                    .iter()
                    .map(|&i| store.entries[i].importance)
                    .fold(f32::NEG_INFINITY, f32::max);
                let max_ts = members
                    .iter()
                    .map(|&i| store.entries[i].timestamp)
                    .max()
                    .unwrap_or(0);
                // Preserve cluster_id from the first member (all share the same label
                // in synthetic data; in production this would be None or merged).
                let cluster_id = store.entries[members[0]].cluster_id;

                let id = next_id;
                next_id += 1;

                MemoryEntry {
                    id,
                    vector: rep_vec,
                    timestamp: max_ts,
                    importance: max_importance,
                    cluster_id,
                }
            })
            .collect();

        // Step 4: trim by importance if we still exceed budget.
        if representatives.len() > budget {
            representatives.sort_unstable_by(|a, b| {
                b.importance
                    .partial_cmp(&a.importance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            representatives.truncate(budget);
        }

        MemoryStore {
            entries: representatives,
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryEntry;

    fn make_store(n: usize, dim: usize) -> MemoryStore {
        let mut store = MemoryStore::new();
        for i in 0..n {
            let vector: Vec<f32> = (0..dim).map(|j| ((i * dim + j) as f32).sin()).collect();
            let ts = i as u64;
            let imp = (i as f32) / (n as f32);
            store.push(MemoryEntry::new(i as u64, vector, ts, imp));
        }
        store
    }

    #[test]
    fn age_eviction_keeps_newest() {
        let store = make_store(10, 4);
        let result = AgeEviction.compact(&store, 3);
        assert_eq!(result.len(), 3);
        // Newest entries have highest timestamps.
        let min_ts = result.entries.iter().map(|e| e.timestamp).min().unwrap();
        assert!(
            min_ts >= 7,
            "expected newest 3 (ts ≥ 7), got min_ts={}",
            min_ts
        );
    }

    #[test]
    fn importance_eviction_keeps_highest() {
        let store = make_store(10, 4);
        let result = ImportanceEviction.compact(&store, 3);
        assert_eq!(result.len(), 3);
        let min_imp = result
            .entries
            .iter()
            .map(|e| e.importance)
            .fold(f32::INFINITY, f32::min);
        // The top-3 entries by importance are i=9,8,7 → importance ≥ 0.7
        assert!(
            min_imp >= 0.65,
            "expected high-importance entries, got min={}",
            min_imp
        );
    }

    #[test]
    fn graph_cut_respects_budget() {
        let store = make_store(20, 8);
        let result = GraphCutCompaction::default().compact(&store, 5);
        assert!(
            result.len() <= 5,
            "got {} entries, expected ≤ 5",
            result.len()
        );
    }

    #[test]
    fn empty_store_returns_empty() {
        let store = MemoryStore::new();
        assert!(AgeEviction.compact(&store, 10).is_empty());
        assert!(ImportanceEviction.compact(&store, 10).is_empty());
        assert!(GraphCutCompaction::default().compact(&store, 10).is_empty());
    }
}
