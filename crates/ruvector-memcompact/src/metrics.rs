//! Recall and quality metrics for compaction evaluation.
//!
//! All metrics compare a compacted store against the original store using
//! semantic cluster labels (present on synthetic data), so the result
//! reflects *cluster-level* coverage rather than exact ID matching.

use std::collections::HashSet;

use crate::graph::cosine_sim;
use crate::memory::{MemoryEntry, MemoryStore};

// ── Top-k retrieval ──────────────────────────────────────────────────────────

/// Return the top-`k` entries from `store` by cosine similarity to `query`,
/// breaking ties by entry id (stable ordering for reproducibility).
pub fn top_k<'s>(query: &[f32], store: &'s MemoryStore, k: usize) -> Vec<&'s MemoryEntry> {
    let mut scored: Vec<(f32, u64, &MemoryEntry)> = store
        .entries
        .iter()
        .map(|e| (cosine_sim(query, &e.vector), e.id, e))
        .collect();
    scored.sort_unstable_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    scored.into_iter().take(k).map(|(_, _, e)| e).collect()
}

// ── Cluster-level recall@k ───────────────────────────────────────────────────

/// Cluster-level recall@k for a single query.
///
/// Truth set:   the semantic cluster IDs of the top-`k` entries in `original`.
/// Predicted:   the semantic cluster IDs of the top-`k` entries in `compacted`.
/// Recall@k  = |truth ∩ predicted| / |truth|
///
/// Entries without a `cluster_id` are skipped in both sets.
/// Returns 1.0 if the truth set is empty (vacuously correct).
pub fn recall_at_k(
    query: &[f32],
    original: &MemoryStore,
    compacted: &MemoryStore,
    k: usize,
) -> f32 {
    let truth: HashSet<u64> = top_k(query, original, k)
        .iter()
        .filter_map(|e| e.cluster_id)
        .collect();

    if truth.is_empty() {
        return 1.0;
    }

    let predicted: HashSet<u64> = top_k(query, compacted, k)
        .iter()
        .filter_map(|e| e.cluster_id)
        .collect();

    let hits = truth.intersection(&predicted).count();
    hits as f32 / truth.len() as f32
}

/// Mean cluster-level recall@k over a slice of query vectors.
pub fn mean_recall_at_k(
    queries: &[Vec<f32>],
    original: &MemoryStore,
    compacted: &MemoryStore,
    k: usize,
) -> f32 {
    if queries.is_empty() {
        return 1.0;
    }
    let total: f32 = queries
        .iter()
        .map(|q| recall_at_k(q, original, compacted, k))
        .sum();
    total / queries.len() as f32
}

// ── Percentiles ──────────────────────────────────────────────────────────────

/// Compute p-th percentile (0.0 … 1.0) from a sorted slice of f64 values.
/// Uses the floor (lower) quantile convention to avoid off-by-one at p50.
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p * (sorted.len() - 1) as f64).floor()) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{MemoryEntry, MemoryStore};

    fn entry(id: u64, vec: Vec<f32>, cluster: u64) -> MemoryEntry {
        let mut e = MemoryEntry::new(id, vec, id, 0.5);
        e.cluster_id = Some(cluster);
        e
    }

    #[test]
    fn perfect_recall_when_compacted_equals_original() {
        let mut store = MemoryStore::new();
        store.push(entry(0, vec![1.0, 0.0, 0.0], 0));
        store.push(entry(1, vec![0.0, 1.0, 0.0], 1));
        store.push(entry(2, vec![0.0, 0.0, 1.0], 2));

        let query = vec![1.0, 0.1, 0.0];
        let recall = recall_at_k(&query, &store, &store, 3);
        assert!((recall - 1.0).abs() < 1e-6, "recall={}", recall);
    }

    #[test]
    fn zero_recall_when_compacted_missing_all_clusters() {
        let mut original = MemoryStore::new();
        original.push(entry(0, vec![1.0, 0.0], 0));
        original.push(entry(1, vec![0.9, 0.1], 0));

        let mut compacted = MemoryStore::new();
        compacted.push(entry(2, vec![0.0, 1.0], 1)); // different cluster

        let query = vec![1.0, 0.0];
        let recall = recall_at_k(&query, &original, &compacted, 2);
        assert_eq!(recall, 0.0, "recall={}", recall);
    }

    #[test]
    fn percentile_sorted_slice() {
        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        assert_eq!(percentile(&data, 0.50), 50.0);
        assert_eq!(percentile(&data, 0.95), 95.0);
        assert_eq!(percentile(&data, 1.00), 100.0);
    }
}
