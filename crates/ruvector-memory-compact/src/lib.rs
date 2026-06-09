//! # ruvector-memory-compact
//!
//! Coherence-gated agent memory compaction for ruvector. Merges semantically
//! redundant memories by building a k-NN coherence graph over embeddings, then
//! clustering via three strategies and replacing each cluster with a centroid
//! vector plus a witness record.
//!
//! ## Three variants
//! - [`NaiveCompactor`]: K-means centroid replacement (baseline)
//! - [`GraphMergeCompactor`]: threshold-based graph merge on similarity graph
//! - [`CoherenceGatedCompactor`]: adaptive-threshold graph merge, coherence-weighted

pub mod coherence;
pub mod graph;
pub mod kmeans;
pub mod merge;

use std::time::Instant;

// ─── Public types ─────────────────────────────────────────────────────────────

/// A single memory entry held by an agent.
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub id: u64,
    pub embedding: Vec<f32>,
    /// Logical timestamp (monotonically increasing insert order).
    pub age: u64,
    pub metadata: String,
}

/// A flat store of memory entries.
#[derive(Debug, Default)]
pub struct MemoryStore {
    pub entries: Vec<MemoryEntry>,
    pub(crate) next_id: u64,
}

/// Summary of one compaction run.
#[derive(Debug, Clone)]
pub struct CompactionResult {
    pub variant: String,
    pub original_count: usize,
    pub compacted_count: usize,
    /// fraction of vectors removed (0.0 = no removal, 1.0 = all removed)
    pub compaction_ratio: f64,
    /// recall@K measured against the pre-compaction exact NN answers
    pub recall_at_k: f64,
    pub duration_ms: u64,
    pub witness_records: Vec<WitnessRecord>,
}

/// Attestation that `merged_ids` were replaced by `centroid_id`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WitnessRecord {
    pub centroid_id: u64,
    pub merged_ids: Vec<u64>,
    /// Average intra-cluster cosine similarity at merge time.
    pub intra_sim: f32,
}

// ─── Compactor trait ──────────────────────────────────────────────────────────

/// Compaction strategy over a [`MemoryStore`].
pub trait Compactor {
    /// Compact `store` toward `target_ratio` (fraction of vectors to keep).
    /// Returns a [`CompactionResult`] including recall@`k` estimated against
    /// `queries` using exact search before and after.
    fn compact(
        &self,
        store: &mut MemoryStore,
        target_ratio: f64,
        queries: &[Vec<f32>],
        k: usize,
    ) -> CompactionResult;

    fn name(&self) -> &'static str;
}

// ─── MemoryStore impl ─────────────────────────────────────────────────────────

impl MemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, embedding: Vec<f32>, metadata: impl Into<String>) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.entries.push(MemoryEntry {
            id,
            embedding,
            age: id,
            metadata: metadata.into(),
        });
        id
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.entries.first().map(|e| e.embedding.len()).unwrap_or(0)
    }
}

// ─── Shared utilities ─────────────────────────────────────────────────────────

/// Cosine similarity in [-1, 1].
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = (na * nb).sqrt();
    if denom < 1e-9 {
        0.0
    } else {
        dot / denom
    }
}

/// Compute centroid of a set of embeddings.
pub fn centroid(embeddings: &[&[f32]]) -> Vec<f32> {
    if embeddings.is_empty() {
        return Vec::new();
    }
    let dim = embeddings[0].len();
    let mut c = vec![0.0_f32; dim];
    let n = embeddings.len() as f32;
    for e in embeddings {
        for (j, &v) in e.iter().enumerate() {
            c[j] += v / n;
        }
    }
    c
}

/// Cluster-aware recall@k: a true neighbour is "hit" if the centroid that
/// absorbed it appears in the compacted top-k. Falls back to exact-match
/// for entries that were not merged.
pub fn recall_clustered(
    queries: &[Vec<f32>],
    before: &[MemoryEntry],
    after: &[MemoryEntry],
    witness: &[WitnessRecord],
    k: usize,
) -> f64 {
    if queries.is_empty() || before.is_empty() || after.is_empty() {
        return 1.0;
    }
    // Build map: original_id -> centroid_id it was merged into.
    let mut id_map: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
    for w in witness {
        for &mid in &w.merged_ids {
            id_map.insert(mid, w.centroid_id);
        }
        id_map.insert(w.centroid_id, w.centroid_id);
    }
    // Identities that remained unchanged get mapped to themselves.
    for e in after {
        id_map.entry(e.id).or_insert(e.id);
    }

    let k = k.min(before.len()).min(after.len());
    if k == 0 {
        return 1.0;
    }

    let mut total = 0.0_f64;
    for q in queries {
        let true_ids = exact_top_k(q, before, k);
        // Map each true neighbour to its representative centroid.
        let mapped: Vec<u64> = true_ids
            .iter()
            .map(|&id| *id_map.get(&id).unwrap_or(&id))
            .collect();
        let found_ids: std::collections::HashSet<u64> =
            exact_top_k(q, after, k).into_iter().collect();
        let hits = mapped.iter().filter(|id| found_ids.contains(id)).count();
        total += hits as f64 / k as f64;
    }
    total / queries.len() as f64
}

/// Exact nearest-neighbour recall@k (no cluster mapping).
pub fn recall_at_k(
    queries: &[Vec<f32>],
    before: &[MemoryEntry],
    after: &[MemoryEntry],
    k: usize,
) -> f64 {
    if queries.is_empty() || before.is_empty() || after.is_empty() {
        return 1.0;
    }
    let k = k.min(before.len()).min(after.len());
    let mut total = 0.0_f64;
    for q in queries {
        let true_ids: std::collections::HashSet<u64> =
            exact_top_k(q, before, k).into_iter().collect();
        let found_ids: std::collections::HashSet<u64> =
            exact_top_k(q, after, k).into_iter().collect();
        let hits = true_ids.intersection(&found_ids).count();
        total += hits as f64 / k as f64;
    }
    total / queries.len() as f64
}

fn exact_top_k(query: &[f32], store: &[MemoryEntry], k: usize) -> Vec<u64> {
    let mut sims: Vec<(f32, u64)> = store
        .iter()
        .map(|e| (cosine_sim(query, &e.embedding), e.id))
        .collect();
    sims.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    sims.truncate(k);
    sims.into_iter().map(|(_, id)| id).collect()
}

/// Wrap a compaction run with timing.
pub fn run_compaction(
    compactor: &dyn Compactor,
    store: &mut MemoryStore,
    target_ratio: f64,
    queries: &[Vec<f32>],
    k: usize,
) -> CompactionResult {
    let start = Instant::now();
    let mut result = compactor.compact(store, target_ratio, queries, k);
    result.duration_ms = start.elapsed().as_millis() as u64;
    result
}

// ─── Re-exports ───────────────────────────────────────────────────────────────

pub use coherence::CoherenceGatedCompactor;
pub use kmeans::NaiveCompactor;
pub use merge::GraphMergeCompactor;

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand::rngs::StdRng;

    fn l2_normalise(mut v: Vec<f32>) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-9 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
        v
    }

    /// Build a clustered store: n_topics centroids, each with vecs_per_topic
    /// noisy neighbours. Also returns one query per topic.
    fn clustered_store(
        n_topics: usize,
        vecs_per_topic: usize,
        dim: usize,
        noise: f32,
        seed: u64,
    ) -> (MemoryStore, Vec<Vec<f32>>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let centroids: Vec<Vec<f32>> = (0..n_topics)
            .map(|_| l2_normalise((0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()))
            .collect();
        let mut store = MemoryStore::new();
        for (t, c) in centroids.iter().enumerate() {
            for _ in 0..vecs_per_topic {
                let noisy: Vec<f32> = c.iter().map(|&x| x + rng.gen::<f32>() * noise).collect();
                store.insert(l2_normalise(noisy), format!("topic-{t}"));
            }
        }
        let queries: Vec<Vec<f32>> = centroids
            .iter()
            .map(|c| {
                l2_normalise(
                    c.iter()
                        .map(|&x| x + rng.gen::<f32>() * noise * 0.5)
                        .collect(),
                )
            })
            .collect();
        (store, queries)
    }

    #[test]
    fn cosine_sim_self_is_one() {
        let v = l2_normalise(vec![1.0, 2.0, 3.0, 4.0]);
        let s = cosine_sim(&v, &v);
        assert!(
            (s - 1.0).abs() < 1e-5,
            "self-similarity should be 1.0, got {s}"
        );
    }

    #[test]
    fn cosine_sim_orthogonal_is_zero() {
        let a = l2_normalise(vec![1.0, 0.0, 0.0]);
        let b = l2_normalise(vec![0.0, 1.0, 0.0]);
        let s = cosine_sim(&a, &b);
        assert!(
            s.abs() < 1e-5,
            "orthogonal vectors should have ~0 similarity, got {s}"
        );
    }

    #[test]
    fn centroid_of_identical_vectors_is_same() {
        let v = vec![0.5_f32, 0.5, 0.5, 0.5];
        let slices: Vec<&[f32]> = vec![v.as_slice(); 4];
        let c = centroid(&slices);
        for (a, b) in c.iter().zip(&v) {
            assert!((a - b).abs() < 1e-5, "centroid mismatch");
        }
    }

    #[test]
    fn memory_store_insert_and_len() {
        let mut store = MemoryStore::new();
        assert_eq!(store.len(), 0);
        store.insert(vec![1.0, 0.0], "a");
        store.insert(vec![0.0, 1.0], "b");
        assert_eq!(store.len(), 2);
        assert_eq!(store.dim(), 2);
    }

    // ── Compaction integration tests ──────────────────────────────────────

    fn assert_compaction_passes(
        compactor: &dyn Compactor,
        n_topics: usize,
        vecs_per_topic: usize,
        target_ratio: f64,
        min_recall: f64,
        label: &str,
    ) {
        let (mut store, queries) = clustered_store(n_topics, vecs_per_topic, 32, 0.15, 1234);
        let n_before = store.len();
        let result = run_compaction(compactor, &mut store, target_ratio, &queries, 5);

        assert!(
            result.compaction_ratio > 0.0,
            "{label}: expected compaction > 0, got {:.3}",
            result.compaction_ratio
        );
        assert!(
            result.compacted_count < n_before,
            "{label}: expected compacted < original ({} vs {})",
            result.compacted_count,
            n_before
        );
        assert!(
            result.recall_at_k >= min_recall,
            "{label}: recall@5={:.3} below threshold {min_recall}",
            result.recall_at_k
        );
        // Witness chain integrity: every original id must appear exactly once.
        let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for w in &result.witness_records {
            for &id in &w.merged_ids {
                assert!(
                    seen.insert(id),
                    "{label}: duplicate id {id} in witness chain"
                );
            }
        }
    }

    #[test]
    fn naive_kmeans_compacts_with_acceptable_recall() {
        let c = NaiveCompactor::default();
        assert_compaction_passes(&c, 10, 20, 0.40, 0.70, "naive-kmeans");
    }

    #[test]
    fn graph_merge_compacts_and_high_recall() {
        let c = GraphMergeCompactor {
            graph_k: 10,
            merge_threshold: None,
        };
        assert_compaction_passes(&c, 10, 20, 0.40, 0.90, "graph-merge");
    }

    #[test]
    fn coherence_gated_compacts_with_high_recall() {
        let c = CoherenceGatedCompactor {
            graph_k: 10,
            coherence_floor: 0.25,
            max_cluster: 15,
        };
        assert_compaction_passes(&c, 10, 20, 0.40, 0.85, "coherence-gated");
    }

    #[test]
    fn recall_at_k_perfect_before_equals_after() {
        let (store, queries) = clustered_store(5, 10, 16, 0.1, 99);
        let entries = store.entries.clone();
        let r = recall_at_k(&queries, &entries, &entries, 5);
        assert!(
            (r - 1.0).abs() < 1e-5,
            "recall against self should be 1.0, got {r}"
        );
    }

    #[test]
    fn witness_records_cover_all_originals() {
        let (mut store, queries) = clustered_store(5, 10, 32, 0.15, 77);
        let n = store.len();
        let c = CoherenceGatedCompactor {
            graph_k: 8,
            coherence_floor: 0.25,
            max_cluster: 15,
        };
        let result = run_compaction(&c, &mut store, 0.40, &queries, 5);
        let mut all_merged: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for w in &result.witness_records {
            for &id in &w.merged_ids {
                all_merged.insert(id);
            }
        }
        // Every vector from before must be in exactly one witness record.
        assert_eq!(
            all_merged.len(),
            n,
            "expected all {n} original ids in witness chain, found {}",
            all_merged.len()
        );
    }

    #[test]
    fn acceptance_recall_passes_threshold() {
        // Integration: all three compactors should achieve ≥55% recall
        // on a 10-topic × 30-vec dataset at 60% compaction.
        let compactors: Vec<(&str, Box<dyn Compactor>)> = vec![
            ("naive-kmeans", Box::new(NaiveCompactor::default())),
            (
                "graph-merge",
                Box::new(GraphMergeCompactor {
                    graph_k: 10,
                    merge_threshold: None,
                }),
            ),
            (
                "coherence-gated",
                Box::new(CoherenceGatedCompactor {
                    graph_k: 10,
                    coherence_floor: 0.25,
                    max_cluster: 20,
                }),
            ),
        ];
        for (name, c) in &compactors {
            let (mut store, queries) = clustered_store(10, 30, 64, 0.15, 42);
            let result = run_compaction(c.as_ref(), &mut store, 0.40, &queries, 5);
            assert!(
                result.recall_at_k >= 0.55,
                "{name}: recall@5={:.3} below acceptance threshold 0.55",
                result.recall_at_k
            );
        }
    }
}
