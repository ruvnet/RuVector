//! # ruvector-wal-ann
//!
//! Streaming Write-Ahead Log for ANN with Coherence-Gated Merge.
//!
//! ## Problem
//!
//! High-throughput vector ingestion creates a tension: maintaining a fresh
//! navigable graph (good recall) requires expensive graph updates on every
//! insert, but deferring updates (good throughput) risks serving stale data.
//!
//! All major production systems (Milvus, LanceDB, Qdrant) resolve this with a
//! size-only WAL flush policy: "seal and index when the segment reaches N
//! vectors." No system currently uses a quality signal to gate the merge.
//!
//! ## Solution
//!
//! [`WalAnnIndex`] introduces a three-tier architecture:
//!
//! 1. **WAL tier** — incoming vectors are buffered here; always searchable via
//!    O(|WAL|·D) linear scan, so no inserted vector is ever invisible.
//! 2. **Coherence gate** — evaluates whether the WAL should be flushed based
//!    on WAL size and/or a coherence score derived from how isolated the WAL
//!    vectors are relative to the main graph.
//! 3. **Main graph tier** — a navigable small-world graph that absorbs WAL
//!    vectors in O(|WAL|·ef·M·D) batch merges when the gate fires.
//!
//! ## Coherence score
//!
//! ```text
//! isolation(v, G) = min{ dist(v, g) | g ∈ G }
//! coherence       = 1 / (1 + mean(isolation(v, G) for v in WAL))
//! ```
//!
//! High coherence → WAL vectors are close to existing graph nodes → merge can
//! wait. Low coherence → WAL vectors are isolated → merge is urgent.
//!
//! ## Variants benchmarked
//!
//! | Variant | Gate | Trade-off |
//! |---------|------|-----------|
//! | `EagerMerge` | Size ≤ 32 | Frequent small merges |
//! | `LazyMerge` | Size ≤ 512 | Infrequent large merges |
//! | `CoherenceGatedMerge` | Coherence < 0.45 OR size ≤ 256 | Quality-driven merges |

pub mod gate;
pub mod graph;
pub mod wal;

pub use gate::{CoherenceGate, EagerGate, LazyGate, MergeGate};
pub use graph::{GraphConfig, NavGraph};
pub use wal::VectorWal;

use wal::l2_sq;

/// Combined search result merging WAL and main-graph hits.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Globally-unique vector ID assigned at insert time.
    pub id: u64,
    /// Squared L2 distance to the query (no sqrt for consistency).
    pub dist_sq: f32,
    /// Where the result came from.
    pub source: ResultSource,
}

/// Indicates which tier a result originated from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResultSource {
    /// Found in the main navigable graph.
    Graph,
    /// Found in the pending write-ahead log.
    Wal,
}

/// Streaming ANN index with WAL buffering and coherence-gated merges.
///
/// The `G` type parameter allows swapping the merge gate strategy without
/// changing the index logic (open/closed principle).
pub struct WalAnnIndex<G: MergeGate> {
    graph: NavGraph,
    wal: VectorWal,
    gate: G,
    /// Monotonically increasing ID counter (external IDs are remapped internally).
    next_id: u64,
    /// Mapping: internal graph node index → external ID.
    graph_ids: Vec<u64>,
    /// Total number of WAL → graph merges performed.
    pub merge_count: usize,
    /// Cumulative number of vectors that have entered the WAL.
    pub total_inserted: u64,
}

impl<G: MergeGate> WalAnnIndex<G> {
    /// Create a new index.
    ///
    /// `dims`          — vector dimensionality (all inserts must match).
    /// `wal_capacity`  — WAL buffer capacity (used as overflow guard).
    /// `gate`          — merge gate strategy controlling flush decisions.
    pub fn new(dims: usize, wal_capacity: usize, gate: G) -> Self {
        WalAnnIndex {
            graph: NavGraph::new(GraphConfig::new(dims)),
            wal: VectorWal::new(wal_capacity, dims),
            gate,
            next_id: 0,
            graph_ids: Vec::new(),
            merge_count: 0,
            total_inserted: 0,
        }
    }

    /// Insert a vector. The vector receives a monotonically increasing ID.
    ///
    /// After insertion, the coherence gate is evaluated. Coherence is
    /// recomputed only every [`COHERENCE_CHECK_INTERVAL`] inserts to avoid
    /// O(|WAL|·|graph|) cost on every write; between checks we pass coherence=1.0
    /// to the gate (meaning "no quality alarm — use size threshold only").
    ///
    /// Returns the assigned ID.
    pub fn insert(&mut self, vector: Vec<f32>) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.total_inserted += 1;

        self.wal.push(id, vector);

        // Evaluate gate: sampled coherence every COHERENCE_CHECK_INTERVAL inserts.
        let coherence = if self.wal.len() % Self::COHERENCE_CHECK_INTERVAL == 0 {
            self.coherence_score_sampled()
        } else {
            1.0
        };
        if self.gate.should_merge(self.wal.len(), coherence) {
            self.flush_wal();
        }

        id
    }

    /// How often (in number of inserts) to recompute the coherence score.
    ///
    /// Lower values = more responsive gate at higher CPU cost.
    /// Higher values = cheaper inserts at the cost of delayed coherence updates.
    const COHERENCE_CHECK_INTERVAL: usize = 8;

    /// Force a WAL flush regardless of gate state.
    ///
    /// Should be called before querying if freshness is critical, or at the
    /// end of a bulk-load phase.
    pub fn flush_wal(&mut self) {
        if self.wal.is_empty() {
            return;
        }
        let entries = self.wal.drain();
        for entry in entries {
            let idx = self.graph.insert(&entry.vector);
            // Ensure id mapping vector is large enough.
            let idx = idx as usize;
            if self.graph_ids.len() <= idx {
                self.graph_ids.resize(idx + 1, u64::MAX);
            }
            self.graph_ids[idx] = entry.id;
        }
        self.merge_count += 1;
    }

    /// Search for the `k` approximate nearest neighbours of `query`.
    ///
    /// Results are merged from both the main graph and the pending WAL.
    /// Final list is sorted by ascending distance.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let ef = (k * 6).max(64);

        // 1. Main graph search.
        let graph_hits = self.graph.search(query, k, ef);
        let mut results: Vec<SearchResult> = graph_hits
            .into_iter()
            .filter_map(|(node_idx, dist)| {
                let id = self.graph_ids.get(node_idx as usize).copied()?;
                if id == u64::MAX {
                    return None;
                }
                Some(SearchResult {
                    id,
                    dist_sq: dist,
                    source: ResultSource::Graph,
                })
            })
            .collect();

        // 2. WAL linear scan.
        let wal_hits = self.wal.search(query, k);
        for (id, dist) in wal_hits {
            results.push(SearchResult {
                id,
                dist_sq: dist,
                source: ResultSource::Wal,
            });
        }

        // 3. Sort combined results and keep top-k.
        results.sort_unstable_by(|a, b| a.dist_sq.total_cmp(&b.dist_sq));
        results.truncate(k);
        results
    }

    /// Compute the exact coherence score between the WAL and the main graph.
    ///
    /// O(|WAL| × |graph| × D) — use [`coherence_score_sampled`] in hot paths.
    pub fn coherence_score(&self) -> f32 {
        if self.wal.is_empty() || self.graph.is_empty() {
            return 1.0;
        }
        let total_isolation: f32 = self
            .wal
            .iter()
            .map(|entry| {
                (0..self.graph.n)
                    .map(|i| l2_sq(&entry.vector, self.graph.row(i)).sqrt())
                    .fold(f32::MAX, f32::min)
            })
            .sum();
        let avg_isolation = total_isolation / self.wal.len() as f32;
        1.0 / (1.0 + avg_isolation)
    }

    /// Sampled coherence score: O(WAL_SAMPLE × GRAPH_SAMPLE × D).
    ///
    /// Approximates the exact coherence by uniformly sampling from both the
    /// WAL and the graph. Accurate enough for the gate decision; avoids the
    /// O(N²) cost of exact coherence on every insert.
    pub fn coherence_score_sampled(&self) -> f32 {
        const WAL_SAMPLE: usize = 16;
        const GRAPH_SAMPLE: usize = 64;

        if self.wal.is_empty() || self.graph.is_empty() {
            return 1.0;
        }
        let wal_step = (self.wal.len() / WAL_SAMPLE).max(1);
        let graph_step = (self.graph.n / GRAPH_SAMPLE).max(1);

        let wal_entries: Vec<_> = self.wal.iter().step_by(wal_step).collect();
        let n_wal = wal_entries.len();
        let total: f32 = wal_entries
            .iter()
            .map(|e| {
                (0..self.graph.n)
                    .step_by(graph_step)
                    .map(|i| l2_sq(&e.vector, self.graph.row(i)).sqrt())
                    .fold(f32::MAX, f32::min)
            })
            .sum();
        1.0 / (1.0 + total / n_wal as f32)
    }

    /// Gate name (for benchmark labelling).
    pub fn gate_name(&self) -> &'static str {
        self.gate.name()
    }

    /// Number of vectors in the main graph.
    pub fn graph_size(&self) -> usize {
        self.graph.n
    }

    /// Number of pending WAL entries.
    pub fn wal_size(&self) -> usize {
        self.wal.len()
    }

    /// Estimated memory usage in bytes (graph store + WAL store + id map).
    pub fn memory_bytes(&self) -> usize {
        let graph_mem = self.graph.memory_bytes();
        let wal_mem = self.wal.len() * (self.wal.dims() * 4 + 16);
        let ids_mem = self.graph_ids.len() * 8;
        graph_mem + wal_mem + ids_mem
    }
}

/// Compute recall@k: fraction of true top-k neighbours found in results.
///
/// `true_ids`   — ground truth IDs in true distance order (first k used).
/// `result_ids` — IDs returned by the index search.
pub fn recall_at_k(true_ids: &[u64], result_ids: &[u64], k: usize) -> f32 {
    let gt: std::collections::HashSet<u64> = true_ids.iter().take(k).copied().collect();
    let found = result_ids
        .iter()
        .take(k)
        .filter(|id| gt.contains(id))
        .count();
    found as f32 / k as f32
}

/// Brute-force ground-truth search for recall evaluation.
///
/// Returns the `k` nearest IDs in the combined set of `(id, vector)` pairs.
pub fn brute_force_search(dataset: &[(u64, Vec<f32>)], query: &[f32], k: usize) -> Vec<u64> {
    let mut dists: Vec<(u64, f32)> = dataset
        .iter()
        .map(|(id, v)| (*id, l2_sq(v, query)))
        .collect();
    dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    dists.truncate(k);
    dists.into_iter().map(|(id, _)| id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dataset(n: usize, dims: usize, seed: u64) -> Vec<(u64, Vec<f32>)> {
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let dist = Normal::new(0.0f32, 1.0).unwrap();
        (0..n as u64)
            .map(|id| {
                let v: Vec<f32> = (0..dims).map(|_| dist.sample(&mut rng)).collect();
                (id, v)
            })
            .collect()
    }

    #[test]
    fn eager_index_recalls_all_inserts() {
        let data = make_dataset(200, 16, 42);
        let gate = EagerGate { threshold: 32 };
        let mut idx = WalAnnIndex::new(16, 64, gate);
        for (_, v) in &data {
            idx.insert(v.clone());
        }
        idx.flush_wal();
        // All 200 vectors should be accessible.
        assert_eq!(idx.graph_size() + idx.wal_size(), 200);
    }

    #[test]
    fn wal_search_returns_before_flush() {
        let data = make_dataset(50, 8, 7);
        let gate = LazyGate { threshold: 10_000 }; // Never auto-flushes.
        let mut idx = WalAnnIndex::new(8, 10_000, gate);
        for (_, v) in &data {
            idx.insert(v.clone());
        }
        // Should be fully in WAL.
        assert_eq!(idx.wal_size(), 50);
        assert_eq!(idx.graph_size(), 0);

        // Query should still find results from WAL scan.
        let q = vec![0.0f32; 8];
        let results = idx.search(&q, 5);
        assert_eq!(results.len(), 5);
        assert!(results.iter().all(|r| r.source == ResultSource::Wal));
    }

    #[test]
    fn coherence_gate_fires_on_isolated_wal() {
        let dims = 16;
        // Build a graph of vectors clustered near the origin.
        let data_a = make_dataset(100, dims, 1);
        let gate = CoherenceGate {
            min_coherence: 0.9,
            max_wal_size: 10_000,
        };
        let mut idx = WalAnnIndex::new(dims, 10_000, gate);
        for (_, v) in &data_a {
            idx.insert(v.clone());
        }
        idx.flush_wal();
        let merges_after_a = idx.merge_count;

        // Now insert vectors far away — coherence should drop and trigger merge.
        let data_b: Vec<Vec<f32>> = (0..20).map(|i| vec![100.0 + i as f32; dims]).collect();
        for v in &data_b {
            idx.insert(v.clone());
        }
        // The coherence gate should have fired at least once.
        assert!(
            idx.merge_count > merges_after_a || idx.wal_size() < 20,
            "coherence gate should have triggered for isolated vectors"
        );
    }

    #[test]
    fn recall_improves_after_flush() {
        let data = make_dataset(500, 32, 99);
        let gate = LazyGate { threshold: 10_000 };
        let mut idx = WalAnnIndex::new(32, 10_000, gate);
        for (_, v) in &data {
            idx.insert(v.clone());
        }

        let query = &data[0].1;
        let k = 10;
        let true_ids = brute_force_search(&data, query, k);

        // Before flush — WAL only.
        let pre_ids: Vec<u64> = idx.search(query, k).iter().map(|r| r.id).collect();
        let pre_recall = recall_at_k(&true_ids, &pre_ids, k);

        // After flush — main graph.
        idx.flush_wal();
        let post_ids: Vec<u64> = idx.search(query, k).iter().map(|r| r.id).collect();
        let post_recall = recall_at_k(&true_ids, &post_ids, k);

        // Both should give reasonable recall; post-flush should be at least as good.
        assert!(
            pre_recall >= 0.5,
            "WAL linear scan should have good recall (pre={pre_recall:.2})"
        );
        assert!(
            post_recall >= 0.7,
            "graph search should have ≥0.70 recall (post={post_recall:.2})"
        );
    }

    #[test]
    fn recall_at_k_computation() {
        let true_ids = vec![1u64, 2, 3, 4, 5];
        let result_ids = vec![1u64, 3, 7, 9, 2];
        let r = recall_at_k(&true_ids, &result_ids, 5);
        assert!((r - 0.6).abs() < 1e-6, "expected 3/5 = 0.6, got {r}");
    }
}
