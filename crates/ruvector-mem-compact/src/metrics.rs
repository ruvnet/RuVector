//! Recall and quality metrics for compaction evaluation.
//!
//! Two recall flavours:
//!
//! - `recall_at_k` — **ID-exact recall**: counts fraction of true top-K IDs
//!   still present in the compacted store.  Measures how well the *exact* same
//!   vectors are retained.  Suitable for datasets where the "best" answer is a
//!   specific ID.
//!
//! - `cluster_coverage_recall` — **semantic/cluster recall**: for each query,
//!   checks whether the compacted store returns at least one vector from the
//!   same cluster as the true top-1 answer.  Suitable for agent memory, where
//!   any representative of the correct concept is a valid answer.
//!
//! Graph-cut compaction is designed to optimise cluster-coverage recall, NOT
//! ID-exact recall.  ID-exact recall on a uniform random 50% subset is
//! theoretically bounded at ~50% regardless of selection strategy.

use crate::compactor::{knn_brute, MemoryStore};
use std::collections::HashSet;

/// Computes ID-exact recall@K: fraction of true top-K IDs recovered after
/// compaction, averaged over all queries.
pub fn recall_at_k(
    queries: &[Vec<f32>],
    original: &MemoryStore,
    compacted: &MemoryStore,
    k: usize,
) -> f32 {
    if queries.is_empty() || compacted.is_empty() {
        return 0.0;
    }
    let effective_k = k.min(compacted.len());
    let mut total_hits = 0u64;
    let total_possible = (queries.len() * effective_k) as u64;

    for q in queries {
        let true_ids: HashSet<u64> = knn_brute(q, original, k)
            .into_iter()
            .map(|(idx, _)| original.entries[idx].id)
            .collect();
        let found_ids: HashSet<u64> = knn_brute(q, compacted, effective_k)
            .into_iter()
            .map(|(idx, _)| compacted.entries[idx].id)
            .collect();
        total_hits += true_ids.intersection(&found_ids).count() as u64;
    }
    total_hits as f32 / total_possible as f32
}

/// Cluster-coverage recall: for each query, checks whether at least one of the
/// compacted store's top-K results belongs to the same "concept cluster" as the
/// original top-1 answer.
///
/// `cluster_of`: maps a vector ID to its concept cluster label (0-indexed).
///
/// This is the primary quality metric for agent memory compaction: agents care
/// whether a relevant memory *exists* in the store, not whether the exact same
/// embedding is retained.
pub fn cluster_coverage_recall(
    queries: &[Vec<f32>],
    original: &MemoryStore,
    compacted: &MemoryStore,
    cluster_of: &[usize], // cluster_of[store_index] = cluster label
    k: usize,
) -> f32 {
    if queries.is_empty() || compacted.is_empty() {
        return 0.0;
    }
    let effective_k = k.min(compacted.len());
    let mut hits = 0u64;

    for q in queries {
        // True top-1 cluster in original store.
        let (true_idx, _) = knn_brute(q, original, 1).into_iter().next().unwrap();
        let true_cluster = cluster_of[true_idx];

        // Check if any of the compacted store's top-K belong to the same cluster.
        let found = knn_brute(q, compacted, effective_k)
            .into_iter()
            .any(|(idx, _)| {
                // Find this compacted entry in the original store to get its cluster.
                let id = compacted.entries[idx].id;
                let orig_idx = original
                    .entries
                    .iter()
                    .position(|e| e.id == id)
                    .unwrap_or(usize::MAX);
                orig_idx < cluster_of.len() && cluster_of[orig_idx] == true_cluster
            });
        if found {
            hits += 1;
        }
    }
    hits as f32 / queries.len() as f32
}

/// Bundle of metrics captured after a compaction pass.
#[derive(Debug, Clone)]
pub struct CompactionMetrics {
    pub variant: &'static str,
    pub n_before: usize,
    pub n_after: usize,
    pub compaction_pct: f32,
    pub recall_id_exact: f32,
    pub recall_cluster_cov: f32,
    pub compact_ms: f32,
    pub query_us_mean: f32,
    pub query_us_p50: f32,
    pub query_us_p95: f32,
    pub memory_kb: f32,
    pub acceptance: bool,
}

impl CompactionMetrics {
    pub fn print_row(&self) {
        println!(
            "{:<22} | {:>6} | {:>6} | {:>9.1}% | {:>9.1}% | {:>12.1}% | {:>9.2} | {:>9.2} | {:>9.2} | {:>9.2} | {:>9.0} | {}",
            self.variant,
            self.n_before,
            self.n_after,
            self.compaction_pct * 100.0,
            self.recall_id_exact * 100.0,
            self.recall_cluster_cov * 100.0,
            self.compact_ms,
            self.query_us_mean,
            self.query_us_p50,
            self.query_us_p95,
            self.memory_kb,
            if self.acceptance { "PASS" } else { "FAIL" },
        );
    }

    pub fn print_header() {
        println!(
            "{:<22} | {:>6} | {:>6} | {:>10} | {:>10} | {:>13} | {:>10} | {:>10} | {:>10} | {:>10} | {:>9} | {}",
            "Variant", "N-orig", "N-kept", "Compact%",
            "Recall(ID)", "Recall(Clust)", "Cmpct(ms)",
            "Qry µs avg", "Qry µs p50", "Qry µs p95", "Mem(KB)", "Accept"
        );
        println!("{}", "-".repeat(155));
    }
}

/// Measure mean / p50 / p95 query latency in microseconds.
pub fn query_latencies(queries: &[Vec<f32>], store: &MemoryStore, k: usize) -> (f32, f32, f32) {
    let mut times_us: Vec<f32> = queries
        .iter()
        .map(|q| {
            let t = std::time::Instant::now();
            let _ = knn_brute(q, store, k);
            t.elapsed().as_secs_f32() * 1_000_000.0
        })
        .collect();
    times_us.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let n = times_us.len();
    let mean = times_us.iter().sum::<f32>() / n as f32;
    let p50 = times_us[n / 2];
    let p95 = times_us[((n as f32 * 0.95) as usize).min(n - 1)];
    (mean, p50, p95)
}

/// Estimated memory: dims × 4 bytes per f32 × n entries, in KB.
pub fn memory_kb(n: usize, dims: usize) -> f32 {
    (n * dims * 4) as f32 / 1024.0
}
