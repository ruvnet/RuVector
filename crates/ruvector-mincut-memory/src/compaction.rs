//! Three compaction strategies that implement [`Compactor`].
//!
//! | Strategy       | Eviction criterion                                    |
//! |----------------|-------------------------------------------------------|
//! | `AgeEvict`     | Oldest entries by logical timestamp (baseline)        |
//! | `CoherenceEvict` | Lowest mean cosine similarity to graph neighbours   |
//! | `MinCutEvict`  | Lowest weighted graph degree (min-cut approximation)  |
//!
//! All three share the same acceptance test: after compaction,
//! `store.len() <= target_size`.

use std::time::Instant;

use crate::{metrics::CompactionResult, store::MemoryStore, Compactor};

fn edge_count(graph: &[Vec<f32>]) -> usize {
    let n = graph.len();
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            if graph[i][j] > 0.0 {
                count += 1;
            }
        }
    }
    count
}

// ─── Baseline: age-based eviction ────────────────────────────────────────────

/// Evict the oldest `N - target_size` entries by logical timestamp.
///
/// This is the simplest possible baseline — no graph reasoning required.
pub struct AgeEvict;

impl Compactor for AgeEvict {
    fn compact(&self, store: &mut MemoryStore, target_size: usize) -> CompactionResult {
        let t0 = Instant::now();
        let entries_before = store.len();
        store.ensure_graph();
        let edges_before = edge_count(&store.graph);

        if store.len() <= target_size {
            let edges_after = edges_before;
            return CompactionResult {
                entries_before,
                entries_after: store.len(),
                edges_before,
                edges_after,
                latency_us: t0.elapsed().as_micros() as u64,
                strategy: "AgeEvict",
            };
        }

        let to_remove = store.len() - target_size;
        // Build (index, timestamp) pairs and sort ascending by timestamp.
        let mut order: Vec<(usize, u64)> = store
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| (i, e.timestamp))
            .collect();
        order.sort_unstable_by_key(|&(_, ts)| ts);
        let evict_indices: Vec<usize> = order[..to_remove].iter().map(|(i, _)| *i).collect();

        store.remove_indices(evict_indices);
        store.ensure_graph();
        let edges_after = edge_count(&store.graph);

        CompactionResult {
            entries_before,
            entries_after: store.len(),
            edges_before,
            edges_after,
            latency_us: t0.elapsed().as_micros() as u64,
            strategy: "AgeEvict",
        }
    }
}

// ─── Coherence-scored eviction ────────────────────────────────────────────────

/// Evict entries with the lowest mean cosine similarity to their graph
/// neighbours.  Isolated entries (no edges) score 0.0 and are evicted first.
pub struct CoherenceEvict;

impl Compactor for CoherenceEvict {
    fn compact(&self, store: &mut MemoryStore, target_size: usize) -> CompactionResult {
        let t0 = Instant::now();
        let entries_before = store.len();
        store.ensure_graph();
        let edges_before = edge_count(&store.graph);

        if store.len() <= target_size {
            return CompactionResult {
                entries_before,
                entries_after: store.len(),
                edges_before,
                edges_after: edges_before,
                latency_us: t0.elapsed().as_micros() as u64,
                strategy: "CoherenceEvict",
            };
        }

        let n = store.len();
        let to_remove = n - target_size;

        // Coherence score = mean weight of incident edges.
        let mut scores: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let neighbours: Vec<f32> = store.graph[i]
                    .iter()
                    .filter(|&&w| w > 0.0)
                    .cloned()
                    .collect();
                let score = if neighbours.is_empty() {
                    0.0
                } else {
                    neighbours.iter().sum::<f32>() / neighbours.len() as f32
                };
                (i, score)
            })
            .collect();

        // Sort ascending — lowest coherence evicted first.
        scores.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let evict_indices: Vec<usize> = scores[..to_remove].iter().map(|(i, _)| *i).collect();

        store.remove_indices(evict_indices);
        store.ensure_graph();
        let edges_after = edge_count(&store.graph);

        CompactionResult {
            entries_before,
            entries_after: store.len(),
            edges_before,
            edges_after,
            latency_us: t0.elapsed().as_micros() as u64,
            strategy: "CoherenceEvict",
        }
    }
}

// ─── MinCut-guided eviction ───────────────────────────────────────────────────

/// Evict entries by lowest *weighted degree* in the similarity graph, which
/// approximates the minimum-cut boundary.
///
/// **Why this approximates min-cut:**  In Karger-Stein and Stoer-Wagner
/// algorithms, the vertices with the lowest weighted degree (sum of incident
/// edge weights) are statistically most likely to appear on a minimum cut.
/// Evicting these vertices removes the weakest-attached memory clusters while
/// preserving the dense, high-coherence core — exactly what an agent needs to
/// keep its most relevant context intact.
///
/// This is a polynomial-time heuristic, not the exact min-cut, but it runs in
/// O(N²) and is deterministic, making it measurable and reproducible.
pub struct MinCutEvict;

impl Compactor for MinCutEvict {
    fn compact(&self, store: &mut MemoryStore, target_size: usize) -> CompactionResult {
        let t0 = Instant::now();
        let entries_before = store.len();
        store.ensure_graph();
        let edges_before = edge_count(&store.graph);

        if store.len() <= target_size {
            return CompactionResult {
                entries_before,
                entries_after: store.len(),
                edges_before,
                edges_after: edges_before,
                latency_us: t0.elapsed().as_micros() as u64,
                strategy: "MinCutEvict",
            };
        }

        let n = store.len();
        let to_remove = n - target_size;

        // Weighted degree = sum of all incident edge weights.
        let mut degrees: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let deg: f32 = store.graph[i].iter().sum();
                (i, deg)
            })
            .collect();

        // Sort ascending — lowest weighted degree is most peripheral.
        degrees.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let evict_indices: Vec<usize> = degrees[..to_remove].iter().map(|(i, _)| *i).collect();

        store.remove_indices(evict_indices);
        store.ensure_graph();
        let edges_after = edge_count(&store.graph);

        CompactionResult {
            entries_before,
            entries_after: store.len(),
            edges_before,
            edges_after,
            latency_us: t0.elapsed().as_micros() as u64,
            strategy: "MinCutEvict",
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::MemoryStore;

    fn clustered_store() -> MemoryStore {
        // Two tight clusters: A (entries 0-3) and B (entries 4-7).
        // Entries within a cluster are similar; across clusters they are
        // orthogonal.  Cluster A: first-half dims active.
        //                Cluster B: second-half dims active.
        let mut s = MemoryStore::new(8, 0.3);
        let ts = |i: u64| i;
        // Cluster A — dimensions 0-3
        for i in 0u64..4 {
            let v = vec![
                1.0 - 0.05 * i as f32,
                0.05 * i as f32,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ];
            s.insert(v, ts(i));
        }
        // Cluster B — dimensions 4-7
        for i in 0u64..4 {
            let v = vec![
                0.0,
                0.0,
                0.0,
                0.0,
                1.0 - 0.05 * i as f32,
                0.05 * i as f32,
                0.0,
                0.0,
            ];
            s.insert(v, ts(i + 10));
        }
        s
    }

    fn no_op_result(store: &mut MemoryStore, target: usize, strategy: &dyn Compactor) {
        let before = store.len();
        let result = strategy.compact(store, target);
        assert_eq!(
            result.entries_after, before,
            "no-op should not remove entries"
        );
    }

    #[test]
    fn age_evict_no_op_when_under_budget() {
        let mut s = clustered_store();
        no_op_result(&mut s, 100, &AgeEvict);
    }

    #[test]
    fn coherence_evict_no_op_when_under_budget() {
        let mut s = clustered_store();
        no_op_result(&mut s, 100, &CoherenceEvict);
    }

    #[test]
    fn mincut_evict_no_op_when_under_budget() {
        let mut s = clustered_store();
        no_op_result(&mut s, 100, &MinCutEvict);
    }

    fn check_compacts_to(strat: &dyn Compactor, target: usize) {
        let mut s = clustered_store();
        let result = strat.compact(&mut s, target);
        assert_eq!(
            s.len(),
            target,
            "store should have exactly {} entries after compaction",
            target
        );
        assert_eq!(result.entries_after, target);
        assert!(
            result.latency_us < 1_000_000,
            "compaction should finish in < 1 second"
        );
    }

    #[test]
    fn age_evict_compacts_correctly() {
        check_compacts_to(&AgeEvict, 4);
    }

    #[test]
    fn coherence_evict_compacts_correctly() {
        check_compacts_to(&CoherenceEvict, 4);
    }

    #[test]
    fn mincut_evict_compacts_correctly() {
        check_compacts_to(&MinCutEvict, 4);
    }

    #[test]
    fn age_evict_removes_oldest() {
        // Timestamps 0-3 are oldest (cluster A).  After evicting 4, only B remains.
        let mut s = clustered_store();
        let _ = AgeEvict.compact(&mut s, 4);
        // All remaining entries should have timestamp >= 10 (cluster B).
        for e in &s.entries {
            assert!(e.timestamp >= 10, "old entries should have been removed");
        }
    }

    #[test]
    fn mincut_reduces_edge_count_or_maintains() {
        let mut s = clustered_store();
        s.rebuild_graph();
        let result = MinCutEvict.compact(&mut s, 4);
        // Removing peripheral nodes should not increase edge count per node.
        assert!(result.edges_after <= result.edges_before);
    }
}
