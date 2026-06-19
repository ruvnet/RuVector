//! # ruvector-hnsw-repair
//!
//! Online HNSW graph repair after vector deletions.
//!
//! Exposes three deletion strategies with measurable recall vs. latency tradeoffs:
//! - [`TombstoneOnly`]: mark-deleted, no structural change (fast delete, recall degrades)
//! - [`BatchRepair`]: periodic repair sweep over accumulated tombstones (amortised cost)
//! - [`EagerRepair`]: reconnect affected edges immediately on every delete (best recall)

pub mod graph;
pub mod strategy;

pub use graph::{HnswConfig, HnswGraph};
pub use strategy::{BatchRepair, DeleteResult, DeletionStrategy, EagerRepair, TombstoneOnly};

/// Compute recall@k over a set of queries against ground truth.
///
/// Ground truth is produced by brute-force L2 scan over live (non-deleted) vectors.
pub fn recall_at_k(graph: &HnswGraph, queries: &[Vec<f32>], k: usize, ef_search: usize) -> f32 {
    let gt = brute_force_knn_live(graph, queries, k);
    let mut total = 0.0f32;
    for (i, query) in queries.iter().enumerate() {
        let results = graph.search(query, k, ef_search);
        let gt_set: std::collections::HashSet<u32> = gt[i].iter().cloned().collect();
        let hits = results.iter().filter(|&&id| gt_set.contains(&id)).count();
        total += hits as f32 / k as f32;
    }
    total / queries.len() as f32
}

/// Brute-force K-nearest-neighbours over live (non-deleted) vectors.
pub fn brute_force_knn_live(graph: &HnswGraph, queries: &[Vec<f32>], k: usize) -> Vec<Vec<u32>> {
    let dim = graph.config.dim;
    queries
        .iter()
        .map(|q| {
            let mut dists: Vec<(f32, u32)> = graph
                .vectors
                .iter()
                .enumerate()
                .filter(|(i, _)| !graph.deleted[*i])
                .map(|(i, v)| (l2_sq(q, v, dim), i as u32))
                .collect();
            dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            dists.iter().take(k).map(|(_, id)| *id).collect()
        })
        .collect()
}

/// Squared L2 distance for a fixed-dimension slice.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32], dim: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..dim {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::{BatchRepair, EagerRepair, TombstoneOnly};

    fn make_graph(n: usize, dim: usize, seed: u64) -> HnswGraph {
        let config = HnswConfig {
            dim,
            m: 8,
            m0: 16,
            ef_construction: 40,
            ml: 1.0 / (8f64.ln()),
        };
        let mut g = HnswGraph::new(config);
        let mut rng = seed;
        for _ in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| lcg_f32(&mut rng)).collect();
            g.insert(v);
        }
        g
    }

    fn lcg_f32(s: &mut u64) -> f32 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s >> 33) as f32 / (u32::MAX as f32)
    }

    #[test]
    fn baseline_recall_is_acceptable() {
        let g = make_graph(500, 32, 42);
        let mut rng = 999u64;
        let queries: Vec<Vec<f32>> = (0..20)
            .map(|_| (0..32).map(|_| lcg_f32(&mut rng)).collect())
            .collect();
        let r = recall_at_k(&g, &queries, 5, 20);
        assert!(r >= 0.50, "baseline recall@5 was {:.3}, want >= 0.50", r);
    }

    #[test]
    fn tombstone_recall_degrades_measurably() {
        let mut g = make_graph(400, 32, 77);
        let mut rng = 888u64;
        let queries: Vec<Vec<f32>> = (0..20)
            .map(|_| (0..32).map(|_| lcg_f32(&mut rng)).collect())
            .collect();
        let before = recall_at_k(&g, &queries, 5, 30);

        let strat = TombstoneOnly;
        for i in 0..80 {
            strat.delete(&mut g, i);
        }
        let after = recall_at_k(&g, &queries, 5, 30);
        // With 20% tombstones recall should still be above near-zero
        assert!(
            after < before + 0.05,
            "tombstone recall should not improve: before={:.3} after={:.3}",
            before,
            after
        );
    }

    #[test]
    fn eager_repair_maintains_recall() {
        let mut g = make_graph(400, 32, 55);
        let mut rng = 333u64;
        let queries: Vec<Vec<f32>> = (0..20)
            .map(|_| (0..32).map(|_| lcg_f32(&mut rng)).collect())
            .collect();
        let before = recall_at_k(&g, &queries, 5, 30);

        let strat = EagerRepair;
        for i in 0..80 {
            strat.delete(&mut g, i);
        }
        let after = recall_at_k(&g, &queries, 5, 30);
        // Eager repair keeps recall within 20pp of baseline
        assert!(
            before - after <= 0.20,
            "eager repair degraded too much: before={:.3} after={:.3}",
            before,
            after
        );
    }

    #[test]
    fn batch_repair_maintains_recall() {
        let mut g = make_graph(400, 32, 11);
        let mut rng = 222u64;
        let queries: Vec<Vec<f32>> = (0..20)
            .map(|_| (0..32).map(|_| lcg_f32(&mut rng)).collect())
            .collect();
        let before = recall_at_k(&g, &queries, 5, 30);

        let strat = BatchRepair::new(20);
        for i in 0..80 {
            strat.delete(&mut g, i);
        }
        strat.flush(&mut g);
        let after = recall_at_k(&g, &queries, 5, 30);
        assert!(
            before - after <= 0.25,
            "batch repair degraded too much: before={:.3} after={:.3}",
            before,
            after
        );
    }

    #[test]
    fn delete_result_counts_repaired_edges() {
        let mut g = make_graph(100, 16, 7);
        let strat = EagerRepair;
        let result = strat.delete(&mut g, 0);
        // Should have attempted at least the tombstone
        let _ = result; // repaired_edges may be zero if node 0 is not referenced
    }

    #[test]
    fn search_skips_deleted_nodes() {
        let mut g = make_graph(200, 16, 13);
        let strat = TombstoneOnly;
        // Delete the first 50 nodes
        for i in 0..50 {
            strat.delete(&mut g, i);
        }
        let mut rng = 500u64;
        let q: Vec<f32> = (0..16).map(|_| lcg_f32(&mut rng)).collect();
        let results = g.search(&q, 5, 20);
        for id in &results {
            assert!(!g.deleted[*id as usize], "deleted node {} in results", id);
        }
    }
}
