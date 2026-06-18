use crate::graph::{cosine_distance, KnnGraph};
use crate::shard::Shard;

/// Brute-force top-k nearest neighbors from the full graph for a query.
/// Ground truth for recall measurement.
pub fn brute_force_knn(graph: &KnnGraph, query: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut dists: Vec<(f32, u32)> = (0..graph.n)
        .map(|i| (cosine_distance(query, graph.get_vector(i)), i as u32))
        .collect();
    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    dists.truncate(k);
    dists.into_iter().map(|(d, id)| (id, d)).collect()
}

/// Brute-force search within a shard.
///
/// Since shards are small (typically ≤ 1 024 nodes), a linear scan over shard
/// vectors is faster in practice than graph traversal and avoids false misses
/// from local graph optima. The local_neighbors stored in the shard are used in
/// production graph-walk mode (future work).
pub fn search_shard(shard: &Shard, query: &[f32], k: usize) -> Vec<(u32, f32)> {
    let n_local = shard.node_ids.len();
    if n_local == 0 {
        return vec![];
    }
    let k_actual = k.min(n_local);

    let mut dists: Vec<(f32, u32)> = (0..n_local)
        .map(|local| {
            let d = cosine_distance(query, shard.get_vector(local));
            (d, shard.node_ids[local])
        })
        .collect();

    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    dists.truncate(k_actual);
    dists.into_iter().map(|(d, id)| (id, d)).collect()
}

/// Recall@k: fraction of ground-truth top-k that appear in the shard results.
pub fn recall_at_k(shard_results: &[(u32, f32)], ground_truth: &[(u32, f32)], k: usize) -> f32 {
    if k == 0 {
        return 1.0;
    }
    let truth_ids: std::collections::HashSet<u32> =
        ground_truth.iter().take(k).map(|(id, _)| *id).collect();
    let found = shard_results
        .iter()
        .take(k)
        .filter(|(id, _)| truth_ids.contains(id))
        .count();
    let denom = k.min(ground_truth.len());
    if denom == 0 {
        return 0.0;
    }
    found as f32 / denom as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphConfig;
    use crate::shard::{BfsShard, ShardExtractor};

    fn sequential_graph(n: usize, dim: usize) -> KnnGraph {
        let vectors: Vec<f32> = (0..n)
            .flat_map(|i| {
                let mut v = vec![0.0f32; dim];
                v[0] = i as f32;
                v
            })
            .collect();
        KnnGraph::build(vectors, dim, &GraphConfig { k_neighbors: 4 }).unwrap()
    }

    #[test]
    fn brute_force_returns_k_results() {
        let g = sequential_graph(20, 4);
        let query = vec![10.0f32, 0.0, 0.0, 0.0];
        let results = brute_force_knn(&g, &query, 5);
        assert_eq!(results.len(), 5);
        // Distances should be non-decreasing.
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1 + 1e-6);
        }
    }

    #[test]
    fn shard_search_returns_correct_count() {
        let g = sequential_graph(16, 4);
        let shard = BfsShard.extract(&g, &[0], 8);
        let query = vec![2.0f32, 0.0, 0.0, 0.0];
        let results = search_shard(&shard, &query, 5);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn recall_perfect_when_sets_match() {
        let results = vec![(1u32, 0.1), (2, 0.2), (3, 0.3)];
        let truth = vec![(1u32, 0.1), (2, 0.2), (3, 0.3)];
        let r = recall_at_k(&results, &truth, 3);
        assert!((r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn recall_zero_when_no_overlap() {
        let results = vec![(10u32, 0.1), (11, 0.2)];
        let truth = vec![(1u32, 0.1), (2, 0.2)];
        let r = recall_at_k(&results, &truth, 2);
        assert!(r < 1e-6);
    }
}
