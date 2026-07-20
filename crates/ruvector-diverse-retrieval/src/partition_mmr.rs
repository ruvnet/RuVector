//! Partition-aware MMR retriever using graph-cut-inspired clustering.
//!
//! Extends standard MMR with a **same-partition penalty**: candidates that
//! belong to the same connectivity component as an already-selected result
//! receive a score deduction.  This forces the result set to span multiple
//! semantic clusters, complementing the mincut approach already used in
//! `ruvector-mincut` for graph-level boundary detection.
//!
//! ## Algorithm
//!
//! 1. Retrieve `POOL_FACTOR * k` candidates by distance.
//! 2. Build a pairwise connectivity graph among candidates using a threshold
//!    derived from the mean intra-pool distance × `THRESHOLD_FRACTION`.
//! 3. Extract connected components via union-find → partition labels.
//! 4. Greedy selection with a modified MMR score:
//!    ```text
//!    score(c) = −λ·dist(c,q) + (1−λ)·min_dist_to_selected
//!               − partition_penalty · same_partition_count(c, selected)
//!    ```
//! 5. `same_partition_count` counts how many selected results share c's partition.

use crate::graph::{estimate_threshold, partition_candidates};
use crate::{l2, DiverseRetriever, RetrievalResult};

/// Candidate pool multiplier.
const POOL_FACTOR: usize = 6;
/// Fraction of mean pairwise pool distance used as connectivity threshold.
/// Applied to the full pool so the bimodal intra/inter-sub-cluster distribution
/// is visible; using only the nearest N would sample purely intra-sub distances
/// and produce a threshold too small to connect within-sub pairs.
const THRESHOLD_FRACTION: f32 = 0.55;

/// MMR retriever augmented with a graph-cut-inspired same-partition penalty.
pub struct PartitionMmrRetriever<'a> {
    vectors: &'a [Vec<f32>],
    /// Relevance-diversity trade-off in [0, 1].
    lambda: f32,
    /// Score penalty per already-selected result in the same partition.
    partition_penalty: f32,
}

impl<'a> PartitionMmrRetriever<'a> {
    /// Create a PartitionMMR retriever.
    ///
    /// # Parameters
    /// - `lambda`: trade-off weight (1.0 = pure relevance, 0.0 = pure diversity)
    /// - `partition_penalty`: per-same-partition-member penalty applied to MMR score
    ///
    /// # Panics
    /// Panics if `lambda` is not in `[0, 1]` or `partition_penalty` is negative.
    pub fn new(vectors: &'a [Vec<f32>], lambda: f32, partition_penalty: f32) -> Self {
        assert!((0.0..=1.0).contains(&lambda), "lambda must be in [0, 1]");
        assert!(
            partition_penalty >= 0.0,
            "partition_penalty must be non-negative"
        );
        Self {
            vectors,
            lambda,
            partition_penalty,
        }
    }
}

impl<'a> DiverseRetriever for PartitionMmrRetriever<'a> {
    fn search(&self, query: &[f32], k: usize) -> Vec<RetrievalResult> {
        let pool_size = (k * POOL_FACTOR).min(self.vectors.len());
        let k = k.min(pool_size);

        // Step 1: retrieve candidate pool sorted by distance.
        let mut scored: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, l2(query, v)))
            .collect();
        scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let candidates: Vec<(usize, f32)> = scored.into_iter().take(pool_size).collect();
        let candidate_ids: Vec<usize> = candidates.iter().map(|&(id, _)| id).collect();

        // Step 2: estimate connectivity threshold and build partitions.
        // Use the full pool so the sample spans both intra- and inter-sub-cluster
        // distances, giving a bimodal distribution whose mean sits between the two
        // modes.  That way the threshold correctly merges within-sub pairs while
        // keeping between-sub pairs in separate components.
        let threshold =
            estimate_threshold(&candidate_ids, self.vectors, THRESHOLD_FRACTION, pool_size);
        let part_labels = partition_candidates(&candidate_ids, self.vectors, threshold);

        // Step 3: attach partition labels to candidates.
        // remaining: (dataset_id, dist_to_query, partition_label)
        let mut remaining: Vec<(usize, f32, usize)> = candidates
            .into_iter()
            .zip(part_labels.into_iter())
            .map(|((id, dist), part)| (id, dist, part))
            .collect();

        // Step 4: greedy PartitionMMR selection.
        // selected: (dataset_id, dist_to_query, partition_label)
        let mut selected: Vec<(usize, f32, usize)> = Vec::with_capacity(k);

        while selected.len() < k && !remaining.is_empty() {
            let best_pos = remaining
                .iter()
                .enumerate()
                .map(|(i, &(id, dist_q, partition))| {
                    let score = if selected.is_empty() {
                        -dist_q
                    } else {
                        let min_dist_to_selected = selected
                            .iter()
                            .map(|&(sid, _, _)| l2(&self.vectors[id], &self.vectors[sid]))
                            .fold(f32::INFINITY, f32::min);

                        let same_count = selected
                            .iter()
                            .filter(|&&(_, _, sp)| sp == partition)
                            .count();

                        -self.lambda * dist_q + (1.0 - self.lambda) * min_dist_to_selected
                            - self.partition_penalty * same_count as f32
                    };
                    (i, score)
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap();

            let chosen = remaining.remove(best_pos);
            selected.push(chosen);
        }

        selected
            .into_iter()
            .map(|(id, distance, _part)| RetrievalResult {
                id,
                distance,
                diversity_score: 0.0,
            })
            .collect()
    }

    fn name(&self) -> &str {
        "PartitionMMR"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mean_diversity;

    /// Two well-separated clusters: A near [0,0], B near [20,0].
    fn two_cluster_vecs() -> Vec<Vec<f32>> {
        let mut v = Vec::new();
        for i in 0..6 {
            v.push(vec![i as f32 * 0.15, 0.0]);
        }
        for i in 0..6 {
            v.push(vec![20.0 + i as f32 * 0.15, 0.0]);
        }
        v
    }

    #[test]
    fn returns_k_results() {
        let v = two_cluster_vecs();
        let r = PartitionMmrRetriever::new(&v, 0.5, 1.5);
        let res = r.search(&[0.0f32, 0.0], 3);
        assert_eq!(res.len(), 3);
    }

    #[test]
    fn partition_mmr_more_diverse_than_topk() {
        let v = two_cluster_vecs();
        let pmr = PartitionMmrRetriever::new(&v, 0.5, 2.0);
        use crate::topk::TopKRetriever;
        let topk = TopKRetriever::new(&v);
        let query = vec![0.0f32, 0.0];

        let pmr_res = pmr.search(&query, 4);
        let topk_res = topk.search(&query, 4);

        let pmr_div = mean_diversity(&pmr_res, &v);
        let topk_div = mean_diversity(&topk_res, &v);

        assert!(
            pmr_div > topk_div,
            "PartitionMMR should be more diverse than TopK: {pmr_div:.2} vs {topk_div:.2}"
        );
    }

    #[test]
    fn high_penalty_forces_cross_cluster() {
        let v = two_cluster_vecs();
        let r = PartitionMmrRetriever::new(&v, 0.5, 100.0);
        let res = r.search(&[0.0f32, 0.0], 2);
        let has_near = res.iter().any(|r| v[r.id][0] < 5.0);
        let has_far = res.iter().any(|r| v[r.id][0] > 15.0);
        assert!(has_near, "should include a near-cluster result");
        assert!(has_far, "should include a far-cluster result");
    }
}
