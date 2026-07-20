//! Standard Maximal Marginal Relevance (MMR) retriever.
//!
//! MMR (Carbonell & Goldstein, 1998) balances relevance and diversity via:
//!
//! ```text
//! MMR(c) = λ · (-dist(c, q)) − (1−λ) · max_{s ∈ S} (−dist(c, s))
//!        = −λ · dist(c, q) + (1−λ) · min_{s ∈ S} dist(c, s)
//! ```
//!
//! A high `lambda` (→ 1) favours relevance; low `lambda` (→ 0) favours
//! diversity.  `lambda = 0.5` is the commonly used balanced setting.
//!
//! Search procedure:
//! 1. Retrieve `POOL_FACTOR * k` candidates by distance.
//! 2. Greedily select the candidate with the highest MMR score.
//! 3. Update the selected set after each pick.

use crate::{l2, DiverseRetriever, RetrievalResult};

/// Number of candidates to retrieve as a multiple of `k`.
const POOL_FACTOR: usize = 6;

/// Maximal Marginal Relevance retriever.
pub struct MmrRetriever<'a> {
    vectors: &'a [Vec<f32>],
    /// Relevance-diversity trade-off: 1.0 = pure relevance, 0.0 = pure diversity.
    lambda: f32,
}

impl<'a> MmrRetriever<'a> {
    /// Create an MMR retriever with the given trade-off weight.
    ///
    /// # Panics
    /// Panics if `lambda` is not in `[0, 1]`.
    pub fn new(vectors: &'a [Vec<f32>], lambda: f32) -> Self {
        assert!((0.0..=1.0).contains(&lambda), "lambda must be in [0, 1]");
        Self { vectors, lambda }
    }
}

impl<'a> DiverseRetriever for MmrRetriever<'a> {
    fn search(&self, query: &[f32], k: usize) -> Vec<RetrievalResult> {
        let pool_size = (k * POOL_FACTOR).min(self.vectors.len());
        let k = k.min(pool_size);

        // Step 1: retrieve candidate pool.
        let mut scored: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, l2(query, v)))
            .collect();
        scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut remaining: Vec<(usize, f32)> = scored.into_iter().take(pool_size).collect();

        // Step 2: greedy MMR selection.
        let mut selected: Vec<(usize, f32)> = Vec::with_capacity(k);

        while selected.len() < k && !remaining.is_empty() {
            let best_pos = remaining
                .iter()
                .enumerate()
                .map(|(i, &(id, dist_q))| {
                    let mmr_score = if selected.is_empty() {
                        -dist_q
                    } else {
                        let min_dist_to_selected = selected
                            .iter()
                            .map(|&(sid, _)| l2(&self.vectors[id], &self.vectors[sid]))
                            .fold(f32::INFINITY, f32::min);

                        -self.lambda * dist_q + (1.0 - self.lambda) * min_dist_to_selected
                    };
                    (i, mmr_score)
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap();

            let (chosen_id, chosen_dist) = remaining.remove(best_pos);
            selected.push((chosen_id, chosen_dist));
        }

        selected
            .into_iter()
            .map(|(id, distance)| RetrievalResult {
                id,
                distance,
                diversity_score: 0.0,
            })
            .collect()
    }

    fn name(&self) -> &str {
        "MMR (λ=0.5)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mean_diversity;

    fn two_cluster_vecs() -> Vec<Vec<f32>> {
        // Cluster A near [0, 0]: indices 0-3
        // Cluster B near [10, 0]: indices 4-7
        let mut v = Vec::new();
        for i in 0..4 {
            v.push(vec![i as f32 * 0.1, 0.0]);
        }
        for i in 0..4 {
            v.push(vec![10.0 + i as f32 * 0.1, 0.0]);
        }
        v
    }

    #[test]
    fn returns_k_results() {
        let v = two_cluster_vecs();
        let r = MmrRetriever::new(&v, 0.5);
        let res = r.search(&[0.0f32, 0.0], 3);
        assert_eq!(res.len(), 3);
    }

    #[test]
    fn lambda_0_maximises_diversity() {
        let v = two_cluster_vecs();
        let r_div = MmrRetriever::new(&v, 0.0);
        let r_rel = MmrRetriever::new(&v, 1.0);
        let query = vec![0.0f32, 0.0];

        let res_div = r_div.search(&query, 2);
        let res_rel = r_rel.search(&query, 2);

        let div_score = mean_diversity(&res_div, &v);
        let rel_score = mean_diversity(&res_rel, &v);

        assert!(
            div_score >= rel_score,
            "λ=0 should yield more diverse results than λ=1, got {div_score:.2} vs {rel_score:.2}"
        );
    }

    #[test]
    fn lambda_1_matches_topk() {
        let v = two_cluster_vecs();
        let r_mmr = MmrRetriever::new(&v, 1.0);

        use crate::topk::TopKRetriever;
        let r_topk = TopKRetriever::new(&v);
        let query = vec![0.0f32, 0.0];

        let res_mmr = r_mmr.search(&query, 3);
        let res_topk = r_topk.search(&query, 3);

        let mmr_ids: Vec<usize> = res_mmr.iter().map(|r| r.id).collect();
        let topk_ids: Vec<usize> = res_topk.iter().map(|r| r.id).collect();
        assert_eq!(mmr_ids, topk_ids, "λ=1 MMR should match TopK");
    }
}
