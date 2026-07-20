//! Rank fusion across lens-specific result lists (the paper's *Sextant*).
//!
//! Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009) combines
//! several ranked lists into one. Its key property here: a record that ranks
//! *consistently well across many independent lenses* outranks a record that
//! ranks highly in only one — exactly the cross-lens corroboration an
//! association-native store wants, and exactly what flattening into a single
//! vector throws away.

use std::collections::BTreeMap;

use crate::constellation::sort_desc;

/// Default RRF damping constant. Smaller `k` rewards top ranks more sharply.
pub const DEFAULT_RRF_K: f32 = 20.0;

/// Reciprocal Rank Fusion over per-lens ranked lists with equal weights.
///
/// Each `ranking` is a list of `(id, _score)` ordered best-first; only the
/// rank position matters. Returns fused `(id, rrf_score)` best-first.
pub fn rrf(rankings: &[Vec<(u64, f32)>], k: f32) -> Vec<(u64, f32)> {
    let weights = vec![1.0_f32; rankings.len()];
    weighted_rrf(rankings, &weights, k)
}

/// Weighted Reciprocal Rank Fusion. `weights[i]` scales lens `i`'s
/// contribution; the anneal optimizer tunes these.
pub fn weighted_rrf(rankings: &[Vec<(u64, f32)>], weights: &[f32], k: f32) -> Vec<(u64, f32)> {
    assert_eq!(rankings.len(), weights.len(), "weights/rankings length");
    let mut acc: BTreeMap<u64, f32> = BTreeMap::new();
    for (lens_idx, ranking) in rankings.iter().enumerate() {
        let w = weights[lens_idx];
        for (rank, (id, _)) in ranking.iter().enumerate() {
            let contribution = w / (k + rank as f32 + 1.0);
            *acc.entry(*id).or_insert(0.0) += contribution;
        }
    }
    let mut fused: Vec<(u64, f32)> = acc.into_iter().collect();
    sort_desc(&mut fused);
    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consistent_record_wins() {
        // Record 7 is rank-2/3/2 across three lenses; record 1 is rank-1 in one
        // lens only. Cross-lens consistency should lift 7 above 1.
        let lens_a = vec![(1, 0.9), (7, 0.8), (2, 0.7)];
        let lens_b = vec![(3, 0.9), (4, 0.8), (7, 0.7)];
        let lens_c = vec![(5, 0.9), (7, 0.8), (1, 0.1)];
        let fused = rrf(&[lens_a, lens_b, lens_c], DEFAULT_RRF_K);
        assert_eq!(fused[0].0, 7, "consistently-ranked record should win");
    }

    #[test]
    fn weights_can_promote_a_lens() {
        let lens_a = vec![(1, 0.9), (2, 0.8)];
        let lens_b = vec![(2, 0.9), (1, 0.8)];
        // Heavily weight lens_b → its top (id 2) wins.
        let fused = weighted_rrf(&[lens_a, lens_b], &[0.1, 5.0], DEFAULT_RRF_K);
        assert_eq!(fused[0].0, 2);
    }
}
