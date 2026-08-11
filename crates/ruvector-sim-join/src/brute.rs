//! Brute-force O(|A|·|B|·d) vector similarity join.
//!
//! Computes exact cosine similarity for every pair (a, b) ∈ A × B.
//! Used as ground-truth baseline for recall measurement of approximate variants.

use crate::{cosine_similarity, Pair, SimJoin};

/// Exact brute-force similarity join.
///
/// Complexity: O(|A| × |B| × d) time, O(|pairs|) space.
/// Guaranteed recall: 1.000.
pub struct BruteJoin;

impl SimJoin for BruteJoin {
    fn join(&self, a: &[Vec<f32>], b: &[Vec<f32>], threshold: f32) -> Vec<Pair> {
        let mut pairs = Vec::new();
        for (ai, av) in a.iter().enumerate() {
            for (bi, bv) in b.iter().enumerate() {
                let sim = cosine_similarity(av, bv);
                if sim >= threshold {
                    pairs.push(Pair {
                        a_idx: ai,
                        b_idx: bi,
                        similarity: sim,
                    });
                }
            }
        }
        pairs.sort_by(|p, q| p.a_idx.cmp(&q.a_idx).then(p.b_idx.cmp(&q.b_idx)));
        pairs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::ClusteredDataset;

    #[test]
    fn exact_pairs_detected() {
        let ds = ClusteredDataset::generate(50, 32, 3, 0.05, 0);
        let pairs = BruteJoin.join(&ds.a, &ds.b, ds.threshold);
        // With tight clusters and low noise, at least some pairs should appear
        assert!(!pairs.is_empty());
        // Every found pair must actually exceed the threshold
        for p in &pairs {
            let sim = cosine_similarity(&ds.a[p.a_idx], &ds.b[p.b_idx]);
            assert!(
                sim >= ds.threshold - 1e-6,
                "pair ({},{}) has sim {sim:.4} below threshold {:.4}",
                p.a_idx,
                p.b_idx,
                ds.threshold
            );
        }
    }

    #[test]
    fn identical_vectors_produce_pair() {
        let v = vec![0.5_f32, 0.5, 0.5, 0.5];
        let a = vec![v.clone()];
        let b = vec![v];
        let pairs = BruteJoin.join(&a, &b, 0.99);
        assert_eq!(pairs.len(), 1);
        assert!((pairs[0].similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn orthogonal_vectors_produce_no_pair() {
        let a = vec![vec![1.0_f32, 0.0]];
        let b = vec![vec![0.0_f32, 1.0]];
        let pairs = BruteJoin.join(&a, &b, 0.5);
        assert!(pairs.is_empty());
    }
}
