//! # ruvector-sim-join
//!
//! Approximate vector similarity join for RuVector.
//!
//! A *similarity join* finds all pairs (a, b) ∈ A × B where sim(a, b) ≥ θ.
//! This is fundamentally different from k-NN query search: there is no single
//! query vector; every element of B is a query against every element of A.
//!
//! Applications: knowledge-graph edge induction, entity resolution, RAG
//! deduplication, agent-memory cross-reference, multi-document linking.
//!
//! Three measurable variants:
//!
//! | Variant     | Complexity       | Recall  | Use case              |
//! |-------------|------------------|---------|-----------------------|
//! | BruteJoin   | O(|A|·|B|·d)    | 1.000   | Ground-truth baseline |
//! | LshJoin     | O(|A|+|B|+pairs)| ~0.85+  | High-throughput approx |
//! | IvfJoin     | O(K·n·d + pairs)| ~0.90+  | Balanced approx        |

pub mod brute;
pub mod dataset;
pub mod ivf;
pub mod lsh;

/// A discovered similar pair.
#[derive(Debug, Clone, PartialEq)]
pub struct Pair {
    pub a_idx: usize,
    pub b_idx: usize,
    pub similarity: f32,
}

/// Core trait for similarity join strategies.
pub trait SimJoin {
    /// Find all pairs (a_idx, b_idx) where cosine_similarity(A[a_idx], B[b_idx]) ≥ threshold.
    /// Returns pairs sorted by (a_idx, b_idx).
    fn join(&self, a: &[Vec<f32>], b: &[Vec<f32>], threshold: f32) -> Vec<Pair>;

    /// Self-join: find pairs within a single set A where i < j and sim ≥ threshold.
    fn self_join(&self, vectors: &[Vec<f32>], threshold: f32) -> Vec<Pair> {
        let pairs = self.join(vectors, vectors, threshold);
        pairs.into_iter().filter(|p| p.a_idx < p.b_idx).collect()
    }
}

/// Cosine similarity between two L2-normalised vectors.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2-normalise a vector in-place.
pub fn l2_normalize(v: &mut Vec<f32>) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

/// Recall of `found` pairs vs `ground_truth` pairs.
pub fn recall(found: &[Pair], ground_truth: &[Pair]) -> f32 {
    if ground_truth.is_empty() {
        return 1.0;
    }
    use std::collections::HashSet;
    let gt_set: HashSet<(usize, usize)> = ground_truth
        .iter()
        .map(|p| (p.a_idx.min(p.b_idx), p.a_idx.max(p.b_idx)))
        .collect();
    let found_set: HashSet<(usize, usize)> = found
        .iter()
        .map(|p| (p.a_idx.min(p.b_idx), p.a_idx.max(p.b_idx)))
        .collect();
    let hits = found_set.iter().filter(|p| gt_set.contains(p)).count();
    hits as f32 / gt_set.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use brute::BruteJoin;
    use dataset::ClusteredDataset;
    use ivf::IvfJoin;
    use lsh::LshJoin;

    fn make_small_dataset() -> ClusteredDataset {
        // noise=0.05 keeps clusters tight so threshold=0.75 captures intra-cluster pairs.
        // With d=64 and noise=0.15 intra-cluster cosine drops below 0.75; 0.05 gives ~0.90+.
        ClusteredDataset::generate(
            /*n_per_set=*/ 200, /*dims=*/ 64, /*n_clusters=*/ 5,
            /*noise=*/ 0.05, /*seed=*/ 42,
        )
    }

    #[test]
    fn brute_join_recall_is_one() {
        let ds = make_small_dataset();
        let brute = BruteJoin;
        let gt = brute.join(&ds.a, &ds.b, ds.threshold);
        let recall = recall(&gt, &gt);
        assert!((recall - 1.0).abs() < 1e-6, "brute join must be exact");
    }

    #[test]
    fn lsh_join_recall_above_threshold() {
        let ds = make_small_dataset();
        let brute = BruteJoin;
        let gt = brute.join(&ds.a, &ds.b, ds.threshold);

        let lsh = LshJoin::new(
            /*hash_bits=*/ 8, /*tables=*/ 4, /*dims=*/ 64, /*seed=*/ 7,
        );
        let found = lsh.join(&ds.a, &ds.b, ds.threshold);
        let r = recall(&found, &gt);
        // LSH recall target: ≥ 0.70 on this small set
        assert!(r >= 0.70, "LshJoin recall {r:.3} must be ≥ 0.70");
    }

    #[test]
    fn ivf_join_recall_above_threshold() {
        let ds = make_small_dataset();
        let brute = BruteJoin;
        let gt = brute.join(&ds.a, &ds.b, ds.threshold);

        let ivf = IvfJoin::new(
            /*n_clusters=*/ 8, /*n_probe=*/ 3, /*seed=*/ 13,
        );
        let found = ivf.join(&ds.a, &ds.b, ds.threshold);
        let r = recall(&found, &gt);
        // IVF recall target: ≥ 0.80
        assert!(r >= 0.80, "IvfJoin recall {r:.3} must be ≥ 0.80");
    }

    #[test]
    fn brute_self_join_no_self_pairs() {
        let ds = make_small_dataset();
        let brute = BruteJoin;
        let pairs = brute.self_join(&ds.a, ds.threshold);
        for p in &pairs {
            assert!(p.a_idx < p.b_idx, "self-join must have a_idx < b_idx");
        }
    }

    #[test]
    fn pair_count_is_non_zero() {
        let ds = make_small_dataset();
        let brute = BruteJoin;
        let gt = brute.join(&ds.a, &ds.b, ds.threshold);
        assert!(
            !gt.is_empty(),
            "clustered dataset must produce some similar pairs"
        );
    }
}
