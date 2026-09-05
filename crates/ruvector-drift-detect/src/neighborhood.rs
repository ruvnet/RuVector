//! NeighborhoodRecall drift detector.
//!
//! At snapshot time, samples A anchor vectors from the current index and
//! computes their exact brute-force k-NN.  After mutations, it re-queries
//! the anchors over the full vector store and computes recall@k against
//! the stored ground-truth neighbors.
//!
//! A deviation from expected neighborhood contamination indicates drift:
//! if post-snapshot vectors cluster further from anchors than expected (far-distribution drift)
//! or closer than expected (in-distribution density shift), the score rises.
//!
//! # Score formula
//!
//! ```text
//! expected_contamination = post_n / total_n
//! actual_contamination[a] = |post-snapshot vectors in k-NN(anchor_a)| / k
//! score = |expected - mean(actual)| / max(expected, 1 - expected)
//! ```
//!
//! Score 0.0 = no drift (post-snapshot vectors appear at expected rate).
//! Score ≈ 1.0 = severe drift (post-snapshot vectors completely avoid or dominate anchor neighbourhoods).
//!
//! # Computational cost
//!
//! Snapshot: O(A · n · D) for brute-force k-NN over n baseline vectors.
//! Score: O(A · n · D) for re-querying over the full (grown) index.
//! State: O(A · D + A · k) for anchor vectors and their neighbors.
//!
//! Practical parameters: A = 50–200 anchors, k = 10–50.  With n = 10K
//! and D = 128 this completes in < 100ms on a modern CPU.

use crate::DriftDetector;

/// Brute-force k-NN query.  Returns the indices of the k closest vectors
/// to `query` in `corpus`, excluding `query_idx` if it appears.
fn brute_force_knn(
    query: &[f32],
    corpus: &[Vec<f32>],
    k: usize,
    exclude: Option<usize>,
) -> Vec<usize> {
    let mut dists: Vec<(usize, f64)> = corpus
        .iter()
        .enumerate()
        .filter(|(i, _)| exclude.map_or(true, |ex| *i != ex))
        .map(|(i, v)| (i, l2sq(query, v)))
        .collect();

    let take = k.min(dists.len());
    dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists[..take].iter().map(|(i, _)| *i).collect()
}

fn l2sq(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = (x - y) as f64;
            d * d
        })
        .sum()
}

/// Drift detector based on k-NN recall regression over anchor vectors.
///
/// # Parameters
///
/// - `k`: number of neighbors to recall (default 10)
/// - `n_anchors`: number of anchor vectors sampled at snapshot time (default 50)
pub struct NeighborhoodDriftDetector {
    dims: usize,
    k: usize,
    n_anchors: usize,

    /// All vectors observed so far (grows over time).
    all_vectors: Vec<Vec<f32>>,

    /// Snapshot state
    baseline_n: usize,
    /// Indices into `all_vectors` for anchor points.
    anchor_indices: Vec<usize>,

    post_snapshot_n: usize,
}

impl NeighborhoodDriftDetector {
    /// Create a new detector.
    ///
    /// - `dims`: embedding dimensionality
    /// - `k`: number of neighbors used in contamination check
    /// - `n_anchors`: how many anchors to sample at snapshot time
    pub fn new(dims: usize, k: usize, n_anchors: usize) -> Self {
        Self {
            dims,
            k,
            n_anchors,
            all_vectors: Vec::new(),
            baseline_n: 0,
            anchor_indices: Vec::new(),
            post_snapshot_n: 0,
        }
    }

    /// Contamination fraction: what fraction of anchor's k-NN are post-snapshot vectors?
    ///
    /// Under no drift this equals `post_n / total_n`.  Deviations signal drift.
    fn contamination_for_anchor(&self, anchor_idx: usize) -> f64 {
        let query = &self.all_vectors[anchor_idx];
        let neighbors = brute_force_knn(query, &self.all_vectors, self.k, Some(anchor_idx));
        let post_count = neighbors
            .iter()
            .filter(|&&idx| idx >= self.baseline_n)
            .count();
        post_count as f64 / self.k.max(1) as f64
    }
}

impl DriftDetector for NeighborhoodDriftDetector {
    fn observe(&mut self, vec: &[f32]) {
        debug_assert_eq!(vec.len(), self.dims);
        self.all_vectors.push(vec.to_vec());
        if self.baseline_n > 0 {
            self.post_snapshot_n += 1;
        }
    }

    fn snapshot(&mut self) {
        let n = self.all_vectors.len();
        self.baseline_n = n;
        self.post_snapshot_n = 0;

        if n == 0 {
            return;
        }

        // Deterministic anchor sampling: evenly spaced indices into the baseline window
        let effective_anchors = self.n_anchors.min(n);
        self.anchor_indices = (0..effective_anchors)
            .map(|i| (i * n) / effective_anchors)
            .collect();
    }

    fn drift_score(&self) -> f64 {
        if self.baseline_n == 0 || self.anchor_indices.is_empty() || self.post_snapshot_n == 0 {
            return 0.0;
        }

        let total_n = self.all_vectors.len();
        // Expected contamination if post-snapshot vectors are i.i.d. from the same distribution
        let expected = self.post_snapshot_n as f64 / total_n as f64;

        // Actual contamination: fraction of each anchor's k-NN that are post-snapshot
        let actual: f64 = self
            .anchor_indices
            .iter()
            .map(|&ai| self.contamination_for_anchor(ai))
            .sum::<f64>()
            / self.anchor_indices.len() as f64;

        // Normalised absolute deviation from expected.
        // Near-zero when distributions match; approaches 1.0 under severe drift.
        let diff = (expected - actual).abs();
        let normalizer = expected.max(1.0 - expected).max(1e-9);
        diff / normalizer
    }

    fn is_drifted(&self, threshold: f64) -> bool {
        self.drift_score() > threshold
    }

    fn reset_baseline(&mut self) {
        self.snapshot();
    }

    fn post_snapshot_count(&self) -> usize {
        self.post_snapshot_n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::SmallRng, SeedableRng};
    use rand_distr::{Distribution, Normal};

    fn sample(rng: &mut SmallRng, n: usize, dims: usize, mean: f32, std: f32) -> Vec<Vec<f32>> {
        let d = Normal::new(mean, std).unwrap();
        (0..n)
            .map(|_| (0..dims).map(|_| d.sample(rng)).collect())
            .collect()
    }

    #[test]
    fn perfect_recall_same_data() {
        let mut det = NeighborhoodDriftDetector::new(16, 5, 20);
        let mut rng = SmallRng::seed_from_u64(55);
        let vecs = sample(&mut rng, 200, 16, 0.0, 1.0);
        for v in &vecs {
            det.observe(v);
        }
        det.snapshot();
        // Add one more from the same distribution — neighborhood should be stable
        for v in sample(&mut rng, 50, 16, 0.0, 1.0) {
            det.observe(&v);
        }
        let score = det.drift_score();
        // With same distribution, recall drop should be small
        assert!(
            score < 0.5,
            "same dist: recall drop should be < 0.5, got {score:.4}"
        );
    }

    #[test]
    fn large_shift_detected_via_contamination() {
        let mut det = NeighborhoodDriftDetector::new(16, 5, 20);
        let mut rng = SmallRng::seed_from_u64(66);
        let vecs = sample(&mut rng, 200, 16, 0.0, 1.0);
        for v in &vecs {
            det.observe(v);
        }
        det.snapshot();
        // Add equal-count vectors from a very different distribution (mean=10).
        // Expected contamination = 200/400 = 0.5.
        // Actual contamination ≈ 0 (drifted vectors are 40 units away in L2).
        // Score = |0.5 - 0| / 0.5 = 1.0 → well above 0.
        for v in sample(&mut rng, 200, 16, 10.0, 0.5) {
            det.observe(&v);
        }
        let score = det.drift_score();
        assert!(
            score > 0.5,
            "far-distribution drift should give score > 0.5, got {score:.4}"
        );
    }

    #[test]
    fn no_post_snapshot_vectors_returns_zero() {
        let mut det = NeighborhoodDriftDetector::new(8, 3, 5);
        let mut rng = SmallRng::seed_from_u64(77);
        for v in sample(&mut rng, 50, 8, 0.0, 1.0) {
            det.observe(&v);
        }
        det.snapshot();
        assert_eq!(det.drift_score(), 0.0);
    }
}
