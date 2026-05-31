//! Graph-neighbourhood drift detector (k-NN two-sample test).
//!
//! Builds a k-nearest-neighbour graph over the union of reference and current
//! windows, then measures the fraction of edges that are "intra-current" versus
//! the fraction expected under the null hypothesis (no drift).
//!
//! Under no drift, a current vector's k nearest neighbours are drawn uniformly
//! from all n = ref_size + cur_size vectors, so the expected intra-current
//! fraction is (cur_size − 1) / (n − 1).  A large deviation indicates that
//! current vectors cluster among themselves — they occupy a different region of
//! the embedding space.
//!
//! This is the embedding-space analogue of Friedman–Rafsky's k-NN test and
//! catches structural (topological) drift that centroid and MMD methods miss.
//!
//! Complexity: O(n·k·d) per [`GraphDriftDetector::report`], where
//!   n = ref_size + cur_size, k = neighbourhood size, d = dimensions.

use std::collections::VecDeque;

use crate::{l2_sq, DriftDetector, DriftReport, DriftScore};

/// Graph-neighbourhood drift detector.
///
/// Maintains a bounded window of reference and current vectors and computes the
/// k-NN two-sample statistic on demand.
pub struct GraphDriftDetector {
    dims: usize,
    k: usize,
    threshold: f32,

    ref_vecs: Vec<Vec<f32>>,
    cur_buf: VecDeque<Vec<f32>>,
    window_size: usize,

    last_score: f32,
}

impl GraphDriftDetector {
    /// Create a new graph drift detector.
    ///
    /// - `reference`: initial reference vectors (retained in full).
    /// - `k`: number of nearest neighbours for the k-NN test.  Typical: 5–15.
    /// - `window_size`: maximum current-window size.
    /// - `threshold`: k-NN test statistic above which drift is signalled.
    pub fn new(reference: &[Vec<f32>], k: usize, window_size: usize, threshold: f32) -> Self {
        assert!(!reference.is_empty());
        assert!(k >= 1);
        let dims = reference[0].len();
        Self {
            dims,
            k,
            threshold,
            ref_vecs: reference.to_vec(),
            cur_buf: VecDeque::with_capacity(window_size),
            window_size,
            last_score: 0.0,
        }
    }

    /// Compute the k-NN two-sample drift score.
    ///
    /// Returns a value in [0, 1] where 0 = no drift and values closer to 1
    /// indicate strong separation between reference and current distributions.
    pub fn knn_score(&self) -> f32 {
        let cur_size = self.cur_buf.len();
        if cur_size == 0 {
            return 0.0;
        }
        let ref_size = self.ref_vecs.len();
        let n = ref_size + cur_size;
        let effective_k = self.k.min(n - 1);

        // Build flat index: all vectors with labels (0=ref, 1=cur)
        let mut all: Vec<(&[f32], u8)> = Vec::with_capacity(n);
        for v in &self.ref_vecs {
            all.push((v, 0));
        }
        for v in &self.cur_buf {
            all.push((v, 1));
        }

        // Count intra-current edges from current vectors
        let mut intra_current: usize = 0;
        let mut total_edges: usize = 0;

        for idx in ref_size..n {
            let (q, _) = all[idx];
            // Find k nearest neighbours (excluding self)
            let mut dists: Vec<(f32, u8)> = all
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != idx)
                .map(|(_, (v, label))| (l2_sq(q, v), *label))
                .collect();
            // Partial sort to find k nearest
            dists.select_nth_unstable_by(effective_k - 1, |a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            for (_, label) in &dists[..effective_k] {
                if *label == 1 {
                    intra_current += 1;
                }
                total_edges += 1;
            }
        }

        if total_edges == 0 {
            return 0.0;
        }

        let observed = intra_current as f32 / total_edges as f32;
        // Expected intra-current fraction under null (excluding self)
        let expected = (cur_size.saturating_sub(1)) as f32 / (n - 1) as f32;
        // Drift statistic: normalised excess intra-current clustering
        // Clamp to [0, 1] — values above expected indicate drift
        ((observed - expected) / (1.0 - expected + 1e-8)).max(0.0)
    }
}

impl DriftDetector for GraphDriftDetector {
    fn observe(&mut self, vec: &[f32]) -> DriftScore {
        assert_eq!(vec.len(), self.dims);

        if self.cur_buf.len() == self.window_size {
            self.cur_buf.pop_front();
        }
        self.cur_buf.push_back(vec.to_vec());

        // Recompute score only when we have enough data for a meaningful test
        if self.cur_buf.len() >= self.k + 1 {
            self.last_score = self.knn_score();
        }

        DriftScore {
            score: self.last_score,
            alert: self.last_score > self.threshold,
        }
    }

    fn report(&self) -> DriftReport {
        let mag = self.knn_score();
        DriftReport {
            drift_detected: mag > self.threshold,
            magnitude: mag,
            window_size: self.cur_buf.len(),
            method: self.name(),
        }
    }

    fn reset_current(&mut self) {
        self.cur_buf.clear();
        self.last_score = 0.0;
    }

    fn promote_current(&mut self) {
        self.ref_vecs = self.cur_buf.iter().cloned().collect();
        self.reset_current();
    }

    fn dims(&self) -> usize {
        self.dims
    }

    fn name(&self) -> &'static str {
        "graph-knn"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normal_vecs(n: usize, d: usize, mean: f32, scale: f32, seed: u64) -> Vec<Vec<f32>> {
        use rand::{rngs::SmallRng, SeedableRng};
        use rand_distr::{Distribution, Normal};
        let mut rng = SmallRng::seed_from_u64(seed);
        let dist = Normal::new(mean, scale).unwrap();
        (0..n)
            .map(|_| (0..d).map(|_| dist.sample(&mut rng)).collect())
            .collect()
    }

    #[test]
    fn no_drift_knn_score_low() {
        let ref_data = normal_vecs(200, 32, 0.0, 1.0, 1);
        let mut det = GraphDriftDetector::new(&ref_data, 7, 100, 0.25);

        let cur = normal_vecs(100, 32, 0.0, 1.0, 2);
        for v in &cur {
            det.observe(v);
        }
        let r = det.report();
        assert!(
            r.magnitude < 0.3,
            "expected low graph drift for same distribution, got {:.4}",
            r.magnitude
        );
    }

    #[test]
    fn large_drift_knn_score_high() {
        let ref_data = normal_vecs(200, 32, 0.0, 1.0, 10);
        let mut det = GraphDriftDetector::new(&ref_data, 7, 100, 0.25);

        // Current vectors from a very different region
        let cur = normal_vecs(100, 32, 5.0, 1.0, 20);
        for v in &cur {
            det.observe(v);
        }
        let r = det.report();
        assert!(
            r.drift_detected,
            "expected graph drift for clearly separated distributions, score={:.4}",
            r.magnitude
        );
    }

    #[test]
    fn promote_updates_reference() {
        let ref_data = normal_vecs(100, 16, 0.0, 1.0, 5);
        let mut det = GraphDriftDetector::new(&ref_data, 5, 80, 0.25);

        // Observe drifted data then promote
        let drifted = normal_vecs(80, 16, 4.0, 1.0, 6);
        for v in &drifted {
            det.observe(v);
        }
        assert!(
            det.report().drift_detected,
            "should detect drift before promote"
        );

        det.promote_current();
        assert_eq!(
            det.ref_vecs.len(),
            80,
            "reference should be updated to current size"
        );
        assert_eq!(det.cur_buf.len(), 0, "current buffer cleared");

        // Same drifted distribution vs new drifted data == no drift
        let same_drifted = normal_vecs(60, 16, 4.0, 1.0, 7);
        for v in &same_drifted {
            det.observe(v);
        }
        let r = det.report();
        assert!(
            !r.drift_detected,
            "after promote same distribution should not drift, score={:.4}",
            r.magnitude
        );
    }
}
