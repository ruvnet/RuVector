//! CentroidDrift detector: tracks K cluster centroids via online k-means.
//!
//! After a baseline snapshot, the detector monitors how far cluster centroids
//! have moved.  A large displacement indicates the data distribution has
//! shifted around a different region of embedding space.
//!
//! # Algorithm
//!
//! Centroid update rule (online k-means with decaying learning rate):
//! ```text
//! closest  = argmin_k ||vec - centroid[k]||²
//! lr       = 1 / (count[k] + 1)
//! centroid[k] += lr * (vec - centroid[k])
//! count[k]    += 1
//! ```
//!
//! # Score formula
//!
//! ```text
//! displacement[k] = ||centroid_baseline[k] - centroid_current[k]||
//! score = Σ_k (count[k] / total) * displacement[k] / baseline_spread
//! ```
//!
//! Where `baseline_spread` is the mean inter-centroid distance at snapshot
//! time, used as a normalizer so the score is unit-free.

use crate::DriftDetector;

/// Drift detector that tracks movement of K cluster centroids.
///
/// # Choosing K
///
/// K = 16–64 works well for 10K–100K vector collections.  Smaller K
/// is faster to update but less sensitive to localised drift.
pub struct CentroidDriftDetector {
    dims: usize,
    k: usize,

    // Current centroids (updated online)
    centroids: Vec<Vec<f32>>,
    counts: Vec<usize>,
    initialized: bool,
    init_buffer: Vec<Vec<f32>>,

    // Frozen at snapshot time
    baseline_centroids: Vec<Vec<f32>>,
    baseline_spread: f64,
    baseline_n: usize,

    post_snapshot_n: usize,
}

impl CentroidDriftDetector {
    /// Create a new detector with `k` clusters for `dims`-dimensional vectors.
    pub fn new(dims: usize, k: usize) -> Self {
        Self {
            dims,
            k,
            centroids: vec![vec![0.0f32; dims]; k],
            counts: vec![0; k],
            initialized: false,
            init_buffer: Vec::new(),
            baseline_centroids: vec![vec![0.0f32; dims]; k],
            baseline_spread: 1.0,
            baseline_n: 0,
            post_snapshot_n: 0,
        }
    }

    /// Assign `vec` to the nearest centroid. Returns (cluster_index, distance²).
    fn nearest(&self, vec: &[f32]) -> (usize, f64) {
        let mut best_idx = 0;
        let mut best_dist = f64::MAX;
        for (k, centroid) in self.centroids.iter().enumerate() {
            let d = l2sq(vec, centroid);
            if d < best_dist {
                best_dist = d;
                best_idx = k;
            }
        }
        (best_idx, best_dist)
    }

    /// Move the nearest centroid toward `vec` using 1/(count+1) learning rate.
    fn update_centroid(&mut self, vec: &[f32]) {
        let (idx, _) = self.nearest(vec);
        self.counts[idx] += 1;
        let lr = 1.0 / self.counts[idx] as f32;
        for (c, &v) in self.centroids[idx].iter_mut().zip(vec.iter()) {
            *c += lr * (v - *c);
        }
    }

    /// Initialize centroids from the first K distinct vectors.
    fn maybe_initialize(&mut self) {
        if self.initialized || self.init_buffer.len() < self.k {
            return;
        }
        // Simple deterministic init: first K buffered vectors as centroids
        for (k, vec) in self.init_buffer.drain(..self.k).enumerate() {
            self.centroids[k] = vec;
            self.counts[k] = 1;
        }
        self.initialized = true;
        // Process remaining buffered vectors
        let remaining: Vec<Vec<f32>> = self.init_buffer.drain(..).collect();
        for v in remaining {
            self.update_centroid(&v);
        }
    }

    /// Mean pairwise L2 distance between all centroid pairs (used as normalizer).
    fn compute_spread(centroids: &[Vec<f32>]) -> f64 {
        let k = centroids.len();
        if k < 2 {
            return 1.0;
        }
        let mut total = 0.0f64;
        let mut pairs = 0usize;
        for i in 0..k {
            for j in (i + 1)..k {
                total += l2sq(&centroids[i], &centroids[j]).sqrt();
                pairs += 1;
            }
        }
        if pairs == 0 {
            1.0
        } else {
            (total / pairs as f64).max(1e-9)
        }
    }
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

impl DriftDetector for CentroidDriftDetector {
    fn observe(&mut self, vec: &[f32]) {
        debug_assert_eq!(vec.len(), self.dims);

        if !self.initialized {
            self.init_buffer.push(vec.to_vec());
            self.maybe_initialize();
        } else {
            self.update_centroid(vec);
        }

        if self.baseline_n > 0 {
            self.post_snapshot_n += 1;
        }
    }

    fn snapshot(&mut self) {
        // Ensure initialization with whatever we have
        if !self.initialized && !self.init_buffer.is_empty() {
            let available = self.init_buffer.len().min(self.k);
            for k in 0..available {
                self.centroids[k] = self.init_buffer[k].clone();
                self.counts[k] = 1;
            }
            self.initialized = true;
            let remaining: Vec<Vec<f32>> = self.init_buffer.drain(..).collect();
            for v in remaining {
                self.update_centroid(&v);
            }
        }

        self.baseline_centroids = self.centroids.clone();
        self.baseline_spread = Self::compute_spread(&self.centroids);
        self.baseline_n = self.counts.iter().sum();
        self.post_snapshot_n = 0;
    }

    fn drift_score(&self) -> f64 {
        if self.baseline_n == 0 || self.post_snapshot_n == 0 {
            return 0.0;
        }

        let total_count: usize = self.counts.iter().sum();
        if total_count == 0 {
            return 0.0;
        }

        let mut weighted_displacement = 0.0f64;
        for k in 0..self.k {
            let disp = l2sq(&self.baseline_centroids[k], &self.centroids[k]).sqrt();
            let weight = self.counts[k] as f64 / total_count as f64;
            weighted_displacement += weight * disp;
        }

        weighted_displacement / self.baseline_spread
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
        let dist = Normal::new(mean, std).unwrap();
        (0..n)
            .map(|_| (0..dims).map(|_| dist.sample(rng)).collect())
            .collect()
    }

    #[test]
    fn stable_distribution_low_score() {
        let mut det = CentroidDriftDetector::new(32, 8);
        let mut rng = SmallRng::seed_from_u64(11);
        for v in sample(&mut rng, 800, 32, 0.0, 1.0) {
            det.observe(&v);
        }
        det.snapshot();
        for v in sample(&mut rng, 400, 32, 0.0, 1.0) {
            det.observe(&v);
        }
        let score = det.drift_score();
        assert!(
            score < 0.5,
            "stable dist should give low centroid score, got {score:.4}"
        );
    }

    #[test]
    fn large_shift_detected() {
        let mut det = CentroidDriftDetector::new(32, 8);
        let mut rng = SmallRng::seed_from_u64(22);
        for v in sample(&mut rng, 800, 32, 0.0, 1.0) {
            det.observe(&v);
        }
        det.snapshot();
        for v in sample(&mut rng, 400, 32, 5.0, 1.0) {
            det.observe(&v);
        }
        let score = det.drift_score();
        assert!(
            score > 0.5,
            "5σ shift should give centroid score > 0.5, got {score:.4}"
        );
    }

    #[test]
    fn k_equals_one_degenerate_case() {
        let mut det = CentroidDriftDetector::new(8, 1);
        let mut rng = SmallRng::seed_from_u64(33);
        for v in sample(&mut rng, 100, 8, 0.0, 1.0) {
            det.observe(&v);
        }
        det.snapshot();
        for v in sample(&mut rng, 100, 8, 3.0, 1.0) {
            det.observe(&v);
        }
        // K=1 should still detect the shift
        assert!(det.drift_score() > 0.0);
    }
}
