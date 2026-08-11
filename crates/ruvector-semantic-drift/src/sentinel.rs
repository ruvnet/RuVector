//! Sentinel-query drift detector.
//!
//! Samples `s` sentinel vectors from the bootstrap phase. Maintains a circular
//! snapshot buffer of the last `snapshot_size` vectors as the "current
//! collection". Every `refresh_every` updates it computes for each sentinel:
//!
//!   mean_knn_dist = avg sq-L2 to the top-k nearest neighbours in snapshot
//!
//! Drift score = mean over sentinels of
//!   |cur_mean_knn_dist - ref_mean_knn_dist| / (ref_mean_knn_dist + ε)
//!
//! When the distribution shifts, the structure of the neighbourhood changes:
//! old sentinel positions are no longer "near" the new data, so mean KNN
//! distance increases (or decreases for convergent drift). This ratio is
//! stable for IID data and sensitive to distributional change.
//!
//! Update cost: O(1) amortised.
//! Refresh cost: O(s · snapshot_size · d) triggered every `refresh_every` updates.
//! Memory: O(s·d + snapshot_size·d).

use crate::DriftDetector;
use rand::{seq::SliceRandom, SeedableRng};

/// Sentinel-query drift detector based on mean-KNN-distance ratios.
pub struct SentinelQueryDetector {
    dim: usize,
    top_k: usize,
    refresh_every: usize,
    snapshot_size: usize,
    threshold: f32,
    seed: u64,

    /// Fixed sentinel vectors sampled during bootstrap.
    sentinels: Vec<Vec<f32>>,
    bootstrap_target: usize,
    bootstrap_buffer: Vec<Vec<f32>>,
    bootstrapped: bool,

    /// Circular buffer of recent vectors (current snapshot).
    snapshot: Vec<Vec<f32>>,
    snapshot_head: usize,
    snapshot_filled: bool,

    /// Reference mean KNN distance per sentinel (from bootstrap snapshot).
    ref_knn_dists: Vec<f32>,
    ref_filled: bool,

    updates_since_refresh: usize,
    last_drift_score: f32,
}

impl SentinelQueryDetector {
    /// Create a new detector.
    ///
    /// * `dim`            — vector dimension
    /// * `num_sentinels`  — number of sentinel query vectors (s)
    /// * `top_k`          — k-nearest neighbours for distance averaging
    /// * `snapshot_size`  — number of recent vectors in the current snapshot
    /// * `threshold`      — mean-KNN-distance relative change threshold (0.0–1.0);
    ///                      0.20 triggers on ≥20% change in neighbourhood structure
    /// * `seed`           — RNG seed for sentinel sampling
    pub fn new(
        dim: usize,
        num_sentinels: usize,
        top_k: usize,
        snapshot_size: usize,
        threshold: f32,
        seed: u64,
    ) -> Self {
        assert!(dim > 0);
        assert!(num_sentinels > 0);
        assert!(top_k > 0);
        assert!(snapshot_size > top_k);
        assert!((0.0..=1.0).contains(&threshold));
        let bootstrap_target = snapshot_size;
        let refresh_every = (snapshot_size / 4).max(1);
        Self {
            dim,
            top_k,
            refresh_every,
            snapshot_size,
            threshold,
            seed,
            sentinels: Vec::with_capacity(num_sentinels),
            bootstrap_target,
            bootstrap_buffer: Vec::new(),
            bootstrapped: false,
            snapshot: Vec::with_capacity(snapshot_size),
            snapshot_head: 0,
            snapshot_filled: false,
            ref_knn_dists: Vec::new(),
            ref_filled: false,
            updates_since_refresh: 0,
            last_drift_score: 0.0,
        }
    }

    fn bootstrap(&mut self, num_sentinels: usize) {
        let n = self.bootstrap_buffer.len();
        // Reserve the first `s` vectors as sentinels — they never enter the
        // snapshot, avoiding zero-distance self-matches that corrupt reference
        // KNN statistics.
        let s = num_sentinels.min(n / 4).max(1);

        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let mut pool: Vec<usize> = (0..s * 2).collect();
        pool.shuffle(&mut rng);
        self.sentinels = pool
            .iter()
            .take(s)
            .map(|&i| self.bootstrap_buffer[i].clone())
            .collect();

        // Insert the remaining bootstrap vectors into the snapshot.
        let rest: Vec<Vec<f32>> = self.bootstrap_buffer.drain(s..).collect();
        self.bootstrap_buffer.clear();
        for v in rest {
            self.insert_snapshot(v);
        }

        self.bootstrapped = true;
        self.capture_reference();
    }

    fn insert_snapshot(&mut self, v: Vec<f32>) {
        if self.snapshot.len() < self.snapshot_size {
            self.snapshot.push(v);
            if self.snapshot.len() == self.snapshot_size {
                self.snapshot_filled = true;
            }
        } else {
            self.snapshot[self.snapshot_head] = v;
            self.snapshot_head = (self.snapshot_head + 1) % self.snapshot_size;
            self.snapshot_filled = true;
        }
    }

    /// Mean sq-L2 from `sentinel` to all vectors in `snapshot`.
    ///
    /// Using all-pairs mean rather than k-NN makes this statistic stable
    /// for small snapshots: the CoV scales as 1/sqrt(n) rather than
    /// depending on the local neighbourhood structure.
    fn mean_all_sq_dist(sentinel: &[f32], snapshot: &[Vec<f32>]) -> f32 {
        if snapshot.is_empty() {
            return 0.0;
        }
        let total: f32 = snapshot
            .iter()
            .map(|v| {
                sentinel
                    .iter()
                    .zip(v.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
            })
            .sum();
        total / snapshot.len() as f32
    }

    fn capture_reference(&mut self) {
        if self.snapshot.is_empty() {
            return;
        }
        self.ref_knn_dists = self
            .sentinels
            .iter()
            .map(|s| Self::mean_all_sq_dist(s, &self.snapshot))
            .collect();
        self.ref_filled = true;
    }

    fn refresh_score(&mut self) {
        if !self.ref_filled || self.sentinels.is_empty() || self.snapshot.is_empty() {
            return;
        }

        let cur_dists: Vec<f32> = self
            .sentinels
            .iter()
            .map(|s| Self::mean_all_sq_dist(s, &self.snapshot))
            .collect();

        let eps = 1.0; // avoid div-by-near-zero for pathological zero-dist sentinels
        let mean_ratio: f32 = self
            .ref_knn_dists
            .iter()
            .zip(cur_dists.iter())
            .map(|(r, c)| (r - c).abs() / (r + eps))
            .sum::<f32>()
            / self.sentinels.len() as f32;

        self.last_drift_score = mean_ratio.min(1.0);
        self.updates_since_refresh = 0;
    }
}

impl DriftDetector for SentinelQueryDetector {
    fn update(&mut self, vector: &[f32]) {
        assert_eq!(vector.len(), self.dim, "dimension mismatch");

        if !self.bootstrapped {
            self.bootstrap_buffer.push(vector.to_vec());
            if self.bootstrap_buffer.len() >= self.bootstrap_target {
                let ns = self.sentinels.capacity().max(1);
                self.bootstrap(ns);
            }
            return;
        }

        self.insert_snapshot(vector.to_vec());
        self.updates_since_refresh += 1;

        if self.updates_since_refresh >= self.refresh_every {
            self.refresh_score();
        }
    }

    fn drift_score(&self) -> f32 {
        self.last_drift_score
    }

    fn is_drifted(&self) -> bool {
        self.ref_filled && self.last_drift_score > self.threshold
    }

    fn name(&self) -> &str {
        "SentinelQuery"
    }

    fn memory_bytes(&self) -> usize {
        let sentinel_bytes = self.sentinels.len() * self.dim * std::mem::size_of::<f32>();
        let snapshot_bytes = self.snapshot_size * self.dim * std::mem::size_of::<f32>();
        sentinel_bytes + snapshot_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{baseline, drifted};

    #[test]
    fn bootstraps_sentinels() {
        let mut det = SentinelQueryDetector::new(16, 3, 3, 50, 0.20, 1);
        for v in baseline(60, 16, 0) {
            det.update(&v);
        }
        assert!(det.bootstrapped);
        assert!(!det.sentinels.is_empty());
    }

    #[test]
    fn no_drift_on_stable() {
        // mean-all-dist CoV ≈ 1/sqrt(n) ≈ 10% for n=100.
        // Threshold 0.20 gives 2σ headroom → negligible FP rate.
        let mut det = SentinelQueryDetector::new(32, 5, 5, 100, 0.20, 2);
        for v in baseline(600, 32, 10) {
            det.update(&v);
        }
        assert!(!det.is_drifted(), "false positive on stable stream");
    }

    #[test]
    fn detects_large_shift() {
        // 10σ shift: mean-all-dist ratio ≈ (64 + 100)/64 - 1 = 1.56 >> 0.20.
        let mut det = SentinelQueryDetector::new(32, 5, 5, 100, 0.20, 3);
        for v in baseline(400, 32, 20) {
            det.update(&v);
        }
        for v in drifted(400, 32, 10.0, 20) {
            det.update(&v);
        }
        assert!(det.is_drifted(), "expected drift for 10σ shift");
    }

    #[test]
    fn mean_all_sq_dist_zero_sentinel() {
        let s = vec![0.0f32; 4];
        let snap: Vec<Vec<f32>> = vec![vec![0.0; 4]; 5];
        let d = SentinelQueryDetector::mean_all_sq_dist(&s, &snap);
        assert!(d < 1e-6, "zero sentinel in zero snapshot should give 0 dist");
    }
}
