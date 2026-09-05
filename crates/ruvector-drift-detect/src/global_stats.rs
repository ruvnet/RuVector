//! GlobalStats drift detector using Welford's online algorithm.
//!
//! Tracks per-dimension mean and variance incrementally.  After a baseline
//! snapshot, reports normalized mean shift plus variance ratio as a combined
//! drift score.
//!
//! # Score formula
//!
//! ```text
//! mean_shift  = Σ_d (μ_base[d] - μ_curr[d])² / σ²_base[d]  / D
//! var_ratio   = Σ_d (max(σ²_curr/σ²_base, σ²_base/σ²_curr) - 1.0)  / D
//! score       = mean_shift + var_ratio
//! ```
//!
//! Under the null (no drift) the score converges to near 0.  A population
//! shift of 1σ in all dimensions yields a score ≈ 1.0.

use crate::DriftDetector;

/// Welford online statistics for one dimension.
#[derive(Clone, Debug, Default)]
struct WelfordState {
    n: usize,
    mean: f64,
    m2: f64,
}

impl WelfordState {
    fn update(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    fn variance(&self) -> f64 {
        if self.n < 2 {
            1.0
        } else {
            self.m2 / (self.n - 1) as f64
        }
    }
}

/// Drift detector based on global distribution statistics.
///
/// Maintains per-dimension Welford running stats over all observed vectors.
/// After `snapshot()`, incoming vectors continue updating the running stats
/// but are compared against the frozen baseline moments.
pub struct GlobalStatsDriftDetector {
    dims: usize,

    // Running stats (all time — baseline + post-snapshot)
    stats: Vec<WelfordState>,
    total_n: usize,

    // Frozen at snapshot time
    baseline_n: usize,
    baseline_mean: Vec<f64>,
    baseline_var: Vec<f64>,

    post_snapshot_n: usize,
    // Separate Welford for post-snapshot data only
    post_stats: Vec<WelfordState>,
}

impl GlobalStatsDriftDetector {
    /// Create a new detector for `dims`-dimensional vectors.
    pub fn new(dims: usize) -> Self {
        Self {
            dims,
            stats: vec![WelfordState::default(); dims],
            total_n: 0,
            baseline_n: 0,
            baseline_mean: vec![0.0; dims],
            baseline_var: vec![1.0; dims],
            post_snapshot_n: 0,
            post_stats: vec![WelfordState::default(); dims],
        }
    }

    pub fn dims(&self) -> usize {
        self.dims
    }
}

impl DriftDetector for GlobalStatsDriftDetector {
    fn observe(&mut self, vec: &[f32]) {
        debug_assert_eq!(vec.len(), self.dims);
        self.total_n += 1;
        for (i, &x) in vec.iter().enumerate() {
            self.stats[i].update(x as f64);
        }
        if self.baseline_n > 0 {
            self.post_snapshot_n += 1;
            for (i, &x) in vec.iter().enumerate() {
                self.post_stats[i].update(x as f64);
            }
        }
    }

    fn snapshot(&mut self) {
        self.baseline_n = self.total_n;
        for i in 0..self.dims {
            self.baseline_mean[i] = self.stats[i].mean;
            self.baseline_var[i] = self.stats[i].variance().max(1e-9);
        }
        self.post_snapshot_n = 0;
        self.post_stats = vec![WelfordState::default(); self.dims];
    }

    fn drift_score(&self) -> f64 {
        if self.baseline_n == 0 || self.post_snapshot_n == 0 {
            return 0.0;
        }

        let mut mean_shift = 0.0f64;
        let mut var_ratio_sum = 0.0f64;

        for i in 0..self.dims {
            let bm = self.baseline_mean[i];
            let bv = self.baseline_var[i];
            let cm = self.post_stats[i].mean;
            let cv = self.post_stats[i].variance().max(1e-9);

            // Normalized squared mean displacement
            let delta = bm - cm;
            mean_shift += (delta * delta) / bv;

            // Variance ratio (symmetric, ≥ 0)
            let ratio = if cv > bv { cv / bv } else { bv / cv };
            var_ratio_sum += ratio - 1.0;
        }

        let d = self.dims as f64;
        mean_shift / d + var_ratio_sum / d
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

    fn sample_normal(
        rng: &mut SmallRng,
        n: usize,
        dims: usize,
        mean: f32,
        std: f32,
    ) -> Vec<Vec<f32>> {
        let dist = Normal::new(mean, std).unwrap();
        (0..n)
            .map(|_| (0..dims).map(|_| dist.sample(rng)).collect())
            .collect()
    }

    #[test]
    fn no_drift_scores_near_zero() {
        let mut det = GlobalStatsDriftDetector::new(32);
        let mut rng = SmallRng::seed_from_u64(42);
        for v in sample_normal(&mut rng, 1000, 32, 0.0, 1.0) {
            det.observe(&v);
        }
        det.snapshot();
        for v in sample_normal(&mut rng, 500, 32, 0.0, 1.0) {
            det.observe(&v);
        }
        let score = det.drift_score();
        assert!(
            score < 0.5,
            "no-drift score should be < 0.5, got {score:.4}"
        );
    }

    #[test]
    fn large_mean_shift_detected() {
        let mut det = GlobalStatsDriftDetector::new(32);
        let mut rng = SmallRng::seed_from_u64(99);
        for v in sample_normal(&mut rng, 1000, 32, 0.0, 1.0) {
            det.observe(&v);
        }
        det.snapshot();
        // Shift mean by 3σ → should trigger
        for v in sample_normal(&mut rng, 500, 32, 3.0, 1.0) {
            det.observe(&v);
        }
        let score = det.drift_score();
        assert!(
            score > 0.5,
            "3σ mean shift should give score > 0.5, got {score:.4}"
        );
        assert!(det.is_drifted(0.5));
    }

    #[test]
    fn pre_snapshot_returns_zero() {
        let mut det = GlobalStatsDriftDetector::new(16);
        let mut rng = SmallRng::seed_from_u64(7);
        for v in sample_normal(&mut rng, 100, 16, 0.0, 1.0) {
            det.observe(&v);
        }
        assert_eq!(det.drift_score(), 0.0, "no snapshot → score must be 0");
    }
}
