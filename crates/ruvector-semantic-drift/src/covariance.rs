//! Covariance-trace drift detector.
//!
//! Uses Welford's online algorithm to track the variance of each embedding
//! dimension independently.  The trace of the covariance matrix (sum of
//! per-dimension variances) captures distributional spread.
//!
//! Drift is detected when the ratio of the current variance trace to the
//! baseline variance trace exceeds a multiplier threshold, OR when the
//! baseline centroid displacement (cosine distance) exceeds a secondary
//! threshold.  Either condition triggers drift.
//!
//! Memory: O(3 * dim * 4 bytes): mean, M2 (variance accumulator), baseline_mean.

use crate::{cosine_distance, DriftDetector};

/// Variance-trace + centroid drift detector using Welford's algorithm.
pub struct CovarianceTrace {
    dim: usize,
    warmup: usize,
    /// Triggers when current_trace / baseline_trace > variance_ratio_threshold
    variance_ratio_threshold: f32,
    /// Triggers on cosine distance from baseline centroid
    centroid_threshold: f32,
    /// Welford running mean
    mean: Vec<f32>,
    /// Welford M2 accumulator (sum of squared deviations)
    m2: Vec<f32>,
    /// Baseline mean captured after warmup
    baseline_mean: Vec<f32>,
    /// Baseline trace captured after warmup
    baseline_trace: f32,
    count: usize,
    current_trace: f32,
    current_centroid_dist: f32,
}

impl CovarianceTrace {
    /// Create a new covariance-trace drift detector.
    ///
    /// # Parameters
    /// - `dim`: embedding dimension
    /// - `warmup`: samples before baseline is locked
    /// - `variance_ratio_threshold`: ratio current/baseline trace triggering drift
    /// - `centroid_threshold`: cosine distance (0–2) triggering drift
    pub fn new(
        dim: usize,
        warmup: usize,
        variance_ratio_threshold: f32,
        centroid_threshold: f32,
    ) -> Self {
        assert!(dim > 0);
        assert!(warmup >= 2, "Welford needs at least 2 samples");
        Self {
            dim,
            warmup,
            variance_ratio_threshold,
            centroid_threshold,
            mean: vec![0.0; dim],
            m2: vec![0.0; dim],
            baseline_mean: vec![0.0; dim],
            baseline_trace: 0.0,
            count: 0,
            current_trace: 0.0,
            current_centroid_dist: 0.0,
        }
    }

    /// Estimated heap bytes used.
    pub fn memory_bytes(&self) -> usize {
        (self.mean.len() + self.m2.len() + self.baseline_mean.len()) * 4
    }

    /// Current sum of per-dimension sample variances (trace of sample covariance).
    pub fn trace(&self) -> f32 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2.iter().sum::<f32>() / (self.count - 1) as f32
    }
}

impl DriftDetector for CovarianceTrace {
    fn feed(&mut self, embedding: &[f32]) {
        debug_assert_eq!(embedding.len(), self.dim);
        self.count += 1;
        let n = self.count as f32;
        // Welford online update
        for i in 0..self.dim {
            let delta = embedding[i] - self.mean[i];
            self.mean[i] += delta / n;
            let delta2 = embedding[i] - self.mean[i];
            self.m2[i] += delta * delta2;
        }

        if self.count == self.warmup {
            self.baseline_mean.copy_from_slice(&self.mean);
            self.baseline_trace = self.trace();
        }

        if self.count >= self.warmup {
            self.current_trace = self.trace();
            self.current_centroid_dist = cosine_distance(&self.mean, &self.baseline_mean);
        }
    }

    fn drift_score(&self) -> f32 {
        if self.count < self.warmup || self.baseline_trace < 1e-9 {
            return 0.0;
        }
        // Combine two signals: variance ratio (clamped at 3x) and centroid distance
        let var_signal = ((self.current_trace / self.baseline_trace) - 1.0)
            .max(0.0)
            .min(2.0)
            / 2.0;
        let centroid_signal = (self.current_centroid_dist / 2.0).clamp(0.0, 1.0);
        (0.5 * var_signal + 0.5 * centroid_signal).clamp(0.0, 1.0)
    }

    fn is_drifted(&self) -> bool {
        if self.count < self.warmup {
            return false;
        }
        let variance_triggered = self.baseline_trace > 1e-9
            && self.current_trace / self.baseline_trace > self.variance_ratio_threshold;
        let centroid_triggered = self.current_centroid_dist > self.centroid_threshold;
        variance_triggered || centroid_triggered
    }

    fn reset_baseline(&mut self) {
        self.baseline_mean.copy_from_slice(&self.mean);
        self.baseline_trace = self.current_trace;
        self.current_centroid_dist = 0.0;
    }

    fn name(&self) -> &'static str {
        "CovarianceTrace"
    }

    fn sample_count(&self) -> usize {
        self.count
    }

    fn memory_bytes(&self) -> usize {
        (self.mean.len() + self.m2.len() + self.baseline_mean.len()) * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn make_vec(dim: usize, val: f32) -> Vec<f32> {
        vec![val; dim]
    }

    #[test]
    fn no_drift_before_warmup() {
        let mut d = CovarianceTrace::new(8, 20, 2.0, 0.1);
        for _ in 0..10 {
            d.feed(&make_vec(8, 1.0));
        }
        assert_eq!(d.drift_score(), 0.0);
        assert!(!d.is_drifted());
    }

    #[test]
    fn detects_centroid_shift() {
        // Use orthogonal rather than opposite vectors: opposite vectors cancel
        // the Welford mean to zero, making cosine distance undefined.
        // Orthogonal vectors produce a mean that is non-zero and clearly
        // displaced from the baseline centroid.
        let mut d = CovarianceTrace::new(4, 20, 5.0, 0.05);
        let a = vec![1.0_f32, 0.0, 0.0, 0.0]; // baseline direction
        let b = vec![0.0_f32, 1.0, 0.0, 0.0]; // orthogonal drift direction
        for _ in 0..20 {
            d.feed(&a);
        }
        assert!(!d.is_drifted(), "stable after warmup");
        // After 20 more orthogonal samples the running mean ≈ [0.5, 0.5, 0, 0],
        // cosine distance from [1,0,0,0] ≈ 0.29 >> threshold 0.05.
        for _ in 0..20 {
            d.feed(&b);
        }
        assert!(d.is_drifted(), "should detect orthogonal centroid shift");
    }

    #[test]
    fn detects_variance_explosion() {
        use rand::{rngs::StdRng, Rng};
        let mut rng = StdRng::seed_from_u64(42);
        let dim = 8;
        let mut d = CovarianceTrace::new(dim, 50, 3.0, 0.5);
        // Tight cluster
        for _ in 0..50 {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(0.99..1.01_f32)).collect();
            d.feed(&v);
        }
        assert!(!d.is_drifted());
        // Explode variance: uniform [-10, 10]
        for _ in 0..50 {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-10.0..10.0_f32)).collect();
            d.feed(&v);
        }
        assert!(d.is_drifted(), "should detect variance explosion");
    }

    #[test]
    fn memory_estimate() {
        let d = CovarianceTrace::new(128, 50, 2.0, 0.1);
        assert_eq!(d.memory_bytes(), 128 * 4 * 3);
    }
}
