//! Centroid-shift semantic drift detector.
//!
//! Tracks the running centroid of the embedding stream using an exponential
//! moving average (EMA).  After `warmup` samples, it snapshots the baseline
//! centroid.  Drift is measured as the cosine distance between the current
//! EMA centroid and that baseline.
//!
//! Memory: O(2 * dim * 4 bytes).

use crate::{cosine_distance, DriftDetector};

/// Centroid-shift drift detector.
///
/// Fast and low-memory: stores only the baseline centroid and current EMA.
pub struct CentroidDrift {
    dim: usize,
    warmup: usize,
    threshold: f32,
    /// Exponential moving average decay.  alpha=1 is a simple running mean.
    alpha: f32,
    ema: Vec<f32>,
    baseline: Vec<f32>,
    count: usize,
    current_score: f32,
}

impl CentroidDrift {
    /// Create a new centroid drift detector.
    ///
    /// # Parameters
    /// - `dim`: embedding dimension
    /// - `warmup`: number of samples before baseline is locked
    /// - `threshold`: cosine distance (0–2) that triggers drift detection
    /// - `alpha`: EMA decay factor in (0, 1]; 0.05 = slow adaptation
    pub fn new(dim: usize, warmup: usize, threshold: f32, alpha: f32) -> Self {
        assert!(dim > 0);
        assert!(warmup > 0);
        assert!((0.0..=1.0).contains(&alpha));
        Self {
            dim,
            warmup,
            threshold,
            alpha,
            ema: vec![0.0; dim],
            baseline: vec![0.0; dim],
            count: 0,
            current_score: 0.0,
        }
    }

    /// Estimated heap bytes used by this detector (excludes stack).
    pub fn memory_bytes(&self) -> usize {
        self.ema.len() * 4 + self.baseline.len() * 4
    }

    fn update_ema(&mut self, embedding: &[f32]) {
        let a = self.alpha;
        for (e, &x) in self.ema.iter_mut().zip(embedding) {
            *e = *e * (1.0 - a) + x * a;
        }
    }
}

impl DriftDetector for CentroidDrift {
    fn feed(&mut self, embedding: &[f32]) {
        debug_assert_eq!(embedding.len(), self.dim);
        self.count += 1;

        if self.count == 1 {
            // Bootstrap EMA with the first sample
            self.ema.copy_from_slice(embedding);
        } else {
            self.update_ema(embedding);
        }

        if self.count == self.warmup {
            self.baseline.copy_from_slice(&self.ema);
        }

        if self.count >= self.warmup {
            self.current_score = cosine_distance(&self.ema, &self.baseline);
        }
    }

    fn drift_score(&self) -> f32 {
        if self.count < self.warmup {
            return 0.0;
        }
        // Normalise to [0, 1]: max cosine distance is 2.0
        (self.current_score / 2.0).clamp(0.0, 1.0)
    }

    fn is_drifted(&self) -> bool {
        self.current_score > self.threshold
    }

    fn reset_baseline(&mut self) {
        self.baseline.copy_from_slice(&self.ema);
        self.current_score = 0.0;
    }

    fn name(&self) -> &'static str {
        "CentroidEMA"
    }

    fn sample_count(&self) -> usize {
        self.count
    }

    fn memory_bytes(&self) -> usize {
        self.ema.len() * 4 + self.baseline.len() * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vec(dim: usize, dir: usize) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        v[dir] = 1.0;
        v
    }

    #[test]
    fn no_drift_before_warmup() {
        let mut d = CentroidDrift::new(4, 10, 0.1, 0.2);
        for _ in 0..5 {
            d.feed(&unit_vec(4, 0));
        }
        assert_eq!(d.drift_score(), 0.0);
        assert!(!d.is_drifted());
    }

    #[test]
    fn detects_direction_shift() {
        let mut d = CentroidDrift::new(4, 20, 0.05, 0.5);
        // Stable window: all vectors in direction 0
        for _ in 0..20 {
            d.feed(&unit_vec(4, 0));
        }
        assert!(!d.is_drifted(), "no drift after warmup");
        // Inject drift: all vectors in direction 1 (orthogonal)
        for _ in 0..20 {
            d.feed(&unit_vec(4, 1));
        }
        assert!(d.is_drifted(), "should detect orthogonal drift");
        assert!(d.drift_score() > 0.3, "drift score={}", d.drift_score());
    }

    #[test]
    fn reset_baseline_clears_drift() {
        let mut d = CentroidDrift::new(4, 10, 0.05, 0.5);
        for _ in 0..10 {
            d.feed(&unit_vec(4, 0));
        }
        for _ in 0..10 {
            d.feed(&unit_vec(4, 1));
        }
        assert!(d.is_drifted());
        d.reset_baseline();
        assert!(!d.is_drifted());
    }

    #[test]
    fn memory_estimate_is_nonzero() {
        let d = CentroidDrift::new(128, 50, 0.1, 0.1);
        assert_eq!(d.memory_bytes(), 128 * 4 * 2);
    }
}
