//! Windowed-variance drift detector.
//!
//! Maintains a sliding window of cosine similarities between incoming
//! vectors and a stable reference centroid (built during warmup).
//! Drift is signalled when the window mean drops or its variance spikes
//! beyond configured thresholds. O(dim + W) per observation.

use std::collections::VecDeque;

use crate::{
    math::{cosine_sim, mean, normalize, variance},
    DriftDetector, DriftScore, DriftSummary,
};

pub struct WindowedVarianceDriftDetector {
    /// Fixed reference centroid from the warmup phase.
    centroid: Vec<f32>,
    /// Whether the centroid has been finalized.
    centroid_ready: bool,
    /// Accumulator for computing the centroid during warmup.
    centroid_acc: Vec<f32>,
    n_warmup: usize,
    /// Sliding window of cosine-sim observations.
    window: VecDeque<f32>,
    window_size: usize,
    /// Baseline mean computed at end of warmup.
    baseline_mean: f32,
    /// Alert when window mean drops more than this below baseline.
    mean_drop_threshold: f32,
    /// Alert when window variance exceeds this value.
    var_threshold: f32,
    n_observed: usize,
    n_alerts: usize,
    current_score: f32,
    peak_score: f32,
}

impl WindowedVarianceDriftDetector {
    /// * `dim`                — embedding dimension
    /// * `n_warmup`           — observations used to set the baseline centroid
    /// * `window_size`        — sliding window length (e.g. 64)
    /// * `mean_drop_threshold`— alert when mean drops by this much (e.g. 0.15)
    /// * `var_threshold`      — alert when variance exceeds this (e.g. 0.04)
    pub fn new(
        dim: usize,
        n_warmup: usize,
        window_size: usize,
        mean_drop_threshold: f32,
        var_threshold: f32,
    ) -> Self {
        Self {
            centroid: vec![0.0; dim],
            centroid_ready: false,
            centroid_acc: vec![0.0; dim],
            n_warmup,
            window: VecDeque::with_capacity(window_size),
            window_size,
            baseline_mean: 0.0,
            mean_drop_threshold,
            var_threshold,
            n_observed: 0,
            n_alerts: 0,
            current_score: 0.0,
            peak_score: 0.0,
        }
    }

    fn finalize_centroid(&mut self) {
        let n = self.n_warmup as f32;
        for (c, a) in self.centroid.iter_mut().zip(self.centroid_acc.iter()) {
            *c = a / n;
        }
        normalize(&mut self.centroid);
        self.centroid_ready = true;

        // The baseline mean is the mean cosine-sim of the warmup window.
        self.baseline_mean = if self.window.is_empty() {
            1.0
        } else {
            let w: Vec<f32> = self.window.iter().copied().collect();
            mean(&w)
        };
    }
}

impl DriftDetector for WindowedVarianceDriftDetector {
    fn observe(&mut self, vector: &[f32]) -> DriftScore {
        self.n_observed += 1;

        // Warmup: accumulate centroid.
        if self.n_observed <= self.n_warmup {
            for (a, v) in self.centroid_acc.iter_mut().zip(vector.iter()) {
                *a += v;
            }
            if self.n_observed == self.n_warmup {
                self.finalize_centroid();
                // Now measure cosine_sim for all warmup vectors against the
                // just-finalised centroid to seed the window with real values.
                // We can only seed with the current vector here; the rest are
                // averaged into the centroid and will appear as the baseline.
                let sim = cosine_sim(vector, &self.centroid);
                if self.window.len() >= self.window_size {
                    self.window.pop_front();
                }
                self.window.push_back(sim);
                // Override baseline_mean to reflect the real within-cluster
                // similarity (not just the last vector).
                // Re-estimate: centroid has absorbed all warmup vectors, so
                // cosine_sim of any warmup vector ≈ the cluster similarity.
                self.baseline_mean = sim;
            }
            return DriftScore {
                score: 0.0,
                alert: false,
                epoch: self.n_observed as u64,
                detector: self.name(),
            };
        }

        if !self.centroid_ready {
            // Edge case: fewer observations than warmup.
            self.finalize_centroid();
        }

        let sim = cosine_sim(vector, &self.centroid);
        if self.window.len() >= self.window_size {
            self.window.pop_front();
        }
        self.window.push_back(sim);

        let w: Vec<f32> = self.window.iter().copied().collect();
        let wm = mean(&w);
        let wv = variance(&w);

        let mean_drift = (self.baseline_mean - wm).max(0.0);
        let var_spike = wv;
        // Score combines normalised mean drop and variance spike.
        let score = (mean_drift / self.mean_drop_threshold.max(1e-8)).min(1.0) * 0.6
            + (var_spike / self.var_threshold.max(1e-8)).min(1.0) * 0.4;

        self.current_score = score;
        self.peak_score = self.peak_score.max(score);

        let alert = mean_drift > self.mean_drop_threshold || var_spike > self.var_threshold;
        if alert {
            self.n_alerts += 1;
        }

        DriftScore {
            score,
            alert,
            epoch: self.n_observed as u64,
            detector: self.name(),
        }
    }

    fn is_drifted(&self) -> bool {
        self.n_observed > self.n_warmup && self.current_score > 0.5
    }

    fn reset(&mut self) {
        let dim = self.centroid.len();
        self.centroid = vec![0.0; dim];
        self.centroid_ready = false;
        self.centroid_acc = vec![0.0; dim];
        self.window.clear();
        self.baseline_mean = 0.0;
        self.n_observed = 0;
        self.n_alerts = 0;
        self.current_score = 0.0;
        self.peak_score = 0.0;
    }

    fn name(&self) -> &'static str {
        "WindowedVariance"
    }

    fn summary(&self) -> DriftSummary {
        DriftSummary {
            detector: self.name(),
            n_observed: self.n_observed,
            n_alerts: self.n_alerts,
            current_score: self.current_score,
            peak_score: self.peak_score,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vec(dim: usize, val: f32) -> Vec<f32> {
        let mut v = vec![val; dim];
        normalize(&mut v);
        v
    }

    #[test]
    fn no_alert_during_warmup() {
        let mut det = WindowedVarianceDriftDetector::new(16, 50, 32, 0.15, 0.04);
        for i in 0..50 {
            let s = det.observe(&make_vec(16, 1.0));
            assert!(!s.alert, "unexpected alert at epoch {}", i + 1);
        }
    }

    #[test]
    fn alert_fires_after_large_drift() {
        let dim = 16;
        let mut det = WindowedVarianceDriftDetector::new(dim, 40, 20, 0.10, 0.03);
        // Warmup with dim-0 unit vectors.
        let stable: Vec<f32> = (0..dim).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        for _ in 0..40 {
            det.observe(&stable);
        }
        // Drift: orthogonal direction.
        let drifted: Vec<f32> = (0..dim).map(|i| if i == 1 { 1.0 } else { 0.0 }).collect();
        let mut alerted = false;
        for _ in 0..60 {
            let s = det.observe(&drifted);
            if s.alert {
                alerted = true;
            }
        }
        assert!(alerted, "WindowedVariance failed to detect orthogonal drift");
    }

    #[test]
    fn reset_clears() {
        let mut det = WindowedVarianceDriftDetector::new(8, 10, 16, 0.15, 0.04);
        for _ in 0..30 {
            det.observe(&make_vec(8, 1.0));
        }
        det.reset();
        assert_eq!(det.n_observed, 0);
        assert!(!det.is_drifted());
    }

    #[test]
    fn summary_peak_ge_current() {
        let mut det = WindowedVarianceDriftDetector::new(8, 10, 16, 0.15, 0.04);
        for _ in 0..50 {
            det.observe(&make_vec(8, 1.0));
        }
        let s = det.summary();
        assert!(s.peak_score >= s.current_score);
    }
}
