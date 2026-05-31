//! Exponential Weighted Average drift detector.
//!
//! Maintains a running centroid of observed vectors. Drift is measured as
//! 1 − cosine_similarity(new_vector, centroid), smoothed with an EWA.
//! O(dim) per observation; suitable for high-throughput write paths.

use crate::{
    math::{cosine_sim, normalize},
    DriftDetector, DriftScore, DriftSummary,
};

pub struct EwaDriftDetector {
    /// EWA learning rate in (0, 1]. Small α → slow adaptation (detects
    /// slow drift); large α → fast adaptation (sensitive to transient noise).
    alpha: f32,
    centroid: Vec<f32>,
    smoothed_score: f32,
    threshold: f32,
    n_observed: usize,
    n_alerts: usize,
    peak_score: f32,
    /// Observations before alerts fire. Used to let centroid warm up.
    warmup: usize,
    recent_raw: Vec<f32>,
}

impl EwaDriftDetector {
    /// * `dim`       — embedding dimension
    /// * `alpha`     — smoothing factor (0.05 is a reasonable default)
    /// * `threshold` — drift score above which `alert` fires (0.15 typical)
    /// * `warmup`    — number of observations before alerts can fire
    pub fn new(dim: usize, alpha: f32, threshold: f32, warmup: usize) -> Self {
        Self {
            alpha,
            centroid: vec![0.0; dim],
            smoothed_score: 0.0,
            threshold,
            n_observed: 0,
            n_alerts: 0,
            peak_score: 0.0,
            warmup,
            recent_raw: Vec::with_capacity(32),
        }
    }
}

impl DriftDetector for EwaDriftDetector {
    fn observe(&mut self, vector: &[f32]) -> DriftScore {
        // On the very first call, seed the centroid with the input.
        if self.n_observed == 0 {
            self.centroid.copy_from_slice(vector);
            normalize(&mut self.centroid);
        }

        let cos = cosine_sim(vector, &self.centroid);
        let raw_score = (1.0 - cos).max(0.0);

        // Update centroid EMA.
        for (c, v) in self.centroid.iter_mut().zip(vector.iter()) {
            *c = (1.0 - self.alpha) * *c + self.alpha * v;
        }
        normalize(&mut self.centroid);

        // Smooth the score.
        self.smoothed_score =
            (1.0 - self.alpha) * self.smoothed_score + self.alpha * raw_score;
        self.n_observed += 1;

        // Track recent raw scores for variance-based secondary signal.
        if self.recent_raw.len() >= 32 {
            self.recent_raw.remove(0);
        }
        self.recent_raw.push(raw_score);

        let alert = self.n_observed > self.warmup && self.smoothed_score > self.threshold;
        if alert {
            self.n_alerts += 1;
        }
        self.peak_score = self.peak_score.max(self.smoothed_score);

        DriftScore {
            score: self.smoothed_score,
            alert,
            epoch: self.n_observed as u64,
            detector: self.name(),
        }
    }

    fn is_drifted(&self) -> bool {
        self.n_observed > self.warmup && self.smoothed_score > self.threshold
    }

    fn reset(&mut self) {
        let dim = self.centroid.len();
        self.centroid = vec![0.0; dim];
        self.smoothed_score = 0.0;
        self.n_observed = 0;
        self.n_alerts = 0;
        self.peak_score = 0.0;
        self.recent_raw.clear();
    }

    fn name(&self) -> &'static str {
        "EWA"
    }

    fn summary(&self) -> DriftSummary {
        DriftSummary {
            detector: self.name(),
            n_observed: self.n_observed,
            n_alerts: self.n_alerts,
            current_score: self.smoothed_score,
            peak_score: self.peak_score,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::mean;

    fn unit_vec(dim: usize, idx: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[idx] = 1.0;
        v
    }

    #[test]
    fn stable_data_no_alert() {
        let mut det = EwaDriftDetector::new(4, 0.1, 0.2, 20);
        for _ in 0..100 {
            let s = det.observe(&unit_vec(4, 0));
            assert!(!s.alert, "false positive at epoch {}", s.epoch);
        }
    }

    #[test]
    fn drifted_data_fires_alert() {
        let mut det = EwaDriftDetector::new(4, 0.2, 0.2, 10);
        // Warmup with dim-0 unit vectors.
        for _ in 0..20 {
            det.observe(&unit_vec(4, 0));
        }
        // Drift to dim-1.
        let mut alerted = false;
        for _ in 0..60 {
            let s = det.observe(&unit_vec(4, 1));
            if s.alert {
                alerted = true;
            }
        }
        assert!(alerted, "drift not detected");
    }

    #[test]
    fn reset_clears_state() {
        let mut det = EwaDriftDetector::new(4, 0.2, 0.2, 5);
        for _ in 0..30 {
            det.observe(&unit_vec(4, 1));
        }
        det.reset();
        assert_eq!(det.n_observed, 0);
        assert!(!det.is_drifted());
    }

    #[test]
    fn summary_fields_consistent() {
        let mut det = EwaDriftDetector::new(8, 0.1, 0.3, 5);
        for _ in 0..20 {
            det.observe(&unit_vec(8, 0));
        }
        let s = det.summary();
        assert_eq!(s.n_observed, 20);
        assert!(s.peak_score >= s.current_score);
        assert!(s.peak_score >= 0.0 && s.peak_score <= 1.5); // score is drift distance, not strictly [0,1]
    }

    #[test]
    fn mean_raw_score_near_zero_on_stable() {
        let mut det = EwaDriftDetector::new(8, 0.1, 0.3, 5);
        let v: Vec<f32> = (0..8).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        for _ in 0..50 {
            det.observe(&v);
        }
        // After warmup on constant data the smoothed score should be near 0.
        assert!(det.smoothed_score < 0.05, "score={}", det.smoothed_score);
    }

    #[test]
    fn mean_helper_used() {
        let v = vec![1.0f32, 2.0, 3.0];
        let m = mean(&v);
        assert!((m - 2.0).abs() < 1e-6);
    }
}
