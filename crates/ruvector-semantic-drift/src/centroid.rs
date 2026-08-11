//! Window centroid drift detector.
//!
//! Maintains a reference window and a current window. Each window accumulates
//! a running mean over `window_size` vectors. When the current window fills,
//! the centroid distance is computed, the window rolls over, and the raw
//! drift score is updated.
//!
//! Update cost: O(d).
//! Memory: O(d) — two centroid vectors of `dim` f32 values.
//! Detection lag: at most `window_size` vectors after onset.

use crate::DriftDetector;

/// Sliding-window centroid drift detector.
///
/// Alerts when the L2 distance between the reference-window centroid and the
/// current-window centroid exceeds `threshold * sqrt(dim)`. Normalising by
/// `sqrt(dim)` makes the threshold dimension-agnostic for unit-variance data.
pub struct WindowCentroidDetector {
    dim: usize,
    window_size: usize,
    threshold: f32,

    ref_centroid: Vec<f32>,
    ref_filled: bool,

    cur_sum: Vec<f32>,
    cur_count: usize,

    last_drift_score: f32,
}

impl WindowCentroidDetector {
    /// Create a new detector.
    ///
    /// * `dim`         — vector dimension
    /// * `window_size` — number of vectors per window (≥ 2)
    /// * `threshold`   — normalised L2 threshold in (0, 1]; 0.10 is a reasonable default
    pub fn new(dim: usize, window_size: usize, threshold: f32) -> Self {
        assert!(dim > 0);
        assert!(window_size >= 2);
        assert!(threshold > 0.0);
        Self {
            dim,
            window_size,
            threshold,
            ref_centroid: vec![0.0; dim],
            ref_filled: false,
            cur_sum: vec![0.0; dim],
            cur_count: 0,
            last_drift_score: 0.0,
        }
    }

    fn current_centroid(&self) -> Vec<f32> {
        let n = self.cur_count as f32;
        self.cur_sum.iter().map(|s| s / n).collect()
    }

    /// Normalised L2 distance between two centroids.
    fn normalised_l2(a: &[f32], b: &[f32]) -> f32 {
        let sq: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        let l2 = sq.sqrt();
        // Normalise by sqrt(dim) so unit-variance data produces drift ~ 1.0 for a 1σ shift.
        l2 / (a.len() as f32).sqrt()
    }
}

impl DriftDetector for WindowCentroidDetector {
    fn update(&mut self, vector: &[f32]) {
        assert_eq!(vector.len(), self.dim, "dimension mismatch");

        for (s, v) in self.cur_sum.iter_mut().zip(vector.iter()) {
            *s += v;
        }
        self.cur_count += 1;

        if self.cur_count >= self.window_size {
            let cur_centroid = self.current_centroid();

            if self.ref_filled {
                self.last_drift_score =
                    Self::normalised_l2(&self.ref_centroid, &cur_centroid);
                // Only advance reference on stable windows. Once drift is
                // detected, keep the last-stable reference so all subsequent
                // windows are compared against the pre-drift baseline.
                if self.last_drift_score <= self.threshold {
                    self.ref_centroid = cur_centroid;
                }
            } else {
                // First complete window — initialise reference.
                self.ref_centroid = cur_centroid;
                self.ref_filled = true;
            }

            self.cur_sum.iter_mut().for_each(|s| *s = 0.0);
            self.cur_count = 0;
        }
    }

    fn drift_score(&self) -> f32 {
        self.last_drift_score
    }

    fn is_drifted(&self) -> bool {
        self.ref_filled && self.last_drift_score > self.threshold
    }

    fn name(&self) -> &str {
        "WindowCentroid"
    }

    fn memory_bytes(&self) -> usize {
        // ref_centroid + cur_sum
        2 * self.dim * std::mem::size_of::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{baseline, drifted};

    #[test]
    fn rolls_without_panic() {
        let mut det = WindowCentroidDetector::new(8, 4, 0.10);
        for v in baseline(20, 8, 0) {
            det.update(&v);
        }
    }

    #[test]
    fn no_drift_before_first_full_window() {
        let mut det = WindowCentroidDetector::new(16, 100, 0.05);
        for v in baseline(99, 16, 1) {
            det.update(&v);
        }
        // Haven't completed first window yet; ref_filled must be false.
        assert!(!det.is_drifted());
    }

    #[test]
    fn detects_large_shift() {
        let mut det = WindowCentroidDetector::new(32, 200, 0.05);
        for v in baseline(400, 32, 10) {
            det.update(&v);
        }
        for v in drifted(400, 32, 8.0, 10) {
            det.update(&v);
        }
        assert!(det.is_drifted(), "expected drift for 8σ shift");
    }

    #[test]
    fn memory_bytes_correct() {
        let det = WindowCentroidDetector::new(128, 500, 0.10);
        assert_eq!(det.memory_bytes(), 2 * 128 * 4);
    }
}
