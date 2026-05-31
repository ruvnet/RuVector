//! Centroid-shift drift detector.
//!
//! Fastest possible detector: tracks the running mean of the reference window
//! and the current sliding window, then scores by the normalised L2 distance
//! between the two centroids.
//!
//! **Sensitivity**: detects mean shift but cannot detect variance-only drift
//! (e.g., the distribution becomes bimodal while keeping the same mean).

use std::collections::VecDeque;

use crate::{centroid, DriftDetector, DriftObservation};

/// Centroid-shift drift detector.
///
/// Drift score = `||centroid_current - centroid_reference|| / sqrt(dim)`.
/// The normalisation by `sqrt(dim)` makes the threshold scale-invariant w.r.t. dimension.
pub struct CentroidDrift {
    dim: usize,
    window_size: usize,
    threshold: f64,
    observations: usize,
    reference: Option<Vec<f64>>,
    window: VecDeque<Vec<f32>>,
}

impl CentroidDrift {
    /// Create a new centroid drift detector.
    ///
    /// # Parameters
    /// * `dim`         – vector dimension
    /// * `window_size` – number of vectors in the sliding window; the first full
    ///                   window is frozen as the reference
    /// * `threshold`   – drift score above which `is_drifted()` returns `true`
    pub fn new(dim: usize, window_size: usize, threshold: f64) -> Self {
        Self {
            dim,
            window_size,
            threshold,
            observations: 0,
            reference: None,
            window: VecDeque::with_capacity(window_size + 1),
        }
    }

    fn recompute_score(&self) -> f64 {
        let ref_c = match &self.reference {
            Some(r) => r,
            None => return 0.0,
        };
        let vecs: Vec<Vec<f32>> = self.window.iter().cloned().collect();
        let cur_c = centroid(&vecs, self.dim);
        let sq: f64 = cur_c
            .iter()
            .zip(ref_c.iter())
            .map(|(c, r)| (c - r).powi(2))
            .sum();
        (sq / self.dim as f64).sqrt()
    }
}

impl DriftDetector for CentroidDrift {
    fn observe(&mut self, vector: &[f32]) -> DriftObservation {
        self.window.push_back(vector.to_vec());
        if self.window.len() > self.window_size {
            self.window.pop_front();
        }
        self.observations += 1;

        // Freeze reference after first full window
        if self.observations == self.window_size && self.reference.is_none() {
            let vecs: Vec<Vec<f32>> = self.window.iter().cloned().collect();
            self.reference = Some(centroid(&vecs, self.dim));
        }

        let score = self.recompute_score();
        DriftObservation {
            observations: self.observations,
            score,
            is_drifted: score > self.threshold,
        }
    }

    fn score(&self) -> f64 {
        self.recompute_score()
    }

    fn is_drifted(&self) -> bool {
        self.recompute_score() > self.threshold
    }

    fn name(&self) -> &str {
        "CentroidDrift"
    }

    fn observations(&self) -> usize {
        self.observations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_drift_identical_windows() {
        let mut det = CentroidDrift::new(4, 50, 0.5);
        // Feed 100 identical vectors
        for _ in 0..100 {
            det.observe(&[1.0, 2.0, 3.0, 4.0]);
        }
        assert!(!det.is_drifted(), "score={}", det.score());
    }

    #[test]
    fn detects_mean_shift() {
        let mut det = CentroidDrift::new(4, 50, 0.5);
        // Reference phase
        for _ in 0..50 {
            det.observe(&[0.0, 0.0, 0.0, 0.0]);
        }
        // Shift phase: new mean = (10, 10, 10, 10)
        for _ in 0..50 {
            det.observe(&[10.0, 10.0, 10.0, 10.0]);
        }
        assert!(
            det.is_drifted(),
            "should detect shift; score={}",
            det.score()
        );
    }
}
