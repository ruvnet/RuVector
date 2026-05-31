//! Diagonal Fréchet distance drift detector.
//!
//! The Fréchet Inception Distance (FID) was introduced for generative model evaluation
//! (Heusel et al., NeurIPS 2017) and requires computing √(Σ_P · Σ_Q) with full
//! covariance matrices — O(D³) per check.  Here we use the **diagonal** approximation:
//!
//! ```text
//! FD_diag(P, Q) = ||μ_P − μ_Q||²
//!              + Σ_d ( σ²_P[d] + σ²_Q[d] − 2·sqrt(σ²_P[d] · σ²_Q[d]) )
//! ```
//!
//! This equals the full Fréchet distance when the covariance matrices are diagonal
//! and runs in O(W·D) per check.  It captures **both** mean shift and variance change,
//! unlike [`crate::CentroidDrift`] which only captures mean shift.

use std::collections::VecDeque;

use crate::{centroid, DriftDetector, DriftObservation};

/// Statistics for one window needed by the Fréchet distance formula.
struct WindowStats {
    mean: Vec<f64>,
    var: Vec<f64>,
}

impl WindowStats {
    fn from_vecs(vecs: &[Vec<f32>], dim: usize) -> Self {
        let n = vecs.len();
        if n == 0 {
            return Self {
                mean: vec![0.0; dim],
                var: vec![1.0; dim],
            };
        }
        let mean = centroid(vecs, dim);
        let mut var = vec![0.0f64; dim];
        for v in vecs {
            for d in 0..dim {
                let diff = v[d] as f64 - mean[d];
                var[d] += diff * diff;
            }
        }
        for v in var.iter_mut() {
            *v /= n as f64;
        }
        Self { mean, var }
    }

    /// Diagonal Fréchet distance to another `WindowStats`.
    fn frechet_diag(&self, other: &WindowStats) -> f64 {
        let mean_term: f64 = self
            .mean
            .iter()
            .zip(other.mean.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        let var_term: f64 = self
            .var
            .iter()
            .zip(other.var.iter())
            .map(|(sp, sq)| sp + sq - 2.0 * (sp * sq).sqrt())
            .sum();

        mean_term + var_term
    }
}

/// Diagonal Fréchet distance drift detector.
///
/// Drift score = `FD_diag(reference_window, current_window)`.
/// Unlike centroid drift, this detector also fires when the **spread** of queries
/// changes (e.g., distribution becomes more concentrated or more diffuse).
pub struct FrechetDrift {
    dim: usize,
    window_size: usize,
    threshold: f64,
    observations: usize,
    reference: Option<WindowStats>,
    window: VecDeque<Vec<f32>>,
}

impl FrechetDrift {
    /// Create a new diagonal Fréchet drift detector.
    ///
    /// # Parameters
    /// * `dim`         – vector dimension
    /// * `window_size` – sliding window length; first full window becomes reference
    /// * `threshold`   – FD_diag score above which `is_drifted()` returns `true`
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

    fn compute_score(&self) -> f64 {
        let ref_stats = match &self.reference {
            Some(r) => r,
            None => return 0.0,
        };
        if self.window.len() < self.window_size / 2 {
            return 0.0;
        }
        let vecs: Vec<Vec<f32>> = self.window.iter().cloned().collect();
        let cur_stats = WindowStats::from_vecs(&vecs, self.dim);
        ref_stats.frechet_diag(&cur_stats)
    }
}

impl DriftDetector for FrechetDrift {
    fn observe(&mut self, vector: &[f32]) -> DriftObservation {
        self.window.push_back(vector.to_vec());
        if self.window.len() > self.window_size {
            self.window.pop_front();
        }
        self.observations += 1;

        if self.observations == self.window_size && self.reference.is_none() {
            let vecs: Vec<Vec<f32>> = self.window.iter().cloned().collect();
            self.reference = Some(WindowStats::from_vecs(&vecs, self.dim));
        }

        let score = self.compute_score();
        DriftObservation {
            observations: self.observations,
            score,
            is_drifted: score > self.threshold,
        }
    }

    fn score(&self) -> f64 {
        self.compute_score()
    }

    fn is_drifted(&self) -> bool {
        self.compute_score() > self.threshold
    }

    fn name(&self) -> &str {
        "FrechetDrift"
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
        let mut det = FrechetDrift::new(4, 50, 5.0);
        for _ in 0..100 {
            det.observe(&[1.0, 2.0, 3.0, 4.0]);
        }
        assert!(det.score() < 1e-9, "score={}", det.score());
    }

    #[test]
    fn detects_mean_shift() {
        let mut det = FrechetDrift::new(4, 50, 1.0);
        for _ in 0..50 {
            det.observe(&[0.0; 4]);
        }
        for _ in 0..50 {
            det.observe(&[10.0; 4]);
        }
        assert!(
            det.is_drifted(),
            "should detect mean shift; score={}",
            det.score()
        );
    }

    #[test]
    fn detects_variance_change_even_at_same_mean() {
        // Same mean (0.5,0.5,0.5,0.5) but very different spread
        let mut det = FrechetDrift::new(4, 80, 0.01);
        // Reference: tightly clustered around 0.5
        for _ in 0..80 {
            det.observe(&[0.5; 4]);
        }
        // Current: spread uniformly 0..1
        let step = 1.0f32 / 80.0;
        for i in 0..80u32 {
            let v = i as f32 * step;
            det.observe(&[v, 1.0 - v, v, 1.0 - v]);
        }
        assert!(
            det.is_drifted(),
            "should detect variance shift; score={}",
            det.score()
        );
    }
}
