//! Random-projection drift detector.
//!
//! Projects each incoming vector onto a fixed random basis of size `k` (k << dim)
//! using a Rademacher matrix (entries ±1/√dim). Maintains running means for the
//! current and reference windows. When the current window fills, the drift score
//! is the RMS normalised mean shift across all projections.
//!
//! Update cost: O(k·d) — one matrix–vector product per insert.
//! Memory: O(k·d + k) — projection matrix plus two statistic vectors.
//! Detection lag: at most `window_size` vectors after onset.
//!
//! This approach approximates a two-sample test based on random projections of
//! the MMD (Maximum Mean Discrepancy) statistic, suitable for high-dimensional
//! embeddings where explicit covariance tracking would cost O(d²).

use crate::DriftDetector;
use rand::{Rng, SeedableRng};

/// Random-projection drift detector with Rademacher basis.
///
/// Alerts when the RMS normalised centroid shift across `k` projections
/// exceeds `threshold`. For unit-variance data, a 1σ centroid shift in
/// one projection dimension produces a signal of order 1.0.
pub struct ProjectionDriftDetector {
    dim: usize,
    k: usize, // number of random projections
    window_size: usize,
    threshold: f32,

    /// k × dim Rademacher projection matrix (rows are projection vectors).
    proj_matrix: Vec<Vec<f32>>,

    /// Reference window mean of projected coordinates.
    ref_mean: Vec<f32>,
    ref_filled: bool,

    /// Current window accumulated projected sum (for mean estimation).
    cur_sum: Vec<f32>,
    cur_count: usize,

    last_drift_score: f32,
}

impl ProjectionDriftDetector {
    /// Create a new detector.
    ///
    /// * `dim`         — vector dimension
    /// * `k`           — number of random projections (16–256 typical)
    /// * `window_size` — number of vectors per window (≥ 2)
    /// * `threshold`   — RMS normalised shift threshold; 0.15 is a reasonable default
    /// * `seed`        — RNG seed for reproducible projection matrix
    pub fn new(dim: usize, k: usize, window_size: usize, threshold: f32, seed: u64) -> Self {
        assert!(dim > 0);
        assert!(k > 0 && k <= dim);
        assert!(window_size >= 2);
        assert!(threshold > 0.0);

        let scale = 1.0 / (dim as f32).sqrt();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let proj_matrix: Vec<Vec<f32>> = (0..k)
            .map(|_| {
                (0..dim)
                    .map(|_| if rng.gen_bool(0.5) { scale } else { -scale })
                    .collect()
            })
            .collect();

        Self {
            dim,
            k,
            window_size,
            threshold,
            proj_matrix,
            ref_mean: vec![0.0; k],
            ref_filled: false,
            cur_sum: vec![0.0; k],
            cur_count: 0,
            last_drift_score: 0.0,
        }
    }

    /// Project a single vector onto all k projection directions.
    fn project(&self, vector: &[f32]) -> Vec<f32> {
        self.proj_matrix
            .iter()
            .map(|row| row.iter().zip(vector.iter()).map(|(r, v)| r * v).sum())
            .collect()
    }

    /// RMS of normalised per-projection mean differences.
    fn rms_shift(ref_mean: &[f32], cur_mean: &[f32]) -> f32 {
        let k = ref_mean.len() as f32;
        let ss: f32 = ref_mean
            .iter()
            .zip(cur_mean.iter())
            .map(|(r, c)| (r - c).powi(2))
            .sum();
        (ss / k).sqrt()
    }
}

impl DriftDetector for ProjectionDriftDetector {
    fn update(&mut self, vector: &[f32]) {
        assert_eq!(vector.len(), self.dim, "dimension mismatch");

        let projected = self.project(vector);
        for (s, p) in self.cur_sum.iter_mut().zip(projected.iter()) {
            *s += p;
        }
        self.cur_count += 1;

        if self.cur_count >= self.window_size {
            let n = self.cur_count as f32;
            let cur_mean: Vec<f32> = self.cur_sum.iter().map(|s| s / n).collect();

            if self.ref_filled {
                self.last_drift_score = Self::rms_shift(&self.ref_mean, &cur_mean);
                // Only advance reference on stable windows (same policy as
                // WindowCentroidDetector): keep pre-drift baseline fixed once
                // drift fires.
                if self.last_drift_score <= self.threshold {
                    self.ref_mean = cur_mean;
                }
            } else {
                self.ref_mean = cur_mean;
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
        "ProjectionDrift"
    }

    fn memory_bytes(&self) -> usize {
        let matrix_bytes = self.k * self.dim * std::mem::size_of::<f32>();
        let stats_bytes = 2 * self.k * std::mem::size_of::<f32>();
        matrix_bytes + stats_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{baseline, drifted};

    #[test]
    fn project_has_correct_dimension() {
        let det = ProjectionDriftDetector::new(64, 16, 100, 0.15, 1);
        let v = vec![1.0f32; 64];
        let p = det.project(&v);
        assert_eq!(p.len(), 16);
    }

    #[test]
    fn no_drift_stable_stream() {
        let mut det = ProjectionDriftDetector::new(32, 8, 100, 0.15, 2);
        for v in baseline(600, 32, 5) {
            det.update(&v);
        }
        assert!(!det.is_drifted(), "false positive on stable stream");
    }

    #[test]
    fn detects_shift_3sigma() {
        let mut det = ProjectionDriftDetector::new(128, 64, 300, 0.10, 7);
        for v in baseline(600, 128, 3) {
            det.update(&v);
        }
        for v in drifted(600, 128, 3.0, 3) {
            det.update(&v);
        }
        assert!(det.is_drifted(), "expected drift for 3σ shift");
    }

    #[test]
    fn memory_bytes_matches() {
        let det = ProjectionDriftDetector::new(128, 64, 500, 0.15, 0);
        let expected = 64 * 128 * 4 + 2 * 64 * 4;
        assert_eq!(det.memory_bytes(), expected);
    }

    #[test]
    fn rms_shift_zero_for_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let s = ProjectionDriftDetector::rms_shift(&v, &v);
        assert!(s < 1e-6, "identical means should give zero shift");
    }
}
