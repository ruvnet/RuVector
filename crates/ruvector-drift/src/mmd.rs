//! Maximum Mean Discrepancy (MMD) drift detector.
//!
//! MMD with a Gaussian RBF kernel is a **kernel two-sample test** that is
//! theoretically sensitive to *any* distributional difference, including ones
//! that leave the mean unchanged (e.g., variance changes, mode splitting).
//!
//! The unbiased U-statistic estimate is:
//! ```text
//! MMD²(P,Q) = (1/n(n-1)) Σ_{i≠j} k(xi,xj)
//!           − (2/nm)     Σ_{i,j}  k(xi,yj)
//!           + (1/m(m-1)) Σ_{i≠j} k(yi,yj)
//! ```
//! where k(x,y) = exp(−||x−y||² / (2σ²)) and σ² is chosen by the median trick.
//!
//! **Cost per check**: O(S²·D) where S is the subsample size (default 150).

use std::collections::VecDeque;

use crate::{l2_sq, DriftDetector, DriftObservation};

/// MMD drift detector.
pub struct MmdDrift {
    dim: usize,
    window_size: usize,
    subsample: usize,
    threshold: f64,
    observations: usize,
    reference: Vec<Vec<f32>>,
    window: VecDeque<Vec<f32>>,
    // Bandwidth σ² estimated from reference window via median trick.
    bandwidth: f64,
}

impl MmdDrift {
    /// Create a new MMD drift detector.
    ///
    /// # Parameters
    /// * `dim`         – vector dimension
    /// * `window_size` – sliding window length; first full window becomes reference
    /// * `subsample`   – number of vectors drawn from each window per check (≤ window_size)
    /// * `threshold`   – MMD² score above which `is_drifted()` returns `true`
    pub fn new(dim: usize, window_size: usize, subsample: usize, threshold: f64) -> Self {
        Self {
            dim,
            window_size,
            subsample: subsample.min(window_size),
            threshold,
            observations: 0,
            reference: Vec::new(),
            window: VecDeque::with_capacity(window_size + 1),
            bandwidth: 1.0,
        }
    }

    /// Estimate σ² from a sample using the median of pairwise squared distances.
    fn estimate_bandwidth(vecs: &[Vec<f32>], subsample: usize) -> f64 {
        let n = vecs.len().min(subsample);
        let mut dists = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                dists.push(l2_sq(&vecs[i], &vecs[j]));
            }
        }
        if dists.is_empty() {
            return 1.0;
        }
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = dists[dists.len() / 2];
        // Guard against degenerate cases
        if median < 1e-12 {
            1.0
        } else {
            median / 2.0 // σ² = median / 2 (standard convention)
        }
    }

    /// RBF kernel value.
    #[inline]
    fn kernel(&self, a: &[f32], b: &[f32]) -> f64 {
        let sq = l2_sq(a, b);
        (-sq / (2.0 * self.bandwidth)).exp()
    }

    /// Unbiased U-statistic estimate of MMD² between two samples.
    fn mmd_sq(&self, x: &[Vec<f32>], y: &[Vec<f32>]) -> f64 {
        let nx = x.len();
        let ny = y.len();
        if nx < 2 || ny < 2 {
            return 0.0;
        }

        // E[k(X,X')] unbiased (i ≠ j)
        let mut kxx = 0.0f64;
        for i in 0..nx {
            for j in 0..nx {
                if i != j {
                    kxx += self.kernel(&x[i], &x[j]);
                }
            }
        }
        kxx /= (nx * (nx - 1)) as f64;

        // E[k(Y,Y')] unbiased
        let mut kyy = 0.0f64;
        for i in 0..ny {
            for j in 0..ny {
                if i != j {
                    kyy += self.kernel(&y[i], &y[j]);
                }
            }
        }
        kyy /= (ny * (ny - 1)) as f64;

        // E[k(X,Y)]
        let mut kxy = 0.0f64;
        for xi in x {
            for yi in y {
                kxy += self.kernel(xi, yi);
            }
        }
        kxy /= (nx * ny) as f64;

        (kxx - 2.0 * kxy + kyy).max(0.0)
    }

    fn take_subsample<'a>(&self, src: &'a VecDeque<Vec<f32>>) -> Vec<Vec<f32>> {
        // Take the most recent `subsample` vectors from the window.
        src.iter().rev().take(self.subsample).cloned().collect()
    }

    fn compute_score(&self) -> f64 {
        if self.reference.is_empty() || self.window.len() < self.subsample {
            return 0.0;
        }
        let ref_sub: Vec<Vec<f32>> = self
            .reference
            .iter()
            .take(self.subsample)
            .cloned()
            .collect();
        let cur_sub = self.take_subsample(&self.window);
        self.mmd_sq(&ref_sub, &cur_sub)
    }
}

impl DriftDetector for MmdDrift {
    fn observe(&mut self, vector: &[f32]) -> DriftObservation {
        self.window.push_back(vector.to_vec());
        if self.window.len() > self.window_size {
            self.window.pop_front();
        }
        self.observations += 1;

        if self.observations == self.window_size && self.reference.is_empty() {
            let vecs: Vec<Vec<f32>> = self.window.iter().cloned().collect();
            self.bandwidth = Self::estimate_bandwidth(&vecs, self.subsample);
            self.reference = vecs.into_iter().take(self.subsample).collect();
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
        "MmdDrift"
    }

    fn observations(&self) -> usize {
        self.observations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_drift_same_distribution() {
        let mut det = MmdDrift::new(4, 60, 40, 0.05);
        let v = vec![1.0f32; 4];
        for _ in 0..120 {
            det.observe(&v);
        }
        // Identical vectors → MMD² should be ~0
        assert!(det.score() < 0.05, "score={}", det.score());
    }

    #[test]
    fn detects_large_shift() {
        let mut det = MmdDrift::new(4, 60, 40, 0.01);
        // Reference: all zeros
        for _ in 0..60 {
            det.observe(&[0.0; 4]);
        }
        // Shift: vectors at (100, 100, 100, 100)
        for _ in 0..60 {
            det.observe(&[100.0; 4]);
        }
        assert!(
            det.is_drifted(),
            "should detect shift; score={}",
            det.score()
        );
    }

    #[test]
    fn dim_used_only_for_reference() {
        let d = MmdDrift::new(8, 20, 10, 0.01);
        assert_eq!(d.dim, 8);
    }
}
