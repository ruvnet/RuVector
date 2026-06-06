//! Alternative B: MMD drift detector via Random Fourier Features.
//!
//! Approximates Maximum Mean Discrepancy (MMD) between the reference and
//! current distributions using random Fourier features (Rahimi & Recht, 2007).
//! Vectors are mapped to an R-dimensional feature space via random cosine
//! projections, then MMD² ≈ ||μ_ref - μ_curr||² in feature space.
//!
//! This gives an unbiased (in expectation) two-sample test statistic that can
//! detect arbitrary distribution shifts — not just mean shifts — at the cost of
//! O(D × R) memory and O(D + R) per-insert work.
//!
//! **Complexity:** O(D + R) insert, O(R) score, O(D × R) memory.

use crate::DriftDetector;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::f64::consts::PI;

/// MMD detector with Random Fourier Feature approximation.
#[derive(Debug, Clone)]
pub struct MmdRffDetector {
    dim: usize,
    num_features: usize,
    warm_up: usize,
    /// Random frequency matrix Ω ∈ ℝ^{R×D} (row-major), sampled ~ N(0, 2γ).
    omega: Vec<f32>,
    /// Random phase offsets b ∈ [0, 2π)^R.
    bias: Vec<f32>,
    /// Reference feature mean (frozen after warm-up).
    ref_feat_mean: Vec<f64>,
    /// Current window feature mean (EMA).
    cur_feat_mean: Vec<f64>,
    /// EMA smoothing factor.
    alpha: f64,
    count: usize,
    warmed_up: bool,
}

impl MmdRffDetector {
    /// Create a new MMD-RFF detector.
    ///
    /// - `dim`: vector dimension
    /// - `num_features`: number of random Fourier features R (typical: 128–512)
    /// - `warm_up`: insertions to build reference
    /// - `bandwidth`: RBF kernel bandwidth γ (controls sensitivity; try 1.0)
    /// - `alpha`: EMA smoothing for current mean
    /// - `seed`: RNG seed for reproducible feature matrix
    pub fn new(
        dim: usize,
        num_features: usize,
        warm_up: usize,
        bandwidth: f32,
        alpha: f64,
        seed: u64,
    ) -> Self {
        assert!(dim > 0);
        assert!(num_features > 0);
        assert!(warm_up > 0);
        assert!(bandwidth > 0.0);
        assert!(alpha > 0.0 && alpha <= 1.0);

        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (2.0 * bandwidth as f64).sqrt();

        // Sample Ω ~ N(0, 2γ·I) via Box-Muller.
        let total = num_features * dim;
        let mut omega = Vec::with_capacity(total);
        while omega.len() < total {
            let u1: f64 = rng.gen::<f64>().max(1e-14);
            let u2: f64 = rng.gen::<f64>();
            let r = scale * (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * PI * u2;
            omega.push((r * theta.cos()) as f32);
            if omega.len() < total {
                omega.push((r * theta.sin()) as f32);
            }
        }
        omega.truncate(total);

        // Sample b ~ Uniform[0, 2π).
        let bias: Vec<f32> = (0..num_features)
            .map(|_| (rng.gen::<f64>() * 2.0 * PI) as f32)
            .collect();

        Self {
            dim,
            num_features,
            warm_up,
            omega,
            bias,
            ref_feat_mean: vec![0.0; num_features],
            cur_feat_mean: vec![0.0; num_features],
            alpha,
            count: 0,
            warmed_up: false,
        }
    }

    /// Map vector → RFF feature vector z(v) = √(2/R) cos(Ω v + b).
    fn map_to_features(&self, vec: &[f32]) -> Vec<f32> {
        let scale = (2.0_f32 / self.num_features as f32).sqrt();
        (0..self.num_features)
            .map(|r| {
                let row = &self.omega[r * self.dim..(r + 1) * self.dim];
                let dot: f32 = row.iter().zip(vec.iter()).map(|(w, x)| w * x).sum();
                scale * (dot + self.bias[r]).cos()
            })
            .collect()
    }
}

impl DriftDetector for MmdRffDetector {
    fn insert(&mut self, vec: &[f32]) {
        debug_assert_eq!(vec.len(), self.dim);
        self.count += 1;

        let feat = self.map_to_features(vec);

        if !self.warmed_up {
            // Welford running mean over reference phase.
            let n = self.count as f64;
            for (i, &f) in feat.iter().enumerate() {
                self.ref_feat_mean[i] += (f as f64 - self.ref_feat_mean[i]) / n;
            }
            if self.count >= self.warm_up {
                self.warmed_up = true;
                self.cur_feat_mean.clone_from(&self.ref_feat_mean);
            }
        } else {
            // EMA update of current feature mean.
            for (i, &f) in feat.iter().enumerate() {
                self.cur_feat_mean[i] =
                    self.alpha * f as f64 + (1.0 - self.alpha) * self.cur_feat_mean[i];
            }
        }
    }

    fn drift_score(&self) -> f32 {
        if !self.warmed_up {
            return 0.0;
        }
        // MMD ≈ ||μ_ref - μ_cur||_2
        self.ref_feat_mean
            .iter()
            .zip(self.cur_feat_mean.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt() as f32
    }

    fn reset_reference(&mut self) {
        self.ref_feat_mean = vec![0.0; self.num_features];
        self.cur_feat_mean = vec![0.0; self.num_features];
        self.count = 0;
        self.warmed_up = false;
    }

    fn count(&self) -> usize {
        self.count
    }

    fn memory_bytes(&self) -> usize {
        self.omega.len() * std::mem::size_of::<f32>()
            + self.bias.len() * std::mem::size_of::<f32>()
            + 2 * self.num_features * std::mem::size_of::<f64>()
    }
}
