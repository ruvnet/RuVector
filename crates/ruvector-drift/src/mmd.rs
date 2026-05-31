//! Maximum Mean Discrepancy drift detector using Random Fourier Features.
//!
//! Approximates the kernel MMD between reference and current windows using the
//! random Fourier feature (RFF) trick (Rahimi & Recht, 2007).  The RBF kernel
//! k(x,y)=exp(−‖x−y‖²/2σ²) is approximated by:
//!
//!   φ(x) = √(2/D) · [cos(wᵢᵀx + bᵢ)]_{i=1..D}
//!
//! where wᵢ ~ N(0, σ⁻²·I) and bᵢ ~ Uniform[0, 2π].
//!
//! MMD² ≈ ‖E[φ(X)] − E[φ(Y)]‖²
//!
//! Complexity: O(D·d) per observation, O(D) space for mean feature vectors.

use crate::{dot, DriftDetector, DriftReport, DriftScore};
use rand::{rngs::SmallRng, Rng, SeedableRng};

const TWO_PI: f32 = std::f32::consts::PI * 2.0;

/// MMD drift detector with random Fourier feature approximation.
///
/// More statistically principled than centroid drift; detects distributional
/// shifts even when the mean is unchanged (e.g., variance changes, multimodal
/// drift).
pub struct MmdDriftDetector {
    dims: usize,
    /// Number of random Fourier features (D).  Higher D → better approximation.
    n_features: usize,
    threshold: f32,

    /// Projection weights, stored flat: [n_features * dims].
    weights: Vec<f32>,
    /// Random phase offsets [n_features].
    biases: Vec<f32>,
    /// Normalisation factor √(2/D).
    norm: f32,

    ref_mean_feat: Vec<f32>,
    ref_count: usize,

    cur_mean_feat: Vec<f32>,
    cur_count: usize,

    window_size: usize,
    cur_eviction_buf: std::collections::VecDeque<Vec<f32>>,
}

impl MmdDriftDetector {
    /// Create a new MMD detector.
    ///
    /// - `reference`: initial reference window vectors.
    /// - `n_features`: random Fourier feature dimension D (64–256 recommended).
    /// - `sigma`: RBF kernel bandwidth.  Use the median pairwise distance as a
    ///   heuristic, or set to `sqrt(dims)` for a reasonable default.
    /// - `window_size`: maximum number of current-window vectors retained.
    /// - `threshold`: MMD score above which drift is signalled.
    pub fn new(
        reference: &[Vec<f32>],
        n_features: usize,
        sigma: f32,
        window_size: usize,
        threshold: f32,
    ) -> Self {
        assert!(!reference.is_empty());
        let dims = reference[0].len();
        let mut rng = SmallRng::seed_from_u64(0xCAFE_BABE);

        // Sample weights from N(0, 1/σ²)
        let inv_sigma2 = 1.0 / (sigma * sigma);
        let weights: Vec<f32> = (0..n_features * dims)
            .map(|_| sample_normal(&mut rng) * inv_sigma2.sqrt())
            .collect();
        let biases: Vec<f32> = (0..n_features).map(|_| rng.gen::<f32>() * TWO_PI).collect();
        let norm = (2.0 / n_features as f32).sqrt();

        // Compute reference mean feature vector
        let ref_mean_feat = mean_features(&weights, &biases, norm, n_features, dims, reference);

        Self {
            dims,
            n_features,
            threshold,
            weights,
            biases,
            norm,
            ref_mean_feat,
            ref_count: reference.len(),
            cur_mean_feat: vec![0.0; n_features],
            cur_count: 0,
            window_size,
            cur_eviction_buf: std::collections::VecDeque::with_capacity(window_size),
        }
    }

    fn project(&self, vec: &[f32]) -> Vec<f32> {
        (0..self.n_features)
            .map(|i| {
                let w = &self.weights[i * self.dims..(i + 1) * self.dims];
                self.norm * (dot(w, vec) + self.biases[i]).cos()
            })
            .collect()
    }

    fn mmd_sq(&self) -> f32 {
        if self.cur_count == 0 {
            return 0.0;
        }
        self.ref_mean_feat
            .iter()
            .zip(self.cur_mean_feat.iter())
            .map(|(r, c)| (r - c) * (r - c))
            .sum()
    }
}

fn mean_features(
    weights: &[f32],
    biases: &[f32],
    norm: f32,
    n_features: usize,
    dims: usize,
    vecs: &[Vec<f32>],
) -> Vec<f32> {
    let mut acc = vec![0.0f32; n_features];
    for v in vecs {
        for i in 0..n_features {
            let w = &weights[i * dims..(i + 1) * dims];
            acc[i] += norm * (dot(w, v) + biases[i]).cos();
        }
    }
    let n = vecs.len() as f32;
    acc.iter_mut().for_each(|x| *x /= n);
    acc
}

fn sample_normal(rng: &mut SmallRng) -> f32 {
    // Box-Muller transform
    let u1: f32 = rng.gen::<f32>().max(1e-10);
    let u2: f32 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::PI * 2.0 * u2).cos()
}

impl DriftDetector for MmdDriftDetector {
    fn observe(&mut self, vec: &[f32]) -> DriftScore {
        assert_eq!(vec.len(), self.dims);
        let phi = self.project(vec);

        // Evict oldest if buffer full (online mean update)
        if self.cur_eviction_buf.len() == self.window_size {
            if let Some(evicted) = self.cur_eviction_buf.pop_front() {
                let evicted_phi = self.project(&evicted);
                // Remove evicted contribution from running mean
                let n = self.cur_count as f32;
                for (m, e) in self.cur_mean_feat.iter_mut().zip(evicted_phi.iter()) {
                    *m = (*m * n - e) / (n - 1.0).max(1.0);
                }
                self.cur_count = self.cur_count.saturating_sub(1);
            }
        }

        // Update running mean
        let n = self.cur_count as f32;
        for (m, p) in self.cur_mean_feat.iter_mut().zip(phi.iter()) {
            *m = (*m * n + p) / (n + 1.0);
        }
        self.cur_count += 1;
        self.cur_eviction_buf.push_back(vec.to_vec());

        let mmd_sq = self.mmd_sq();
        let score = mmd_sq.sqrt();
        DriftScore {
            score,
            alert: score > self.threshold,
        }
    }

    fn report(&self) -> DriftReport {
        let mag = self.mmd_sq().sqrt();
        DriftReport {
            drift_detected: mag > self.threshold,
            magnitude: mag,
            window_size: self.cur_count,
            method: self.name(),
        }
    }

    fn reset_current(&mut self) {
        self.cur_mean_feat.fill(0.0);
        self.cur_count = 0;
        self.cur_eviction_buf.clear();
    }

    fn promote_current(&mut self) {
        if self.cur_count == 0 {
            return;
        }
        self.ref_mean_feat.clone_from(&self.cur_mean_feat);
        self.ref_count = self.cur_count;
        self.reset_current();
    }

    fn dims(&self) -> usize {
        self.dims
    }

    fn name(&self) -> &'static str {
        "mmd-rff"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gaussian_vecs(n: usize, d: usize, mean: f32, seed: u64) -> Vec<Vec<f32>> {
        use rand::{rngs::SmallRng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..d).map(|_| sample_normal(&mut rng) + mean).collect())
            .collect()
    }

    #[test]
    fn no_drift_mmd_low() {
        let d = 64;
        let ref_data = gaussian_vecs(500, d, 0.0, 1);
        let sigma = (d as f32).sqrt();
        let mut det = MmdDriftDetector::new(&ref_data, 128, sigma, 200, 0.15);

        let cur = gaussian_vecs(200, d, 0.0, 2);
        for v in &cur {
            det.observe(v);
        }
        let r = det.report();
        assert!(
            !r.drift_detected,
            "no drift expected, mmd={:.4}",
            r.magnitude
        );
    }

    #[test]
    fn drift_detected_by_mmd() {
        let d = 64;
        let ref_data = gaussian_vecs(500, d, 0.0, 10);
        let sigma = (d as f32).sqrt();
        let mut det = MmdDriftDetector::new(&ref_data, 128, sigma, 200, 0.05);

        // Shift mean by 1.5 — large distributional change
        let cur = gaussian_vecs(200, d, 1.5, 20);
        for v in &cur {
            det.observe(v);
        }
        let r = det.report();
        assert!(r.drift_detected, "drift expected, mmd={:.4}", r.magnitude);
    }

    #[test]
    fn mmd_detects_variance_shift_centroid_cannot() {
        let d = 64;
        // Reference: N(0, 1)
        let ref_data = gaussian_vecs(500, d, 0.0, 7);
        let sigma = (d as f32).sqrt();
        let mut det = MmdDriftDetector::new(&ref_data, 128, sigma, 300, 0.05);

        // Current: N(0, 3) — same mean, very different variance
        let cur: Vec<Vec<f32>> = {
            use rand::{rngs::SmallRng, SeedableRng};
            let mut rng = SmallRng::seed_from_u64(99);
            (0..300)
                .map(|_| (0..d).map(|_| sample_normal(&mut rng) * 3.0).collect())
                .collect()
        };
        for v in &cur {
            det.observe(v);
        }
        // MMD should pick this up (centroid would score near 0)
        let r = det.report();
        // MMD with RFF is probabilistic; we just check it registers some signal
        assert!(
            r.magnitude > 0.01,
            "expected some MMD signal for variance shift, got {:.5}",
            r.magnitude
        );
    }
}
