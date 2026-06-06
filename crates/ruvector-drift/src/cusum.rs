//! Alternative A: CUSUM-based drift detector.
//!
//! Runs a standard CUSUM (Page 1954) control chart on the L2 norm of each
//! incoming vector.  For vectors from N(μ, I), E[||v||²] = D + ||μ||², so a
//! shift in mean always increases the expected squared norm — giving CUSUM a
//! reliable scalar channel that is sign-agnostic and dimension-agnostic.
//!
//! The reference phase fits a running mean and unbiased variance of ||v||² via
//! Welford's algorithm.  After warm-up, each new vector contributes one
//! z-scored observation to both an upper CUSUM (increase) and a lower CUSUM
//! (decrease) statistic.  Either arm triggering indicates distribution shift.
//!
//! **Complexity:** O(D) insert (norm computation), O(1) score, O(1) memory.

use crate::DriftDetector;

/// CUSUM drift detector operating on per-vector L2 squared norms.
#[derive(Debug, Clone)]
pub struct CusumDetector {
    dim: usize,
    warm_up: usize,
    /// Allowance (slack) for normal variability; typically 0.5 σ.
    slack: f64,
    /// Welford running mean of ||v||².
    ref_mean: f64,
    /// Welford M2 accumulator for ||v||².
    ref_m2: f64,
    /// Welford running count during reference phase.
    ref_n: usize,
    /// Reference std of ||v||² (frozen after warm-up).
    ref_std: f64,
    /// Upper CUSUM statistic (detects upward shifts).
    cusum_up: f64,
    /// Lower CUSUM statistic (detects downward shifts).
    cusum_down: f64,
    count: usize,
    warmed_up: bool,
}

impl CusumDetector {
    /// Create a new norm-based CUSUM detector.
    ///
    /// - `dim`: vector dimension (used for memory reporting only)
    /// - `warm_up`: insertions to build reference statistics
    /// - `slack`: CUSUM allowance in units of reference σ (try 0.5–1.0)
    pub fn new(dim: usize, warm_up: usize, slack: f64) -> Self {
        assert!(dim > 0);
        assert!(warm_up >= 2, "need at least 2 samples for variance");
        assert!(slack >= 0.0);
        Self {
            dim,
            warm_up,
            slack,
            ref_mean: 0.0,
            ref_m2: 0.0,
            ref_n: 0,
            ref_std: 1.0,
            cusum_up: 0.0,
            cusum_down: 0.0,
            count: 0,
            warmed_up: false,
        }
    }

    /// L2 squared norm of `vec`.
    fn sq_norm(vec: &[f32]) -> f64 {
        vec.iter().map(|&x| (x as f64).powi(2)).sum()
    }
}

impl DriftDetector for CusumDetector {
    fn insert(&mut self, vec: &[f32]) {
        debug_assert_eq!(vec.len(), self.dim);
        self.count += 1;

        let norm_sq = Self::sq_norm(vec);

        if !self.warmed_up {
            // Welford online update for mean and M2 of ||v||²
            self.ref_n += 1;
            let delta = norm_sq - self.ref_mean;
            self.ref_mean += delta / self.ref_n as f64;
            let delta2 = norm_sq - self.ref_mean;
            self.ref_m2 += delta * delta2;

            if self.count >= self.warm_up {
                self.warmed_up = true;
                // Sample std; floor at 1.0 to avoid division by near-zero.
                self.ref_std = if self.ref_n >= 2 {
                    (self.ref_m2 / (self.ref_n - 1) as f64).sqrt().max(1.0)
                } else {
                    1.0
                };
            }
        } else {
            // Z-score the squared norm relative to reference statistics.
            let z = (norm_sq - self.ref_mean) / self.ref_std;
            self.cusum_up = (self.cusum_up + z - self.slack).max(0.0);
            self.cusum_down = (self.cusum_down - z - self.slack).max(0.0);
        }
    }

    fn drift_score(&self) -> f32 {
        if !self.warmed_up {
            return 0.0;
        }
        self.cusum_up.max(self.cusum_down) as f32
    }

    fn reset_reference(&mut self) {
        self.ref_mean = 0.0;
        self.ref_m2 = 0.0;
        self.ref_n = 0;
        self.ref_std = 1.0;
        self.cusum_up = 0.0;
        self.cusum_down = 0.0;
        self.count = 0;
        self.warmed_up = false;
    }

    fn count(&self) -> usize {
        self.count
    }

    fn memory_bytes(&self) -> usize {
        // All state is stack-allocated scalars.
        6 * std::mem::size_of::<f64>()
    }
}
