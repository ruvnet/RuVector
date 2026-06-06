//! Baseline drift detector: EMA mean-shift distance.
//!
//! Compares the exponential moving average of recent insertions against a
//! frozen reference mean.  When their L2 distance exceeds a threshold the
//! agent memory is considered to have semantically drifted.
//!
//! **Complexity:** O(D) insert, O(D) score, O(D) memory.

use crate::{stats::OnlineStats, DriftDetector};

/// Exponential-moving-average mean-shift detector.
///
/// During the warm-up phase (`count < warm_up`), every inserted vector feeds
/// a reference [`OnlineStats`].  After warm-up, new vectors update a separate
/// EMA mean.  The drift score is the L2 distance between the reference mean
/// and the EMA mean.
#[derive(Debug, Clone)]
pub struct MeanShiftDetector {
    dim: usize,
    warm_up: usize,
    alpha: f64,
    reference: OnlineStats,
    ema: Vec<f64>,
    count: usize,
    warmed_up: bool,
}

impl MeanShiftDetector {
    /// Create a new detector.
    ///
    /// - `dim`: vector dimension
    /// - `warm_up`: number of insertions to build the reference distribution
    /// - `alpha`: EMA smoothing factor (0 < alpha ≤ 1; smaller = slower adaptation)
    pub fn new(dim: usize, warm_up: usize, alpha: f64) -> Self {
        assert!(dim > 0);
        assert!(warm_up > 0);
        assert!(alpha > 0.0 && alpha <= 1.0);
        Self {
            dim,
            warm_up,
            alpha,
            reference: OnlineStats::new(dim),
            ema: vec![0.0; dim],
            count: 0,
            warmed_up: false,
        }
    }
}

impl DriftDetector for MeanShiftDetector {
    fn insert(&mut self, vec: &[f32]) {
        debug_assert_eq!(vec.len(), self.dim);
        self.count += 1;

        if !self.warmed_up {
            self.reference.push(vec);
            if self.count >= self.warm_up {
                self.warmed_up = true;
                // seed EMA from the reference mean
                self.ema.clone_from(&self.reference.mean);
            }
        } else {
            // EMA update
            for (i, &x) in vec.iter().enumerate() {
                self.ema[i] = self.alpha * x as f64 + (1.0 - self.alpha) * self.ema[i];
            }
        }
    }

    fn drift_score(&self) -> f32 {
        if !self.warmed_up {
            return 0.0;
        }
        self.reference.mean_l2_to(&self.ema) as f32
    }

    fn reset_reference(&mut self) {
        self.reference = OnlineStats::new(self.dim);
        self.ema = vec![0.0; self.dim];
        self.count = 0;
        self.warmed_up = false;
    }

    fn count(&self) -> usize {
        self.count
    }

    fn memory_bytes(&self) -> usize {
        self.reference.memory_bytes() + self.dim * std::mem::size_of::<f64>()
    }
}
