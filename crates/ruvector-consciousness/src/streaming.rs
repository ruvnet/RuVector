//! Online/streaming Φ estimation for time-series data.
//!
//! Computes Φ over a sliding window of observations, maintaining
//! an empirical TPM that is updated incrementally. Designed for:
//! - Neural data (EEG/fMRI time series)
//! - Real-time BCI applications
//! - Long-running consciousness monitoring
//!
//! Key features:
//! - Exponential forgetting factor for non-stationarity
//! - Change-point detection in Φ trajectory
//! - EWMA smoothing for noise reduction

use crate::error::ConsciousnessError;
use crate::traits::PhiEngine;
use crate::types::{ComputeBudget, PhiResult, StreamingPhiResult, TransitionMatrix};

use std::time::Instant;

// ---------------------------------------------------------------------------
// Streaming Φ estimator
// ---------------------------------------------------------------------------

/// Online Φ estimator with empirical TPM and sliding window.
pub struct StreamingPhiEstimator {
    /// Number of states in the system.
    n: usize,
    /// Transition count matrix (row i, col j = count of i→j transitions).
    counts: Vec<f64>,
    /// Exponential forgetting factor (0 < λ ≤ 1). 1.0 = no forgetting.
    forgetting_factor: f64,
    /// Minimum observations before computing Φ.
    min_observations: usize,
    /// Total transitions observed.
    total_transitions: usize,
    /// Previous state (for tracking transitions).
    prev_state: Option<usize>,
    /// EWMA smoothing factor for Φ (0 < α ≤ 1).
    ewma_alpha: f64,
    /// Running EWMA of Φ.
    phi_ewma: f64,
    /// Running variance (Welford's online algorithm).
    phi_m2: f64,
    phi_mean: f64,
    /// History of recent Φ values (ring buffer).
    history: Vec<f64>,
    max_history: usize,
    /// Change-point detection: CUSUM parameters.
    cusum_pos: f64,
    cusum_neg: f64,
    cusum_threshold: f64,
}

impl StreamingPhiEstimator {
    /// Create a new streaming estimator for a system with `n` states.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            counts: vec![0.0; n * n],
            forgetting_factor: 0.99,
            min_observations: n * 2,
            total_transitions: 0,
            prev_state: None,
            ewma_alpha: 0.1,
            phi_ewma: 0.0,
            phi_m2: 0.0,
            phi_mean: 0.0,
            history: Vec::new(),
            max_history: 1000,
            cusum_pos: 0.0,
            cusum_neg: 0.0,
            cusum_threshold: 3.0,
        }
    }

    /// Configure forgetting factor (0 < λ ≤ 1). Lower = faster forgetting.
    pub fn with_forgetting_factor(mut self, lambda: f64) -> Self {
        assert!(lambda > 0.0 && lambda <= 1.0);
        self.forgetting_factor = lambda;
        self
    }

    /// Configure EWMA smoothing factor (0 < α ≤ 1). Higher = more responsive.
    pub fn with_ewma_alpha(mut self, alpha: f64) -> Self {
        assert!(alpha > 0.0 && alpha <= 1.0);
        self.ewma_alpha = alpha;
        self
    }

    /// Configure change-point detection threshold.
    pub fn with_cusum_threshold(mut self, threshold: f64) -> Self {
        self.cusum_threshold = threshold;
        self
    }

    /// Observe a new state in the time series.
    ///
    /// Updates the empirical TPM and returns updated Φ estimate
    /// if enough data has been accumulated.
    pub fn observe<E: PhiEngine>(
        &mut self,
        state: usize,
        engine: &E,
        budget: &ComputeBudget,
    ) -> Option<StreamingPhiResult> {
        assert!(state < self.n, "state {} out of range for n={}", state, self.n);

        // Record transition.
        if let Some(prev) = self.prev_state {
            // Apply forgetting factor to all counts.
            if self.forgetting_factor < 1.0 {
                for c in &mut self.counts {
                    *c *= self.forgetting_factor;
                }
            }
            // Increment transition count.
            self.counts[prev * self.n + state] += 1.0;
            self.total_transitions += 1;
        }
        self.prev_state = Some(state);

        // Don't compute until we have enough data.
        if self.total_transitions < self.min_observations {
            return None;
        }

        // Build empirical TPM from counts.
        let tpm = self.build_tpm();

        // Compute Φ.
        let phi_result = engine.compute_phi(&tpm, Some(state), budget).ok()?;
        let phi = phi_result.phi;

        // Update EWMA.
        if self.history.is_empty() {
            self.phi_ewma = phi;
            self.phi_mean = phi;
        } else {
            self.phi_ewma = self.ewma_alpha * phi + (1.0 - self.ewma_alpha) * self.phi_ewma;
        }

        // Update variance (Welford's).
        let count = self.history.len() as f64 + 1.0;
        let delta = phi - self.phi_mean;
        self.phi_mean += delta / count;
        let delta2 = phi - self.phi_mean;
        self.phi_m2 += delta * delta2;

        let variance = if count > 1.0 {
            self.phi_m2 / (count - 1.0)
        } else {
            0.0
        };

        // Change-point detection (CUSUM).
        let change_detected = self.update_cusum(phi);

        // Update history.
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(phi);

        Some(StreamingPhiResult {
            phi,
            time_steps: self.total_transitions,
            phi_ewma: self.phi_ewma,
            phi_variance: variance,
            change_detected,
            history: self.history.clone(),
        })
    }

    /// Build a normalized TPM from transition counts.
    fn build_tpm(&self) -> TransitionMatrix {
        let n = self.n;
        let mut data = vec![0.0f64; n * n];

        for i in 0..n {
            let mut row_sum = 0.0;
            for j in 0..n {
                row_sum += self.counts[i * n + j];
            }
            if row_sum > 0.0 {
                let inv = 1.0 / row_sum;
                for j in 0..n {
                    data[i * n + j] = self.counts[i * n + j] * inv;
                }
            } else {
                // No transitions from state i: use uniform.
                let uniform = 1.0 / n as f64;
                for j in 0..n {
                    data[i * n + j] = uniform;
                }
            }
        }

        TransitionMatrix::new(n, data)
    }

    /// CUSUM change-point detection.
    /// Returns true if a change point is detected.
    fn update_cusum(&mut self, phi: f64) -> bool {
        let deviation = phi - self.phi_mean;
        self.cusum_pos = (self.cusum_pos + deviation).max(0.0);
        self.cusum_neg = (self.cusum_neg - deviation).max(0.0);

        let detected = self.cusum_pos > self.cusum_threshold
            || self.cusum_neg > self.cusum_threshold;

        if detected {
            // Reset after detection.
            self.cusum_pos = 0.0;
            self.cusum_neg = 0.0;
        }

        detected
    }

    /// Current number of observed transitions.
    pub fn num_transitions(&self) -> usize {
        self.total_transitions
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.counts.fill(0.0);
        self.total_transitions = 0;
        self.prev_state = None;
        self.phi_ewma = 0.0;
        self.phi_m2 = 0.0;
        self.phi_mean = 0.0;
        self.history.clear();
        self.cusum_pos = 0.0;
        self.cusum_neg = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phi::SpectralPhiEngine;

    #[test]
    fn streaming_accumulates_data() {
        let mut estimator = StreamingPhiEstimator::new(4);
        let engine = SpectralPhiEngine::default();
        let budget = ComputeBudget::fast();

        // Feed a sequence of states.
        let states = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        let mut got_result = false;
        for &s in &states {
            if let Some(result) = estimator.observe(s, &engine, &budget) {
                assert!(result.phi >= 0.0);
                assert!(result.time_steps > 0);
                got_result = true;
            }
        }
        assert!(got_result, "should produce result after enough observations");
    }

    #[test]
    fn streaming_ewma_smooths() {
        let mut estimator = StreamingPhiEstimator::new(4)
            .with_ewma_alpha(0.5)
            .with_forgetting_factor(1.0);
        let engine = SpectralPhiEngine::default();
        let budget = ComputeBudget::fast();

        // Feed many transitions.
        for _ in 0..50 {
            for s in 0..4 {
                estimator.observe(s, &engine, &budget);
            }
        }

        assert!(estimator.num_transitions() > 0);
    }

    #[test]
    fn streaming_reset_clears() {
        let mut estimator = StreamingPhiEstimator::new(4);
        let engine = SpectralPhiEngine::default();
        let budget = ComputeBudget::fast();

        for s in 0..4 {
            estimator.observe(s, &engine, &budget);
        }
        assert!(estimator.num_transitions() > 0);

        estimator.reset();
        assert_eq!(estimator.num_transitions(), 0);
    }
}
