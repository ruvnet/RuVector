//! EMA-based CSI preprocessor — extracts body-modulated residuals
//! from raw amplitudes by suppressing the static room baseline.
//!
//! Mirrors `wifi_densepose_vitals::CsiVitalPreprocessor`. The
//! algorithm: for each subcarrier maintain an exponentially-weighted
//! moving average of the amplitude. The current sample minus the
//! previous-step prediction is the residual we feed to vital-sign
//! extractors. Lower `alpha` ⇒ slower tracking, stronger static-
//! component suppression; higher `alpha` ⇒ faster adaptation, less
//! suppression.
//!
//! For ESP32 indoor sensing the upstream default is α = 0.05, which
//! keeps a multi-second memory of the room's static structure while
//! letting breathing-band (0.1–0.5 Hz) variation pass through.

use crate::csi::CsiFrame;

#[derive(Debug, Clone)]
pub struct CsiVitalPreprocessor {
    predictions: Vec<f64>,
    initialized: Vec<bool>,
    alpha: f64,
    n_subcarriers: usize,
}

impl CsiVitalPreprocessor {
    /// Allocate a preprocessor for `n_subcarriers` channels with EMA
    /// smoothing factor `alpha`. `alpha` is clamped to (0.001, 0.999).
    #[must_use]
    pub fn new(n_subcarriers: usize, alpha: f64) -> Self {
        Self {
            predictions: vec![0.0; n_subcarriers],
            initialized: vec![false; n_subcarriers],
            alpha: alpha.clamp(0.001, 0.999),
            n_subcarriers,
        }
    }

    /// 56 subcarriers × α = 0.05 — the upstream ESP32 vitals default.
    #[must_use]
    pub fn esp32_default() -> Self {
        Self::new(56, 0.05)
    }

    #[must_use]
    pub const fn n_subcarriers(&self) -> usize {
        self.n_subcarriers
    }

    #[must_use]
    pub const fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Process one CSI frame and return per-subcarrier amplitude
    /// residuals.
    ///
    /// Returns `None` if the frame has zero subcarriers. The first
    /// frame for each subcarrier seeds the EMA prediction and the
    /// returned residual is exactly zero — clinical-grade data only
    /// after the EMA has had a few samples to settle (≈ 5 / α frames
    /// for 95 % settling).
    pub fn process(&mut self, frame: &CsiFrame) -> Option<Vec<f64>> {
        let n = frame.amplitudes.len().min(self.n_subcarriers);
        if n == 0 {
            return None;
        }

        let mut residuals = vec![0.0; n];
        for (i, residual) in residuals.iter_mut().enumerate().take(n) {
            if self.initialized[i] {
                *residual = frame.amplitudes[i] - self.predictions[i];
                self.predictions[i] =
                    self.alpha * frame.amplitudes[i] + (1.0 - self.alpha) * self.predictions[i];
            } else {
                self.predictions[i] = frame.amplitudes[i];
                self.initialized[i] = true;
                *residual = 0.0;
            }
        }
        Some(residuals)
    }

    /// Discard the EMA state; the next [`Self::process`] call will
    /// re-seed each subcarrier from its first observation.
    pub fn reset(&mut self) {
        self.predictions.fill(0.0);
        self.initialized.fill(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(amps: Vec<f64>) -> CsiFrame {
        let n = amps.len();
        let phases = vec![0.0; n];
        CsiFrame {
            amplitudes: amps,
            phases,
            n_subcarriers: n,
            sample_index: 0,
            sample_rate_hz: 30.0,
        }
    }

    #[test]
    fn first_frame_seeds_zero_residual() {
        let mut p = CsiVitalPreprocessor::new(3, 0.05);
        let r = p.process(&frame(vec![10.0, 20.0, 30.0])).unwrap();
        assert_eq!(r, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn static_input_drives_residual_to_zero() {
        let mut p = CsiVitalPreprocessor::new(2, 0.5);
        // First frame seeds; subsequent identical frames → zero
        // residuals because predictions equal the inputs.
        for _ in 0..5 {
            let r = p.process(&frame(vec![10.0, 20.0])).unwrap();
            for v in r {
                assert!(v.abs() < 1e-9);
            }
        }
    }

    #[test]
    fn step_change_produces_signed_residual() {
        let mut p = CsiVitalPreprocessor::new(1, 0.05);
        // Seed
        p.process(&frame(vec![10.0])).unwrap();
        // Step up by 1.0; residual should be +1.0 (observed - predicted).
        let r = p.process(&frame(vec![11.0])).unwrap();
        assert!((r[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn alpha_clamped_to_safe_range() {
        let p = CsiVitalPreprocessor::new(1, -10.0);
        assert!(p.alpha() >= 0.001);
        let p = CsiVitalPreprocessor::new(1, 10.0);
        assert!(p.alpha() <= 0.999);
    }

    #[test]
    fn reset_restores_seeding_behaviour() {
        let mut p = CsiVitalPreprocessor::new(1, 0.5);
        p.process(&frame(vec![10.0])).unwrap();
        let r = p.process(&frame(vec![20.0])).unwrap();
        assert!(r[0] > 0.0);
        p.reset();
        let r = p.process(&frame(vec![20.0])).unwrap();
        // First frame post-reset → zero residual again.
        assert!(r[0].abs() < 1e-9);
    }

    #[test]
    fn empty_frame_returns_none() {
        let mut p = CsiVitalPreprocessor::new(0, 0.5);
        assert!(p.process(&frame(vec![])).is_none());
    }
}
