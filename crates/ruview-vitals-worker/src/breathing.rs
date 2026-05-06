//! Respiratory-rate extractor — IIR bandpass (0.1–0.5 Hz default) +
//! zero-crossing rate over a multi-second history window.
//!
//! Mirrors the public surface of
//! `wifi_densepose_vitals::BreathingExtractor`: each frame the worker
//! calls `extract(residuals, weights)` with the per-subcarrier
//! residual + variance-based fusion weights; the extractor returns
//! `None` until enough history has accumulated, then a
//! [`VitalEstimate`] every call.

use std::collections::VecDeque;

use crate::biquad::{zero_crossings, BandpassParams, Biquad};
use crate::types::{VitalEstimate, VitalStatus};

/// Default lower bound of the breathing band (Hz). 0.1 Hz ≈ 6 BPM.
pub const DEFAULT_LOW_HZ: f64 = 0.1;
/// Default upper bound of the breathing band (Hz). 0.5 Hz ≈ 30 BPM.
pub const DEFAULT_HIGH_HZ: f64 = 0.5;

#[derive(Debug, Clone)]
pub struct BreathingExtractor {
    biquad: Biquad,
    /// Filtered-signal history; capped at `window_samples`.
    history: VecDeque<f64>,
    sample_rate_hz: f64,
    window_secs: f64,
    /// Maximum samples retained. Equal to `sample_rate_hz * window_secs`.
    window_samples: usize,
    /// Frequency-band edges in Hz.
    low_hz: f64,
    high_hz: f64,
    /// Number of subcarriers the extractor expects in `extract`. Used
    /// only for length validation; subcarrier fusion happens per-call
    /// against the supplied `weights` slice.
    n_subcarriers: usize,
}

impl BreathingExtractor {
    #[must_use]
    pub fn new(n_subcarriers: usize, sample_rate_hz: f64, window_secs: f64) -> Self {
        Self::with_band(
            n_subcarriers,
            sample_rate_hz,
            window_secs,
            DEFAULT_LOW_HZ,
            DEFAULT_HIGH_HZ,
        )
    }

    /// Like [`Self::new`] but with a custom band. Useful for unit
    /// tests with high-frequency synthetic signals.
    #[must_use]
    pub fn with_band(
        n_subcarriers: usize,
        sample_rate_hz: f64,
        window_secs: f64,
        low_hz: f64,
        high_hz: f64,
    ) -> Self {
        let center = (low_hz + high_hz) * 0.5;
        let bandwidth = (high_hz - low_hz).max(f64::EPSILON);
        let biquad = Biquad::bandpass(BandpassParams {
            center_hz: center,
            bandwidth_hz: bandwidth,
            sample_rate_hz,
        });
        let window_samples = ((sample_rate_hz * window_secs).round() as usize).max(8);
        Self {
            biquad,
            history: VecDeque::with_capacity(window_samples),
            sample_rate_hz,
            window_secs,
            window_samples,
            low_hz,
            high_hz,
            n_subcarriers,
        }
    }

    /// Push one frame into the extractor and (when ready) emit an
    /// estimate.
    ///
    /// `residuals` and `weights` are equal-length; weights need not
    /// sum to 1 (we re-normalise internally). Returns `None` while
    /// the history is filling, or when the in-band oscillation count
    /// drops to zero (rare — usually means no person is in front of
    /// the sensor).
    pub fn extract(
        &mut self,
        residuals: &[f64],
        weights: &[f64],
    ) -> Option<VitalEstimate> {
        let n = residuals.len().min(weights.len()).min(self.n_subcarriers);
        if n == 0 {
            return None;
        }

        // Weighted fusion across subcarriers — re-normalise on the fly
        // in case caller passed non-unit weights or a degenerate set.
        let weight_sum: f64 = weights.iter().take(n).sum();
        let fused = if weight_sum > f64::EPSILON {
            (0..n)
                .map(|i| residuals[i] * weights[i] / weight_sum)
                .sum::<f64>()
        } else {
            // Equal-weight fallback when the caller couldn't compute
            // useful weights (e.g. silent room).
            residuals.iter().take(n).sum::<f64>() / n as f64
        };

        // Bandpass.
        let y = self.biquad.step(fused);
        if self.history.len() == self.window_samples {
            self.history.pop_front();
        }
        self.history.push_back(y);

        // Need a settled window before we trust the rate estimate.
        // 80 % full is a reasonable threshold for a 30 s window with
        // 30 fps → 720 samples.
        let min_for_estimate = (self.window_samples * 8) / 10;
        if self.history.len() < min_for_estimate {
            return None;
        }

        let samples: Vec<f64> = self.history.iter().copied().collect();
        let crossings = zero_crossings(&samples);
        let duration_secs = samples.len() as f64 / self.sample_rate_hz;

        // 2 zero-crossings per cycle; convert cycles/sec → BPM.
        let bpm = (crossings as f64 / 2.0) / duration_secs * 60.0;

        // Reject out-of-band BPM estimates (e.g. transient noise spike).
        let band_lo_bpm = self.low_hz * 60.0;
        let band_hi_bpm = self.high_hz * 60.0;
        if !bpm.is_finite() || bpm < band_lo_bpm || bpm > band_hi_bpm {
            return Some(VitalEstimate::unavailable());
        }

        // Confidence proxy: signal RMS vs. window length, normalised
        // and clamped. Higher RMS = stronger oscillation.
        let rms = (samples.iter().map(|v| v * v).sum::<f64>() / samples.len() as f64).sqrt();
        let confidence = (rms * 4.0).min(1.0).max(0.05);

        // Status proxy: high-confidence + plausible band → Valid.
        let status = if confidence > 0.6 {
            VitalStatus::Valid
        } else if confidence > 0.3 {
            VitalStatus::Degraded
        } else {
            VitalStatus::Unreliable
        };

        Some(VitalEstimate {
            value_bpm: bpm,
            confidence,
            status,
        })
    }

    pub fn reset(&mut self) {
        self.biquad.reset();
        self.history.clear();
    }

    #[must_use]
    pub const fn sample_rate_hz(&self) -> f64 {
        self.sample_rate_hz
    }

    #[must_use]
    pub const fn window_secs(&self) -> f64 {
        self.window_secs
    }

    #[must_use]
    pub fn history_len(&self) -> usize {
        self.history.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    /// Drive a single subcarrier with a sinusoid at `freq_hz`.
    fn drive_sinusoid(
        ex: &mut BreathingExtractor,
        freq_hz: f64,
        n_subcarriers: usize,
    ) -> Option<VitalEstimate> {
        let weights = vec![1.0 / n_subcarriers as f64; n_subcarriers];
        let mut last = None;
        let total = (ex.sample_rate_hz() * ex.window_secs() * 1.5) as usize;
        for i in 0..total {
            let t = i as f64 / ex.sample_rate_hz();
            let x = (TAU * freq_hz * t).sin();
            let residuals = vec![x; n_subcarriers];
            last = ex.extract(&residuals, &weights);
        }
        last
    }

    #[test]
    fn settles_at_breathing_rate_15bpm() {
        // 0.25 Hz × 60 = 15 BPM.
        let mut ex = BreathingExtractor::new(8, 30.0, 30.0);
        let est = drive_sinusoid(&mut ex, 0.25, 8).expect("estimate");
        assert!(
            (est.value_bpm - 15.0).abs() <= 2.0,
            "expected ~15 BPM ±2, got {}",
            est.value_bpm
        );
        assert!(matches!(
            est.status,
            VitalStatus::Valid | VitalStatus::Degraded
        ));
    }

    #[test]
    fn settles_at_breathing_rate_24bpm() {
        // 0.4 Hz × 60 = 24 BPM.
        let mut ex = BreathingExtractor::new(4, 30.0, 30.0);
        let est = drive_sinusoid(&mut ex, 0.4, 4).expect("estimate");
        assert!(
            (est.value_bpm - 24.0).abs() <= 2.0,
            "expected ~24 BPM ±2, got {}",
            est.value_bpm
        );
    }

    #[test]
    fn returns_none_until_history_is_settled() {
        let mut ex = BreathingExtractor::new(1, 30.0, 30.0);
        let weights = vec![1.0];
        // First few frames should produce None.
        for i in 0..10 {
            let r = vec![(i as f64).sin()];
            assert!(
                ex.extract(&r, &weights).is_none(),
                "early extract should be None"
            );
        }
    }

    #[test]
    fn degenerate_weights_use_equal_fallback() {
        // All-zero weights should still produce some result once the
        // history is full (no NaNs, no panics).
        let mut ex = BreathingExtractor::new(2, 30.0, 6.0);
        let weights = vec![0.0, 0.0];
        let mut got_any = false;
        for i in 0..400 {
            let t = i as f64 / 30.0;
            let r = vec![(TAU * 0.25 * t).sin(), (TAU * 0.25 * t).sin()];
            if ex.extract(&r, &weights).is_some() {
                got_any = true;
            }
        }
        assert!(got_any);
    }

    #[test]
    fn reset_clears_history() {
        let mut ex = BreathingExtractor::new(1, 30.0, 6.0);
        for _ in 0..50 {
            ex.extract(&[0.5], &[1.0]);
        }
        assert!(ex.history_len() > 0);
        ex.reset();
        assert_eq!(ex.history_len(), 0);
    }
}
