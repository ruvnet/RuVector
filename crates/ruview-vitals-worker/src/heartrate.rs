//! Heart-rate extractor — IIR bandpass (0.8-2.0 Hz default) +
//! biased autocorrelation peak detection over a multi-second history.
//!
//! The cardiac signal at chest-surface displacement is ~10× weaker
//! than the respiratory signal, so a simple zero-crossing count picks
//! up too much breathing-band leakage. Autocorrelation is more robust
//! because it amplifies any periodicity at the candidate lag.
//!
//! Mirrors the public surface of `wifi_densepose_vitals::HeartRateExtractor`.

use std::collections::VecDeque;

use crate::biquad::{BandpassParams, Biquad};
use crate::types::{VitalEstimate, VitalStatus};

/// Default lower bound of the heart-rate band (Hz). 0.8 Hz ≈ 48 BPM.
pub const DEFAULT_LOW_HZ: f64 = 0.8;
/// Default upper bound of the heart-rate band (Hz). 2.0 Hz ≈ 120 BPM.
pub const DEFAULT_HIGH_HZ: f64 = 2.0;

#[derive(Debug, Clone)]
pub struct HeartRateExtractor {
    biquad: Biquad,
    history: VecDeque<f64>,
    sample_rate_hz: f64,
    window_secs: f64,
    window_samples: usize,
    low_hz: f64,
    high_hz: f64,
    n_subcarriers: usize,
}

impl HeartRateExtractor {
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
        let window_samples = ((sample_rate_hz * window_secs).round() as usize).max(16);
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

    /// Push one frame; emit an estimate once the history is settled.
    pub fn extract(&mut self, residuals: &[f64], phases: &[f64]) -> Option<VitalEstimate> {
        let n = residuals.len().min(self.n_subcarriers);
        if n == 0 {
            return None;
        }

        // Heart-rate fusion is a phase-coherence proxy: we average
        // residuals weighted by per-subcarrier phase coherence
        // (proxied by `cos(phase)` magnitude). When the upstream
        // `phases` vector is short or empty we fall back to plain
        // mean — keeps the worker robust during cold start.
        let fused = if phases.len() >= n {
            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..n {
                let w = phases[i].cos().abs();
                num += residuals[i] * w;
                den += w;
            }
            if den > f64::EPSILON {
                num / den
            } else {
                residuals.iter().take(n).sum::<f64>() / n as f64
            }
        } else {
            residuals.iter().take(n).sum::<f64>() / n as f64
        };

        let y = self.biquad.step(fused);
        if self.history.len() == self.window_samples {
            self.history.pop_front();
        }
        self.history.push_back(y);

        let min_for_estimate = (self.window_samples * 8) / 10;
        if self.history.len() < min_for_estimate {
            return None;
        }

        let samples: Vec<f64> = self.history.iter().copied().collect();

        // Lag bounds from the band edges.
        let lag_min = ((self.sample_rate_hz / self.high_hz).floor() as usize).max(2);
        let lag_max = ((self.sample_rate_hz / self.low_hz).ceil() as usize).min(samples.len() / 2);
        if lag_max <= lag_min {
            return Some(VitalEstimate::unavailable());
        }

        // Centre the signal to remove residual DC offset.
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let centred: Vec<f64> = samples.iter().map(|v| v - mean).collect();

        let r0 = centred.iter().map(|v| v * v).sum::<f64>().max(f64::EPSILON);

        let mut best_lag = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for lag in lag_min..=lag_max {
            let mut acc = 0.0;
            for i in 0..centred.len() - lag {
                acc += centred[i] * centred[i + lag];
            }
            // Biased autocorrelation, normalised by r0.
            let score = acc / r0;
            if score > best_score {
                best_score = score;
                best_lag = lag;
            }
        }

        if best_lag == 0 {
            return Some(VitalEstimate::unavailable());
        }
        let freq_hz = self.sample_rate_hz / best_lag as f64;
        let bpm = freq_hz * 60.0;

        let band_lo_bpm = self.low_hz * 60.0;
        let band_hi_bpm = self.high_hz * 60.0;
        if !bpm.is_finite() || bpm < band_lo_bpm || bpm > band_hi_bpm {
            return Some(VitalEstimate::unavailable());
        }

        // Confidence: normalised autocorrelation peak, clamped to a
        // useful range. r̂(τ) ∈ [-1, 1] in theory; we bias positive.
        let confidence = best_score.clamp(0.0, 1.0).max(0.05);

        let status = if confidence > 0.5 {
            VitalStatus::Valid
        } else if confidence > 0.25 {
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

    fn drive_sinusoid(
        ex: &mut HeartRateExtractor,
        freq_hz: f64,
        n_subcarriers: usize,
    ) -> Option<VitalEstimate> {
        let phases = vec![0.0; n_subcarriers];
        let mut last = None;
        let total = (ex.sample_rate_hz() * ex.window_secs() * 1.5) as usize;
        for i in 0..total {
            let t = i as f64 / ex.sample_rate_hz();
            let x = (TAU * freq_hz * t).sin();
            let residuals = vec![x; n_subcarriers];
            last = ex.extract(&residuals, &phases);
        }
        last
    }

    #[test]
    fn settles_at_60bpm() {
        // 1.0 Hz × 60 = 60 BPM.
        let mut ex = HeartRateExtractor::new(8, 30.0, 10.0);
        let est = drive_sinusoid(&mut ex, 1.0, 8).expect("estimate");
        assert!(
            (est.value_bpm - 60.0).abs() <= 4.0,
            "expected ~60 BPM ±4, got {}",
            est.value_bpm
        );
    }

    #[test]
    fn settles_at_90bpm() {
        // 1.5 Hz × 60 = 90 BPM.
        let mut ex = HeartRateExtractor::new(4, 30.0, 10.0);
        let est = drive_sinusoid(&mut ex, 1.5, 4).expect("estimate");
        assert!(
            (est.value_bpm - 90.0).abs() <= 6.0,
            "expected ~90 BPM ±6, got {}",
            est.value_bpm
        );
    }

    #[test]
    fn cold_start_yields_none() {
        let mut ex = HeartRateExtractor::new(1, 30.0, 10.0);
        let phases = vec![0.0];
        for i in 0..30 {
            let r = vec![(i as f64 * 0.1).sin()];
            assert!(ex.extract(&r, &phases).is_none(), "early extract Some");
        }
    }

    #[test]
    fn missing_phase_vector_uses_plain_mean() {
        // Empty phases slice should not panic.
        let mut ex = HeartRateExtractor::new(2, 30.0, 6.0);
        for i in 0..400 {
            let t = i as f64 / 30.0;
            let v = (TAU * 1.0 * t).sin();
            let _ = ex.extract(&[v, v], &[]);
        }
        // No assertion beyond "didn't panic" — the math path is tested
        // elsewhere; this verifies the fallback branch is safe.
    }

    #[test]
    fn reset_clears_history() {
        let mut ex = HeartRateExtractor::new(1, 30.0, 6.0);
        for _ in 0..200 {
            ex.extract(&[0.5], &[0.0]);
        }
        assert!(ex.history_len() > 0);
        ex.reset();
        assert_eq!(ex.history_len(), 0);
    }
}
