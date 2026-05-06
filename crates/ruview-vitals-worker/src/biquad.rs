//! 2nd-order biquad bandpass — RBJ "constant-skirt-gain" cookbook
//! variant, tuned per [`BandpassParams`] for vital-sign frequency bands.
//!
//! Direct-Form-I implementation:
//!
//! ```text
//!   y[n] = (b0/a0)·x[n] + (b1/a0)·x[n-1] + (b2/a0)·x[n-2]
//!        − (a1/a0)·y[n-1] − (a2/a0)·y[n-2]
//! ```
//!
//! For a bandpass filter (Robert Bristow-Johnson cookbook):
//!
//! ```text
//!   ω₀ = 2π · f_c / f_s
//!   α  = sin(ω₀) / (2Q)
//!   b0 =  α        b1 =  0       b2 = -α
//!   a0 = 1 + α     a1 = -2cos ω₀ a2 = 1 − α
//! ```
//!
//! Quality factor `Q = f_c / Δf`, where `Δf` is the −3 dB bandwidth.
//! Higher Q = narrower band, sharper rolloff, longer settling time.

use std::f64::consts::TAU;

/// Filter design parameters in Hz.
#[derive(Debug, Clone, Copy)]
pub struct BandpassParams {
    pub center_hz: f64,
    pub bandwidth_hz: f64,
    pub sample_rate_hz: f64,
}

impl BandpassParams {
    /// Quality factor `Q = f_c / Δf`. Saturates `bandwidth_hz` to a
    /// tiny epsilon so we never divide by zero.
    #[must_use]
    pub fn quality_factor(&self) -> f64 {
        self.center_hz / self.bandwidth_hz.max(f64::EPSILON)
    }
}

/// Direct-Form-I 2nd-order biquad. Coefficients are precomputed and
/// stored; the filter holds two samples of history for both input and
/// output.
#[derive(Debug, Clone)]
pub struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl Biquad {
    /// Build a biquad from a bandpass design.
    ///
    /// Returns a no-op pass-through filter when `sample_rate_hz` is
    /// non-positive or when `center_hz` is at / above Nyquist —
    /// exotic configurations should not crash the worker; they should
    /// just produce zero-output.
    #[must_use]
    pub fn bandpass(params: BandpassParams) -> Self {
        let fs = params.sample_rate_hz;
        let fc = params.center_hz;
        let q = params.quality_factor();

        if fs <= 0.0 || fc <= 0.0 || fc >= fs * 0.5 || q <= 0.0 {
            return Self::pass_through();
        }

        let w0 = TAU * fc / fs;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);

        let a0 = 1.0 + alpha;
        let inv = 1.0 / a0;
        let b0 = alpha * inv;
        let b1 = 0.0;
        let b2 = -alpha * inv;
        let a1 = (-2.0 * cos_w0) * inv;
        let a2 = (1.0 - alpha) * inv;

        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// `y = x` filter, used as a fallback for invalid params.
    #[must_use]
    pub const fn pass_through() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Run one input sample through the filter and return the output.
    pub fn step(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        // Shift state.
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }

    /// Reset filter history to zero.
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// Count zero-crossings (sign changes) in `samples`. A leading zero is
/// not counted; we only count transitions where the sign actually
/// changes.
#[must_use]
pub fn zero_crossings(samples: &[f64]) -> usize {
    let mut count = 0;
    let mut last_sign: i8 = 0;
    for &v in samples {
        let s = if v > 0.0 {
            1
        } else if v < 0.0 {
            -1
        } else {
            0
        };
        if s != 0 && last_sign != 0 && s != last_sign {
            count += 1;
        }
        if s != 0 {
            last_sign = s;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_through_is_identity() {
        let mut bq = Biquad::pass_through();
        assert_eq!(bq.step(2.5), 2.5);
        assert_eq!(bq.step(-1.0), -1.0);
    }

    #[test]
    fn bandpass_attenuates_dc() {
        let mut bq = Biquad::bandpass(BandpassParams {
            center_hz: 0.25,
            bandwidth_hz: 0.4,
            sample_rate_hz: 30.0,
        });
        // Drive with constant 1.0 for 200 samples; output should
        // converge to ~0 (dc is fully rejected).
        let mut last = 0.0;
        for _ in 0..200 {
            last = bq.step(1.0);
        }
        assert!(last.abs() < 1e-3, "dc not rejected, |y|={}", last.abs());
    }

    #[test]
    fn bandpass_passes_in_band_sinusoid() {
        // 0.25 Hz sinusoid at 30 fps → in the breathing band.
        let mut bq = Biquad::bandpass(BandpassParams {
            center_hz: 0.25,
            bandwidth_hz: 0.4,
            sample_rate_hz: 30.0,
        });
        // Drive long enough for the filter to settle, then look at
        // the peak amplitude over a final cycle.
        let n = 600usize; // 20 s
        let mut max_after_settle = 0.0_f64;
        for i in 0..n {
            let t = i as f64 / 30.0;
            let x = (TAU * 0.25 * t).sin();
            let y = bq.step(x).abs();
            if i > 300 {
                max_after_settle = max_after_settle.max(y);
            }
        }
        assert!(
            max_after_settle > 0.3,
            "in-band signal heavily attenuated, peak={max_after_settle}"
        );
    }

    #[test]
    fn invalid_params_yield_pass_through() {
        let bq = Biquad::bandpass(BandpassParams {
            center_hz: 100.0, // above Nyquist for fs=30
            bandwidth_hz: 1.0,
            sample_rate_hz: 30.0,
        });
        // pass-through has b0=1.0, others 0 — assert via step.
        let mut bq = bq;
        assert_eq!(bq.step(0.7), 0.7);
    }

    #[test]
    fn zero_crossings_counts_sign_flips_only() {
        assert_eq!(zero_crossings(&[1.0, 2.0, 3.0]), 0);
        assert_eq!(zero_crossings(&[1.0, -1.0, 1.0, -1.0]), 3);
        assert_eq!(zero_crossings(&[0.0, 1.0, -1.0, 0.0, 1.0]), 2);
        assert_eq!(zero_crossings(&[]), 0);
    }
}
