//! Biquad filter — Audio EQ Cookbook implementation.
//!
//! Ported from Tympan's `AudioFilterBiquad_F32`. Implements all standard
//! filter types using Robert Bristow-Johnson's Audio EQ Cookbook formulas
//! with Direct Form II Transposed structure.

use super::block::{AudioBlock, AudioProcessor};
use std::f64::consts::PI;

/// Filter type for the biquad.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterType {
    LowPass,
    HighPass,
    BandPass,
    Notch,
    AllPass,
    PeakingEQ,
    LowShelf,
    HighShelf,
}

/// Second-order IIR biquad filter (Audio EQ Cookbook).
///
/// Uses Direct Form II Transposed for numerical stability. Supports stereo
/// processing with independent state per channel.
pub struct BiquadFilter {
    filter_type: FilterType,
    frequency: f32,
    q: f32,
    gain_db: f32,
    // Normalized coefficients (divided by a0)
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    // State — Direct Form II Transposed (per channel)
    z1_l: f64,
    z2_l: f64,
    z1_r: f64,
    z2_r: f64,
    sample_rate: f32,
}

impl BiquadFilter {
    /// Create a new biquad filter. Coefficients are computed when `prepare()` is called.
    ///
    /// For filter types that don't use gain (LowPass, HighPass, BandPass, Notch,
    /// AllPass), pass 0.0 for `gain_db` or use the 3-argument `new()`.
    pub fn new(filter_type: FilterType, frequency: f32, q: f32) -> Self {
        Self::with_gain(filter_type, frequency, q, 0.0)
    }

    /// Create a biquad filter with explicit gain (for PeakingEQ, LowShelf, HighShelf).
    pub fn with_gain(filter_type: FilterType, frequency: f32, q: f32, gain_db: f32) -> Self {
        Self {
            filter_type,
            frequency,
            q,
            gain_db,
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            z1_l: 0.0,
            z2_l: 0.0,
            z1_r: 0.0,
            z2_r: 0.0,
            sample_rate: 0.0,
        }
    }

    /// Update the cutoff/center frequency and recalculate coefficients.
    pub fn set_frequency(&mut self, freq: f32) {
        self.frequency = freq;
        if self.sample_rate > 0.0 {
            self.compute_coefficients();
        }
    }

    /// Update Q factor and recalculate coefficients.
    pub fn set_q(&mut self, q: f32) {
        self.q = q;
        if self.sample_rate > 0.0 {
            self.compute_coefficients();
        }
    }

    /// Update gain (only affects PeakingEQ, LowShelf, HighShelf).
    pub fn set_gain_db(&mut self, gain_db: f32) {
        self.gain_db = gain_db;
        if self.sample_rate > 0.0 {
            self.compute_coefficients();
        }
    }

    /// Reset filter state (clear delay lines).
    pub fn reset(&mut self) {
        self.z1_l = 0.0;
        self.z2_l = 0.0;
        self.z1_r = 0.0;
        self.z2_r = 0.0;
    }

    /// Compute biquad coefficients from the Audio EQ Cookbook.
    fn compute_coefficients(&mut self) {
        let sr = self.sample_rate as f64;
        let freq = (self.frequency as f64).min(sr * 0.499); // Nyquist guard
        let w0 = 2.0 * PI * freq / sr;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * self.q as f64);

        let (b0, b1, b2, a0, a1, a2);

        match self.filter_type {
            FilterType::LowPass => {
                b1 = 1.0 - cos_w0;
                b0 = b1 / 2.0;
                b2 = b0;
                a0 = 1.0 + alpha;
                a1 = -2.0 * cos_w0;
                a2 = 1.0 - alpha;
            }
            FilterType::HighPass => {
                b0 = (1.0 + cos_w0) / 2.0;
                b1 = -(1.0 + cos_w0);
                b2 = b0;
                a0 = 1.0 + alpha;
                a1 = -2.0 * cos_w0;
                a2 = 1.0 - alpha;
            }
            FilterType::BandPass => {
                b0 = alpha;
                b1 = 0.0;
                b2 = -alpha;
                a0 = 1.0 + alpha;
                a1 = -2.0 * cos_w0;
                a2 = 1.0 - alpha;
            }
            FilterType::Notch => {
                b0 = 1.0;
                b1 = -2.0 * cos_w0;
                b2 = 1.0;
                a0 = 1.0 + alpha;
                a1 = -2.0 * cos_w0;
                a2 = 1.0 - alpha;
            }
            FilterType::AllPass => {
                b0 = 1.0 - alpha;
                b1 = -2.0 * cos_w0;
                b2 = 1.0 + alpha;
                a0 = 1.0 + alpha;
                a1 = -2.0 * cos_w0;
                a2 = 1.0 - alpha;
            }
            FilterType::PeakingEQ => {
                let a_lin = 10.0f64.powf(self.gain_db as f64 / 40.0);
                b0 = 1.0 + alpha * a_lin;
                b1 = -2.0 * cos_w0;
                b2 = 1.0 - alpha * a_lin;
                a0 = 1.0 + alpha / a_lin;
                a1 = -2.0 * cos_w0;
                a2 = 1.0 - alpha / a_lin;
            }
            FilterType::LowShelf => {
                let a_lin = 10.0f64.powf(self.gain_db as f64 / 40.0);
                let two_sqrt_a_alpha = 2.0 * a_lin.sqrt() * alpha;
                b0 = a_lin * ((a_lin + 1.0) - (a_lin - 1.0) * cos_w0 + two_sqrt_a_alpha);
                b1 = 2.0 * a_lin * ((a_lin - 1.0) - (a_lin + 1.0) * cos_w0);
                b2 = a_lin * ((a_lin + 1.0) - (a_lin - 1.0) * cos_w0 - two_sqrt_a_alpha);
                a0 = (a_lin + 1.0) + (a_lin - 1.0) * cos_w0 + two_sqrt_a_alpha;
                a1 = -2.0 * ((a_lin - 1.0) + (a_lin + 1.0) * cos_w0);
                a2 = (a_lin + 1.0) + (a_lin - 1.0) * cos_w0 - two_sqrt_a_alpha;
            }
            FilterType::HighShelf => {
                let a_lin = 10.0f64.powf(self.gain_db as f64 / 40.0);
                let two_sqrt_a_alpha = 2.0 * a_lin.sqrt() * alpha;
                b0 = a_lin * ((a_lin + 1.0) + (a_lin - 1.0) * cos_w0 + two_sqrt_a_alpha);
                b1 = -2.0 * a_lin * ((a_lin - 1.0) + (a_lin + 1.0) * cos_w0);
                b2 = a_lin * ((a_lin + 1.0) + (a_lin - 1.0) * cos_w0 - two_sqrt_a_alpha);
                a0 = (a_lin + 1.0) - (a_lin - 1.0) * cos_w0 + two_sqrt_a_alpha;
                a1 = 2.0 * ((a_lin - 1.0) - (a_lin + 1.0) * cos_w0);
                a2 = (a_lin + 1.0) - (a_lin - 1.0) * cos_w0 - two_sqrt_a_alpha;
            }
        }

        // Normalize by a0
        self.b0 = b0 / a0;
        self.b1 = b1 / a0;
        self.b2 = b2 / a0;
        self.a1 = a1 / a0;
        self.a2 = a2 / a0;
    }

    /// Process a single sample (Direct Form II Transposed).
    #[inline]
    fn process_sample(
        b0: f64,
        b1: f64,
        b2: f64,
        a1: f64,
        a2: f64,
        z1: &mut f64,
        z2: &mut f64,
        x: f32,
    ) -> f32 {
        let x64 = x as f64;
        let y = b0 * x64 + *z1;
        *z1 = b1 * x64 - a1 * y + *z2;
        *z2 = b2 * x64 - a2 * y;
        y as f32
    }
}

impl AudioProcessor for BiquadFilter {
    fn prepare(&mut self, sample_rate: f32, _block_size: usize) {
        self.sample_rate = sample_rate;
        self.compute_coefficients();
        self.reset();
    }

    fn process(&mut self, block: &mut AudioBlock) {
        let (b0, b1, b2, a1, a2) = (self.b0, self.b1, self.b2, self.a1, self.a2);

        for s in block.left.iter_mut() {
            *s = Self::process_sample(b0, b1, b2, a1, a2, &mut self.z1_l, &mut self.z2_l, *s);
        }
        for s in block.right.iter_mut() {
            *s = Self::process_sample(b0, b1, b2, a1, a2, &mut self.z1_r, &mut self.z2_r, *s);
        }
    }

    fn name(&self) -> &str {
        "BiquadFilter"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI as PI32;

    fn make_tone(freq: f32, sample_rate: f32, num_samples: usize, amplitude: f32) -> Vec<f32> {
        (0..num_samples)
            .map(|i| amplitude * (2.0 * PI32 * freq * i as f32 / sample_rate).sin())
            .collect()
    }

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }

    fn rms_db(samples: &[f32]) -> f32 {
        20.0 * rms(samples).max(1e-10).log10()
    }

    fn make_block(samples: &[f32], sr: f32) -> AudioBlock {
        AudioBlock {
            left: samples.to_vec(),
            right: samples.to_vec(),
            sample_rate: sr,
            block_size: samples.len(),
            metadata: super::super::block::BlockMetadata::default(),
        }
    }

    #[test]
    fn lowpass_attenuates_high_frequency() {
        let sr = 48000.0;
        let n = 4096;

        // Generate 4 kHz tone
        let tone = make_tone(4000.0, sr, n, 0.5);
        let input_db = rms_db(&tone);

        let mut filter = BiquadFilter::new(FilterType::LowPass, 1000.0, 0.707);
        filter.prepare(sr, n);

        let mut block = make_block(&tone, sr);
        // Process multiple blocks to get past transient
        filter.process(&mut block);
        let mut block2 = make_block(&tone, sr);
        filter.process(&mut block2);

        let output_db = rms_db(&block2.left);
        let attenuation = input_db - output_db;

        assert!(
            attenuation > 12.0,
            "LPF at 1kHz should attenuate 4kHz by >12dB, got {:.1}dB",
            attenuation
        );
    }

    #[test]
    fn highpass_attenuates_low_frequency() {
        let sr = 48000.0;
        let n = 4096;

        // Generate 250 Hz tone
        let tone = make_tone(250.0, sr, n, 0.5);
        let input_db = rms_db(&tone);

        let mut filter = BiquadFilter::new(FilterType::HighPass, 1000.0, 0.707);
        filter.prepare(sr, n);

        let mut block = make_block(&tone, sr);
        filter.process(&mut block);
        let mut block2 = make_block(&tone, sr);
        filter.process(&mut block2);

        let output_db = rms_db(&block2.left);
        let attenuation = input_db - output_db;

        assert!(
            attenuation > 12.0,
            "HPF at 1kHz should attenuate 250Hz by >12dB, got {:.1}dB",
            attenuation
        );
    }

    #[test]
    fn peaking_eq_boosts_target_frequency() {
        let sr = 48000.0;
        let n = 4096;
        let boost_db = 12.0;

        // Generate 1 kHz tone
        let tone = make_tone(1000.0, sr, n, 0.1);
        let input_db = rms_db(&tone);

        let mut filter = BiquadFilter::with_gain(FilterType::PeakingEQ, 1000.0, 1.0, boost_db);
        filter.prepare(sr, n);

        let mut block = make_block(&tone, sr);
        filter.process(&mut block);
        let mut block2 = make_block(&tone, sr);
        filter.process(&mut block2);

        let output_db = rms_db(&block2.left);
        let gain = output_db - input_db;

        assert!(
            gain > 8.0,
            "PeakingEQ +12dB at 1kHz should boost 1kHz by >8dB, got {:.1}dB",
            gain
        );
    }

    #[test]
    fn passband_signal_passes_through_lowpass() {
        let sr = 48000.0;
        let n = 4096;

        // Generate 100 Hz tone — well below 1kHz cutoff
        let tone = make_tone(100.0, sr, n, 0.5);
        let input_db = rms_db(&tone);

        let mut filter = BiquadFilter::new(FilterType::LowPass, 1000.0, 0.707);
        filter.prepare(sr, n);

        let mut block = make_block(&tone, sr);
        filter.process(&mut block);
        let mut block2 = make_block(&tone, sr);
        filter.process(&mut block2);

        let output_db = rms_db(&block2.left);
        let diff = (output_db - input_db).abs();

        assert!(
            diff < 1.0,
            "100Hz through LPF@1kHz should pass with <1dB change, got {:.1}dB",
            diff
        );
    }
}
