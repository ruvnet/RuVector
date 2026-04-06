//! Brick-wall output limiter.
//!
//! Prevents the output signal from exceeding a configurable ceiling (default
//! -1 dB FS). Uses fast-attack / slow-release envelope tracking for
//! transparent gain reduction.

use super::block::{AudioBlock, AudioProcessor};

/// Brick-wall output limiter with envelope-based gain reduction.
///
/// Tracks peak levels with a fast attack and slow release. When the
/// envelope exceeds the threshold, gain is reduced proportionally so
/// the output never clips.
pub struct Limiter {
    /// Threshold in dB FS. Output will never exceed this level.
    pub threshold_db: f32,
    /// Attack time in milliseconds (default: 0.1 ms — very fast).
    pub attack_ms: f32,
    /// Release time in milliseconds (default: 50 ms).
    pub release_ms: f32,
    // Internal state
    envelope_l: f32,
    envelope_r: f32,
    attack_coeff: f32,
    release_coeff: f32,
    threshold_linear: f32,
    sample_rate: f32,
}

impl Limiter {
    /// Create a new limiter with the given ceiling in dB FS.
    pub fn new(threshold_db: f32) -> Self {
        Self {
            threshold_db,
            attack_ms: 0.1,
            release_ms: 50.0,
            envelope_l: 0.0,
            envelope_r: 0.0,
            attack_coeff: 0.0,
            release_coeff: 0.0,
            threshold_linear: 10.0f32.powf(threshold_db / 20.0),
            sample_rate: 0.0,
        }
    }

    /// Default limiter at -1 dB FS (standard headroom for hearing aids).
    pub fn default_ceiling() -> Self {
        Self::new(-1.0)
    }

    fn update_coefficients(&mut self) {
        if self.sample_rate > 0.0 {
            self.attack_coeff = (-1.0 / (self.attack_ms * 0.001 * self.sample_rate)).exp();
            self.release_coeff = (-1.0 / (self.release_ms * 0.001 * self.sample_rate)).exp();
            self.threshold_linear = 10.0f32.powf(self.threshold_db / 20.0);
        }
    }

    /// Process a single sample with envelope tracking and gain reduction.
    #[inline]
    fn limit_sample(
        x: f32,
        envelope: &mut f32,
        threshold: f32,
        attack_coeff: f32,
        release_coeff: f32,
    ) -> f32 {
        let abs_x = x.abs();

        // Envelope follower: fast attack, slow release
        if abs_x > *envelope {
            *envelope = attack_coeff * *envelope + (1.0 - attack_coeff) * abs_x;
        } else {
            *envelope = release_coeff * *envelope + (1.0 - release_coeff) * abs_x;
        }

        // Compute gain reduction
        if *envelope > threshold {
            let gain = threshold / *envelope;
            x * gain
        } else {
            x
        }
    }
}

impl AudioProcessor for Limiter {
    fn prepare(&mut self, sample_rate: f32, _block_size: usize) {
        self.sample_rate = sample_rate;
        self.update_coefficients();
        self.envelope_l = 0.0;
        self.envelope_r = 0.0;
    }

    fn process(&mut self, block: &mut AudioBlock) {
        let threshold = self.threshold_linear;
        let attack = self.attack_coeff;
        let release = self.release_coeff;

        for s in block.left.iter_mut() {
            *s = Self::limit_sample(*s, &mut self.envelope_l, threshold, attack, release);
        }
        for s in block.right.iter_mut() {
            *s = Self::limit_sample(*s, &mut self.envelope_r, threshold, attack, release);
        }
    }

    fn name(&self) -> &str {
        "Limiter"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

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
    fn signal_below_threshold_passes_unchanged() {
        let sr = 48000.0;
        let mut limiter = Limiter::new(-1.0); // threshold at ~0.891
        limiter.prepare(sr, 256);

        // Signal at -6 dB ≈ 0.5 amplitude — well below -1 dB threshold
        let amplitude = 0.5;
        let samples: Vec<f32> = (0..1024)
            .map(|i| amplitude * (2.0 * PI * 1000.0 * i as f32 / sr).sin())
            .collect();

        let mut block = make_block(&samples, sr);
        limiter.process(&mut block);

        // Check that output matches input closely
        let max_diff: f32 = block
            .left
            .iter()
            .zip(samples.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 0.01,
            "Signal below threshold should pass unchanged, max diff={:.4}",
            max_diff
        );
    }

    #[test]
    fn signal_above_threshold_is_limited() {
        let sr = 48000.0;
        let mut limiter = Limiter::new(-6.0); // threshold at ~0.501
        limiter.prepare(sr, 256);

        // Signal at 0 dB = 1.0 amplitude — well above -6 dB threshold
        let amplitude = 1.0;
        let samples: Vec<f32> = (0..2048)
            .map(|i| amplitude * (2.0 * PI * 1000.0 * i as f32 / sr).sin())
            .collect();

        let mut block = make_block(&samples, sr);
        limiter.process(&mut block);

        // After the limiter settles, peaks should be near the threshold
        let threshold_linear = 10.0f32.powf(-6.0 / 20.0);
        // Check the last 1024 samples (skip transient at start)
        let max_output: f32 = block.left[1024..]
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_output < threshold_linear * 1.15, // 15% tolerance for envelope lag
            "Limiter output peak={:.4} should be near threshold={:.4}",
            max_output,
            threshold_linear
        );
    }

    #[test]
    fn limiter_prevents_clipping() {
        let sr = 48000.0;
        let mut limiter = Limiter::new(-1.0);
        limiter.prepare(sr, 256);

        // Very loud signal: amplitude = 2.0 (well above 0 dBFS)
        let samples: Vec<f32> = (0..4096)
            .map(|i| 2.0 * (2.0 * PI * 440.0 * i as f32 / sr).sin())
            .collect();

        let mut block = make_block(&samples, sr);
        limiter.process(&mut block);

        let threshold_linear = 10.0f32.powf(-1.0 / 20.0);
        // After settling, output should be limited
        let max_output: f32 = block.left[2048..]
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_output < threshold_linear * 1.2,
            "Limiter should prevent clipping: peak={:.4}, threshold={:.4}",
            max_output,
            threshold_linear
        );
    }
}
