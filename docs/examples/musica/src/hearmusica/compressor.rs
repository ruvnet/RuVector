//! Wide Dynamic Range Compression (WDRC) — multi-band compressor for hearing aids.
//!
//! Ported from Tympan's `AudioEffectCompressor_F32`. Splits the signal into
//! frequency bands using Linkwitz-Riley crossover filters, applies per-band
//! compression with soft-knee curves, and sums the bands back together.

use super::block::{AudioBlock, AudioProcessor};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Biquad coefficients (internal helper for crossover filters)
// ---------------------------------------------------------------------------

/// Second-order IIR biquad filter state used internally by the crossover.
#[derive(Clone, Debug)]
struct BiquadCoeffs {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    // Direct Form II Transposed state
    z1: f64,
    z2: f64,
}

impl BiquadCoeffs {
    fn new() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Design a 2nd-order Butterworth low-pass filter.
    fn low_pass(freq: f32, sample_rate: f32) -> Self {
        let w0 = 2.0 * PI as f64 * freq as f64 / sample_rate as f64;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * std::f64::consts::FRAC_1_SQRT_2); // Q = 1/sqrt(2)

        let b1 = 1.0 - cos_w0;
        let b0 = b1 / 2.0;
        let b2 = b0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Design a 2nd-order Butterworth high-pass filter.
    fn high_pass(freq: f32, sample_rate: f32) -> Self {
        let w0 = 2.0 * PI as f64 * freq as f64 / sample_rate as f64;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * std::f64::consts::FRAC_1_SQRT_2);

        let b0 = (1.0 + cos_w0) / 2.0;
        let b1 = -(1.0 + cos_w0);
        let b2 = b0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Process a single sample through this biquad (Direct Form II Transposed).
    #[inline]
    fn process_sample(&mut self, x: f32) -> f32 {
        let x64 = x as f64;
        let y = self.b0 * x64 + self.z1;
        self.z1 = self.b1 * x64 - self.a1 * y + self.z2;
        self.z2 = self.b2 * x64 - self.a2 * y;
        y as f32
    }

    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Crossover filter pair (Linkwitz-Riley = cascaded Butterworth)
// ---------------------------------------------------------------------------

/// A Linkwitz-Riley crossover: two cascaded Butterworth filters for flat
/// magnitude response when low + high are summed.
#[derive(Clone, Debug)]
struct CrossoverFilter {
    lp1: BiquadCoeffs,
    lp2: BiquadCoeffs,
    hp1: BiquadCoeffs,
    hp2: BiquadCoeffs,
}

impl CrossoverFilter {
    fn new(freq: f32, sample_rate: f32) -> Self {
        Self {
            lp1: BiquadCoeffs::low_pass(freq, sample_rate),
            lp2: BiquadCoeffs::low_pass(freq, sample_rate),
            hp1: BiquadCoeffs::high_pass(freq, sample_rate),
            hp2: BiquadCoeffs::high_pass(freq, sample_rate),
        }
    }

    /// Split a sample into (low, high) components.
    #[inline]
    fn split_sample(&mut self, x: f32) -> (f32, f32) {
        let lo = self.lp2.process_sample(self.lp1.process_sample(x));
        let hi = self.hp2.process_sample(self.hp1.process_sample(x));
        (lo, hi)
    }

    fn reset(&mut self) {
        self.lp1.reset();
        self.lp2.reset();
        self.hp1.reset();
        self.hp2.reset();
    }
}

// ---------------------------------------------------------------------------
// Compressor band
// ---------------------------------------------------------------------------

/// Per-band compression parameters and state.
#[derive(Clone, Debug)]
pub struct CompressorBand {
    /// Compression threshold in dB FS.
    pub threshold_db: f32,
    /// Compression ratio (e.g. 3.0 means 3:1).
    pub ratio: f32,
    /// Attack time in milliseconds.
    pub attack_ms: f32,
    /// Release time in milliseconds.
    pub release_ms: f32,
    /// Soft-knee width in dB.
    pub knee_db: f32,
    /// Post-compression makeup gain in dB.
    pub makeup_gain_db: f32,
    // Internal state
    envelope: f32,
    attack_coeff: f32,
    release_coeff: f32,
}

impl CompressorBand {
    /// Create a band with default hearing-aid parameters.
    pub fn new() -> Self {
        Self {
            threshold_db: -40.0,
            ratio: 3.0,
            attack_ms: 5.0,
            release_ms: 50.0,
            knee_db: 10.0,
            makeup_gain_db: 0.0,
            envelope: 0.0,
            attack_coeff: 0.0,
            release_coeff: 0.0,
        }
    }

    /// Recalculate time constants for the given sample rate.
    fn update_coefficients(&mut self, sample_rate: f32) {
        self.attack_coeff = (-1.0 / (self.attack_ms * 0.001 * sample_rate)).exp();
        self.release_coeff = (-1.0 / (self.release_ms * 0.001 * sample_rate)).exp();
    }

    /// Compute the gain reduction in dB for a given input level in dB.
    fn compute_gain_db(&self, level_db: f32) -> f32 {
        let t = self.threshold_db;
        let r = self.ratio;
        let w = self.knee_db;
        let half_w = w / 2.0;

        if level_db < (t - half_w) {
            // Below knee — no compression
            0.0
        } else if level_db > (t + half_w) {
            // Above knee — full compression
            t + (level_db - t) / r - level_db
        } else {
            // In the knee — quadratic interpolation
            let x = level_db - t + half_w;
            let gain = (1.0 / r - 1.0) * x * x / (2.0 * w);
            gain
        }
    }

    /// Process a single sample: envelope tracking + gain computation.
    #[inline]
    fn process_sample(&mut self, x: f32) -> f32 {
        let abs_x = x.abs();

        // Envelope follower
        if abs_x > self.envelope {
            self.envelope =
                self.attack_coeff * self.envelope + (1.0 - self.attack_coeff) * abs_x;
        } else {
            self.envelope =
                self.release_coeff * self.envelope + (1.0 - self.release_coeff) * abs_x;
        }

        // Convert to dB
        let level_db = 20.0 * (self.envelope + 1e-10).log10();

        // Compute gain in dB
        let gain_db = self.compute_gain_db(level_db) + self.makeup_gain_db;

        // Apply gain
        let gain_linear = 10.0f32.powf(gain_db / 20.0);
        x * gain_linear
    }
}

impl Default for CompressorBand {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Multi-band WDRC
// ---------------------------------------------------------------------------

/// Multi-band Wide Dynamic Range Compressor.
///
/// Splits the input into frequency bands via Linkwitz-Riley crossover filters,
/// applies independent compression to each band, then sums back together.
pub struct WDRCompressor {
    bands: Vec<CompressorBand>,
    crossover_freqs: Vec<f32>,
    /// Left-channel crossover filters.
    crossovers_l: Vec<CrossoverFilter>,
    /// Right-channel crossover filters.
    crossovers_r: Vec<CrossoverFilter>,
    /// Per-band left/right compressor states (separate L/R envelopes).
    bands_r: Vec<CompressorBand>,
    sample_rate: f32,
    block_size: usize,
}

impl WDRCompressor {
    /// Create a single-band compressor with the given threshold and ratio.
    ///
    /// This matches the Tympan `AudioEffectCompressor_F32` constructor signature.
    pub fn new(threshold_db: f32, ratio: f32) -> Self {
        let mut band = CompressorBand::new();
        band.threshold_db = threshold_db;
        band.ratio = ratio;
        Self::with_bands(vec![band], vec![])
    }

    /// Create a multi-band compressor with default band settings.
    ///
    /// Default crossover frequencies: <250 Hz, 250-1k Hz, 1k-4k Hz, >4k Hz (for 4 bands).
    pub fn multi_band(num_bands: usize) -> Self {
        let freqs = match num_bands {
            1 => vec![],
            2 => vec![1000.0],
            3 => vec![500.0, 2000.0],
            _ => vec![250.0, 1000.0, 4000.0], // 4 bands
        };
        let actual_bands = freqs.len() + 1;
        let bands: Vec<CompressorBand> = (0..actual_bands).map(|_| CompressorBand::new()).collect();
        Self::with_bands(bands, freqs)
    }

    /// Create a compressor with explicit band configs and crossover frequencies.
    ///
    /// `crossover_freqs.len()` must equal `bands.len() - 1`.
    pub fn with_bands(bands: Vec<CompressorBand>, crossover_freqs: Vec<f32>) -> Self {
        assert_eq!(
            crossover_freqs.len() + 1,
            bands.len(),
            "Need exactly N-1 crossover frequencies for N bands"
        );
        let bands_r = bands.clone();
        let n_cross = crossover_freqs.len();
        Self {
            bands,
            bands_r,
            crossover_freqs,
            crossovers_l: vec![CrossoverFilter::new(1000.0, 48000.0); n_cross],
            crossovers_r: vec![CrossoverFilter::new(1000.0, 48000.0); n_cross],
            sample_rate: 48000.0,
            block_size: 0,
        }
    }

    /// Access a band's parameters mutably.
    pub fn band_mut(&mut self, index: usize) -> &mut CompressorBand {
        &mut self.bands[index]
    }

    /// Number of bands.
    pub fn num_bands(&self) -> usize {
        self.bands.len()
    }

    /// Split a mono sample into N bands using cascaded crossovers.
    fn split_into_bands(crossovers: &mut [CrossoverFilter], sample: f32, out: &mut [f32]) {
        let n_bands = crossovers.len() + 1;
        if n_bands == 1 {
            out[0] = sample;
            return;
        }
        // Recursive split: first crossover splits into low / rest
        let (lo, hi) = crossovers[0].split_sample(sample);
        out[0] = lo;
        if n_bands == 2 {
            out[1] = hi;
        } else {
            // Recursively split the high portion
            Self::split_into_bands(&mut crossovers[1..], hi, &mut out[1..]);
        }
    }
}

impl AudioProcessor for WDRCompressor {
    fn prepare(&mut self, sample_rate: f32, block_size: usize) {
        self.sample_rate = sample_rate;
        self.block_size = block_size;

        // Rebuild crossover filters
        self.crossovers_l = self
            .crossover_freqs
            .iter()
            .map(|&f| CrossoverFilter::new(f, sample_rate))
            .collect();
        self.crossovers_r = self
            .crossover_freqs
            .iter()
            .map(|&f| CrossoverFilter::new(f, sample_rate))
            .collect();

        // Reset crossover state
        for c in &mut self.crossovers_l {
            c.reset();
        }
        for c in &mut self.crossovers_r {
            c.reset();
        }

        // Update band coefficients
        for band in &mut self.bands {
            band.update_coefficients(sample_rate);
            band.envelope = 0.0;
        }
        for band in &mut self.bands_r {
            band.update_coefficients(sample_rate);
            band.envelope = 0.0;
        }
    }

    fn process(&mut self, block: &mut AudioBlock) {
        let n_bands = self.bands.len();
        let mut band_samples = vec![0.0f32; n_bands];

        // Process left channel
        for i in 0..block.left.len() {
            let x = block.left[i];
            Self::split_into_bands(&mut self.crossovers_l, x, &mut band_samples);
            let mut sum = 0.0;
            for (b, &s) in self.bands.iter_mut().zip(band_samples.iter()) {
                sum += b.process_sample(s);
            }
            block.left[i] = sum;
        }

        // Process right channel
        for i in 0..block.right.len() {
            let x = block.right[i];
            Self::split_into_bands(&mut self.crossovers_r, x, &mut band_samples);
            let mut sum = 0.0;
            for (b, &s) in self.bands_r.iter_mut().zip(band_samples.iter()) {
                sum += b.process_sample(s);
            }
            block.right[i] = sum;
        }
    }

    fn name(&self) -> &str {
        "WDRCompressor"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
    fn quiet_signal_passes_unchanged() {
        // A very quiet signal (well below -40 dB threshold) should pass through
        // with negligible change.
        let sr = 48000.0;
        let mut comp = WDRCompressor::multi_band(4);
        comp.prepare(sr, 256);

        // -80 dB signal ≈ 0.0001 amplitude
        let amplitude = 0.0001f32;
        let samples: Vec<f32> = (0..256)
            .map(|i| amplitude * (2.0 * PI * 1000.0 * i as f32 / sr).sin())
            .collect();
        let mut block = make_block(&samples, sr);

        // Feed some silence first to let crossovers settle, then test
        let mut warmup = make_block(&vec![0.0; 512], sr);
        comp.process(&mut warmup);

        let input_energy: f32 = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
        comp.process(&mut block);
        let output_energy: f32 =
            block.left.iter().map(|s| s * s).sum::<f32>() / block.left.len() as f32;

        // Should be within 3 dB
        let ratio_db = 10.0 * (output_energy / (input_energy + 1e-20)).log10();
        assert!(
            ratio_db.abs() < 3.0,
            "Quiet signal changed by {:.1} dB, expected ~0 dB",
            ratio_db
        );
    }

    #[test]
    fn loud_signal_is_compressed() {
        // A loud signal (above threshold) should be reduced in level.
        let sr = 48000.0;
        let mut comp = WDRCompressor::new(-20.0, 4.0);
        comp.bands[0].knee_db = 0.0; // Hard knee for predictability
        comp.bands[0].makeup_gain_db = 0.0;
        comp.bands_r[0].knee_db = 0.0;
        comp.bands_r[0].makeup_gain_db = 0.0;
        comp.prepare(sr, 2048);

        // -6 dB signal ≈ 0.5 amplitude — well above -20 dB threshold
        let amplitude = 0.5f32;
        let samples: Vec<f32> = (0..2048)
            .map(|i| amplitude * (2.0 * PI * 1000.0 * i as f32 / sr).sin())
            .collect();
        let input_rms: f32 =
            (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();

        let mut block = make_block(&samples, sr);
        comp.process(&mut block);

        let output_rms: f32 = (block.left.iter().map(|s| s * s).sum::<f32>()
            / block.left.len() as f32)
            .sqrt();

        // Output should be quieter than input
        assert!(
            output_rms < input_rms * 0.9,
            "Expected compression: input RMS={:.4}, output RMS={:.4}",
            input_rms,
            output_rms
        );
    }

    #[test]
    fn attack_release_envelope_tracking() {
        // Verify that the envelope tracks transients: a sudden loud burst
        // after silence should show the envelope rising.
        let sr = 48000.0;
        let mut band = CompressorBand::new();
        band.attack_ms = 1.0; // 1ms attack
        band.release_ms = 50.0;
        band.threshold_db = -60.0; // Very low so compression is active
        band.ratio = 10.0;
        band.knee_db = 0.0;
        band.makeup_gain_db = 0.0;
        band.update_coefficients(sr);

        // Feed silence — envelope should be near zero
        for _ in 0..480 {
            band.process_sample(0.0);
        }
        let env_after_silence = band.envelope;

        // Feed loud signal
        for i in 0..480 {
            let x = 0.5 * (2.0 * PI * 1000.0 * i as f32 / sr).sin();
            band.process_sample(x);
        }
        let env_after_loud = band.envelope;

        assert!(
            env_after_loud > env_after_silence + 0.01,
            "Envelope should rise with loud signal: silence={:.6}, loud={:.6}",
            env_after_silence,
            env_after_loud
        );

        // Feed silence again — envelope should decay
        let env_before_release = band.envelope;
        for _ in 0..4800 {
            // ~100ms of silence
            band.process_sample(0.0);
        }
        let env_after_release = band.envelope;

        assert!(
            env_after_release < env_before_release * 0.5,
            "Envelope should decay: before={:.6}, after={:.6}",
            env_before_release,
            env_after_release
        );
    }

    #[test]
    fn soft_knee_gain_computation() {
        let band = CompressorBand {
            threshold_db: -30.0,
            ratio: 3.0,
            knee_db: 10.0,
            ..CompressorBand::new()
        };

        // Well below threshold — zero gain reduction
        assert!((band.compute_gain_db(-50.0)).abs() < 0.01);

        // Well above threshold — full compression
        let gain = band.compute_gain_db(-10.0);
        let expected = -30.0 + (-10.0 - -30.0) / 3.0 - (-10.0);
        assert!(
            (gain - expected).abs() < 0.1,
            "gain={:.2}, expected={:.2}",
            gain,
            expected
        );

        // At threshold — should be in knee region, gain between 0 and full
        let gain_at_threshold = band.compute_gain_db(-30.0);
        assert!(
            gain_at_threshold < 0.0 && gain_at_threshold > -5.0,
            "Knee gain at threshold={:.2}, expected small negative",
            gain_at_threshold
        );
    }
}
