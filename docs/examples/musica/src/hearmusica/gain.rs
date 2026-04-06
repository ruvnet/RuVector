//! Gain processor with audiogram-based frequency shaping.
//!
//! Port of Tympan's `AudioEffectGain_F32` plus audiogram fitting.
//! Supports flat gain, audiogram-shaped gain via peaking EQ filters,
//! and NAL-R hearing aid prescription.

use super::block::{AudioBlock, AudioProcessor};

/// Biquad filter state for a single second-order section.
struct BiquadState {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    // Filter delay elements (transposed Direct Form II)
    z1: f64,
    z2: f64,
}

impl BiquadState {
    /// Create a peaking EQ biquad filter.
    ///
    /// # Arguments
    /// * `freq_hz` - Center frequency in Hz.
    /// * `gain_db` - Gain at center frequency in dB.
    /// * `q` - Quality factor (bandwidth control).
    /// * `sample_rate` - Sample rate in Hz.
    fn peaking_eq(freq_hz: f32, gain_db: f32, q: f32, sample_rate: f32) -> Self {
        let a = 10.0_f64.powf(gain_db as f64 / 40.0);
        let w0 = 2.0 * std::f64::consts::PI * freq_hz as f64 / sample_rate as f64;
        let sin_w0 = w0.sin();
        let cos_w0 = w0.cos();
        let alpha = sin_w0 / (2.0 * q as f64);

        let b0 = 1.0 + alpha * a;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0 - alpha * a;
        let a0 = 1.0 + alpha / a;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha / a;

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

    /// Process a single sample through the biquad (Transposed Direct Form II).
    fn process_sample(&mut self, input: f32) -> f32 {
        let x = input as f64;
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y as f32
    }

    /// Reset filter state.
    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

/// Gain processor with flat gain and optional audiogram-based frequency shaping.
pub struct GainProcessor {
    /// Flat gain in dB (applied on top of any audiogram shaping).
    gain_db: f32,
    /// Audiogram data: (frequency_hz, hearing_threshold_dB) pairs.
    audiogram_gains: Option<Vec<(f32, f32)>>,
    /// Linear gain per band (derived from audiogram).
    band_gains: Vec<f32>,
    /// Peaking EQ filters for audiogram shaping (left channel).
    band_filters_l: Vec<BiquadState>,
    /// Peaking EQ filters for audiogram shaping (right channel).
    band_filters_r: Vec<BiquadState>,
    /// Whether this processor uses NAL-R prescription.
    use_nal_r: bool,
    /// Configured sample rate.
    sample_rate: f32,
    /// Configured block size.
    block_size: usize,
}

impl GainProcessor {
    /// Create a flat gain processor.
    pub fn new(gain_db: f32) -> Self {
        Self {
            gain_db,
            audiogram_gains: None,
            band_gains: Vec::new(),
            band_filters_l: Vec::new(),
            band_filters_r: Vec::new(),
            use_nal_r: false,
            sample_rate: 16000.0,
            block_size: 128,
        }
    }

    /// Create a gain processor shaped by an audiogram.
    ///
    /// Uses the half-gain rule: prescribed gain = hearing_loss * 0.5 at each frequency.
    ///
    /// # Arguments
    /// * `audiogram` - Slice of (frequency_hz, hearing_threshold_dB_HL) pairs.
    pub fn with_audiogram(audiogram: &[(f32, f32)]) -> Self {
        Self {
            gain_db: 0.0,
            audiogram_gains: Some(audiogram.to_vec()),
            band_gains: Vec::new(),
            band_filters_l: Vec::new(),
            band_filters_r: Vec::new(),
            use_nal_r: false,
            sample_rate: 16000.0,
            block_size: 128,
        }
    }

    /// Create a gain processor using the NAL-R prescription formula.
    ///
    /// NAL-R: `gain(f) = X + 0.31 * HTL(f) + correction(f)`
    /// where `X = 0.05 * (HTL_500 + HTL_1k + HTL_2k)`
    ///
    /// # Arguments
    /// * `audiogram` - Slice of (frequency_hz, hearing_threshold_dB_HL) pairs.
    pub fn with_nal_r(audiogram: &[(f32, f32)]) -> Self {
        Self {
            gain_db: 0.0,
            audiogram_gains: Some(audiogram.to_vec()),
            band_gains: Vec::new(),
            band_filters_l: Vec::new(),
            band_filters_r: Vec::new(),
            use_nal_r: true,
            sample_rate: 16000.0,
            block_size: 128,
        }
    }

    /// Convert dB to linear gain.
    fn db_to_linear(db: f32) -> f32 {
        10.0f32.powf(db / 20.0)
    }

    /// Interpolate the audiogram to find the threshold at a given frequency.
    fn interpolate_audiogram(audiogram: &[(f32, f32)], freq: f32) -> f32 {
        if audiogram.is_empty() {
            return 0.0;
        }
        if freq <= audiogram[0].0 {
            return audiogram[0].1;
        }
        if freq >= audiogram[audiogram.len() - 1].0 {
            return audiogram[audiogram.len() - 1].1;
        }
        for i in 0..audiogram.len() - 1 {
            if freq >= audiogram[i].0 && freq <= audiogram[i + 1].0 {
                let t = (freq - audiogram[i].0) / (audiogram[i + 1].0 - audiogram[i].0);
                return audiogram[i].1 + t * (audiogram[i + 1].1 - audiogram[i].1);
            }
        }
        0.0
    }

    /// Compute NAL-R prescribed gain for a given frequency and audiogram.
    fn nal_r_gain(audiogram: &[(f32, f32)], freq_hz: f32) -> f32 {
        // Find thresholds at 500, 1000, 2000 Hz
        let htl_500 = Self::interpolate_audiogram(audiogram, 500.0);
        let htl_1k = Self::interpolate_audiogram(audiogram, 1000.0);
        let htl_2k = Self::interpolate_audiogram(audiogram, 2000.0);

        let three_freq_avg = htl_500 + htl_1k + htl_2k;
        let x = 0.05 * three_freq_avg;

        let htl_f = Self::interpolate_audiogram(audiogram, freq_hz);

        // Frequency-dependent correction
        let correction = if freq_hz <= 375.0 {
            1.0 // ~250 Hz region
        } else if freq_hz <= 750.0 {
            0.0 // 500 Hz
        } else if freq_hz <= 1500.0 {
            0.0 // 1000 Hz
        } else if freq_hz <= 3000.0 {
            0.0 // 2000 Hz
        } else if freq_hz <= 5000.0 {
            -1.0 // 4000 Hz
        } else {
            -2.0 // 6000+ Hz
        };

        let gain = x + 0.31 * htl_f + correction;
        gain.max(0.0) // Don't apply negative gain
    }

    /// Build the filter bank from the audiogram data.
    fn build_filters(&mut self) {
        self.band_filters_l.clear();
        self.band_filters_r.clear();
        self.band_gains.clear();

        let audiogram = match &self.audiogram_gains {
            Some(a) => a.clone(),
            None => return,
        };

        if audiogram.is_empty() {
            return;
        }

        for &(freq, threshold) in &audiogram {
            // Skip frequencies above Nyquist
            if freq >= self.sample_rate * 0.5 {
                continue;
            }

            let gain_db = if self.use_nal_r {
                Self::nal_r_gain(&audiogram, freq)
            } else {
                // Half-gain rule
                threshold * 0.5
            };

            self.band_gains.push(gain_db);

            let q = 1.0;
            self.band_filters_l.push(BiquadState::peaking_eq(freq, gain_db, q, self.sample_rate));
            self.band_filters_r.push(BiquadState::peaking_eq(freq, gain_db, q, self.sample_rate));
        }
    }
}

impl AudioProcessor for GainProcessor {
    fn prepare(&mut self, sample_rate: f32, block_size: usize) {
        self.sample_rate = sample_rate;
        self.block_size = block_size;
        self.build_filters();
    }

    fn process(&mut self, block: &mut AudioBlock) {
        if !self.band_filters_l.is_empty() {
            // Apply audiogram-shaped filtering
            for i in 0..block.left.len() {
                let mut out_l = 0.0f32;
                let mut out_r = 0.0f32;
                let num_bands = self.band_filters_l.len();

                for b in 0..num_bands {
                    out_l += self.band_filters_l[b].process_sample(block.left[i]);
                    out_r += self.band_filters_r[b].process_sample(block.right[i]);
                }

                // Average the parallel filter outputs (prevents excessive gain buildup)
                if num_bands > 0 {
                    block.left[i] = out_l / num_bands as f32;
                    block.right[i] = out_r / num_bands as f32;
                }
            }
        }

        // Apply flat gain on top
        let flat_gain = Self::db_to_linear(self.gain_db);
        if (flat_gain - 1.0).abs() > 1e-7 {
            for s in block.left.iter_mut() {
                *s *= flat_gain;
            }
            for s in block.right.iter_mut() {
                *s *= flat_gain;
            }
        }
    }

    fn name(&self) -> &str {
        "Gain"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_gain_6db_doubles_amplitude() {
        let mut gp = GainProcessor::new(6.0206); // exact +6dB ~= 2x
        gp.prepare(16000.0, 8);

        let input = vec![0.1, 0.2, -0.3, 0.4, 0.0, -0.5, 0.6, 0.25];
        let mut block = AudioBlock::new(8, 16000.0);
        block.left = input.clone();
        block.right = input.clone();

        gp.process(&mut block);

        for (i, &inp) in input.iter().enumerate() {
            let expected = inp * 2.0;
            assert!(
                (block.left[i] - expected).abs() < 0.01,
                "Sample {}: expected ~{}, got {}",
                i,
                expected,
                block.left[i]
            );
        }
    }

    #[test]
    fn audiogram_boosts_high_frequencies() {
        // Typical sloping hearing loss: normal lows, increasing loss at highs
        let audiogram = vec![
            (250.0, 10.0),
            (500.0, 15.0),
            (1000.0, 25.0),
            (2000.0, 40.0),
            (4000.0, 60.0),
        ];

        let mut gp = GainProcessor::with_audiogram(&audiogram);
        gp.prepare(16000.0, 512);

        // Generate a high-frequency tone (4 kHz) and a low-frequency tone (250 Hz)
        let sr = 16000.0;
        let n = 512;

        // Test high-freq tone
        let high_input: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 4000.0 * i as f32 / sr).sin() * 0.1)
            .collect();

        let mut block_high = AudioBlock::new(n, sr);
        block_high.left = high_input.clone();
        block_high.right = high_input.clone();

        gp.process(&mut block_high);

        // Reset filters for the next test
        for f in gp.band_filters_l.iter_mut() {
            f.reset();
        }
        for f in gp.band_filters_r.iter_mut() {
            f.reset();
        }

        // Test low-freq tone
        let low_input: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 250.0 * i as f32 / sr).sin() * 0.1)
            .collect();

        let mut block_low = AudioBlock::new(n, sr);
        block_low.left = low_input.clone();
        block_low.right = low_input.clone();

        gp.process(&mut block_low);

        // Compute RMS of output for each — skip transient startup (first 128 samples)
        let rms_high: f32 = block_high.left[128..]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            / (n - 128) as f32;
        let rms_high = rms_high.sqrt();

        let rms_low: f32 = block_low.left[128..]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            / (n - 128) as f32;
        let rms_low = rms_low.sqrt();

        // High-frequency loss is greater (60 dB vs 10 dB), so the boost should be larger
        // meaning the high-freq output RMS should be greater relative to input than low-freq
        let input_rms = 0.1 / 2.0_f32.sqrt(); // RMS of sine with amplitude 0.1
        let gain_ratio_high = rms_high / input_rms;
        let gain_ratio_low = rms_low / input_rms;

        assert!(
            gain_ratio_high > gain_ratio_low,
            "High-freq gain ratio ({:.3}) should exceed low-freq ({:.3}) for sloping loss",
            gain_ratio_high,
            gain_ratio_low
        );
    }

    #[test]
    fn nal_r_produces_reasonable_gains() {
        // Typical mild-to-moderate presbycusis
        let audiogram = vec![
            (250.0, 20.0),
            (500.0, 25.0),
            (1000.0, 35.0),
            (2000.0, 50.0),
            (4000.0, 65.0),
            (6000.0, 75.0),
        ];

        let gp = GainProcessor::with_nal_r(&audiogram);

        // Verify NAL-R gain values are in reasonable range
        let three_freq_avg = 25.0 + 35.0 + 50.0; // 110
        let _x = 0.05 * three_freq_avg; // 5.5

        // At 1000 Hz: X + 0.31 * 35 + 0 = 5.5 + 10.85 = 16.35
        let gain_1k = GainProcessor::nal_r_gain(&audiogram, 1000.0);
        assert!(
            (gain_1k - 16.35).abs() < 1.0,
            "NAL-R gain at 1kHz should be ~16.35 dB, got {:.2}",
            gain_1k
        );

        // At 4000 Hz: X + 0.31 * 65 - 1 = 5.5 + 20.15 - 1 = 24.65
        let gain_4k = GainProcessor::nal_r_gain(&audiogram, 4000.0);
        assert!(
            (gain_4k - 24.65).abs() < 1.0,
            "NAL-R gain at 4kHz should be ~24.65 dB, got {:.2}",
            gain_4k
        );

        // Gains should increase with frequency (following the loss pattern)
        let gain_500 = GainProcessor::nal_r_gain(&audiogram, 500.0);
        assert!(
            gain_1k > gain_500,
            "NAL-R gain should increase with frequency for sloping loss. 500Hz={:.1}, 1kHz={:.1}",
            gain_500,
            gain_1k
        );
        assert!(
            gain_4k > gain_1k,
            "NAL-R gain should increase with frequency for sloping loss. 1kHz={:.1}, 4kHz={:.1}",
            gain_1k,
            gain_4k
        );

        // All gains should be positive and under 50 dB for this audiogram
        for freq in [250.0, 500.0, 1000.0, 2000.0, 4000.0, 6000.0] {
            let g = GainProcessor::nal_r_gain(&audiogram, freq);
            assert!(g >= 0.0, "NAL-R gain at {} Hz should be non-negative: {}", freq, g);
            assert!(g < 50.0, "NAL-R gain at {} Hz should be < 50 dB: {}", freq, g);
        }

        // Verify it constructs without panicking and the flag is set
        assert!(gp.use_nal_r);
    }

    #[test]
    fn zero_gain_is_passthrough() {
        let mut gp = GainProcessor::new(0.0);
        gp.prepare(16000.0, 4);

        let input = vec![0.1, -0.2, 0.3, -0.4];
        let mut block = AudioBlock::new(4, 16000.0);
        block.left = input.clone();
        block.right = input.clone();

        gp.process(&mut block);

        assert_eq!(block.left, input, "0 dB gain should be pass-through");
    }
}
