//! Phase-Aware Reconstruction via Griffin-Lim.
//!
//! The Griffin-Lim algorithm iteratively estimates phase from a magnitude
//! spectrogram, producing higher-quality reconstructions than using the
//! original (potentially corrupted) phase after masking.
//!
//! Algorithm:
//! 1. Start with random phase
//! 2. Synthesize time-domain signal (ISTFT)
//! 3. Re-analyze (STFT) — keep target magnitude, update phase
//! 4. Repeat until convergence

use crate::stft::{self, StftResult, TfBin};
use std::f64::consts::PI;

/// Configuration for the Griffin-Lim algorithm.
#[derive(Debug, Clone)]
pub struct GriffinLimConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Stop early if mean magnitude error drops below this threshold.
    pub convergence_tolerance: f64,
}

impl Default for GriffinLimConfig {
    fn default() -> Self {
        Self {
            max_iterations: 32,
            convergence_tolerance: 1e-6,
        }
    }
}

/// Result of Griffin-Lim phase estimation.
#[derive(Debug)]
pub struct GriffinLimResult {
    /// Reconstructed time-domain signal.
    pub signal: Vec<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final mean magnitude reconstruction error.
    pub final_error: f64,
}

/// Simple pseudo-random number generator (xorshift64).
/// Avoids external dependencies for deterministic phase initialization.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 as f64) / (u64::MAX as f64)
    }
}

/// Run the Griffin-Lim algorithm to estimate phase from a magnitude spectrogram.
///
/// - `magnitudes`: magnitude values indexed `[frame * num_freq_bins + freq_bin]`
/// - `num_frames`: number of time frames
/// - `num_freq_bins`: frequency bins per frame (window_size/2 + 1)
/// - `window_size`: FFT window size
/// - `hop_size`: hop size between frames
/// - `sample_rate`: sample rate
/// - `output_len`: desired output signal length
/// - `config`: algorithm parameters
pub fn griffin_lim(
    magnitudes: &[f64],
    num_frames: usize,
    num_freq_bins: usize,
    window_size: usize,
    hop_size: usize,
    sample_rate: f64,
    output_len: usize,
    config: &GriffinLimConfig,
) -> GriffinLimResult {
    assert_eq!(magnitudes.len(), num_frames * num_freq_bins);
    assert!(window_size.is_power_of_two());

    // Initialize with random phase
    let mut rng = Rng::new(42);
    let mut phases: Vec<f64> = (0..magnitudes.len())
        .map(|_| rng.next_f64() * 2.0 * PI - PI)
        .collect();

    let mut signal = vec![0.0; output_len];
    let mut final_error = f64::MAX;
    let mut iterations = 0;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        // Build an StftResult from current magnitudes + phases
        let stft_result = build_stft_result(
            magnitudes,
            &phases,
            num_frames,
            num_freq_bins,
            window_size,
            hop_size,
            sample_rate,
        );

        // ISTFT with all-ones mask to get time-domain signal
        let ones = vec![1.0; magnitudes.len()];
        signal = stft::istft(&stft_result, &ones, output_len);

        // Re-analyze to get updated phases
        let re_analyzed = stft::stft(&signal, window_size, hop_size, sample_rate);

        // Compute error and update phases
        let mut total_error = 0.0;
        let mut count = 0;
        let usable_frames = re_analyzed.num_frames.min(num_frames);
        let usable_bins = re_analyzed.num_freq_bins.min(num_freq_bins);

        for frame in 0..usable_frames {
            for bin in 0..usable_bins {
                let orig_idx = frame * num_freq_bins + bin;
                let re_idx = frame * re_analyzed.num_freq_bins + bin;

                if re_idx < re_analyzed.bins.len() {
                    phases[orig_idx] = re_analyzed.bins[re_idx].phase;
                    let mag_err = magnitudes[orig_idx] - re_analyzed.bins[re_idx].magnitude;
                    total_error += mag_err * mag_err;
                    count += 1;
                }
            }
        }

        final_error = if count > 0 {
            (total_error / count as f64).sqrt()
        } else {
            0.0
        };

        if final_error < config.convergence_tolerance {
            break;
        }
    }

    GriffinLimResult {
        signal,
        iterations,
        final_error,
    }
}

/// Reconstruct a signal using Griffin-Lim phase estimation instead of
/// the original phase from the STFT result.
///
/// This applies the given mask to the STFT magnitudes, then uses
/// Griffin-Lim to find a consistent phase, producing smoother output
/// than using potentially corrupted original phase.
pub fn phase_aware_istft(
    stft_result: &StftResult,
    mask: &[f64],
    output_len: usize,
    config: &GriffinLimConfig,
) -> GriffinLimResult {
    let n = stft_result.num_frames * stft_result.num_freq_bins;
    assert_eq!(mask.len(), n);

    // Extract masked magnitudes
    let magnitudes: Vec<f64> = stft_result
        .bins
        .iter()
        .zip(mask.iter())
        .map(|(bin, &m)| bin.magnitude * m)
        .collect();

    griffin_lim(
        &magnitudes,
        stft_result.num_frames,
        stft_result.num_freq_bins,
        stft_result.window_size,
        stft_result.hop_size,
        stft_result.sample_rate,
        output_len,
        config,
    )
}

/// Build an `StftResult` from separate magnitude and phase arrays.
fn build_stft_result(
    magnitudes: &[f64],
    phases: &[f64],
    num_frames: usize,
    num_freq_bins: usize,
    window_size: usize,
    hop_size: usize,
    sample_rate: f64,
) -> StftResult {
    let bins: Vec<TfBin> = magnitudes
        .iter()
        .zip(phases.iter())
        .enumerate()
        .map(|(i, (&mag, &phase))| TfBin {
            frame: i / num_freq_bins,
            freq_bin: i % num_freq_bins,
            magnitude: mag,
            phase,
        })
        .collect();

    StftResult {
        bins,
        num_frames,
        num_freq_bins,
        hop_size,
        window_size,
        sample_rate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_signal(freq: f64, sample_rate: f64, len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin())
            .collect()
    }

    #[test]
    fn test_convergence() {
        // Error should decrease over iterations.
        let sr = 8000.0;
        let len = 2048;
        let signal = sine_signal(440.0, sr, len);

        let result = stft::stft(&signal, 256, 128, sr);
        let magnitudes: Vec<f64> = result.bins.iter().map(|b| b.magnitude).collect();

        let few = griffin_lim(
            &magnitudes,
            result.num_frames,
            result.num_freq_bins,
            256,
            128,
            sr,
            len,
            &GriffinLimConfig {
                max_iterations: 2,
                convergence_tolerance: 0.0,
            },
        );

        let many = griffin_lim(
            &magnitudes,
            result.num_frames,
            result.num_freq_bins,
            256,
            128,
            sr,
            len,
            &GriffinLimConfig {
                max_iterations: 32,
                convergence_tolerance: 0.0,
            },
        );

        assert!(
            many.final_error <= few.final_error + 1e-10,
            "More iterations should reduce error: {} (32 iter) vs {} (2 iter)",
            many.final_error,
            few.final_error
        );
    }

    #[test]
    fn test_roundtrip_quality() {
        // Griffin-Lim from a known STFT should reconstruct a signal whose
        // dominant frequency matches the original.
        let sr = 8000.0;
        let len = 4096;
        let freq = 440.0;
        let signal = sine_signal(freq, sr, len);

        let result = stft::stft(&signal, 256, 128, sr);
        let mask = vec![1.0; result.bins.len()];

        let gl = phase_aware_istft(
            &result,
            &mask,
            len,
            &GriffinLimConfig {
                max_iterations: 50,
                convergence_tolerance: 1e-8,
            },
        );

        assert!(gl.iterations > 0);

        // Verify the dominant frequency is preserved by re-analyzing
        let re = stft::stft(&gl.signal, 256, 128, sr);
        let mut freq_energy = vec![0.0f64; re.num_freq_bins];
        for bin in &re.bins {
            freq_energy[bin.freq_bin] += bin.magnitude * bin.magnitude;
        }
        let peak_bin = freq_energy
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let peak_freq = peak_bin as f64 * sr / 256.0;
        let freq_err = (peak_freq - freq).abs();

        assert!(
            freq_err < sr / 256.0 * 2.0,
            "Peak frequency {peak_freq:.1} Hz too far from {freq} Hz"
        );
    }

    #[test]
    fn test_known_signal_reconstruction() {
        // A DC-like low-frequency signal should be well-reconstructed.
        let sr = 8000.0;
        let len = 4096;
        let freq = 100.0;
        let signal = sine_signal(freq, sr, len);

        let result = stft::stft(&signal, 512, 256, sr);
        let magnitudes: Vec<f64> = result.bins.iter().map(|b| b.magnitude).collect();

        let gl = griffin_lim(
            &magnitudes,
            result.num_frames,
            result.num_freq_bins,
            512,
            256,
            sr,
            len,
            &GriffinLimConfig {
                max_iterations: 50,
                convergence_tolerance: 1e-8,
            },
        );

        // Verify the dominant frequency is correct by finding peak in re-analyzed STFT
        let re = stft::stft(&gl.signal, 512, 256, sr);
        let mut freq_energy = vec![0.0f64; re.num_freq_bins];
        for bin in &re.bins {
            freq_energy[bin.freq_bin] += bin.magnitude * bin.magnitude;
        }
        let peak_bin = freq_energy
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        let peak_freq = peak_bin as f64 * sr / 512.0;
        let freq_err = (peak_freq - freq).abs();

        assert!(
            freq_err < sr / 512.0 * 2.0, // within 2 bins
            "Peak frequency {peak_freq:.1} Hz too far from {freq} Hz (err={freq_err:.1})"
        );
    }
}
