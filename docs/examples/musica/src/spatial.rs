//! Spatial covariance model for stereo/multichannel source separation.
//!
//! Uses inter-channel phase difference (IPD) and inter-channel level difference
//! (ILD) to build spatial masks. Combined with the graph-based separator for
//! joint spectro-spatial separation.
//!
//! Key equations:
//! - IPD(f,t) = angle(X_L(f,t) * conj(X_R(f,t)))
//! - ILD(f,t) = 20*log10(|X_L(f,t)| / |X_R(f,t)|)
//! - Spatial mask: M(f,t) = exp(-IPD^2/(2*sigma_ipd^2)) * sigmoid(ILD/sigma_ild)

use crate::stft::{self, StftResult};
use std::f64::consts::PI;

/// Configuration for spatial separation.
#[derive(Debug, Clone)]
pub struct SpatialConfig {
    /// Expected source directions in degrees (-90 to +90).
    pub source_directions: Vec<f64>,
    /// IPD bandwidth parameter (radians).
    pub ipd_sigma: f64,
    /// ILD bandwidth parameter (dB).
    pub ild_sigma: f64,
    /// Window size for STFT.
    pub window_size: usize,
    /// Hop size for STFT.
    pub hop_size: usize,
    /// Sample rate.
    pub sample_rate: f64,
}

impl Default for SpatialConfig {
    fn default() -> Self {
        Self {
            source_directions: vec![-30.0, 30.0],
            ipd_sigma: 0.5,
            ild_sigma: 6.0,
            window_size: 512,
            hop_size: 256,
            sample_rate: 16000.0,
        }
    }
}

/// Result from spatial separation.
#[derive(Debug, Clone)]
pub struct SpatialResult {
    /// Separated mono sources.
    pub sources: Vec<Vec<f64>>,
    /// Per-source spatial masks.
    pub masks: Vec<Vec<f64>>,
    /// IPD map (frames x freq_bins).
    pub ipd_map: Vec<f64>,
    /// ILD map (frames x freq_bins).
    pub ild_map: Vec<f64>,
    /// Processing time in ms.
    pub processing_ms: f64,
}

/// Compute inter-channel phase difference between left and right STFT.
fn compute_ipd(left: &StftResult, right: &StftResult) -> Vec<f64> {
    let total = left.num_frames * left.num_freq_bins;
    let mut ipd = vec![0.0; total];

    for i in 0..total {
        // IPD = phase_left - phase_right, wrapped to [-pi, pi]
        let mut diff = left.bins[i].phase - right.bins[i].phase;
        while diff > PI { diff -= 2.0 * PI; }
        while diff < -PI { diff += 2.0 * PI; }
        ipd[i] = diff;
    }
    ipd
}

/// Compute inter-channel level difference between left and right STFT.
fn compute_ild(left: &StftResult, right: &StftResult) -> Vec<f64> {
    let total = left.num_frames * left.num_freq_bins;
    let mut ild = vec![0.0; total];

    for i in 0..total {
        let mag_l = left.bins[i].magnitude.max(1e-12);
        let mag_r = right.bins[i].magnitude.max(1e-12);
        ild[i] = 20.0 * (mag_l / mag_r).log10();
    }
    ild
}

/// Expected IPD for a source at a given direction and frequency.
/// Based on the far-field model: IPD = 2*pi*f*d*sin(theta)/c
/// where d = microphone spacing (~0.15m for headphones), c = 343 m/s.
fn expected_ipd(direction_deg: f64, freq_hz: f64, mic_spacing: f64) -> f64 {
    let theta = direction_deg * PI / 180.0;
    let c = 343.0; // speed of sound
    let ipd = 2.0 * PI * freq_hz * mic_spacing * theta.sin() / c;
    // Wrap to [-pi, pi]
    let mut wrapped = ipd % (2.0 * PI);
    if wrapped > PI { wrapped -= 2.0 * PI; }
    if wrapped < -PI { wrapped += 2.0 * PI; }
    wrapped
}

/// Expected ILD for a source at a given direction.
/// Simple model: ILD ~ 8 * sin(theta) dB at high frequencies.
fn expected_ild(direction_deg: f64) -> f64 {
    let theta = direction_deg * PI / 180.0;
    8.0 * theta.sin()
}

/// Separate stereo signal using spatial cues.
pub fn spatial_separate(
    left: &[f64],
    right: &[f64],
    config: &SpatialConfig,
) -> SpatialResult {
    let start = std::time::Instant::now();
    let n = left.len().min(right.len());
    let num_sources = config.source_directions.len();

    let left_stft = stft::stft(left, config.window_size, config.hop_size, config.sample_rate);
    let right_stft = stft::stft(right, config.window_size, config.hop_size, config.sample_rate);

    let ipd_map = compute_ipd(&left_stft, &right_stft);
    let ild_map = compute_ild(&left_stft, &right_stft);

    let total_tf = left_stft.num_frames * left_stft.num_freq_bins;
    let mic_spacing = 0.15; // 15cm typical head width

    // Compute spatial masks for each source
    let mut masks: Vec<Vec<f64>> = vec![vec![0.0; total_tf]; num_sources];

    for s in 0..num_sources {
        let dir = config.source_directions[s];
        let expected_ild_val = expected_ild(dir);

        for i in 0..total_tf {
            let freq_bin = i % left_stft.num_freq_bins;
            let freq_hz = freq_bin as f64 * config.sample_rate / (config.window_size as f64);

            let expected_ipd_val = expected_ipd(dir, freq_hz, mic_spacing);

            // IPD likelihood: Gaussian around expected IPD
            let ipd_diff = ipd_map[i] - expected_ipd_val;
            let ipd_score = (-ipd_diff * ipd_diff / (2.0 * config.ipd_sigma * config.ipd_sigma)).exp();

            // ILD likelihood: sigmoid around expected ILD
            let ild_diff = ild_map[i] - expected_ild_val;
            let ild_score = 1.0 / (1.0 + (-ild_diff / config.ild_sigma).exp());
            // Symmetric: closer to expected ILD = higher score
            let ild_proximity = (-ild_diff.abs() / config.ild_sigma).exp();

            masks[s][i] = ipd_score * ild_proximity;
        }
    }

    // Normalize masks to sum to 1
    for i in 0..total_tf {
        let sum: f64 = (0..num_sources).map(|s| masks[s][i]).sum();
        if sum > 1e-12 {
            for s in 0..num_sources {
                masks[s][i] /= sum;
            }
        } else {
            for s in 0..num_sources {
                masks[s][i] = 1.0 / num_sources as f64;
            }
        }
    }

    // Reconstruct mono sources from left+right average
    let mono_stft_bins: Vec<_> = (0..total_tf)
        .map(|i| crate::stft::TfBin {
            frame: left_stft.bins[i].frame,
            freq_bin: left_stft.bins[i].freq_bin,
            magnitude: (left_stft.bins[i].magnitude + right_stft.bins[i].magnitude) / 2.0,
            phase: left_stft.bins[i].phase, // Use left channel phase
        })
        .collect();

    let mono_stft = StftResult {
        bins: mono_stft_bins,
        num_frames: left_stft.num_frames,
        num_freq_bins: left_stft.num_freq_bins,
        hop_size: left_stft.hop_size,
        window_size: left_stft.window_size,
        sample_rate: left_stft.sample_rate,
    };

    let sources: Vec<Vec<f64>> = masks.iter()
        .map(|mask| stft::istft(&mono_stft, mask, n))
        .collect();

    let processing_ms = start.elapsed().as_secs_f64() * 1000.0;

    SpatialResult {
        sources,
        masks,
        ipd_map,
        ild_map,
        processing_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_separate_basic() {
        let sr = 16000.0;
        let n = 4000; // 250ms
        let config = SpatialConfig {
            source_directions: vec![-30.0, 30.0],
            sample_rate: sr,
            window_size: 512,
            hop_size: 256,
            ..SpatialConfig::default()
        };

        // Speech from left (-30 deg): louder in left ear
        let speech: Vec<f64> = (0..n).map(|i| {
            let t = i as f64 / sr;
            0.5 * (2.0 * PI * 200.0 * t).sin() + 0.2 * (2.0 * PI * 400.0 * t).sin()
        }).collect();

        // Noise from right (+30 deg): louder in right ear
        let noise: Vec<f64> = (0..n).map(|i| {
            let t = i as f64 / sr;
            0.3 * (2.0 * PI * 1500.0 * t).sin()
        }).collect();

        let left: Vec<f64> = speech.iter().zip(noise.iter())
            .map(|(s, n)| s * 1.2 + n * 0.5).collect();
        let right: Vec<f64> = speech.iter().zip(noise.iter())
            .map(|(s, n)| s * 0.5 + n * 1.2).collect();

        let result = spatial_separate(&left, &right, &config);

        assert_eq!(result.sources.len(), 2);
        assert_eq!(result.sources[0].len(), n);
        assert!(result.processing_ms > 0.0);

        // Masks should sum to ~1
        let total = result.masks[0].len();
        for i in 0..total.min(100) {
            let sum: f64 = result.masks.iter().map(|m| m[i]).sum();
            assert!((sum - 1.0).abs() < 0.01, "Mask sum = {sum} at {i}");
        }
    }

    #[test]
    fn test_ipd_computation() {
        // Same signal in both channels -> IPD should be near 0
        let signal: Vec<f64> = (0..2000).map(|i| (i as f64 * 0.01).sin()).collect();
        let left = stft::stft(&signal, 256, 128, 8000.0);
        let right = stft::stft(&signal, 256, 128, 8000.0);

        let ipd = compute_ipd(&left, &right);
        let max_ipd = ipd.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        assert!(max_ipd < 0.01, "Same signal IPD should be ~0, got {max_ipd}");
    }

    #[test]
    fn test_expected_ipd_symmetry() {
        let freq = 1000.0;
        let spacing = 0.15;
        let ipd_left = expected_ipd(-30.0, freq, spacing);
        let ipd_right = expected_ipd(30.0, freq, spacing);
        assert!((ipd_left + ipd_right).abs() < 0.01,
            "IPD should be antisymmetric: {ipd_left} vs {ipd_right}");
    }
}
