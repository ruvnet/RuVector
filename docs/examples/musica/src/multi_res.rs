//! Multi-Resolution STFT for improved transient and tonal separation.
//!
//! Different frequency ranges benefit from different window sizes:
//! - Short windows (256) capture transients (drums, percussive attacks)
//! - Medium windows (1024) balance time/frequency resolution for mid-range
//! - Long windows (4096) resolve tonal content (bass, sustained vocals)
//!
//! This module runs multiple STFTs in parallel and merges the results.

use crate::stft::{self, StftResult, TfBin};

/// Frequency band definition for multi-resolution analysis.
#[derive(Debug, Clone)]
pub struct BandConfig {
    /// Lower frequency bound (Hz).
    pub freq_lo: f64,
    /// Upper frequency bound (Hz).
    pub freq_hi: f64,
    /// FFT window size for this band (must be power of 2).
    pub window_size: usize,
    /// Hop size for this band.
    pub hop_size: usize,
}

/// Configuration for multi-resolution STFT.
#[derive(Debug, Clone)]
pub struct MultiResConfig {
    /// Per-band configurations ordered low-to-high.
    pub bands: Vec<BandConfig>,
    /// Sample rate.
    pub sample_rate: f64,
}

impl Default for MultiResConfig {
    fn default() -> Self {
        Self {
            bands: vec![
                BandConfig {
                    freq_lo: 0.0,
                    freq_hi: 500.0,
                    window_size: 4096,
                    hop_size: 2048,
                },
                BandConfig {
                    freq_lo: 500.0,
                    freq_hi: 4000.0,
                    window_size: 1024,
                    hop_size: 512,
                },
                BandConfig {
                    freq_lo: 4000.0,
                    freq_hi: 22050.0,
                    window_size: 256,
                    hop_size: 128,
                },
            ],
            sample_rate: 44100.0,
        }
    }
}

/// STFT result for a single frequency band.
pub struct BandResult {
    /// The underlying STFT result.
    pub stft: StftResult,
    /// Lower frequency bound (Hz).
    pub freq_lo: f64,
    /// Upper frequency bound (Hz).
    pub freq_hi: f64,
    /// Starting frequency bin index (inclusive) within this STFT.
    pub bin_lo: usize,
    /// Ending frequency bin index (exclusive) within this STFT.
    pub bin_hi: usize,
}

/// Complete multi-resolution STFT result.
pub struct MultiResResult {
    /// Per-band results.
    pub bands: Vec<BandResult>,
    /// Sample rate.
    pub sample_rate: f64,
    /// Original signal length.
    pub signal_len: usize,
}

/// Convert a frequency in Hz to an FFT bin index.
fn freq_to_bin(freq: f64, window_size: usize, sample_rate: f64) -> usize {
    let bin = (freq * window_size as f64 / sample_rate).round() as usize;
    bin.min(window_size / 2)
}

/// Perform multi-resolution STFT on a signal.
///
/// Runs a separate STFT for each configured band and tags
/// each result with the relevant frequency bin range.
pub fn multi_res_stft(signal: &[f64], config: &MultiResConfig) -> MultiResResult {
    let mut bands = Vec::with_capacity(config.bands.len());

    for band in &config.bands {
        assert!(
            band.window_size.is_power_of_two(),
            "window_size must be power of 2"
        );
        let result = stft::stft(signal, band.window_size, band.hop_size, config.sample_rate);

        let bin_lo = freq_to_bin(band.freq_lo, band.window_size, config.sample_rate);
        let bin_hi = freq_to_bin(band.freq_hi, band.window_size, config.sample_rate)
            .min(result.num_freq_bins);

        bands.push(BandResult {
            stft: result,
            freq_lo: band.freq_lo,
            freq_hi: band.freq_hi,
            bin_lo,
            bin_hi,
        });
    }

    MultiResResult {
        bands,
        sample_rate: config.sample_rate,
        signal_len: signal.len(),
    }
}

/// Merge per-band masks from different resolutions into a single unified mask.
///
/// Each element of `band_masks` corresponds to a `BandResult` in the
/// `MultiResResult` and contains mask values in `[0, 1]` for every bin
/// in that band's STFT (full size, including out-of-band bins).
///
/// The output is a unified mask at the resolution of the **first** band
/// (typically the longest window / finest frequency resolution). For each
/// output bin we find which band owns that frequency and interpolate the
/// nearest mask value from that band's time grid.
///
/// `target_window_size` and `target_hop_size` define the output resolution.
pub fn merge_multi_res_masks(
    multi_res: &MultiResResult,
    band_masks: &[Vec<f64>],
    target_window_size: usize,
    target_hop_size: usize,
) -> Vec<f64> {
    assert_eq!(multi_res.bands.len(), band_masks.len());

    let target_num_freq = target_window_size / 2 + 1;
    let num_target_frames =
        if multi_res.signal_len >= target_window_size {
            (multi_res.signal_len - target_window_size) / target_hop_size + 1
        } else {
            0
        };

    let mut merged = vec![0.0; num_target_frames * target_num_freq];

    for target_frame in 0..num_target_frames {
        let target_time_sample = target_frame * target_hop_size;

        for target_bin in 0..target_num_freq {
            let freq_hz =
                target_bin as f64 * multi_res.sample_rate / target_window_size as f64;

            // Find owning band
            let band_idx = multi_res
                .bands
                .iter()
                .position(|b| freq_hz >= b.freq_lo && freq_hz < b.freq_hi)
                .unwrap_or_else(|| {
                    // If beyond all bands, use the last one
                    multi_res.bands.len() - 1
                });

            let band = &multi_res.bands[band_idx];
            let mask = &band_masks[band_idx];
            let band_stft = &band.stft;

            // Map target_bin to this band's bin index
            let band_bin = freq_to_bin(freq_hz, band_stft.window_size, multi_res.sample_rate)
                .min(band_stft.num_freq_bins - 1);

            // Map target time to this band's frame index (nearest)
            let band_frame = if band_stft.hop_size > 0 && band_stft.num_frames > 0 {
                let f = target_time_sample as f64 / band_stft.hop_size as f64;
                (f.round() as usize).min(band_stft.num_frames - 1)
            } else {
                0
            };

            let mask_idx = band_frame * band_stft.num_freq_bins + band_bin;
            let val = if mask_idx < mask.len() {
                mask[mask_idx]
            } else {
                1.0
            };

            merged[target_frame * target_num_freq + target_bin] = val;
        }
    }

    merged
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn sine_signal(freq: f64, sample_rate: f64, len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin())
            .collect()
    }

    #[test]
    fn test_multi_res_roundtrip_consistency() {
        // A pure tone should appear in the correct band with consistent energy.
        let sr = 44100.0;
        let len = 16384;
        let signal = sine_signal(440.0, sr, len);

        let config = MultiResConfig::default();
        let result = multi_res_stft(&signal, &config);

        assert_eq!(result.bands.len(), 3);

        // 440 Hz should be in the low band (0-500 Hz)
        let low = &result.bands[0];
        assert!(low.freq_lo <= 440.0 && low.freq_hi >= 440.0);

        // Compute total energy in the low band's relevant bins
        let mut energy_in_band = 0.0;
        let mut energy_total = 0.0;
        for bin in &low.stft.bins {
            let e = bin.magnitude * bin.magnitude;
            energy_total += e;
            if bin.freq_bin >= low.bin_lo && bin.freq_bin < low.bin_hi {
                energy_in_band += e;
            }
        }
        // Most energy should be within the band
        assert!(
            energy_in_band / energy_total > 0.9,
            "Expected >90% energy in band, got {:.1}%",
            100.0 * energy_in_band / energy_total
        );
    }

    #[test]
    fn test_transient_detection_improvement() {
        // Short windows should give better time resolution for a click.
        let sr = 44100.0;
        let len = 16384;
        let mut signal = vec![0.0; len];
        // Insert a sharp click at sample 8192
        signal[8192] = 1.0;

        // Short-window STFT (better transient localization)
        let short = stft::stft(&signal, 256, 128, sr);
        // Long-window STFT (worse transient localization)
        let long = stft::stft(&signal, 4096, 2048, sr);

        // Find the frame with max energy in each
        let max_frame_energy = |res: &StftResult| -> (usize, f64) {
            let mut frame_energy = vec![0.0f64; res.num_frames];
            for bin in &res.bins {
                frame_energy[bin.frame] += bin.magnitude * bin.magnitude;
            }
            frame_energy
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, &e)| (i, e))
                .unwrap()
        };

        let (short_peak_frame, _) = max_frame_energy(&short);
        let (long_peak_frame, _) = max_frame_energy(&long);

        // Short window should localize the click to a narrower time range.
        // The time resolution of the short window is hop_size/sr.
        let short_time = short_peak_frame as f64 * 128.0 / sr;
        let long_time = long_peak_frame as f64 * 2048.0 / sr;
        let click_time = 8192.0 / sr;

        let short_err = (short_time - click_time).abs();
        let long_err = (long_time - click_time).abs();

        // Short window error should be smaller (better localization)
        assert!(
            short_err <= long_err + 1e-6,
            "Short window err {short_err:.4}s should be <= long window err {long_err:.4}s"
        );
    }

    #[test]
    fn test_band_merging() {
        let sr = 8000.0;
        let len = 8192;
        let signal = sine_signal(440.0, sr, len);

        let config = MultiResConfig {
            bands: vec![
                BandConfig {
                    freq_lo: 0.0,
                    freq_hi: 1000.0,
                    window_size: 1024,
                    hop_size: 512,
                },
                BandConfig {
                    freq_lo: 1000.0,
                    freq_hi: 4000.0,
                    window_size: 256,
                    hop_size: 128,
                },
            ],
            sample_rate: sr,
        };
        let result = multi_res_stft(&signal, &config);

        // Create all-ones masks for both bands
        let masks: Vec<Vec<f64>> = result
            .bands
            .iter()
            .map(|b| vec![1.0; b.stft.bins.len()])
            .collect();

        let target_win = 512;
        let target_hop = 256;
        let merged = merge_multi_res_masks(&result, &masks, target_win, target_hop);

        let target_num_freq = target_win / 2 + 1;
        let expected_frames = (len - target_win) / target_hop + 1;

        assert_eq!(merged.len(), expected_frames * target_num_freq);
        // All masks were 1.0, so merged should be all 1.0
        for &v in &merged {
            assert!(
                (v - 1.0).abs() < 1e-10,
                "Merged mask should be 1.0 for all-ones input"
            );
        }
    }
}
