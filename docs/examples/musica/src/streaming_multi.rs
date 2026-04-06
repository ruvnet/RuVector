//! Streaming multitrack 6-stem separation.
//!
//! Combines the frame-by-frame streaming approach from `hearing_aid` with the
//! 6-stem separation logic from `multitrack`. Audio is processed one frame at
//! a time while maintaining rolling state for temporal coherence.
//!
//! Pipeline per frame:
//! 1. Append to rolling buffer (keep last `num_rolling_frames` frames)
//! 2. Run STFT on rolling buffer
//! 3. For each stem: compute frequency prior score + transient/harmonic features
//! 4. Build per-stem graph over rolling window
//! 5. Compute Fiedler vector for coherence grouping
//! 6. Generate soft masks with Wiener normalization (sum to 1.0)
//! 7. Apply temporal smoothing with previous frame's masks (EMA)
//! 8. Return per-stem masks for current frame only

use crate::multitrack::{default_stem_priors, Stem, StemPrior};
use crate::stft;
use std::collections::HashMap;

/// Configuration for streaming multitrack separation.
#[derive(Debug, Clone)]
pub struct StreamingMultiConfig {
    /// STFT window size in samples.
    pub window_size: usize,
    /// STFT hop size in samples.
    pub hop_size: usize,
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Number of rolling frames to keep for temporal context.
    pub num_rolling_frames: usize,
    /// Temporal mask smoothing factor (0 = no smoothing, 1 = frozen).
    pub mask_smoothing: f64,
}

impl Default for StreamingMultiConfig {
    fn default() -> Self {
        Self {
            window_size: 2048,
            hop_size: 512,
            sample_rate: 44100.0,
            num_rolling_frames: 4,
            mask_smoothing: 0.3,
        }
    }
}

/// Per-stem mask data for a single frame.
#[derive(Debug, Clone)]
pub struct StemFrame {
    /// Stem type.
    pub stem: Stem,
    /// Soft mask values for frequency bins of the current frame.
    pub mask: Vec<f64>,
    /// Confidence score (average mask value in the stem's primary frequency range).
    pub confidence: f64,
}

/// Result of processing one audio frame across all 6 stems.
#[derive(Debug, Clone)]
pub struct MultiFrame {
    /// Per-stem frame data.
    pub stems: Vec<StemFrame>,
    /// Processing latency in microseconds.
    pub latency_us: u64,
}

/// Rolling state for streaming multitrack separation.
pub struct StreamingMultiState {
    /// Rolling buffer of audio frames (each frame = `window_size` samples).
    rolling_buffer: Vec<Vec<f64>>,
    /// Previous masks per stem, indexed [stem_idx][freq_bin].
    prev_masks: Vec<Vec<f64>>,
    /// Frame counter.
    pub frame_count: u64,
    /// Cached stem priors.
    priors: Vec<(Stem, StemPrior)>,
    /// Accumulated output samples per stem for reconstruction.
    accumulated: Vec<Vec<f64>>,
}

impl StreamingMultiState {
    /// Create a new streaming state from the given config.
    pub fn new(config: &StreamingMultiConfig) -> Self {
        let num_freq = config.window_size / 2 + 1;
        let priors = default_stem_priors();
        let num_stems = priors.len();

        Self {
            rolling_buffer: Vec::new(),
            prev_masks: vec![vec![0.0; num_freq]; num_stems],
            frame_count: 0,
            priors,
            accumulated: vec![Vec::new(); num_stems],
        }
    }

    /// Process a single audio frame and return per-stem masks.
    ///
    /// `samples` should be one hop's worth of audio (config.hop_size samples).
    /// Internally the rolling buffer accumulates enough context for STFT analysis.
    pub fn process_frame(&mut self, samples: &[f64], config: &StreamingMultiConfig) -> MultiFrame {
        let start = std::time::Instant::now();
        let num_freq = config.window_size / 2 + 1;

        // 1. Append to rolling buffer
        self.rolling_buffer.push(samples.to_vec());
        if self.rolling_buffer.len() > config.num_rolling_frames {
            self.rolling_buffer.remove(0);
        }

        // Flatten rolling buffer into a contiguous signal for STFT
        let rolling_signal: Vec<f64> = self.rolling_buffer.iter().flat_map(|f| f.iter().copied()).collect();

        // 2. Run STFT on rolling buffer
        let stft_result = stft::stft(&rolling_signal, config.window_size, config.hop_size, config.sample_rate);
        let num_frames = stft_result.num_frames;
        let stft_num_freq = stft_result.num_freq_bins;
        let total_bins = num_frames * stft_num_freq;

        // Extract magnitudes
        let magnitudes: Vec<f64> = stft_result.bins.iter().map(|b| b.magnitude).collect();

        // 3. Compute transient and harmonic features
        let transient_scores = compute_transient_scores(&magnitudes, num_frames, stft_num_freq);
        let harmonicity_scores = compute_harmonicity_scores(&magnitudes, num_frames, stft_num_freq);

        // Build raw masks per stem
        let mut raw_masks: Vec<Vec<f64>> = Vec::with_capacity(self.priors.len());

        for (_stem, prior) in &self.priors {
            let freq_bin_min = freq_to_bin(prior.freq_range.0, config.sample_rate, config.window_size);
            let freq_bin_max = freq_to_bin(prior.freq_range.1, config.sample_rate, config.window_size);

            let mut mask = vec![0.0f64; total_bins];

            // Frequency prior scoring
            for frame in 0..num_frames {
                for f in 0..stft_num_freq {
                    let idx = frame * stft_num_freq + f;
                    if f >= freq_bin_min && f <= freq_bin_max {
                        mask[idx] = 1.0;
                    } else {
                        let dist = if f < freq_bin_min {
                            (freq_bin_min - f) as f64
                        } else {
                            (f - freq_bin_max) as f64
                        };
                        mask[idx] = (-dist / 10.0).exp();
                    }
                }
            }

            // Weight by harmonic/transient character
            for idx in 0..total_bins {
                let h_weight = harmonicity_scores[idx] * prior.harmonic_strength;
                let t_weight = transient_scores[idx] * prior.transient_weight;
                mask[idx] *= (1.0 + h_weight + t_weight) / 2.0;
            }

            // 4. Build per-stem graph over rolling window
            let window_bins = collect_active_bins(
                &magnitudes, num_frames, stft_num_freq, freq_bin_min, freq_bin_max,
            );

            if window_bins.len() >= 4 {
                let (edges, num_nodes) = build_stem_graph(
                    &window_bins, &magnitudes, &harmonicity_scores, &transient_scores,
                    stft_num_freq, prior,
                );

                // 5. Compute Fiedler vector for coherence grouping
                if num_nodes > 2 && !edges.is_empty() {
                    let fiedler = compute_fiedler(num_nodes, &edges);
                    let median = {
                        let mut sorted = fiedler.clone();
                        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        sorted[sorted.len() / 2]
                    };

                    for (local_i, &(frame, freq)) in window_bins.iter().enumerate() {
                        let idx = frame * stft_num_freq + freq;
                        let fiedler_val = if local_i < fiedler.len() { fiedler[local_i] } else { 0.0 };
                        let boost = if fiedler_val > median { 1.2 } else { 0.8 };
                        mask[idx] *= boost;
                    }
                }
            }

            raw_masks.push(mask);
        }

        // 6. Wiener normalization: masks sum to 1.0 per T-F bin
        let normalized = wiener_normalize(&raw_masks, &magnitudes, total_bins);

        // Extract only the last frame's frequency bins
        let last_frame = if num_frames > 0 { num_frames - 1 } else { 0 };
        let frame_offset = last_frame * stft_num_freq;

        let mut stem_frames = Vec::with_capacity(self.priors.len());
        for (s, (stem, prior)) in self.priors.iter().enumerate() {
            let mut frame_mask = vec![0.0; num_freq];
            for f in 0..num_freq.min(stft_num_freq) {
                let src = frame_offset + f;
                frame_mask[f] = if src < normalized[s].len() { normalized[s][src] } else { 0.0 };
            }

            // 7. Temporal smoothing (EMA with previous frame's mask)
            let alpha = config.mask_smoothing;
            for f in 0..num_freq {
                frame_mask[f] = alpha * self.prev_masks[s][f] + (1.0 - alpha) * frame_mask[f];
            }

            // Re-normalize after smoothing to maintain sum-to-1 property
            // (done below after collecting all stems)

            // Confidence: average mask in primary frequency range
            let freq_bin_min = freq_to_bin(prior.freq_range.0, config.sample_rate, config.window_size);
            let freq_bin_max = freq_to_bin(prior.freq_range.1, config.sample_rate, config.window_size);
            let range_len = (freq_bin_max - freq_bin_min + 1).max(1);
            let confidence: f64 = (freq_bin_min..=freq_bin_max.min(num_freq - 1))
                .map(|f| frame_mask[f])
                .sum::<f64>()
                / range_len as f64;

            self.prev_masks[s] = frame_mask.clone();

            stem_frames.push(StemFrame {
                stem: *stem,
                mask: frame_mask,
                confidence,
            });
        }

        // Re-normalize smoothed masks so they sum to 1.0 per bin
        for f in 0..num_freq {
            let sum: f64 = stem_frames.iter().map(|sf| sf.mask[f]).sum();
            if sum > 1e-10 {
                for sf in stem_frames.iter_mut() {
                    sf.mask[f] /= sum;
                }
            }
        }

        // Update prev_masks after renormalization
        for (s, sf) in stem_frames.iter().enumerate() {
            self.prev_masks[s] = sf.mask.clone();
        }

        // 8. Accumulate per-stem audio (apply mask to last hop of input via simple gain)
        for (s, sf) in stem_frames.iter().enumerate() {
            let avg_gain: f64 = sf.mask.iter().sum::<f64>() / num_freq.max(1) as f64;
            let stem_samples: Vec<f64> = samples.iter().map(|&x| x * avg_gain).collect();
            self.accumulated[s].extend_from_slice(&stem_samples);
        }

        self.frame_count += 1;
        let latency_us = start.elapsed().as_micros() as u64;

        MultiFrame {
            stems: stem_frames,
            latency_us,
        }
    }

    /// Reconstruct accumulated audio per stem from all processed frames.
    pub fn get_accumulated_stems(&self) -> Vec<(Stem, Vec<f64>)> {
        self.priors
            .iter()
            .enumerate()
            .map(|(i, (stem, _))| (*stem, self.accumulated[i].clone()))
            .collect()
    }
}

// ── Internal helpers ────────────────────────────────────────────────────

fn freq_to_bin(freq_hz: f64, sample_rate: f64, window_size: usize) -> usize {
    let bin = (freq_hz * window_size as f64 / sample_rate).round() as usize;
    bin.min(window_size / 2)
}

fn compute_transient_scores(magnitudes: &[f64], num_frames: usize, num_freq: usize) -> Vec<f64> {
    let mut scores = vec![0.0; magnitudes.len()];
    for f in 0..num_freq {
        for frame in 1..num_frames {
            let curr = magnitudes[frame * num_freq + f];
            let prev = magnitudes[(frame - 1) * num_freq + f];
            let diff = (curr - prev).max(0.0);
            scores[frame * num_freq + f] = (diff / (prev + 1e-8)).min(1.0);
        }
    }
    scores
}

fn compute_harmonicity_scores(magnitudes: &[f64], num_frames: usize, num_freq: usize) -> Vec<f64> {
    let mut scores = vec![0.0; magnitudes.len()];
    for frame in 0..num_frames {
        for f in 1..num_freq / 4 {
            let base = frame * num_freq;
            let fund = magnitudes[base + f];
            if fund < 1e-6 {
                continue;
            }
            let mut harmonic_energy = 0.0;
            let mut count = 0;
            for h in [2, 3, 4] {
                let hf = f * h;
                if hf < num_freq {
                    harmonic_energy += magnitudes[base + hf];
                    count += 1;
                }
            }
            if count > 0 {
                let ratio = harmonic_energy / (count as f64 * fund);
                scores[base + f] = ratio.min(1.0);
                for h in [2, 3, 4] {
                    let hf = f * h;
                    if hf < num_freq {
                        scores[base + hf] = scores[base + hf].max(ratio * 0.5);
                    }
                }
            }
        }
    }
    scores
}

fn collect_active_bins(
    magnitudes: &[f64],
    num_frames: usize,
    num_freq: usize,
    freq_min: usize,
    freq_max: usize,
) -> Vec<(usize, usize)> {
    let mut bins = Vec::new();
    let threshold = 0.001;
    for frame in 0..num_frames {
        for f in freq_min..=freq_max.min(num_freq - 1) {
            let idx = frame * num_freq + f;
            if idx < magnitudes.len() && magnitudes[idx] > threshold {
                bins.push((frame, f));
            }
        }
    }
    bins
}

fn build_stem_graph(
    bins: &[(usize, usize)],
    magnitudes: &[f64],
    harmonicity: &[f64],
    transients: &[f64],
    num_freq: usize,
    prior: &StemPrior,
) -> (Vec<(usize, usize, f64)>, usize) {
    let n = bins.len();
    let mut edges = Vec::new();
    let bin_map: HashMap<(usize, usize), usize> =
        bins.iter().enumerate().map(|(i, &b)| (b, i)).collect();

    for (i, &(frame_i, freq_i)) in bins.iter().enumerate() {
        let idx_i = frame_i * num_freq + freq_i;

        // Spectral neighbor
        if let Some(&j) = bin_map.get(&(frame_i, freq_i + 1)) {
            let idx_j = frame_i * num_freq + freq_i + 1;
            let w = (magnitudes[idx_i] * magnitudes[idx_j]).sqrt() * 0.5;
            if w > 1e-4 {
                edges.push((i, j, w));
            }
        }

        // Temporal neighbor
        if let Some(&j) = bin_map.get(&(frame_i + 1, freq_i)) {
            let idx_j = (frame_i + 1) * num_freq + freq_i;
            if idx_j < magnitudes.len() {
                let w = (magnitudes[idx_i] * magnitudes[idx_j]).sqrt() * prior.temporal_smoothness;
                if w > 1e-4 {
                    edges.push((i, j, w));
                }
            }
        }

        // Harmonic neighbors
        for h in [2, 3] {
            let hf = freq_i * h;
            if let Some(&j) = bin_map.get(&(frame_i, hf)) {
                let idx_j = frame_i * num_freq + hf;
                if idx_j < harmonicity.len() {
                    let w = (harmonicity[idx_i] * harmonicity[idx_j]).sqrt()
                        * prior.harmonic_strength
                        * 0.3;
                    if w > 1e-4 {
                        edges.push((i, j, w));
                    }
                }
            }
        }
    }

    (edges, n)
}

/// Power-iteration Fiedler vector computation (fast, no external deps).
fn compute_fiedler(n: usize, edges: &[(usize, usize, f64)]) -> Vec<f64> {
    if n <= 2 || edges.is_empty() {
        return vec![0.0; n];
    }

    let mut degree = vec![0.0f64; n];
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];

    for &(u, v, w) in edges {
        if u < n && v < n {
            degree[u] += w;
            degree[v] += w;
            adj[u].push((v, w));
            adj[v].push((u, w));
        }
    }

    let d_inv: Vec<f64> = degree
        .iter()
        .map(|&d| if d > 1e-12 { 1.0 / d } else { 0.0 })
        .collect();

    let mut v: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64) - 0.5).collect();
    let mean: f64 = v.iter().sum::<f64>() / n as f64;
    for x in &mut v {
        *x -= mean;
    }

    // 15 iterations is enough for per-frame coherence (vs 20 in batch mode)
    for _ in 0..15 {
        let mut new_v = vec![0.0; n];
        for i in 0..n {
            let mut sum = 0.0;
            for &(j, w) in &adj[i] {
                sum += w * v[j];
            }
            new_v[i] = d_inv[i] * sum;
        }

        let mean: f64 = new_v.iter().sum::<f64>() / n as f64;
        for x in &mut new_v {
            *x -= mean;
        }

        let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for x in &mut new_v {
                *x /= norm;
            }
        }

        v = new_v;
    }

    v
}

fn wiener_normalize(raw_masks: &[Vec<f64>], magnitudes: &[f64], total_bins: usize) -> Vec<Vec<f64>> {
    let k = raw_masks.len();
    let mut masks = vec![vec![0.0; total_bins]; k];

    for i in 0..total_bins {
        let mag = if i < magnitudes.len() { magnitudes[i] } else { 0.0 };
        let sum: f64 = raw_masks.iter().map(|m| {
            let v = if i < m.len() { m[i] } else { 0.0 };
            v * v * mag * mag + 1e-10
        }).sum();

        for s in 0..k {
            let v = if i < raw_masks[s].len() { raw_masks[s][i] } else { 0.0 };
            masks[s][i] = (v * v * mag * mag + 1e-10) / sum;
        }
    }

    masks
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn make_test_signal(config: &StreamingMultiConfig, num_hops: usize) -> Vec<Vec<f64>> {
        let hop = config.hop_size;
        (0..num_hops)
            .map(|h| {
                (0..hop)
                    .map(|i| {
                        let t = (h * hop + i) as f64 / config.sample_rate;
                        0.5 * (2.0 * PI * 200.0 * t).sin()
                            + 0.3 * (2.0 * PI * 1500.0 * t).sin()
                            + 0.1 * (2.0 * PI * 5000.0 * t).sin()
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_mask_normalization() {
        let config = StreamingMultiConfig {
            window_size: 1024,
            hop_size: 256,
            ..StreamingMultiConfig::default()
        };
        let mut state = StreamingMultiState::new(&config);
        let frames = make_test_signal(&config, 8);

        // Process enough frames to have valid output
        let mut last_result = None;
        for frame in &frames {
            last_result = Some(state.process_frame(frame, &config));
        }

        let result = last_result.unwrap();
        assert_eq!(result.stems.len(), 6);

        let num_freq = config.window_size / 2 + 1;
        for f in 0..num_freq {
            let sum: f64 = result.stems.iter().map(|sf| sf.mask[f]).sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Mask sum at bin {f} = {sum:.6}, expected ~1.0"
            );
        }
    }

    #[test]
    fn test_streaming_consistency() {
        let config = StreamingMultiConfig {
            window_size: 1024,
            hop_size: 256,
            ..StreamingMultiConfig::default()
        };
        let mut state = StreamingMultiState::new(&config);
        let frames = make_test_signal(&config, 20);

        for (i, frame) in frames.iter().enumerate() {
            let result = state.process_frame(frame, &config);
            assert_eq!(result.stems.len(), 6, "Frame {i} should produce 6 stems");
            for sf in &result.stems {
                assert_eq!(
                    sf.mask.len(),
                    config.window_size / 2 + 1,
                    "Mask length mismatch at frame {i}"
                );
            }
        }

        assert_eq!(state.frame_count, 20);

        let accumulated = state.get_accumulated_stems();
        assert_eq!(accumulated.len(), 6);
        for (stem, samples) in &accumulated {
            assert!(
                !samples.is_empty(),
                "Accumulated audio for {:?} should not be empty",
                stem
            );
        }
    }

    #[test]
    fn test_frame_latency() {
        let config = StreamingMultiConfig {
            window_size: 1024,
            hop_size: 256,
            ..StreamingMultiConfig::default()
        };
        let mut state = StreamingMultiState::new(&config);
        let frames = make_test_signal(&config, 10);

        for frame in &frames {
            let result = state.process_frame(frame, &config);
            assert!(
                result.latency_us < 50_000,
                "Frame latency {}us exceeds 50ms budget",
                result.latency_us
            );
        }
    }

    #[test]
    fn test_temporal_smoothing() {
        let config = StreamingMultiConfig {
            window_size: 1024,
            hop_size: 256,
            mask_smoothing: 0.5,
            ..StreamingMultiConfig::default()
        };
        let mut state = StreamingMultiState::new(&config);

        // First: process steady-state frames to warm up
        let steady_frames = make_test_signal(&config, 6);
        let mut prev_result = None;
        for frame in &steady_frames {
            prev_result = Some(state.process_frame(frame, &config));
        }
        let prev = prev_result.unwrap();

        // Now process one more frame with the same signal
        let next_frame: Vec<f64> = (0..config.hop_size)
            .map(|i| {
                let t = (6 * config.hop_size + i) as f64 / config.sample_rate;
                0.5 * (2.0 * PI * 200.0 * t).sin()
                    + 0.3 * (2.0 * PI * 1500.0 * t).sin()
                    + 0.1 * (2.0 * PI * 5000.0 * t).sin()
            })
            .collect();
        let curr = state.process_frame(&next_frame, &config);

        // L2 distance between consecutive masks should be bounded for each stem
        for s in 0..6 {
            let l2: f64 = prev.stems[s]
                .mask
                .iter()
                .zip(curr.stems[s].mask.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();

            let num_freq = config.window_size / 2 + 1;
            let normalized_l2 = l2 / (num_freq as f64).sqrt();

            assert!(
                normalized_l2 < 0.5,
                "Stem {:?} mask changed too abruptly between frames: normalized L2 = {:.4}",
                curr.stems[s].stem,
                normalized_l2
            );
        }
    }
}
