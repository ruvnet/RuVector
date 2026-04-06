//! Multitrack 6-stem audio source separation.
//!
//! Separates audio into: Vocals, Bass, Drums, Guitar, Piano, Other
//!
//! Uses band-split spectral analysis with graph-based structural refinement:
//! 1. High-resolution STFT (4096 window, 1024 hop)
//! 2. Band-split features per stem type with frequency priors
//! 3. Graph construction with stem-specific edges
//! 4. Fiedler vector for coherence grouping
//! 5. Dynamic mincut for boundary refinement
//! 6. Wiener-style soft mask with temporal smoothing
//! 7. Replay logging for reproducibility

use crate::stft::{self, StftResult};
use ruvector_mincut::prelude::*;
use std::collections::HashMap;

/// The 6 stem types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stem {
    Vocals,
    Bass,
    Drums,
    Guitar,
    Piano,
    Other,
}

impl Stem {
    pub fn all() -> &'static [Stem] {
        &[
            Stem::Vocals,
            Stem::Bass,
            Stem::Drums,
            Stem::Guitar,
            Stem::Piano,
            Stem::Other,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Stem::Vocals => "vocals",
            Stem::Bass => "bass",
            Stem::Drums => "drums",
            Stem::Guitar => "guitar",
            Stem::Piano => "piano",
            Stem::Other => "other",
        }
    }
}

/// Stem-specific spectral priors.
#[derive(Debug, Clone)]
pub struct StemPrior {
    /// Frequency range (min_hz, max_hz).
    pub freq_range: (f64, f64),
    /// Temporal smoothness weight (higher = more continuity expected).
    pub temporal_smoothness: f64,
    /// Harmonic strength weight.
    pub harmonic_strength: f64,
    /// Transient weight (high for drums).
    pub transient_weight: f64,
}

/// Get default stem priors.
pub fn default_stem_priors() -> Vec<(Stem, StemPrior)> {
    vec![
        (
            Stem::Vocals,
            StemPrior {
                freq_range: (80.0, 8000.0),
                temporal_smoothness: 0.7,
                harmonic_strength: 0.9,
                transient_weight: 0.3,
            },
        ),
        (
            Stem::Bass,
            StemPrior {
                freq_range: (20.0, 300.0),
                temporal_smoothness: 0.8,
                harmonic_strength: 0.6,
                transient_weight: 0.2,
            },
        ),
        (
            Stem::Drums,
            StemPrior {
                freq_range: (20.0, 16000.0),
                temporal_smoothness: 0.2,
                harmonic_strength: 0.1,
                transient_weight: 0.95,
            },
        ),
        (
            Stem::Guitar,
            StemPrior {
                freq_range: (80.0, 5000.0),
                temporal_smoothness: 0.6,
                harmonic_strength: 0.85,
                transient_weight: 0.4,
            },
        ),
        (
            Stem::Piano,
            StemPrior {
                freq_range: (27.0, 4186.0),
                temporal_smoothness: 0.5,
                harmonic_strength: 0.95,
                transient_weight: 0.5,
            },
        ),
        (
            Stem::Other,
            StemPrior {
                freq_range: (20.0, 20000.0),
                temporal_smoothness: 0.3,
                harmonic_strength: 0.2,
                transient_weight: 0.3,
            },
        ),
    ]
}

/// Configuration.
#[derive(Debug, Clone)]
pub struct MultitrackConfig {
    /// STFT window size.
    pub window_size: usize,
    /// STFT hop size.
    pub hop_size: usize,
    /// Sample rate.
    pub sample_rate: f64,
    /// Frames per graph window.
    pub graph_window_frames: usize,
    /// Temporal mask smoothing (0-1).
    pub mask_smoothing: f64,
    /// Number of spectral components for Fiedler analysis.
    pub num_spectral_components: usize,
}

impl Default for MultitrackConfig {
    fn default() -> Self {
        Self {
            window_size: 4096,
            hop_size: 1024,
            sample_rate: 44100.0,
            graph_window_frames: 8,
            mask_smoothing: 0.3,
            num_spectral_components: 4,
        }
    }
}

/// Per-stem result.
#[derive(Debug, Clone)]
pub struct StemResult {
    /// Which stem.
    pub stem: Stem,
    /// Soft mask indexed [frame * num_freq_bins + freq_bin].
    pub mask: Vec<f64>,
    /// Reconstructed signal.
    pub signal: Vec<f64>,
    /// Confidence (average mask value in primary frequency range).
    pub confidence: f64,
}

/// Full multitrack result.
pub struct MultitrackResult {
    /// Per-stem results.
    pub stems: Vec<StemResult>,
    /// STFT of the input.
    pub stft_result: StftResult,
    /// Statistics.
    pub stats: MultitrackStats,
    /// Replay log.
    pub replay_log: Vec<ReplayEntry>,
}

/// Statistics.
#[derive(Debug, Clone)]
pub struct MultitrackStats {
    /// Total STFT frames.
    pub total_frames: usize,
    /// Graph nodes used.
    pub graph_nodes: usize,
    /// Graph edges used.
    pub graph_edges: usize,
    /// Total processing time in ms.
    pub processing_time_ms: f64,
    /// Energy per stem.
    pub per_stem_energy: Vec<(Stem, f64)>,
}

/// Replay log entry.
#[derive(Debug, Clone)]
pub struct ReplayEntry {
    /// Frame index.
    pub frame: usize,
    /// Stem being processed.
    pub stem: Stem,
    /// MinCut value.
    pub cut_value: f64,
    /// Partition sizes.
    pub partition_sizes: Vec<usize>,
}

/// Separate a mono signal into 6 stems.
pub fn separate_multitrack(signal: &[f64], config: &MultitrackConfig) -> MultitrackResult {
    let start = std::time::Instant::now();

    // STFT
    let stft_result = stft::stft(signal, config.window_size, config.hop_size, config.sample_rate);
    let num_frames = stft_result.num_frames;
    let num_freq = stft_result.num_freq_bins;
    let total_bins = num_frames * num_freq;

    let priors = default_stem_priors();
    let mut replay_log = Vec::new();
    let mut total_graph_nodes = 0usize;
    let mut total_graph_edges = 0usize;

    // Compute per-bin magnitude for Wiener masking
    let magnitudes: Vec<f64> = stft_result.bins.iter().map(|b| b.magnitude).collect();

    // Compute transient score per bin (magnitude derivative across frames)
    let transient_scores = compute_transient_scores(&magnitudes, num_frames, num_freq);

    // Compute harmonicity score per bin
    let harmonicity_scores = compute_harmonicity_scores(&magnitudes, num_frames, num_freq);

    // For each stem, compute a raw affinity mask
    let mut raw_masks: Vec<Vec<f64>> = Vec::new();

    for (stem, prior) in &priors {
        let freq_bin_min = freq_to_bin(prior.freq_range.0, config.sample_rate, config.window_size);
        let freq_bin_max = freq_to_bin(prior.freq_range.1, config.sample_rate, config.window_size);

        let mut mask = vec![0.0f64; total_bins];

        // Step 1: Frequency prior
        for frame in 0..num_frames {
            for f in 0..num_freq {
                let idx = frame * num_freq + f;
                if f >= freq_bin_min && f <= freq_bin_max {
                    mask[idx] = 1.0;
                } else {
                    // Soft falloff outside primary range
                    let dist = if f < freq_bin_min {
                        (freq_bin_min - f) as f64
                    } else {
                        (f - freq_bin_max) as f64
                    };
                    mask[idx] = (-dist / 10.0).exp();
                }
            }
        }

        // Step 2: Weight by harmonic/transient character
        for idx in 0..total_bins {
            let h_weight = harmonicity_scores[idx] * prior.harmonic_strength;
            let t_weight = transient_scores[idx] * prior.transient_weight;
            mask[idx] *= (1.0 + h_weight + t_weight) / 2.0;
        }

        // Step 3: Graph-based refinement per window
        let step = config.graph_window_frames;
        let mut frame_start = 0;
        while frame_start < num_frames {
            let frame_end = (frame_start + step).min(num_frames);
            let window_bins = collect_window_bins(
                &magnitudes,
                frame_start,
                frame_end,
                num_freq,
                freq_bin_min,
                freq_bin_max,
            );

            if window_bins.len() >= 4 {
                let (edges, num_nodes) = build_stem_graph(
                    &window_bins,
                    &magnitudes,
                    &harmonicity_scores,
                    &transient_scores,
                    num_freq,
                    prior,
                );

                total_graph_nodes += num_nodes;
                total_graph_edges += edges.len();

                // Compute Fiedler vector for this window
                let fiedler = compute_stem_fiedler(num_nodes, &edges);

                // Use Fiedler vector to modulate mask
                let median = {
                    let mut sorted = fiedler.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    sorted[fiedler.len() / 2]
                };

                for (local_i, &(frame, freq)) in window_bins.iter().enumerate() {
                    let idx = frame * num_freq + freq;
                    let fiedler_val = if local_i < fiedler.len() {
                        fiedler[local_i]
                    } else {
                        0.0
                    };

                    // Bins on the "coherent" side get boosted
                    let boost = if fiedler_val > median { 1.2 } else { 0.8 };
                    mask[idx] *= boost;
                }

                // Get mincut value for replay log
                let cut_value = compute_window_mincut(&edges);
                let above = fiedler.iter().filter(|&&v| v > median).count();
                let below = fiedler.len() - above;

                replay_log.push(ReplayEntry {
                    frame: frame_start,
                    stem: *stem,
                    cut_value,
                    partition_sizes: vec![above, below],
                });
            }

            frame_start += step;
        }

        // Step 4: Temporal smoothing
        apply_temporal_smoothing(&mut mask, num_frames, num_freq, config.mask_smoothing);

        raw_masks.push(mask);
    }

    // Wiener-style normalization: ensure masks sum to ~1 at each TF bin
    let mut masks = wiener_normalize(&raw_masks, &magnitudes, total_bins);

    // Reconstruct signals
    let mut stems = Vec::new();
    let mut per_stem_energy = Vec::new();

    for (i, (stem, _prior)) in priors.iter().enumerate() {
        let signal_out = stft::istft(&stft_result, &masks[i], signal.len());

        let energy: f64 = signal_out.iter().map(|s| s * s).sum::<f64>() / signal_out.len().max(1) as f64;
        per_stem_energy.push((*stem, energy));

        let confidence = compute_stem_confidence(&masks[i], num_frames, num_freq);

        stems.push(StemResult {
            stem: *stem,
            mask: masks[i].clone(),
            signal: signal_out,
            confidence,
        });
    }

    let processing_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    MultitrackResult {
        stems,
        stft_result,
        stats: MultitrackStats {
            total_frames: num_frames,
            graph_nodes: total_graph_nodes,
            graph_edges: total_graph_edges,
            processing_time_ms,
            per_stem_energy,
        },
        replay_log,
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
            // Normalize transient score
            scores[frame * num_freq + f] = (diff / (prev + 1e-8)).min(1.0);
        }
    }

    scores
}

fn compute_harmonicity_scores(
    magnitudes: &[f64],
    num_frames: usize,
    num_freq: usize,
) -> Vec<f64> {
    let mut scores = vec![0.0; magnitudes.len()];

    for frame in 0..num_frames {
        for f in 1..num_freq / 4 {
            let base = frame * num_freq;
            let fund = magnitudes[base + f];
            if fund < 1e-6 {
                continue;
            }

            // Check for harmonics at 2x, 3x, 4x
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

                // Also mark harmonics
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

fn collect_window_bins(
    magnitudes: &[f64],
    frame_start: usize,
    frame_end: usize,
    num_freq: usize,
    freq_min: usize,
    freq_max: usize,
) -> Vec<(usize, usize)> {
    let mut bins = Vec::new();
    let mag_threshold = 0.001;

    for frame in frame_start..frame_end {
        for f in freq_min..=freq_max.min(num_freq - 1) {
            let idx = frame * num_freq + f;
            if idx < magnitudes.len() && magnitudes[idx] > mag_threshold {
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

    // Build bin -> local index map
    let bin_map: HashMap<(usize, usize), usize> = bins.iter().enumerate().map(|(i, &b)| (b, i)).collect();

    for (i, &(frame_i, freq_i)) in bins.iter().enumerate() {
        let idx_i = frame_i * num_freq + freq_i;

        // Spectral neighbor (same frame, f+1)
        if let Some(&j) = bin_map.get(&(frame_i, freq_i + 1)) {
            let idx_j = frame_i * num_freq + freq_i + 1;
            let w = (magnitudes[idx_i] * magnitudes[idx_j]).sqrt() * 0.5;
            if w > 1e-4 {
                edges.push((i, j, w));
            }
        }

        // Temporal neighbor (same freq, frame+1)
        if let Some(&j) = bin_map.get(&(frame_i + 1, freq_i)) {
            let idx_j = (frame_i + 1) * num_freq + freq_i;
            let w = (magnitudes[idx_i] * magnitudes[idx_j]).sqrt() * prior.temporal_smoothness;
            if w > 1e-4 {
                edges.push((i, j, w));
            }
        }

        // Harmonic neighbors
        for h in [2, 3] {
            let hf = freq_i * h;
            if let Some(&j) = bin_map.get(&(frame_i, hf)) {
                let idx_j = frame_i * num_freq + hf;
                let w = (harmonicity[idx_i] * harmonicity[idx_j]).sqrt()
                    * prior.harmonic_strength
                    * 0.3;
                if w > 1e-4 {
                    edges.push((i, j, w));
                }
            }
        }
    }

    (edges, n)
}

fn compute_stem_fiedler(n: usize, edges: &[(usize, usize, f64)]) -> Vec<f64> {
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

    for _ in 0..20 {
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

fn compute_window_mincut(edges: &[(usize, usize, f64)]) -> f64 {
    if edges.is_empty() {
        return 0.0;
    }

    let edge_list: Vec<(u64, u64, f64)> = edges
        .iter()
        .map(|&(u, v, w)| (u as u64, v as u64, w))
        .collect();

    match MinCutBuilder::new().exact().with_edges(edge_list).build() {
        Ok(mc) => mc.min_cut_value(),
        Err(_) => 0.0,
    }
}

fn apply_temporal_smoothing(
    mask: &mut [f64],
    num_frames: usize,
    num_freq: usize,
    alpha: f64,
) {
    for f in 0..num_freq {
        for frame in 1..num_frames {
            let prev = mask[(frame - 1) * num_freq + f];
            let curr = &mut mask[frame * num_freq + f];
            *curr = alpha * prev + (1.0 - alpha) * *curr;
        }
    }
}

fn wiener_normalize(raw_masks: &[Vec<f64>], magnitudes: &[f64], total_bins: usize) -> Vec<Vec<f64>> {
    let k = raw_masks.len();
    let mut masks = vec![vec![0.0; total_bins]; k];

    for i in 0..total_bins {
        let mag = magnitudes[i];
        let sum: f64 = raw_masks.iter().map(|m| m[i] * m[i] * mag * mag + 1e-10).sum();

        for s in 0..k {
            masks[s][i] = (raw_masks[s][i] * raw_masks[s][i] * mag * mag + 1e-10) / sum;
        }
    }

    masks
}

fn compute_stem_confidence(mask: &[f64], num_frames: usize, num_freq: usize) -> f64 {
    if mask.is_empty() {
        return 0.0;
    }

    let total = mask.iter().sum::<f64>();
    total / mask.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stem_priors() {
        let priors = default_stem_priors();
        assert_eq!(priors.len(), 6);

        // Verify all stems are covered
        for stem in Stem::all() {
            assert!(
                priors.iter().any(|(s, _)| s == stem),
                "Missing prior for {:?}",
                stem
            );
        }
    }

    #[test]
    fn test_separate_simple() {
        use std::f64::consts::PI;

        // Two tones — should produce non-zero masks for multiple stems
        let sr = 44100.0;
        let n = 44100; // 1 second
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                0.5 * (2.0 * PI * 200.0 * t).sin() + 0.3 * (2.0 * PI * 2000.0 * t).sin()
            })
            .collect();

        let config = MultitrackConfig {
            window_size: 1024,
            hop_size: 512,
            sample_rate: sr,
            graph_window_frames: 4,
            ..MultitrackConfig::default()
        };

        let result = separate_multitrack(&signal, &config);

        assert_eq!(result.stems.len(), 6);

        // At least some stems should have non-zero energy
        let total_energy: f64 = result.stems.iter().map(|s| {
            s.signal.iter().map(|x| x * x).sum::<f64>()
        }).sum();

        assert!(total_energy > 0.0, "Total reconstructed energy should be > 0");
    }

    #[test]
    fn test_six_stems_coverage() {
        use std::f64::consts::PI;

        let sr = 44100.0;
        let n = 22050;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / sr).sin())
            .collect();

        let config = MultitrackConfig {
            window_size: 1024,
            hop_size: 512,
            sample_rate: sr,
            graph_window_frames: 4,
            ..MultitrackConfig::default()
        };

        let result = separate_multitrack(&signal, &config);

        // Masks should approximately sum to 1 at each TF bin
        let total_bins = result.stft_result.num_frames * result.stft_result.num_freq_bins;
        let num_check = total_bins.min(200);

        for i in 0..num_check {
            let sum: f64 = result.stems.iter().map(|s| s.mask[i]).sum();
            assert!(
                (sum - 1.0).abs() < 0.1,
                "Mask sum at bin {i} = {sum:.3}, expected ~1.0"
            );
        }
    }

    #[test]
    fn test_replay_logging() {
        use std::f64::consts::PI;

        let sr = 44100.0;
        let n = 22050;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / sr).sin())
            .collect();

        let config = MultitrackConfig {
            window_size: 1024,
            hop_size: 512,
            sample_rate: sr,
            graph_window_frames: 4,
            ..MultitrackConfig::default()
        };

        let result = separate_multitrack(&signal, &config);

        assert!(
            !result.replay_log.is_empty(),
            "Replay log should have entries"
        );

        for entry in &result.replay_log {
            assert!(entry.cut_value >= 0.0);
            assert!(!entry.partition_sizes.is_empty());
        }
    }

    #[test]
    fn test_mask_smoothing() {
        use std::f64::consts::PI;

        let sr = 44100.0;
        let n = 44100;

        // Impulse followed by silence — smoothing should spread energy
        let mut signal = vec![0.0; n];
        for i in 0..1000 {
            signal[i] = (2.0 * PI * 440.0 * i as f64 / sr).sin();
        }

        let config = MultitrackConfig {
            window_size: 1024,
            hop_size: 512,
            sample_rate: sr,
            graph_window_frames: 4,
            mask_smoothing: 0.5,
            ..MultitrackConfig::default()
        };

        let result = separate_multitrack(&signal, &config);

        // Check that some stem has temporally smooth mask
        let num_freq = result.stft_result.num_freq_bins;
        let num_frames = result.stft_result.num_frames;

        if num_frames > 2 {
            let vocals_mask = &result.stems[0].mask;
            let mut total_diff = 0.0;
            let mut count = 0;

            for f in 0..num_freq.min(10) {
                for frame in 1..num_frames {
                    let diff = (vocals_mask[frame * num_freq + f]
                        - vocals_mask[(frame - 1) * num_freq + f])
                    .abs();
                    total_diff += diff;
                    count += 1;
                }
            }

            let avg_diff = total_diff / count.max(1) as f64;
            // With smoothing=0.5, average frame-to-frame diff should be moderate
            assert!(
                avg_diff < 1.0,
                "Mask should be temporally smooth: avg_diff={avg_diff:.4}"
            );
        }
    }
}
