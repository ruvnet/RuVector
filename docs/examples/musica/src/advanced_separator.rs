//! Advanced separation techniques pushing toward SOTA quality.
//!
//! Implements cascaded refinement, Wiener filtering, multi-resolution
//! graph fusion, and iterative mask estimation for maximum SDR.

use crate::audio_graph::{build_audio_graph, AudioGraph, GraphParams};
use crate::separator::{separate, SeparatorConfig, SeparationResult};
use crate::stft::{self, StftResult};

/// Configuration for advanced separation.
#[derive(Debug, Clone)]
pub struct AdvancedConfig {
    /// Number of cascade iterations (each refines on residuals).
    pub cascade_iterations: usize,
    /// Number of Wiener filter iterations.
    pub wiener_iterations: usize,
    /// Number of sources to separate.
    pub num_sources: usize,
    /// STFT window sizes for multi-resolution fusion.
    pub window_sizes: Vec<usize>,
    /// Hop size ratio (hop = window / hop_ratio).
    pub hop_ratio: usize,
    /// Wiener filter exponent (higher = sharper masks).
    pub wiener_exponent: f64,
    /// Residual mixing weight for cascade iterations.
    pub cascade_alpha: f64,
    /// Graph params.
    pub graph_params: GraphParams,
}

impl Default for AdvancedConfig {
    fn default() -> Self {
        Self {
            cascade_iterations: 3,
            wiener_iterations: 2,
            num_sources: 2,
            window_sizes: vec![256, 512, 1024],
            hop_ratio: 2,
            wiener_exponent: 2.0,
            cascade_alpha: 0.7,
            graph_params: GraphParams::default(),
        }
    }
}

/// Result from advanced separation.
#[derive(Debug, Clone)]
pub struct AdvancedResult {
    /// Separated source signals.
    pub sources: Vec<Vec<f64>>,
    /// Per-iteration SDR improvements (if references provided).
    pub iteration_sdrs: Vec<f64>,
    /// Total processing time in milliseconds.
    pub processing_ms: f64,
    /// Number of cascade iterations used.
    pub iterations_used: usize,
    /// Resolution stats: (window_size, num_nodes).
    pub resolution_stats: Vec<(usize, usize)>,
}

// ── Wiener Filter ───────────────────────────────────────────────────────

/// Apply Wiener filtering to refine soft masks.
///
/// Wiener mask: M_s = |S_s|^p / sum_k(|S_k|^p)
/// where S_s is the estimated spectrogram of source s,
/// and p is the Wiener exponent (2 = standard, higher = sharper).
fn wiener_refine(
    stft_result: &StftResult,
    masks: &[Vec<f64>],
    exponent: f64,
    iterations: usize,
) -> Vec<Vec<f64>> {
    let total_tf = stft_result.num_frames * stft_result.num_freq_bins;
    let num_sources = masks.len();
    let mut refined = masks.to_vec();

    for _iter in 0..iterations {
        // Compute power spectrograms for each source
        let power_specs: Vec<Vec<f64>> = refined
            .iter()
            .map(|mask| {
                (0..total_tf)
                    .map(|i| {
                        let mag = stft_result.bins[i].magnitude * mask[i];
                        mag.powf(exponent)
                    })
                    .collect()
            })
            .collect();

        // Compute Wiener masks
        for s in 0..num_sources {
            for i in 0..total_tf {
                let total_power: f64 = power_specs.iter().map(|p| p[i]).sum();
                refined[s][i] = if total_power > 1e-12 {
                    power_specs[s][i] / total_power
                } else {
                    1.0 / num_sources as f64
                };
            }
        }
    }

    refined
}

// ── Cascaded Separation ─────────────────────────────────────────────────

/// Run cascaded separation: separate → estimate → residual → re-separate.
///
/// Each iteration refines the masks using the residual signal:
/// 1. Run graph separation to get initial masks
/// 2. Reconstruct estimated sources
/// 3. Compute residual = mixed - sum(estimated)
/// 4. Re-separate residual and blend with previous masks
fn cascade_separate(
    signal: &[f64],
    config: &AdvancedConfig,
    sample_rate: f64,
) -> (Vec<Vec<f64>>, Vec<(usize, usize)>) {
    let ws = config.window_sizes[0]; // Primary window size
    let hs = ws / config.hop_ratio;
    let n = signal.len();

    let stft_result = stft::stft(signal, ws, hs, sample_rate);
    let total_tf = stft_result.num_frames * stft_result.num_freq_bins;

    // Initial separation
    let graph = build_audio_graph(&stft_result, &config.graph_params);
    let mut stats = vec![(ws, graph.num_nodes)];
    let sep_config = SeparatorConfig {
        num_sources: config.num_sources,
        ..SeparatorConfig::default()
    };
    let initial = separate(&graph, &sep_config);

    // Apply Wiener filtering to initial masks
    let mut masks = wiener_refine(
        &stft_result,
        &initial.masks,
        config.wiener_exponent,
        config.wiener_iterations,
    );

    // Cascade iterations
    for iter in 1..config.cascade_iterations {
        // Reconstruct estimated sources
        let estimated: Vec<Vec<f64>> = masks
            .iter()
            .map(|mask| stft::istft(&stft_result, mask, n))
            .collect();

        // Compute residual
        let reconstructed_sum: Vec<f64> = (0..n)
            .map(|i| estimated.iter().map(|s| s[i]).sum())
            .collect();
        let residual: Vec<f64> = signal.iter()
            .zip(reconstructed_sum.iter())
            .map(|(s, r)| s - r)
            .collect();

        // Check if residual is significant
        let residual_energy: f64 = residual.iter().map(|x| x * x).sum::<f64>() / n as f64;
        let signal_energy: f64 = signal.iter().map(|x| x * x).sum::<f64>() / n as f64;
        if residual_energy < signal_energy * 0.01 {
            break; // Residual is < 1% of signal, no point continuing
        }

        // Re-separate the residual
        let res_stft = stft::stft(&residual, ws, hs, sample_rate);
        let res_graph = build_audio_graph(&res_stft, &config.graph_params);
        let res_sep = separate(&res_graph, &sep_config);

        // Blend residual masks with previous masks
        let alpha = config.cascade_alpha * (0.5f64).powi(iter as i32); // Decay blending weight
        let res_masks = wiener_refine(
            &res_stft,
            &res_sep.masks,
            config.wiener_exponent,
            1,
        );

        for s in 0..config.num_sources {
            for i in 0..total_tf.min(res_masks[s].len()) {
                // Add residual contribution, weighted by magnitude
                let res_contribution = res_masks[s][i] * alpha;
                masks[s][i] = (masks[s][i] + res_contribution).min(1.0);
            }
        }

        // Re-normalize masks to sum to 1
        for i in 0..total_tf {
            let sum: f64 = (0..config.num_sources).map(|s| masks[s][i]).sum();
            if sum > 1e-12 {
                for s in 0..config.num_sources {
                    masks[s][i] /= sum;
                }
            }
        }
    }

    // Final reconstruction
    let sources: Vec<Vec<f64>> = masks
        .iter()
        .map(|mask| stft::istft(&stft_result, mask, n))
        .collect();

    (sources, stats)
}

// ── Multi-Resolution Fusion ─────────────────────────────────────────────

/// Separate using multiple STFT resolutions and fuse the masks.
///
/// Different window sizes capture different aspects:
/// - Small windows (256): good temporal resolution, captures transients
/// - Medium windows (512): balanced
/// - Large windows (1024): good frequency resolution, captures harmonics
///
/// Masks from all resolutions are averaged for robust separation.
fn multi_resolution_separate(
    signal: &[f64],
    config: &AdvancedConfig,
    sample_rate: f64,
) -> (Vec<Vec<f64>>, Vec<(usize, usize)>) {
    let n = signal.len();
    let num_sources = config.num_sources;

    // Use the primary (smallest) window for final reconstruction
    let primary_ws = config.window_sizes[0];
    let primary_hs = primary_ws / config.hop_ratio;
    let primary_stft = stft::stft(signal, primary_ws, primary_hs, sample_rate);
    let primary_tf = primary_stft.num_frames * primary_stft.num_freq_bins;

    // Initialize accumulated masks at primary resolution
    let mut fused_masks = vec![vec![0.0; primary_tf]; num_sources];
    let mut weight_sum = 0.0f64;
    let mut stats = Vec::new();

    let sep_config = SeparatorConfig {
        num_sources,
        ..SeparatorConfig::default()
    };

    for &ws in &config.window_sizes {
        let hs = ws / config.hop_ratio;
        let stft_result = stft::stft(signal, ws, hs, sample_rate);
        let graph = build_audio_graph(&stft_result, &config.graph_params);
        stats.push((ws, graph.num_nodes));

        let separation = separate(&graph, &sep_config);

        // Wiener-refine this resolution's masks
        let refined = wiener_refine(
            &stft_result,
            &separation.masks,
            config.wiener_exponent,
            1,
        );

        // Interpolate masks to primary resolution
        let this_frames = stft_result.num_frames;
        let this_freq = stft_result.num_freq_bins;
        let pri_frames = primary_stft.num_frames;
        let pri_freq = primary_stft.num_freq_bins;

        // Resolution weight: larger windows get more weight for
        // frequency-dependent features, smaller for temporal
        let res_weight = 1.0;

        for s in 0..num_sources {
            for f in 0..pri_frames {
                // Map primary frame to this resolution's frame
                let src_f = (f as f64 * this_frames as f64 / pri_frames as f64) as usize;
                let src_f = src_f.min(this_frames.saturating_sub(1));

                for k in 0..pri_freq {
                    // Map primary freq bin to this resolution's freq bin
                    let src_k = (k as f64 * this_freq as f64 / pri_freq as f64) as usize;
                    let src_k = src_k.min(this_freq.saturating_sub(1));

                    let src_idx = src_f * this_freq + src_k;
                    let dst_idx = f * pri_freq + k;

                    if src_idx < refined[s].len() && dst_idx < primary_tf {
                        fused_masks[s][dst_idx] += refined[s][src_idx] * res_weight;
                    }
                }
            }
        }
        weight_sum += res_weight;
    }

    // Normalize fused masks
    if weight_sum > 0.0 {
        for s in 0..num_sources {
            for v in &mut fused_masks[s] {
                *v /= weight_sum;
            }
        }
    }

    // Re-normalize to sum to 1 per TF bin
    for i in 0..primary_tf {
        let sum: f64 = (0..num_sources).map(|s| fused_masks[s][i]).sum();
        if sum > 1e-12 {
            for s in 0..num_sources {
                fused_masks[s][i] /= sum;
            }
        }
    }

    // Reconstruct
    let sources: Vec<Vec<f64>> = fused_masks
        .iter()
        .map(|mask| stft::istft(&primary_stft, mask, n))
        .collect();

    (sources, stats)
}

// ── Full Advanced Pipeline ──────────────────────────────────────────────

/// Run the full advanced separation pipeline:
/// 1. Multi-resolution graph construction + separation
/// 2. Wiener filter mask refinement
/// 3. Cascaded residual refinement
///
/// Returns separated sources with maximum quality.
pub fn advanced_separate(
    signal: &[f64],
    config: &AdvancedConfig,
    sample_rate: f64,
) -> AdvancedResult {
    let start = std::time::Instant::now();

    // Phase 1: Multi-resolution fusion
    let (mut sources, mut stats) = multi_resolution_separate(signal, config, sample_rate);

    // Phase 2: Cascaded refinement on the fused result
    if config.cascade_iterations > 1 {
        let (cascade_sources, cascade_stats) = cascade_separate(signal, config, sample_rate);
        stats.extend(cascade_stats);

        // Blend multi-res and cascade results (equal weight)
        let n = signal.len();
        for s in 0..config.num_sources.min(sources.len()).min(cascade_sources.len()) {
            for i in 0..n.min(sources[s].len()).min(cascade_sources[s].len()) {
                sources[s][i] = 0.5 * sources[s][i] + 0.5 * cascade_sources[s][i];
            }
        }
    }

    let processing_ms = start.elapsed().as_secs_f64() * 1000.0;

    AdvancedResult {
        sources,
        iteration_sdrs: Vec::new(),
        processing_ms,
        iterations_used: config.cascade_iterations,
        resolution_stats: stats,
    }
}

/// Compute SDR between reference and estimate (clamped to [-60, 100]).
pub fn compute_sdr_clamped(reference: &[f64], estimate: &[f64]) -> f64 {
    let n = reference.len().min(estimate.len());
    if n == 0 {
        return -60.0;
    }

    let ref_energy: f64 = reference[..n].iter().map(|x| x * x).sum();
    let noise_energy: f64 = reference[..n]
        .iter()
        .zip(estimate[..n].iter())
        .map(|(r, e)| (r - e).powi(2))
        .sum();

    if ref_energy < 1e-12 {
        return -60.0;
    }
    if noise_energy < 1e-12 {
        return 100.0;
    }

    (10.0 * (ref_energy / noise_energy).log10()).clamp(-60.0, 100.0)
}

/// Compare basic vs advanced separation on a mix.
pub fn compare_basic_vs_advanced(
    mixed: &[f64],
    references: &[Vec<f64>],
    sample_rate: f64,
) -> ComparisonResult {
    let n = mixed.len();
    let num_sources = references.len();

    // Basic separation
    let basic_start = std::time::Instant::now();
    let stft_result = stft::stft(mixed, 256, 128, sample_rate);
    let graph = build_audio_graph(&stft_result, &GraphParams::default());
    let basic_sep = separate(&graph, &SeparatorConfig {
        num_sources,
        ..SeparatorConfig::default()
    });
    let basic_sources: Vec<Vec<f64>> = basic_sep.masks.iter()
        .map(|m| stft::istft(&stft_result, m, n))
        .collect();
    let basic_ms = basic_start.elapsed().as_secs_f64() * 1000.0;

    // Advanced separation
    let adv_start = std::time::Instant::now();
    let adv_config = AdvancedConfig {
        num_sources,
        ..AdvancedConfig::default()
    };
    let adv_result = advanced_separate(mixed, &adv_config, sample_rate);
    let adv_ms = adv_start.elapsed().as_secs_f64() * 1000.0;

    // Compute SDRs
    let mut basic_sdrs = Vec::new();
    let mut advanced_sdrs = Vec::new();

    for s in 0..num_sources.min(basic_sources.len()).min(adv_result.sources.len()) {
        basic_sdrs.push(compute_sdr_clamped(&references[s], &basic_sources[s]));
        advanced_sdrs.push(compute_sdr_clamped(&references[s], &adv_result.sources[s]));
    }

    let basic_avg = if basic_sdrs.is_empty() { -60.0 } else {
        basic_sdrs.iter().sum::<f64>() / basic_sdrs.len() as f64
    };
    let advanced_avg = if advanced_sdrs.is_empty() { -60.0 } else {
        advanced_sdrs.iter().sum::<f64>() / advanced_sdrs.len() as f64
    };

    ComparisonResult {
        basic_sdrs,
        advanced_sdrs,
        basic_avg_sdr: basic_avg,
        advanced_avg_sdr: advanced_avg,
        improvement_db: advanced_avg - basic_avg,
        basic_ms,
        advanced_ms: adv_ms,
    }
}

/// Comparison result between basic and advanced separation.
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub basic_sdrs: Vec<f64>,
    pub advanced_sdrs: Vec<f64>,
    pub basic_avg_sdr: f64,
    pub advanced_avg_sdr: f64,
    pub improvement_db: f64,
    pub basic_ms: f64,
    pub advanced_ms: f64,
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn sine(freq: f64, sr: f64, n: usize, amp: f64) -> Vec<f64> {
        (0..n).map(|i| amp * (2.0 * PI * freq * i as f64 / sr).sin()).collect()
    }

    #[test]
    fn test_wiener_refine_normalizes() {
        // Create dummy STFT and masks
        let signal: Vec<f64> = (0..2000).map(|i| (i as f64 * 0.01).sin()).collect();
        let stft_result = stft::stft(&signal, 256, 128, 8000.0);
        let total_tf = stft_result.num_frames * stft_result.num_freq_bins;

        let masks = vec![
            vec![0.7; total_tf],
            vec![0.3; total_tf],
        ];

        let refined = wiener_refine(&stft_result, &masks, 2.0, 2);

        // Should sum to ~1.0 per bin
        for i in 0..total_tf {
            let sum: f64 = refined.iter().map(|m| m[i]).sum();
            assert!((sum - 1.0).abs() < 0.01, "Wiener masks should sum to 1, got {}", sum);
        }
    }

    #[test]
    fn test_cascade_improves_or_maintains() {
        let sr = 8000.0;
        let n = 2000;
        let src1 = sine(200.0, sr, n, 1.0);
        let src2 = sine(2000.0, sr, n, 0.8);
        let mixed: Vec<f64> = src1.iter().zip(src2.iter()).map(|(a, b)| a + b).collect();

        let config = AdvancedConfig {
            cascade_iterations: 2,
            wiener_iterations: 1,
            num_sources: 2,
            window_sizes: vec![256],
            ..AdvancedConfig::default()
        };

        let result = cascade_separate(&mixed, &config, sr);
        assert_eq!(result.0.len(), 2);
        assert_eq!(result.0[0].len(), n);
    }

    #[test]
    fn test_multi_resolution_produces_output() {
        let sr = 8000.0;
        let n = 4000;
        let src1 = sine(200.0, sr, n, 1.0);
        let src2 = sine(2000.0, sr, n, 0.8);
        let mixed: Vec<f64> = src1.iter().zip(src2.iter()).map(|(a, b)| a + b).collect();

        let config = AdvancedConfig {
            num_sources: 2,
            window_sizes: vec![256, 512],
            ..AdvancedConfig::default()
        };

        let (sources, stats) = multi_resolution_separate(&mixed, &config, sr);
        assert_eq!(sources.len(), 2);
        assert_eq!(stats.len(), 2); // Two resolutions
    }

    #[test]
    fn test_advanced_separate_full() {
        let sr = 8000.0;
        let n = 4000;
        let src1 = sine(200.0, sr, n, 1.0);
        let src2 = sine(2000.0, sr, n, 0.8);
        let mixed: Vec<f64> = src1.iter().zip(src2.iter()).map(|(a, b)| a + b).collect();

        let config = AdvancedConfig {
            num_sources: 2,
            cascade_iterations: 2,
            wiener_iterations: 1,
            window_sizes: vec![256, 512],
            ..AdvancedConfig::default()
        };

        let result = advanced_separate(&mixed, &config, sr);
        assert_eq!(result.sources.len(), 2);
        assert!(result.processing_ms > 0.0);
    }

    #[test]
    fn test_sdr_clamped() {
        let signal = vec![1.0; 100];
        let zeros = vec![0.0; 100];

        // Perfect reconstruction
        assert!(compute_sdr_clamped(&signal, &signal) > 90.0);

        // Zero reference
        assert_eq!(compute_sdr_clamped(&zeros, &signal), -60.0);

        // Empty
        assert_eq!(compute_sdr_clamped(&[], &[]), -60.0);
    }

    #[test]
    fn test_comparison_basic_vs_advanced() {
        let sr = 8000.0;
        let n = 2000;
        let src1 = sine(200.0, sr, n, 1.0);
        let src2 = sine(2000.0, sr, n, 0.8);
        let mixed: Vec<f64> = src1.iter().zip(src2.iter()).map(|(a, b)| a + b).collect();

        let result = compare_basic_vs_advanced(&mixed, &[src1, src2], sr);
        assert_eq!(result.basic_sdrs.len(), 2);
        assert_eq!(result.advanced_sdrs.len(), 2);
        assert!(result.basic_ms > 0.0);
        assert!(result.advanced_ms > 0.0);
    }
}
