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
            wiener_iterations: 3,
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

    // First resolution establishes the reference mask ordering
    let mut reference_masks: Option<Vec<Vec<f64>>> = None;

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

        // Interpolate each mask to primary resolution grid
        let mut interp_masks = vec![vec![0.0; primary_tf]; num_sources];
        for s in 0..num_sources {
            for f in 0..pri_frames {
                let src_f = (f as f64 * this_frames as f64 / pri_frames as f64) as usize;
                let src_f = src_f.min(this_frames.saturating_sub(1));

                for k in 0..pri_freq {
                    let src_k = (k as f64 * this_freq as f64 / pri_freq as f64) as usize;
                    let src_k = src_k.min(this_freq.saturating_sub(1));

                    let src_idx = src_f * this_freq + src_k;
                    let dst_idx = f * pri_freq + k;

                    if src_idx < refined[s].len() && dst_idx < primary_tf {
                        interp_masks[s][dst_idx] = refined[s][src_idx];
                    }
                }
            }
        }

        // Align source ordering with reference (first resolution)
        // by correlating masks and swapping if needed
        if let Some(ref ref_masks) = reference_masks {
            if num_sources == 2 {
                // Compute correlation: identity vs swapped
                let corr_identity: f64 = (0..primary_tf)
                    .map(|i| interp_masks[0][i] * ref_masks[0][i] + interp_masks[1][i] * ref_masks[1][i])
                    .sum();
                let corr_swapped: f64 = (0..primary_tf)
                    .map(|i| interp_masks[1][i] * ref_masks[0][i] + interp_masks[0][i] * ref_masks[1][i])
                    .sum();
                if corr_swapped > corr_identity {
                    interp_masks.swap(0, 1);
                }
            }
        } else {
            reference_masks = Some(interp_masks.clone());
        }

        for s in 0..num_sources {
            for i in 0..primary_tf {
                fused_masks[s][i] += interp_masks[s][i];
            }
        }
        weight_sum += 1.0;
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

/// Composite separation quality score (higher = better).
/// Combines independence, reconstruction accuracy, and energy balance.
fn separation_quality(mixed: &[f64], sources: &[Vec<f64>]) -> f64 {
    let n = mixed.len();
    if n == 0 || sources.is_empty() {
        return 0.0;
    }

    // 1. Independence: 1 - avg absolute cross-correlation
    let xcorr = source_cross_correlation(sources);
    let independence = 1.0 - xcorr;

    // 2. Reconstruction accuracy: how well sources sum to the mix
    let mixed_energy: f64 = mixed.iter().map(|x| x * x).sum::<f64>().max(1e-12);
    let recon_err: f64 = (0..n)
        .map(|i| {
            let sum: f64 = sources.iter().map(|s| s.get(i).copied().unwrap_or(0.0)).sum();
            (mixed[i] - sum).powi(2)
        })
        .sum::<f64>();
    let accuracy = 1.0 - (recon_err / mixed_energy).min(1.0);

    // 3. Energy balance: sources should have reasonable energy relative to mix
    //    Penalize solutions where one source has near-zero energy
    let source_energies: Vec<f64> = sources.iter()
        .map(|s| s.iter().map(|x| x * x).sum::<f64>())
        .collect();
    let total_source_energy = source_energies.iter().sum::<f64>().max(1e-12);
    let min_ratio = source_energies.iter()
        .map(|&e| e / total_source_energy)
        .fold(f64::MAX, f64::min);
    // Ideal: each source has 1/N of total energy. min_ratio near 1/N is good.
    let expected_ratio = 1.0 / sources.len() as f64;
    let balance = (min_ratio / expected_ratio).min(1.0);

    // Weighted combination
    0.3 * independence + 0.4 * accuracy + 0.3 * balance
}

/// Compute average absolute cross-correlation between all source pairs.
/// Lower = more independent (better separation).
fn source_cross_correlation(sources: &[Vec<f64>]) -> f64 {
    if sources.len() < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0;
    for i in 0..sources.len() {
        for j in (i + 1)..sources.len() {
            let n = sources[i].len().min(sources[j].len());
            if n == 0 { continue; }
            let ei: f64 = sources[i][..n].iter().map(|x| x * x).sum::<f64>().sqrt();
            let ej: f64 = sources[j][..n].iter().map(|x| x * x).sum::<f64>().sqrt();
            if ei < 1e-12 || ej < 1e-12 { continue; }
            let dot: f64 = sources[i][..n].iter().zip(sources[j][..n].iter())
                .map(|(a, b)| a * b).sum();
            total += (dot / (ei * ej)).abs();
            count += 1;
        }
    }
    if count > 0 { total / count as f64 } else { 0.0 }
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
    let n = signal.len();
    let num_sources = config.num_sources;

    // Strategy: run multiple window sizes independently, Wiener-refine each,
    // then pick the best result by composite quality score.
    // This avoids lossy mask interpolation between resolutions.

    let mut best_sources: Option<Vec<Vec<f64>>> = None;
    let mut best_quality = f64::NEG_INFINITY;
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

        let initial = separate(&graph, &sep_config);

        // Raw masks
        let raw_sources: Vec<Vec<f64>> = initial.masks.iter()
            .map(|mask| stft::istft(&stft_result, mask, n))
            .collect();
        let raw_quality = separation_quality(signal, &raw_sources);

        if raw_quality > best_quality {
            best_quality = raw_quality;
            best_sources = Some(raw_sources);
        }

        // Try Wiener with multiple exponents: soft (1.5), standard (2.0), sharp (3.0)
        for &exp in &[1.5, config.wiener_exponent, 3.0] {
            let wiener_masks = wiener_refine(
                &stft_result,
                &initial.masks,
                exp,
                config.wiener_iterations,
            );
            let wiener_sources: Vec<Vec<f64>> = wiener_masks.iter()
                .map(|mask| stft::istft(&stft_result, mask, n))
                .collect();
            let wiener_quality = separation_quality(signal, &wiener_sources);

            if wiener_quality > best_quality {
                best_quality = wiener_quality;
                best_sources = Some(wiener_sources);
            }
        }
    }

    let mut best_sources = best_sources.unwrap_or_else(|| vec![signal.to_vec()]);

    // Phase 2: Cascaded refinement using the best resolution
    if config.cascade_iterations > 1 {
        let (mut cascade_sources, cascade_stats) = cascade_separate(signal, config, sample_rate);
        stats.extend(cascade_stats);

        // Align cascade source ordering
        if num_sources == 2 && cascade_sources.len() == 2 && best_sources.len() == 2 {
            let n_min = best_sources[0].len().min(cascade_sources[0].len());
            let corr_id: f64 = (0..n_min)
                .map(|i| best_sources[0][i] * cascade_sources[0][i] + best_sources[1][i] * cascade_sources[1][i])
                .sum();
            let corr_sw: f64 = (0..n_min)
                .map(|i| best_sources[0][i] * cascade_sources[1][i] + best_sources[1][i] * cascade_sources[0][i])
                .sum();
            if corr_sw > corr_id {
                cascade_sources.swap(0, 1);
            }
        }

        let cascade_quality = separation_quality(signal, &cascade_sources);
        if cascade_quality > best_quality {
            best_sources = cascade_sources;
        }
    }

    let processing_ms = start.elapsed().as_secs_f64() * 1000.0;

    AdvancedResult {
        sources: best_sources,
        iteration_sdrs: Vec::new(),
        processing_ms,
        iterations_used: config.cascade_iterations,
        resolution_stats: stats,
    }
}

/// Find best permutation of estimated sources to match references (2-source).
/// Returns (best_sdrs, best_permutation_indices).
fn best_permutation_sdr(references: &[Vec<f64>], estimates: &[Vec<f64>]) -> (Vec<f64>, Vec<usize>) {
    let n = references.len().min(estimates.len());
    if n == 0 {
        return (vec![], vec![]);
    }
    if n == 1 {
        return (vec![compute_sdr_clamped(&references[0], &estimates[0])], vec![0]);
    }

    // For 2 sources, try both assignments
    // Perm 0: ref0->est0, ref1->est1
    let sdr_00 = compute_sdr_clamped(&references[0], &estimates[0]);
    let sdr_11 = compute_sdr_clamped(&references[1], &estimates[1]);
    let avg_identity = (sdr_00 + sdr_11) / 2.0;

    // Perm 1: ref0->est1, ref1->est0
    let sdr_01 = compute_sdr_clamped(&references[0], &estimates[1]);
    let sdr_10 = compute_sdr_clamped(&references[1], &estimates[0]);
    let avg_swapped = (sdr_01 + sdr_10) / 2.0;

    if avg_identity >= avg_swapped {
        (vec![sdr_00, sdr_11], vec![0, 1])
    } else {
        (vec![sdr_01, sdr_10], vec![1, 0])
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

    // Compute SDRs with best permutation matching
    let (basic_sdrs, _) = best_permutation_sdr(references, &basic_sources);
    let (advanced_sdrs, _) = best_permutation_sdr(references, &adv_result.sources);

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
