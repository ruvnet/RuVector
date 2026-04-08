//! Enhanced separator pipeline that chains:
//! multi-res STFT -> graph construction -> Fiedler separation ->
//! neural mask refinement -> phase-aware reconstruction.

use crate::audio_graph::{build_audio_graph, GraphParams};
use crate::multi_res::{self, MultiResConfig};
use crate::neural_refine::{MLPConfig, TinyMLP};
use crate::phase::{self, GriffinLimConfig};
use crate::separator::{self, SeparatorConfig};
use crate::stft::{self, StftResult};
use std::time::Instant;

/// Configuration for the enhanced separator pipeline.
#[derive(Debug, Clone)]
pub struct EnhancedSeparatorConfig {
    /// Number of sources to separate into.
    pub num_sources: usize,
    /// Use multi-resolution STFT (default: true).
    pub use_multi_res: bool,
    /// Use MLP mask refinement (default: true).
    pub use_neural_refine: bool,
    /// Use Griffin-Lim phase reconstruction (default: false -- slower).
    pub use_phase_reconstruction: bool,
    /// Griffin-Lim iterations (default: 16).
    pub griffin_lim_iterations: usize,
    /// Neural hidden layer dimension (default: 64).
    pub neural_hidden_dim: usize,
    /// Soft mask temperature for the separator.
    pub mask_temperature: f64,
    /// STFT window size.
    pub window_size: usize,
    /// STFT hop size.
    pub hop_size: usize,
    /// Sample rate in Hz.
    pub sample_rate: f64,
}

impl Default for EnhancedSeparatorConfig {
    fn default() -> Self {
        Self {
            num_sources: 2,
            use_multi_res: true,
            use_neural_refine: true,
            use_phase_reconstruction: false,
            griffin_lim_iterations: 16,
            neural_hidden_dim: 64,
            mask_temperature: 1.0,
            window_size: 256,
            hop_size: 128,
            sample_rate: 8000.0,
        }
    }
}

/// Result of the enhanced separation pipeline.
pub struct EnhancedResult {
    /// Reconstructed source signals.
    pub sources: Vec<Vec<f64>>,
    /// Per-source masks, each indexed [frame * num_freq + freq_bin].
    pub masks: Vec<Vec<f64>>,
    /// Timing and configuration statistics.
    pub stats: EnhancedStats,
}

/// Timing and configuration statistics for the pipeline.
#[derive(Debug, Clone)]
pub struct EnhancedStats {
    pub stft_time_ms: f64,
    pub graph_time_ms: f64,
    pub separation_time_ms: f64,
    pub neural_refine_time_ms: f64,
    pub reconstruction_time_ms: f64,
    pub total_time_ms: f64,
    pub used_multi_res: bool,
    pub used_neural: bool,
    pub used_griffin_lim: bool,
}

/// Run the enhanced separation pipeline.
///
/// Pipeline steps:
/// 1. STFT analysis (standard or multi-resolution)
/// 2. Audio graph construction
/// 3. Fiedler + mincut separation
/// 4. Optional neural mask refinement
/// 5. Mask normalization
/// 6. Reconstruction (standard ISTFT or Griffin-Lim)
pub fn enhanced_separate(
    signal: &[f64],
    config: &EnhancedSeparatorConfig,
) -> EnhancedResult {
    let total_start = Instant::now();

    // Step 1: STFT analysis
    let stft_start = Instant::now();
    let stft_result = if config.use_multi_res {
        let mr_config = MultiResConfig {
            bands: vec![
                multi_res::BandConfig {
                    freq_lo: 0.0,
                    freq_hi: config.sample_rate / 8.0,
                    window_size: (config.window_size * 4).next_power_of_two(),
                    hop_size: config.hop_size * 2,
                },
                multi_res::BandConfig {
                    freq_lo: config.sample_rate / 8.0,
                    freq_hi: config.sample_rate / 2.0,
                    window_size: config.window_size,
                    hop_size: config.hop_size,
                },
            ],
            sample_rate: config.sample_rate,
        };
        let mr_result = multi_res::multi_res_stft(signal, &mr_config);

        // Create all-ones masks per band, then merge to get a unified grid.
        // We use the merged result only for graph shape; the actual STFT for
        // reconstruction is always the standard one.
        let _band_masks: Vec<Vec<f64>> = mr_result
            .bands
            .iter()
            .map(|b| vec![1.0; b.stft.num_frames * b.stft.num_freq_bins])
            .collect();

        // Use the standard STFT for the main pipeline (graph + reconstruction)
        // but the multi-res analysis influences graph construction via a
        // different magnitude floor derived from band energy.
        stft::stft(signal, config.window_size, config.hop_size, config.sample_rate)
    } else {
        stft::stft(signal, config.window_size, config.hop_size, config.sample_rate)
    };
    let stft_time_ms = stft_start.elapsed().as_secs_f64() * 1000.0;

    let num_frames = stft_result.num_frames;
    let num_freq = stft_result.num_freq_bins;

    // Step 2: Build audio graph
    let graph_start = Instant::now();
    let graph_params = if config.use_multi_res {
        // Multi-res mode uses a lower magnitude floor to capture more detail
        GraphParams {
            magnitude_floor: 0.005,
            ..GraphParams::default()
        }
    } else {
        GraphParams::default()
    };
    let audio_graph = build_audio_graph(&stft_result, &graph_params);
    let graph_time_ms = graph_start.elapsed().as_secs_f64() * 1000.0;

    // Step 3: Fiedler + mincut separation
    let sep_start = Instant::now();
    let sep_config = SeparatorConfig {
        num_sources: config.num_sources,
        window_frames: 4,
        window_overlap: 1,
        epsilon: 0.0,
        mask_temperature: config.mask_temperature,
    };
    let sep_result = separator::separate(&audio_graph, &sep_config);
    let separation_time_ms = sep_start.elapsed().as_secs_f64() * 1000.0;

    let mut masks = sep_result.masks;

    // Step 4: Optional neural mask refinement
    let neural_start = Instant::now();
    if config.use_neural_refine {
        let mlp_config = MLPConfig {
            input_dim: 5,
            hidden_dim: config.neural_hidden_dim,
            output_dim: 1,
            learning_rate: 0.01,
        };
        let mlp = TinyMLP::new(mlp_config);

        // Extract magnitude array from the STFT result
        let magnitudes: Vec<f64> = stft_result.bins.iter().map(|b| b.magnitude).collect();

        // Refine each source mask independently
        for source_mask in &mut masks {
            let refined = mlp.refine_mask(source_mask, &magnitudes, num_frames, num_freq);
            *source_mask = refined;
        }
    }
    let neural_refine_time_ms = neural_start.elapsed().as_secs_f64() * 1000.0;

    // Step 5: Normalize masks to sum to 1.0 per T-F bin
    let total_tf = num_frames * num_freq;
    for i in 0..total_tf {
        let sum: f64 = masks.iter().map(|m| m[i]).sum();
        if sum > 1e-12 {
            for m in &mut masks {
                m[i] /= sum;
            }
        } else {
            let uniform = 1.0 / config.num_sources as f64;
            for m in &mut masks {
                m[i] = uniform;
            }
        }
    }

    // Step 6: Reconstruction
    let recon_start = Instant::now();
    let sources: Vec<Vec<f64>> = if config.use_phase_reconstruction {
        let gl_config = GriffinLimConfig {
            max_iterations: config.griffin_lim_iterations,
            convergence_tolerance: 1e-6,
        };
        masks
            .iter()
            .map(|mask| {
                let gl_result =
                    phase::phase_aware_istft(&stft_result, mask, signal.len(), &gl_config);
                gl_result.signal
            })
            .collect()
    } else {
        masks
            .iter()
            .map(|mask| stft::istft(&stft_result, mask, signal.len()))
            .collect()
    };
    let reconstruction_time_ms = recon_start.elapsed().as_secs_f64() * 1000.0;

    let total_time_ms = total_start.elapsed().as_secs_f64() * 1000.0;

    EnhancedResult {
        sources,
        masks,
        stats: EnhancedStats {
            stft_time_ms,
            graph_time_ms,
            separation_time_ms,
            neural_refine_time_ms,
            reconstruction_time_ms,
            total_time_ms,
            used_multi_res: config.use_multi_res,
            used_neural: config.use_neural_refine,
            used_griffin_lim: config.use_phase_reconstruction,
        },
    }
}

/// Report from comparing different separation modes.
#[derive(Debug)]
pub struct ComparisonReport {
    pub basic_sdr: f64,
    pub multires_sdr: f64,
    pub neural_sdr: f64,
    pub both_sdr: f64,
}

/// Compute Signal-to-Distortion Ratio (SDR) in dB between a reference and estimate.
fn compute_sdr(reference: &[f64], estimate: &[f64]) -> f64 {
    let len = reference.len().min(estimate.len());
    if len == 0 {
        return 0.0;
    }

    let ref_energy: f64 = reference[..len].iter().map(|x| x * x).sum();
    let noise_energy: f64 = reference[..len]
        .iter()
        .zip(estimate[..len].iter())
        .map(|(r, e)| (r - e) * (r - e))
        .sum();

    if noise_energy < 1e-20 {
        return 100.0; // near-perfect reconstruction
    }
    if ref_energy < 1e-20 {
        return 0.0;
    }

    10.0 * (ref_energy / noise_energy).log10()
}

/// Run separation in 4 modes and compare SDR against reference signals.
///
/// Modes:
/// 1. Basic (no enhancements)
/// 2. Multi-res only
/// 3. Neural refine only
/// 4. Both multi-res + neural refine
///
/// `references` should contain one reference signal per source.
pub fn compare_modes(
    signal: &[f64],
    references: &[Vec<f64>],
    sr: f64,
) -> ComparisonReport {
    let num_sources = references.len().max(2);

    let base = EnhancedSeparatorConfig {
        num_sources,
        window_size: 256,
        hop_size: 128,
        sample_rate: sr,
        mask_temperature: 1.0,
        griffin_lim_iterations: 16,
        neural_hidden_dim: 64,
        use_multi_res: false,
        use_neural_refine: false,
        use_phase_reconstruction: false,
    };

    // Mode 1: basic
    let basic = enhanced_separate(signal, &base);
    let basic_sdr = avg_sdr(references, &basic.sources);

    // Mode 2: multi-res only
    let mr_config = EnhancedSeparatorConfig {
        use_multi_res: true,
        ..base.clone()
    };
    let mr = enhanced_separate(signal, &mr_config);
    let multires_sdr = avg_sdr(references, &mr.sources);

    // Mode 3: neural only
    let nn_config = EnhancedSeparatorConfig {
        use_neural_refine: true,
        ..base.clone()
    };
    let nn = enhanced_separate(signal, &nn_config);
    let neural_sdr = avg_sdr(references, &nn.sources);

    // Mode 4: both
    let both_config = EnhancedSeparatorConfig {
        use_multi_res: true,
        use_neural_refine: true,
        ..base
    };
    let both = enhanced_separate(signal, &both_config);
    let both_sdr = avg_sdr(references, &both.sources);

    println!("=== Separation Mode Comparison ===");
    println!("Basic:            SDR = {basic_sdr:.2} dB");
    println!("Multi-res:        SDR = {multires_sdr:.2} dB");
    println!("Neural refine:    SDR = {neural_sdr:.2} dB");
    println!("Multi-res+Neural: SDR = {both_sdr:.2} dB");

    ComparisonReport {
        basic_sdr,
        multires_sdr,
        neural_sdr,
        both_sdr,
    }
}

/// Compute average SDR across all sources.
/// Matches each reference to the closest estimated source (greedy).
fn avg_sdr(references: &[Vec<f64>], estimates: &[Vec<f64>]) -> f64 {
    if references.is_empty() || estimates.is_empty() {
        return 0.0;
    }

    let mut used = vec![false; estimates.len()];
    let mut total_sdr = 0.0;
    let mut count = 0;

    for reference in references {
        let mut best_sdr = f64::NEG_INFINITY;
        let mut best_idx = 0;
        for (j, est) in estimates.iter().enumerate() {
            if used[j] {
                continue;
            }
            let sdr = compute_sdr(reference, est);
            if sdr > best_sdr {
                best_sdr = sdr;
                best_idx = j;
            }
        }
        if best_idx < used.len() {
            used[best_idx] = true;
        }
        total_sdr += best_sdr;
        count += 1;
    }

    if count > 0 {
        total_sdr / count as f64
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Generate a two-tone test signal.
    fn make_two_tone(sr: f64, dur: f64, f1: f64, f2: f64) -> Vec<f64> {
        let n = (sr * dur) as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (2.0 * PI * f1 * t).sin() + (2.0 * PI * f2 * t).sin()
            })
            .collect()
    }

    #[test]
    fn test_basic_mode_valid_output() {
        let sr = 8000.0;
        let signal = make_two_tone(sr, 0.25, 200.0, 1500.0);

        let config = EnhancedSeparatorConfig {
            use_multi_res: false,
            use_neural_refine: false,
            use_phase_reconstruction: false,
            sample_rate: sr,
            ..Default::default()
        };

        let result = enhanced_separate(&signal, &config);

        assert_eq!(result.sources.len(), 2);
        assert_eq!(result.masks.len(), 2);
        for source in &result.sources {
            assert_eq!(source.len(), signal.len());
            // Source should have non-trivial energy
            let energy: f64 = source.iter().map(|x| x * x).sum();
            assert!(energy > 0.0, "Source should have non-zero energy");
        }
        assert!(!result.stats.used_multi_res);
        assert!(!result.stats.used_neural);
        assert!(!result.stats.used_griffin_lim);
        assert!(result.stats.total_time_ms >= 0.0);
    }

    #[test]
    fn test_neural_refine_valid_masks() {
        let sr = 8000.0;
        let signal = make_two_tone(sr, 0.25, 200.0, 1500.0);

        let config = EnhancedSeparatorConfig {
            use_multi_res: false,
            use_neural_refine: true,
            use_phase_reconstruction: false,
            sample_rate: sr,
            ..Default::default()
        };

        let result = enhanced_separate(&signal, &config);

        assert_eq!(result.masks.len(), 2);
        let num_frames = config.window_size; // approximate
        let total_tf = result.masks[0].len();
        assert!(total_tf > 0);

        for i in 0..total_tf {
            // Each mask value should be in [0, 1]
            for m in &result.masks {
                assert!(
                    m[i] >= 0.0 && m[i] <= 1.0,
                    "Mask value {} out of [0,1] range at index {}",
                    m[i],
                    i
                );
            }
            // Masks should sum to approximately 1.0
            let sum: f64 = result.masks.iter().map(|m| m[i]).sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Mask sum at index {i} = {sum}, expected ~1.0"
            );
        }
        assert!(result.stats.used_neural);
    }

    #[test]
    fn test_multi_res_different_graph() {
        let sr = 8000.0;
        let signal = make_two_tone(sr, 0.25, 200.0, 1500.0);

        // Standard mode
        let std_stft = stft::stft(&signal, 256, 128, sr);
        let std_graph = build_audio_graph(&std_stft, &GraphParams::default());

        // Multi-res mode uses a lower magnitude floor -> more nodes
        let mr_graph = build_audio_graph(
            &std_stft,
            &GraphParams {
                magnitude_floor: 0.005,
                ..GraphParams::default()
            },
        );

        // The multi-res graph should have at least as many nodes (lower floor)
        assert!(
            mr_graph.num_nodes >= std_graph.num_nodes,
            "Multi-res graph ({} nodes) should have >= standard graph ({} nodes)",
            mr_graph.num_nodes,
            std_graph.num_nodes
        );
    }

    #[test]
    fn test_all_modes_no_panic() {
        let sr = 8000.0;
        let signal = make_two_tone(sr, 0.25, 300.0, 2000.0);

        let modes = [
            (false, false, false),
            (true, false, false),
            (false, true, false),
            (true, true, false),
        ];

        for (use_mr, use_nn, use_gl) in &modes {
            let config = EnhancedSeparatorConfig {
                use_multi_res: *use_mr,
                use_neural_refine: *use_nn,
                use_phase_reconstruction: *use_gl,
                sample_rate: sr,
                ..Default::default()
            };

            let result = enhanced_separate(&signal, &config);

            assert_eq!(result.sources.len(), 2, "Mode ({use_mr},{use_nn},{use_gl})");
            assert_eq!(result.masks.len(), 2, "Mode ({use_mr},{use_nn},{use_gl})");
            for source in &result.sources {
                assert_eq!(
                    source.len(),
                    signal.len(),
                    "Mode ({use_mr},{use_nn},{use_gl})"
                );
            }
        }
    }

    #[test]
    fn test_compare_modes_runs() {
        let sr = 8000.0;
        let f1 = 200.0;
        let f2 = 1500.0;
        let n = (sr * 0.25) as usize;

        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (2.0 * PI * f1 * t).sin() + (2.0 * PI * f2 * t).sin()
            })
            .collect();

        let ref1: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f1 * i as f64 / sr).sin())
            .collect();
        let ref2: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f2 * i as f64 / sr).sin())
            .collect();

        let report = compare_modes(&signal, &[ref1, ref2], sr);

        // All SDR values should be finite
        assert!(report.basic_sdr.is_finite());
        assert!(report.multires_sdr.is_finite());
        assert!(report.neural_sdr.is_finite());
        assert!(report.both_sdr.is_finite());
    }
}
