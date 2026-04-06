//! Benchmark harness for audio mincut separation.
//!
//! Measures SDR (Signal-to-Distortion Ratio), SIR (Signal-to-Interference Ratio),
//! and processing time. Compares mincut separation against a frequency-band baseline.

use std::f64::consts::PI;
use std::time::Instant;

use crate::audio_graph::{build_audio_graph, GraphParams};
use crate::separator::{separate, SeparationResult, SeparatorConfig};
use crate::stft::{self, StftResult};

/// Signal quality metrics (BSS_EVAL style).
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Signal-to-Distortion Ratio (dB). Higher is better.
    pub sdr: f64,
    /// Signal-to-Interference Ratio (dB). Higher is better.
    pub sir: f64,
    /// Signal-to-Artifact Ratio (dB). Higher is better.
    pub sar: f64,
    /// Energy ratio between recovered and original.
    pub energy_ratio: f64,
}

/// Benchmark result.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Method name.
    pub method: String,
    /// Quality per source.
    pub quality: Vec<QualityMetrics>,
    /// Processing time in milliseconds.
    pub elapsed_ms: f64,
    /// Graph construction time in milliseconds.
    pub graph_build_ms: f64,
    /// Separation time in milliseconds.
    pub separation_ms: f64,
    /// Number of graph nodes.
    pub num_nodes: usize,
    /// Number of graph edges.
    pub num_edges: usize,
}

/// Generate a synthetic test signal: sum of N sinusoids.
pub fn generate_test_signal(
    sample_rate: f64,
    duration: f64,
    frequencies: &[f64],
    amplitudes: &[f64],
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = (sample_rate * duration) as usize;
    let mut mixed = vec![0.0; n];
    let mut sources = Vec::new();

    for (i, (&freq, &amp)) in frequencies.iter().zip(amplitudes.iter()).enumerate() {
        let source: Vec<f64> = (0..n)
            .map(|j| {
                let t = j as f64 / sample_rate;
                // Add some harmonics for realism
                amp * (2.0 * PI * freq * t).sin()
                    + amp * 0.3 * (2.0 * PI * freq * 2.0 * t).sin()
                    + amp * 0.1 * (2.0 * PI * freq * 3.0 * t).sin()
                    + amp * 0.05 * (i as f64 * 0.1 * t).sin() // Slow modulation
            })
            .collect();

        for (j, &s) in source.iter().enumerate() {
            mixed[j] += s;
        }
        sources.push(source);
    }

    (mixed, sources)
}

/// Compute SDR between reference and estimated signals.
fn compute_sdr(reference: &[f64], estimated: &[f64]) -> f64 {
    let n = reference.len().min(estimated.len());
    if n == 0 {
        return f64::NEG_INFINITY;
    }

    let ref_energy: f64 = reference[..n].iter().map(|x| x * x).sum();
    let noise_energy: f64 = reference[..n]
        .iter()
        .zip(estimated[..n].iter())
        .map(|(r, e)| (r - e) * (r - e))
        .sum();

    if noise_energy < 1e-12 {
        return 100.0; // Perfect reconstruction
    }
    if ref_energy < 1e-12 {
        return f64::NEG_INFINITY;
    }

    10.0 * (ref_energy / noise_energy).log10()
}

/// Compute SIR: how much of the target signal leaks into other sources.
fn compute_sir(reference: &[f64], estimated: &[f64], interference: &[f64]) -> f64 {
    let n = reference.len().min(estimated.len()).min(interference.len());
    if n == 0 {
        return f64::NEG_INFINITY;
    }

    // Project estimated onto reference
    let ref_energy: f64 = reference[..n].iter().map(|x| x * x).sum();
    if ref_energy < 1e-12 {
        return f64::NEG_INFINITY;
    }

    let cross: f64 = reference[..n]
        .iter()
        .zip(estimated[..n].iter())
        .map(|(r, e)| r * e)
        .sum();
    let scale = cross / ref_energy;

    let target_proj: Vec<f64> = reference[..n].iter().map(|r| r * scale).collect();
    let target_energy: f64 = target_proj.iter().map(|x| x * x).sum();

    // Interference component
    let interf_energy: f64 = interference[..n].iter().map(|x| x * x).sum();

    if interf_energy < 1e-12 {
        return 100.0;
    }

    10.0 * (target_energy / interf_energy).log10()
}

/// Compute SAR (artifact ratio).
fn compute_sar(reference: &[f64], estimated: &[f64]) -> f64 {
    let n = reference.len().min(estimated.len());
    if n == 0 {
        return f64::NEG_INFINITY;
    }

    let est_energy: f64 = estimated[..n].iter().map(|x| x * x).sum();
    let artifact_energy: f64 = reference[..n]
        .iter()
        .zip(estimated[..n].iter())
        .map(|(r, e)| {
            let diff = e - r;
            diff * diff
        })
        .sum();

    if artifact_energy < 1e-12 {
        return 100.0;
    }

    10.0 * (est_energy / artifact_energy).log10()
}

/// Run mincut separation benchmark.
pub fn benchmark_mincut(
    mixed: &[f64],
    ground_truth: &[Vec<f64>],
    sample_rate: f64,
    window_size: usize,
    hop_size: usize,
    graph_params: &GraphParams,
    sep_config: &SeparatorConfig,
) -> BenchmarkResult {
    let start = Instant::now();

    // STFT
    let stft_result = stft::stft(mixed, window_size, hop_size, sample_rate);

    // Build audio graph
    let t_graph = Instant::now();
    let ag = build_audio_graph(&stft_result, graph_params);
    let graph_build_ms = t_graph.elapsed().as_secs_f64() * 1000.0;

    // Separate
    let t_sep = Instant::now();
    let sep_result = separate(&ag, sep_config);
    let separation_ms = t_sep.elapsed().as_secs_f64() * 1000.0;

    // Reconstruct sources
    let quality = evaluate_separation(
        &stft_result,
        &sep_result,
        mixed.len(),
        ground_truth,
    );

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    BenchmarkResult {
        method: "Dynamic MinCut".to_string(),
        quality,
        elapsed_ms,
        graph_build_ms,
        separation_ms,
        num_nodes: ag.num_nodes,
        num_edges: ag.num_edges,
    }
}

/// Frequency-band baseline: simple high/low pass split.
pub fn benchmark_freq_baseline(
    mixed: &[f64],
    ground_truth: &[Vec<f64>],
    sample_rate: f64,
    window_size: usize,
    hop_size: usize,
    num_sources: usize,
) -> BenchmarkResult {
    let start = Instant::now();

    let stft_result = stft::stft(mixed, window_size, hop_size, sample_rate);
    let num_freq = stft_result.num_freq_bins;
    let total_bins = stft_result.bins.len();

    // Create masks by splitting frequency range evenly
    let bins_per_source = num_freq / num_sources.max(1);
    let mut masks = vec![vec![0.0; total_bins]; num_sources];

    for frame in 0..stft_result.num_frames {
        for f in 0..num_freq {
            let source = (f / bins_per_source).min(num_sources - 1);
            let idx = frame * num_freq + f;
            masks[source][idx] = 1.0;
        }
    }

    // Reconstruct and evaluate
    let quality = evaluate_with_masks(
        &stft_result,
        &masks,
        mixed.len(),
        ground_truth,
    );

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    BenchmarkResult {
        method: "Frequency Band Split".to_string(),
        quality,
        elapsed_ms,
        graph_build_ms: 0.0,
        separation_ms: elapsed_ms,
        num_nodes: 0,
        num_edges: 0,
    }
}

/// Evaluate separation quality.
fn evaluate_separation(
    stft_result: &StftResult,
    sep_result: &SeparationResult,
    signal_len: usize,
    ground_truth: &[Vec<f64>],
) -> Vec<QualityMetrics> {
    evaluate_with_masks(stft_result, &sep_result.masks, signal_len, ground_truth)
}

/// Evaluate quality given masks.
fn evaluate_with_masks(
    stft_result: &StftResult,
    masks: &[Vec<f64>],
    signal_len: usize,
    ground_truth: &[Vec<f64>],
) -> Vec<QualityMetrics> {
    let num_sources = masks.len().min(ground_truth.len());
    let mut quality = Vec::new();

    for s in 0..num_sources {
        let recovered = stft::istft(stft_result, &masks[s], signal_len);
        let reference = &ground_truth[s];

        let sdr = compute_sdr(reference, &recovered);
        let sir = if num_sources > 1 {
            let other_idx = if s == 0 { 1 } else { 0 };
            compute_sir(reference, &recovered, &ground_truth[other_idx])
        } else {
            100.0
        };
        let sar = compute_sar(reference, &recovered);

        let ref_energy: f64 = reference.iter().map(|x| x * x).sum();
        let rec_energy: f64 = recovered.iter().map(|x| x * x).sum();
        let energy_ratio = if ref_energy > 1e-12 {
            rec_energy / ref_energy
        } else {
            0.0
        };

        quality.push(QualityMetrics {
            sdr,
            sir,
            sar,
            energy_ratio,
        });
    }

    quality
}

/// Print benchmark comparison table.
pub fn print_comparison(results: &[BenchmarkResult]) {
    println!("\n{:=<80}", "");
    println!("  BENCHMARK COMPARISON");
    println!("{:=<80}", "");

    for result in results {
        println!("\n  Method: {}", result.method);
        println!("  {:-<60}", "");
        println!("  Time:  total={:.1}ms  graph={:.1}ms  sep={:.1}ms",
            result.elapsed_ms, result.graph_build_ms, result.separation_ms);
        if result.num_nodes > 0 {
            println!("  Graph: {} nodes, {} edges", result.num_nodes, result.num_edges);
        }

        for (i, q) in result.quality.iter().enumerate() {
            println!(
                "  Source {}: SDR={:+.1}dB  SIR={:+.1}dB  SAR={:+.1}dB  energy={:.2}",
                i, q.sdr, q.sir, q.sar, q.energy_ratio
            );
        }
    }

    // Side-by-side if 2 results
    if results.len() >= 2 {
        println!("\n  {:-<60}", "");
        println!("  DELTA (MinCut vs Baseline)");
        let mc = &results[0];
        let bl = &results[1];

        let n = mc.quality.len().min(bl.quality.len());
        for i in 0..n {
            let dsdr = mc.quality[i].sdr - bl.quality[i].sdr;
            let dsir = mc.quality[i].sir - bl.quality[i].sir;
            println!(
                "  Source {}: dSDR={:+.1}dB  dSIR={:+.1}dB",
                i, dsdr, dsir
            );
        }
    }

    println!("\n{:=<80}\n", "");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdr_perfect() {
        let signal: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let sdr = compute_sdr(&signal, &signal);
        assert!(sdr > 90.0, "Perfect reconstruction should have high SDR");
    }

    #[test]
    fn test_sdr_noisy() {
        let signal: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.1).sin()).collect();
        let noisy: Vec<f64> = signal.iter().map(|&s| s + 0.1).collect();
        let sdr = compute_sdr(&signal, &noisy);
        assert!(sdr > 0.0, "SDR should be positive with small noise");
        assert!(sdr < 50.0, "SDR should be finite with noise");
    }

    #[test]
    fn test_generate_test_signal() {
        let (mixed, sources) = generate_test_signal(
            8000.0, 0.5,
            &[200.0, 1000.0],
            &[1.0, 0.8],
        );
        assert_eq!(sources.len(), 2);
        assert_eq!(mixed.len(), 4000);
        assert_eq!(sources[0].len(), 4000);
    }
}
