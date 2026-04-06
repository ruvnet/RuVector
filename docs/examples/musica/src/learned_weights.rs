//! Gradient-free optimization of graph construction weights.
//!
//! Uses Nelder-Mead simplex search to optimize spectral_weight, temporal_weight,
//! harmonic_weight, phase_threshold, and onset_weight to maximize SDR on
//! a training set. No neural network required — just direct parameter search.

use crate::audio_graph::{build_audio_graph, GraphParams};
use crate::separator::{separate, SeparatorConfig};
use crate::stft;

/// Training scenario: a mixed signal with known source references.
pub struct TrainingSample {
    pub mixed: Vec<f64>,
    pub references: Vec<Vec<f64>>,
    pub sample_rate: f64,
}

/// Result of weight optimization.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub best_params: GraphParams,
    pub best_sdr: f64,
    pub iterations: usize,
    pub history: Vec<f64>,
}

/// Evaluate a set of graph params on a training sample, returning average SDR.
fn evaluate_params(
    params: &GraphParams,
    sample: &TrainingSample,
    window_size: usize,
    hop_size: usize,
) -> f64 {
    let stft_result = stft::stft(&sample.mixed, window_size, hop_size, sample.sample_rate);
    let graph = build_audio_graph(&stft_result, params);
    let config = SeparatorConfig {
        num_sources: sample.references.len(),
        ..SeparatorConfig::default()
    };
    let result = separate(&graph, &config);
    let n = sample.mixed.len();

    // Compute SDR with best permutation (for 2 sources)
    let sources: Vec<Vec<f64>> = result.masks.iter()
        .map(|m| stft::istft(&stft_result, m, n))
        .collect();

    best_permutation_avg_sdr(&sample.references, &sources)
}

/// Compute average SDR with best permutation for 2 sources.
fn best_permutation_avg_sdr(references: &[Vec<f64>], estimates: &[Vec<f64>]) -> f64 {
    let k = references.len().min(estimates.len());
    if k == 0 { return -60.0; }
    if k == 1 {
        return compute_sdr(&references[0], &estimates[0]);
    }

    // Try both permutations
    let sdr_id = (compute_sdr(&references[0], &estimates[0])
        + compute_sdr(&references[1], &estimates[1])) / 2.0;
    let sdr_sw = (compute_sdr(&references[0], &estimates[1])
        + compute_sdr(&references[1], &estimates[0])) / 2.0;
    sdr_id.max(sdr_sw)
}

fn compute_sdr(reference: &[f64], estimate: &[f64]) -> f64 {
    let n = reference.len().min(estimate.len());
    if n == 0 { return -60.0; }
    let ref_e: f64 = reference[..n].iter().map(|x| x * x).sum();
    let noise_e: f64 = reference[..n].iter().zip(estimate[..n].iter())
        .map(|(r, e)| (r - e).powi(2)).sum();
    if ref_e < 1e-12 { return -60.0; }
    if noise_e < 1e-12 { return 100.0; }
    (10.0 * (ref_e / noise_e).log10()).clamp(-60.0, 100.0)
}

/// Convert GraphParams to a parameter vector for optimization.
fn params_to_vec(p: &GraphParams) -> Vec<f64> {
    vec![
        p.spectral_weight,
        p.temporal_weight,
        p.harmonic_weight,
        p.phase_threshold,
        p.onset_weight,
        p.magnitude_floor,
    ]
}

/// Convert parameter vector back to GraphParams.
fn vec_to_params(v: &[f64]) -> GraphParams {
    GraphParams {
        spectral_weight: v[0].max(0.01),
        temporal_weight: v[1].max(0.01),
        harmonic_weight: v[2].max(0.0),
        phase_threshold: v[3].clamp(0.1, std::f64::consts::PI),
        onset_weight: v[4].max(0.0),
        magnitude_floor: v[5].clamp(0.001, 0.1),
        ..GraphParams::default()
    }
}

/// Optimize graph weights using Nelder-Mead simplex search.
pub fn optimize_weights(
    samples: &[TrainingSample],
    max_iterations: usize,
    window_size: usize,
    hop_size: usize,
) -> OptimizationResult {
    let initial = GraphParams::default();
    let dim = 6;

    // Initialize simplex: initial point + dim perturbations
    let x0 = params_to_vec(&initial);
    let mut simplex: Vec<(Vec<f64>, f64)> = Vec::with_capacity(dim + 1);

    let eval = |v: &[f64]| -> f64 {
        let params = vec_to_params(v);
        let mut total_sdr = 0.0;
        for sample in samples {
            total_sdr += evaluate_params(&params, sample, window_size, hop_size);
        }
        total_sdr / samples.len() as f64
    };

    // Initial point
    let f0 = eval(&x0);
    simplex.push((x0.clone(), f0));

    // Perturbed points
    for i in 0..dim {
        let mut xi = x0.clone();
        xi[i] *= 1.3; // 30% perturbation
        let fi = eval(&xi);
        simplex.push((xi, fi));
    }

    let mut history = vec![f0];

    // Nelder-Mead iterations
    let alpha = 1.0; // reflection
    let gamma = 2.0; // expansion
    let rho = 0.5;   // contraction
    let sigma = 0.5;  // shrink

    for iter in 0..max_iterations {
        // Sort by objective (higher = better, so sort descending)
        simplex.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        history.push(simplex[0].1);

        let best = &simplex[0].1;
        let worst_idx = simplex.len() - 1;
        let second_worst_idx = worst_idx - 1;

        // Check convergence
        let spread = simplex[0].1 - simplex[worst_idx].1;
        if spread < 0.01 && iter > 10 {
            break;
        }

        // Centroid (excluding worst)
        let mut centroid = vec![0.0; dim];
        for i in 0..worst_idx {
            for d in 0..dim {
                centroid[d] += simplex[i].0[d];
            }
        }
        for d in 0..dim {
            centroid[d] /= worst_idx as f64;
        }

        // Reflection
        let reflected: Vec<f64> = (0..dim)
            .map(|d| centroid[d] + alpha * (centroid[d] - simplex[worst_idx].0[d]))
            .collect();
        let f_reflected = eval(&reflected);

        if f_reflected > simplex[second_worst_idx].1 && f_reflected <= *best {
            simplex[worst_idx] = (reflected, f_reflected);
            continue;
        }

        if f_reflected > *best {
            // Expansion
            let expanded: Vec<f64> = (0..dim)
                .map(|d| centroid[d] + gamma * (reflected[d] - centroid[d]))
                .collect();
            let f_expanded = eval(&expanded);
            if f_expanded > f_reflected {
                simplex[worst_idx] = (expanded, f_expanded);
            } else {
                simplex[worst_idx] = (reflected, f_reflected);
            }
            continue;
        }

        // Contraction
        let contracted: Vec<f64> = (0..dim)
            .map(|d| centroid[d] + rho * (simplex[worst_idx].0[d] - centroid[d]))
            .collect();
        let f_contracted = eval(&contracted);
        if f_contracted > simplex[worst_idx].1 {
            simplex[worst_idx] = (contracted, f_contracted);
            continue;
        }

        // Shrink
        let best_point = simplex[0].0.clone();
        for i in 1..simplex.len() {
            for d in 0..dim {
                simplex[i].0[d] = best_point[d] + sigma * (simplex[i].0[d] - best_point[d]);
            }
            simplex[i].1 = eval(&simplex[i].0);
        }
    }

    simplex.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    OptimizationResult {
        best_params: vec_to_params(&simplex[0].0),
        best_sdr: simplex[0].1,
        iterations: history.len(),
        history,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn sine(freq: f64, sr: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| (2.0 * PI * freq * i as f64 / sr).sin()).collect()
    }

    #[test]
    fn test_evaluate_params() {
        let sr = 8000.0;
        let n = 2000;
        let s1 = sine(200.0, sr, n);
        let s2 = sine(2000.0, sr, n);
        let mixed: Vec<f64> = s1.iter().zip(s2.iter()).map(|(a, b)| a + b).collect();

        let sample = TrainingSample {
            mixed,
            references: vec![s1, s2],
            sample_rate: sr,
        };

        let sdr = evaluate_params(&GraphParams::default(), &sample, 256, 128);
        assert!(sdr.is_finite());
    }

    #[test]
    fn test_optimize_weights_runs() {
        let sr = 8000.0;
        let n = 2000;
        let s1 = sine(200.0, sr, n);
        let s2 = sine(2000.0, sr, n);
        let mixed: Vec<f64> = s1.iter().zip(s2.iter()).map(|(a, b)| a + b).collect();

        let samples = vec![TrainingSample {
            mixed,
            references: vec![s1, s2],
            sample_rate: sr,
        }];

        let result = optimize_weights(&samples, 5, 256, 128);
        assert!(result.best_sdr.is_finite());
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_nelder_mead_improves() {
        let sr = 8000.0;
        let n = 2000;

        // Create a harder scenario: close tones
        let s1 = sine(400.0, sr, n);
        let s2 = sine(600.0, sr, n);
        let mixed: Vec<f64> = s1.iter().zip(s2.iter()).map(|(a, b)| a + b).collect();

        let samples = vec![TrainingSample {
            mixed,
            references: vec![s1, s2],
            sample_rate: sr,
        }];

        let default_sdr = evaluate_params(&GraphParams::default(), &samples[0], 256, 128);
        let result = optimize_weights(&samples, 15, 256, 128);

        // Should not get worse (may not always improve on simple scenarios)
        assert!(result.best_sdr >= default_sdr - 1.0,
            "Optimized {:.2} should be >= default {:.2} - 1.0",
            result.best_sdr, default_sdr);
    }
}
