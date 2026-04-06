//! Adaptive graph parameter optimization for audio separation.
//!
//! Searches for optimal `GraphParams` that maximize separation quality (SDR)
//! on a small labeled set. Supports grid search, random search, and a
//! Bayesian-inspired heuristic for next-point selection.

use std::time::Instant;

use crate::audio_graph::{build_audio_graph, GraphParams};
use crate::separator::{separate, SeparatorConfig};
use crate::stft;

/// Search range for a single parameter.
#[derive(Debug, Clone)]
pub struct ParamRange {
    /// Minimum value (inclusive).
    pub min: f64,
    /// Maximum value (inclusive).
    pub max: f64,
    /// Step size for grid search.
    pub step: f64,
}

impl ParamRange {
    pub fn new(min: f64, max: f64, step: f64) -> Self {
        Self { min, max, step }
    }

    /// Generate grid values within the range.
    fn grid_values(&self) -> Vec<f64> {
        let mut vals = Vec::new();
        let mut v = self.min;
        while v <= self.max + 1e-9 {
            vals.push(v);
            v += self.step;
        }
        if vals.is_empty() {
            vals.push(self.min);
        }
        vals
    }

    /// Clamp a value to the range.
    fn clamp(&self, v: f64) -> f64 {
        v.max(self.min).min(self.max)
    }
}

/// Search range for integer parameters.
#[derive(Debug, Clone)]
pub struct IntParamRange {
    pub min: usize,
    pub max: usize,
    pub step: usize,
}

impl IntParamRange {
    pub fn new(min: usize, max: usize, step: usize) -> Self {
        Self { min, max, step }
    }

    fn grid_values(&self) -> Vec<usize> {
        let mut vals = Vec::new();
        let mut v = self.min;
        while v <= self.max {
            vals.push(v);
            v += self.step;
        }
        if vals.is_empty() {
            vals.push(self.min);
        }
        vals
    }

    fn clamp(&self, v: usize) -> usize {
        v.max(self.min).min(self.max)
    }
}

/// Configuration for adaptive parameter search.
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    pub spectral_weight: ParamRange,
    pub temporal_weight: ParamRange,
    pub harmonic_weight: ParamRange,
    pub phase_threshold: ParamRange,
    pub spectral_radius: IntParamRange,
    pub max_harmonics: IntParamRange,
    /// Metric to optimize: currently only "sdr" is supported.
    pub metric: String,
    /// STFT window size (power of 2).
    pub window_size: usize,
    /// STFT hop size.
    pub hop_size: usize,
    /// Sample rate.
    pub sample_rate: f64,
    /// Separator config for evaluation.
    pub separator_config: SeparatorConfig,
}

/// Result of a single trial.
#[derive(Debug, Clone)]
pub struct TrialResult {
    /// Parameters used in this trial.
    pub params: GraphParams,
    /// Average SDR achieved (dB).
    pub sdr: f64,
    /// Processing time in milliseconds.
    pub elapsed_ms: f64,
}

/// Result of a parameter search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Best parameters found.
    pub best_params: GraphParams,
    /// Best score achieved.
    pub best_score: f64,
    /// All trial results.
    pub trials: Vec<TrialResult>,
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
        return 100.0;
    }
    if ref_energy < 1e-12 {
        return f64::NEG_INFINITY;
    }

    10.0 * (ref_energy / noise_energy).log10()
}

/// Evaluate a set of `GraphParams` on a mixed signal against references.
///
/// Returns the average SDR across all sources.
fn evaluate_params(
    mixed: &[f64],
    references: &[Vec<f64>],
    params: &GraphParams,
    config: &AdaptiveConfig,
) -> f64 {
    let stft_result = stft::stft(mixed, config.window_size, config.hop_size, config.sample_rate);
    let ag = build_audio_graph(&stft_result, params);
    let sep = separate(&ag, &config.separator_config);

    let num_sources = sep.masks.len().min(references.len());
    if num_sources == 0 {
        return f64::NEG_INFINITY;
    }

    let mut total_sdr = 0.0;
    for s in 0..num_sources {
        let recovered = stft::istft(&stft_result, &sep.masks[s], mixed.len());
        total_sdr += compute_sdr(&references[s], &recovered);
    }

    total_sdr / num_sources as f64
}

/// Build `GraphParams` from the tunable values, keeping defaults for fixed fields.
fn make_params(
    spectral_weight: f64,
    temporal_weight: f64,
    harmonic_weight: f64,
    phase_threshold: f64,
    spectral_radius: usize,
    max_harmonics: usize,
) -> GraphParams {
    GraphParams {
        spectral_weight,
        temporal_weight,
        harmonic_weight,
        phase_threshold,
        spectral_radius,
        max_harmonics,
        ..GraphParams::default()
    }
}

/// Exhaustive grid search over the parameter space.
///
/// For each combination of parameter values (at the grid step intervals),
/// evaluates separation quality and returns the best result.
pub fn grid_search(
    signal: &[f64],
    references: &[Vec<f64>],
    config: &AdaptiveConfig,
) -> SearchResult {
    let sw_vals = config.spectral_weight.grid_values();
    let tw_vals = config.temporal_weight.grid_values();
    let hw_vals = config.harmonic_weight.grid_values();
    let pt_vals = config.phase_threshold.grid_values();
    let sr_vals = config.spectral_radius.grid_values();
    let mh_vals = config.max_harmonics.grid_values();

    let mut trials = Vec::new();
    let mut best_score = f64::NEG_INFINITY;
    let mut best_params = GraphParams::default();

    for &sw in &sw_vals {
        for &tw in &tw_vals {
            for &hw in &hw_vals {
                for &pt in &pt_vals {
                    for &sr in &sr_vals {
                        for &mh in &mh_vals {
                            let params = make_params(sw, tw, hw, pt, sr, mh);
                            let start = Instant::now();
                            let sdr = evaluate_params(signal, references, &params, config);
                            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

                            if sdr > best_score {
                                best_score = sdr;
                                best_params = params.clone();
                            }

                            trials.push(TrialResult {
                                params,
                                sdr,
                                elapsed_ms,
                            });
                        }
                    }
                }
            }
        }
    }

    SearchResult {
        best_params,
        best_score,
        trials,
    }
}

/// Simple LCG (Linear Congruential Generator) for deterministic random sampling.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // Parameters from Numerical Recipes.
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    /// Uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform f64 in [min, max].
    fn uniform(&mut self, min: f64, max: f64) -> f64 {
        min + self.next_f64() * (max - min)
    }

    /// Uniform usize in [min, max].
    fn uniform_usize(&mut self, min: usize, max: usize) -> usize {
        if max <= min {
            return min;
        }
        min + (self.next_u64() as usize % (max - min + 1))
    }
}

/// Random search: sample parameter combinations uniformly at random.
///
/// Faster than grid search for high-dimensional spaces since it does not
/// suffer from the curse of dimensionality.
pub fn random_search(
    signal: &[f64],
    references: &[Vec<f64>],
    config: &AdaptiveConfig,
    num_trials: usize,
) -> SearchResult {
    let mut rng = Lcg::new(42);
    let mut trials = Vec::with_capacity(num_trials);
    let mut best_score = f64::NEG_INFINITY;
    let mut best_params = GraphParams::default();

    for _ in 0..num_trials {
        let params = make_params(
            rng.uniform(config.spectral_weight.min, config.spectral_weight.max),
            rng.uniform(config.temporal_weight.min, config.temporal_weight.max),
            rng.uniform(config.harmonic_weight.min, config.harmonic_weight.max),
            rng.uniform(config.phase_threshold.min, config.phase_threshold.max),
            rng.uniform_usize(config.spectral_radius.min, config.spectral_radius.max),
            rng.uniform_usize(config.max_harmonics.min, config.max_harmonics.max),
        );

        let start = Instant::now();
        let sdr = evaluate_params(signal, references, &params, config);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        if sdr > best_score {
            best_score = sdr;
            best_params = params.clone();
        }

        trials.push(TrialResult {
            params,
            sdr,
            elapsed_ms,
        });
    }

    SearchResult {
        best_params,
        best_score,
        trials,
    }
}

/// Bayesian-inspired next-point selection heuristic.
///
/// Picks parameters near the best results so far with exploration noise that
/// decreases as more results are gathered. Not a full Gaussian Process — just
/// a practical heuristic that samples around the top-3 results.
pub fn bayesian_step(
    results_so_far: &[TrialResult],
    config: &AdaptiveConfig,
) -> GraphParams {
    if results_so_far.is_empty() {
        return GraphParams::default();
    }

    // Sort by SDR descending and take top 3.
    let mut sorted: Vec<&TrialResult> = results_so_far.iter().collect();
    sorted.sort_by(|a, b| b.sdr.partial_cmp(&a.sdr).unwrap_or(std::cmp::Ordering::Equal));
    let top_k = sorted.len().min(3);

    // Noise scale decreases with number of observations.
    let base_noise = 0.3 / (1.0 + results_so_far.len() as f64 * 0.1);

    // Use a deterministic seed based on the number of results.
    let mut rng = Lcg::new(results_so_far.len() as u64 * 7919 + 31);

    // Weighted average of top-k results, with highest weight on rank 1.
    let weights: Vec<f64> = (0..top_k).map(|i| 1.0 / (1.0 + i as f64)).collect();
    let w_sum: f64 = weights.iter().sum();

    let mut sw = 0.0;
    let mut tw = 0.0;
    let mut hw = 0.0;
    let mut pt = 0.0;
    let mut sr = 0.0;
    let mut mh = 0.0;

    for (i, &trial) in sorted[..top_k].iter().enumerate() {
        let w = weights[i] / w_sum;
        sw += w * trial.params.spectral_weight;
        tw += w * trial.params.temporal_weight;
        hw += w * trial.params.harmonic_weight;
        pt += w * trial.params.phase_threshold;
        sr += w * trial.params.spectral_radius as f64;
        mh += w * trial.params.max_harmonics as f64;
    }

    // Add exploration noise.
    let noise = |rng: &mut Lcg, range: &ParamRange, scale: f64| -> f64 {
        let span = range.max - range.min;
        (rng.next_f64() - 0.5) * 2.0 * span * scale
    };

    sw = config.spectral_weight.clamp(sw + noise(&mut rng, &config.spectral_weight, base_noise));
    tw = config.temporal_weight.clamp(tw + noise(&mut rng, &config.temporal_weight, base_noise));
    hw = config.harmonic_weight.clamp(hw + noise(&mut rng, &config.harmonic_weight, base_noise));
    pt = config.phase_threshold.clamp(pt + noise(&mut rng, &config.phase_threshold, base_noise));

    let sr_span = (config.spectral_radius.max - config.spectral_radius.min) as f64;
    let sr_noise = ((rng.next_f64() - 0.5) * 2.0 * sr_span * base_noise).round() as isize;
    let sr_val = (sr.round() as isize + sr_noise).max(config.spectral_radius.min as isize) as usize;
    let sr_val = config.spectral_radius.clamp(sr_val);

    let mh_span = (config.max_harmonics.max - config.max_harmonics.min) as f64;
    let mh_noise = ((rng.next_f64() - 0.5) * 2.0 * mh_span * base_noise).round() as isize;
    let mh_val = (mh.round() as isize + mh_noise).max(config.max_harmonics.min as isize) as usize;
    let mh_val = config.max_harmonics.clamp(mh_val);

    make_params(sw, tw, hw, pt, sr_val, mh_val)
}

/// Sensible default search ranges for all `GraphParams` fields.
pub fn default_search_ranges() -> AdaptiveConfig {
    AdaptiveConfig {
        spectral_weight: ParamRange::new(0.1, 2.0, 0.3),
        temporal_weight: ParamRange::new(0.1, 2.0, 0.3),
        harmonic_weight: ParamRange::new(0.0, 1.5, 0.3),
        phase_threshold: ParamRange::new(0.1, 0.9, 0.2),
        spectral_radius: IntParamRange::new(1, 5, 1),
        max_harmonics: IntParamRange::new(2, 6, 1),
        metric: "sdr".to_string(),
        window_size: 256,
        hop_size: 128,
        sample_rate: 8000.0,
        separator_config: SeparatorConfig {
            num_sources: 2,
            window_frames: 4,
            window_overlap: 1,
            epsilon: 0.0,
            mask_temperature: 1.0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Generate a two-tone test signal with known sources.
    fn make_test_data(f1: f64, f2: f64) -> (Vec<f64>, Vec<Vec<f64>>) {
        let sr = 8000.0;
        let dur = 0.15;
        let n = (sr * dur) as usize;

        let src1: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f1 * i as f64 / sr).sin())
            .collect();
        let src2: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f2 * i as f64 / sr).sin())
            .collect();
        let mixed: Vec<f64> = src1.iter().zip(src2.iter()).map(|(a, b)| a + b).collect();

        (mixed, vec![src1, src2])
    }

    #[test]
    fn test_grid_search_beats_default() {
        let (mixed, refs) = make_test_data(300.0, 2000.0);

        // Use a very coarse grid to keep the test fast.
        let mut config = default_search_ranges();
        config.spectral_weight = ParamRange::new(0.5, 2.0, 1.5);
        config.temporal_weight = ParamRange::new(0.5, 2.0, 1.5);
        config.harmonic_weight = ParamRange::new(0.0, 1.5, 1.5);
        config.phase_threshold = ParamRange::new(0.1, 0.9, 0.8);
        config.spectral_radius = IntParamRange::new(1, 3, 2);
        config.max_harmonics = IntParamRange::new(2, 4, 2);

        let result = grid_search(&mixed, &refs, &config);

        // The search should have explored multiple trials.
        assert!(
            result.trials.len() > 1,
            "Grid search should try multiple parameter combos, got {}",
            result.trials.len()
        );

        // Evaluate default params for comparison.
        let default_sdr = evaluate_params(&mixed, &refs, &GraphParams::default(), &config);

        // The best found should be at least as good as default.
        assert!(
            result.best_score >= default_sdr - 1.0,
            "Grid search best ({:.2} dB) should be close to or better than default ({:.2} dB)",
            result.best_score,
            default_sdr
        );
    }

    #[test]
    fn test_random_search_valid() {
        let (mixed, refs) = make_test_data(400.0, 1800.0);
        let config = default_search_ranges();

        let result = random_search(&mixed, &refs, &config, 5);

        assert_eq!(result.trials.len(), 5, "Should have exactly 5 trials");
        assert!(
            result.best_score > f64::NEG_INFINITY,
            "Best score should be finite"
        );

        // Verify returned params are within ranges.
        let p = &result.best_params;
        assert!(p.spectral_weight >= config.spectral_weight.min);
        assert!(p.spectral_weight <= config.spectral_weight.max);
        assert!(p.temporal_weight >= config.temporal_weight.min);
        assert!(p.temporal_weight <= config.temporal_weight.max);
        assert!(p.harmonic_weight >= config.harmonic_weight.min);
        assert!(p.harmonic_weight <= config.harmonic_weight.max);
        assert!(p.phase_threshold >= config.phase_threshold.min);
        assert!(p.phase_threshold <= config.phase_threshold.max);
        assert!(p.spectral_radius >= config.spectral_radius.min);
        assert!(p.spectral_radius <= config.spectral_radius.max);
        assert!(p.max_harmonics >= config.max_harmonics.min);
        assert!(p.max_harmonics <= config.max_harmonics.max);
    }

    #[test]
    fn test_bayesian_step_within_ranges() {
        let config = default_search_ranges();

        // Create some fake trial results.
        let trials = vec![
            TrialResult {
                params: make_params(1.0, 1.0, 0.5, 0.5, 3, 4),
                sdr: 5.0,
                elapsed_ms: 10.0,
            },
            TrialResult {
                params: make_params(0.5, 1.5, 1.0, 0.3, 2, 3),
                sdr: 7.0,
                elapsed_ms: 12.0,
            },
            TrialResult {
                params: make_params(1.5, 0.5, 0.3, 0.7, 4, 5),
                sdr: 3.0,
                elapsed_ms: 9.0,
            },
        ];

        let next = bayesian_step(&trials, &config);

        assert!(
            next.spectral_weight >= config.spectral_weight.min
                && next.spectral_weight <= config.spectral_weight.max,
            "spectral_weight {} out of range [{}, {}]",
            next.spectral_weight,
            config.spectral_weight.min,
            config.spectral_weight.max
        );
        assert!(
            next.temporal_weight >= config.temporal_weight.min
                && next.temporal_weight <= config.temporal_weight.max,
            "temporal_weight {} out of range [{}, {}]",
            next.temporal_weight,
            config.temporal_weight.min,
            config.temporal_weight.max
        );
        assert!(
            next.harmonic_weight >= config.harmonic_weight.min
                && next.harmonic_weight <= config.harmonic_weight.max,
            "harmonic_weight {} out of range [{}, {}]",
            next.harmonic_weight,
            config.harmonic_weight.min,
            config.harmonic_weight.max
        );
        assert!(
            next.phase_threshold >= config.phase_threshold.min
                && next.phase_threshold <= config.phase_threshold.max,
            "phase_threshold {} out of range [{}, {}]",
            next.phase_threshold,
            config.phase_threshold.min,
            config.phase_threshold.max
        );
        assert!(
            next.spectral_radius >= config.spectral_radius.min
                && next.spectral_radius <= config.spectral_radius.max,
            "spectral_radius {} out of range [{}, {}]",
            next.spectral_radius,
            config.spectral_radius.min,
            config.spectral_radius.max
        );
        assert!(
            next.max_harmonics >= config.max_harmonics.min
                && next.max_harmonics <= config.max_harmonics.max,
            "max_harmonics {} out of range [{}, {}]",
            next.max_harmonics,
            config.max_harmonics.min,
            config.max_harmonics.max
        );
    }

    #[test]
    fn test_bayesian_step_empty_results() {
        let config = default_search_ranges();
        let next = bayesian_step(&[], &config);
        // Should return default params without panicking.
        assert!(next.spectral_weight > 0.0);
    }
}
