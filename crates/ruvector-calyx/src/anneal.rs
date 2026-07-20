//! Reversible self-optimization of lens fusion weights (the paper's *Anneal*).
//!
//! "Compose" is only as good as the weights behind the fusion. Anneal tunes the
//! per-lens fusion weights to maximize an objective — here, *accepted answers
//! per unit cost* — using simulated annealing. Every proposal is **reversible**:
//! a rejected move restores the previous weights exactly, so the optimizer
//! never corrupts the deployed configuration.

use crate::rng::Rng;

/// Annealing schedule and proposal parameters.
#[derive(Debug, Clone)]
pub struct AnnealConfig {
    pub iterations: usize,
    /// Initial temperature for Metropolis acceptance.
    pub t_start: f32,
    /// Final temperature.
    pub t_end: f32,
    /// Max absolute perturbation applied to one weight per proposal.
    pub step: f32,
    /// Weights are clamped to `[0, max_weight]`.
    pub max_weight: f32,
    pub seed: u64,
}

impl Default for AnnealConfig {
    fn default() -> Self {
        Self {
            iterations: 300,
            t_start: 0.10,
            t_end: 0.001,
            step: 0.5,
            max_weight: 8.0,
            seed: 7,
        }
    }
}

/// The optimization result.
#[derive(Debug, Clone)]
pub struct AnnealOutcome {
    pub best_weights: Vec<f32>,
    pub best_utility: f32,
    pub initial_utility: f32,
    pub accepted_moves: usize,
    pub reverted_moves: usize,
}

/// Anneal `initial` weights to maximize `utility`.
///
/// `utility` maps a weight vector to a scalar (higher is better); in the
/// benchmark it is `accepted_answers / total_cost`. Proposals that fail the
/// Metropolis test are fully reverted, demonstrating the reversibility property.
pub fn anneal_weights<F>(initial: Vec<f32>, cfg: &AnnealConfig, utility: F) -> AnnealOutcome
where
    F: Fn(&[f32]) -> f32,
{
    let mut rng = Rng::new(cfg.seed);
    let n = initial.len();

    let mut current = initial.clone();
    let mut current_u = utility(&current);
    let initial_utility = current_u;

    let mut best = current.clone();
    let mut best_u = current_u;

    let mut accepted = 0usize;
    let mut reverted = 0usize;

    let iters = cfg.iterations.max(1);
    for i in 0..iters {
        // Geometric temperature schedule.
        let frac = i as f32 / iters as f32;
        let temp = cfg.t_start * (cfg.t_end / cfg.t_start).powf(frac);

        // Propose: perturb one weight. Snapshot for reversibility.
        let idx = rng.range(n);
        let prev = current[idx];
        let delta = rng.signed() * cfg.step;
        current[idx] = (prev + delta).clamp(0.0, cfg.max_weight);

        let candidate_u = utility(&current);
        let d = candidate_u - current_u;
        let accept = d >= 0.0 || rng.f32() < (d / temp.max(1e-6)).exp();

        if accept {
            current_u = candidate_u;
            accepted += 1;
            if candidate_u > best_u {
                best_u = candidate_u;
                best.copy_from_slice(&current);
            }
        } else {
            // Reversible: restore exactly.
            current[idx] = prev;
            reverted += 1;
        }
    }

    AnnealOutcome {
        best_weights: best,
        best_utility: best_u,
        initial_utility,
        accepted_moves: accepted,
        reverted_moves: reverted,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anneal_improves_a_simple_objective() {
        // Objective maximized when weights ≈ [3, 1]: a smooth concave bump.
        let utility = |w: &[f32]| -> f32 {
            let a = -(w[0] - 3.0).powi(2);
            let b = -(w[1] - 1.0).powi(2);
            a + b
        };
        let cfg = AnnealConfig::default();
        let out = anneal_weights(vec![0.0, 0.0], &cfg, utility);
        assert!(out.best_utility >= out.initial_utility);
        assert!((out.best_weights[0] - 3.0).abs() < 0.6, "w0={}", out.best_weights[0]);
        assert!((out.best_weights[1] - 1.0).abs() < 0.6, "w1={}", out.best_weights[1]);
        assert!(out.reverted_moves > 0, "some moves should be reverted");
    }
}
