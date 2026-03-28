//! Causal emergence and effective information computation.
//!
//! Implements Erik Hoel's causal emergence framework:
//! - **Effective Information (EI)**: measures the causal power of a system
//! - **Determinism**: how precisely causes map to effects
//! - **Degeneracy**: how many causes lead to the same effect
//! - **Causal Emergence**: EI_macro - EI_micro > 0 means the macro level
//!   is more causally informative than the micro level
//!
//! # References
//!
//! Hoel, E.P. (2017). "When the Map is Better Than the Territory."
//! Entropy, 19(5), 188.

use crate::error::{ConsciousnessError, ValidationError};
use crate::simd::{entropy, kl_divergence};
use crate::traits::EmergenceEngine;
use crate::types::{ComputeBudget, EmergenceResult, TransitionMatrix};

use std::time::Instant;

// ---------------------------------------------------------------------------
// Effective Information
// ---------------------------------------------------------------------------

/// Compute effective information for a TPM.
///
/// EI = H_max(cause) - <H(cause | effect)>
///    = log2(n) - (determinism_deficit + degeneracy)
///
/// Concretely:
///   EI = (1/n) * Σ_state D_KL( P(future|state) || uniform )
///
/// This measures how much knowing the current state reduces uncertainty
/// about the future state, compared to a uniform prior.
pub fn effective_information(tpm: &TransitionMatrix) -> Result<f64, ConsciousnessError> {
    let n = tpm.n;
    if n < 2 {
        return Err(ValidationError::EmptySystem.into());
    }

    let uniform: Vec<f64> = vec![1.0 / n as f64; n];
    let mut ei_sum = 0.0f64;

    for state in 0..n {
        let row = &tpm.data[state * n..(state + 1) * n];
        ei_sum += kl_divergence(row, &uniform);
    }

    Ok(ei_sum / n as f64)
}

/// Compute determinism: how precisely states map to unique outcomes.
///
/// det = (1/n) * Σ_state H_max - H(P(future|state))
///     = log(n) - (1/n) * Σ_state H(row)
pub fn determinism(tpm: &TransitionMatrix) -> f64 {
    let n = tpm.n;
    let h_max = (n as f64).ln();
    let mut avg_entropy = 0.0f64;

    for state in 0..n {
        let row = &tpm.data[state * n..(state + 1) * n];
        avg_entropy += entropy(row);
    }
    avg_entropy /= n as f64;

    h_max - avg_entropy
}

/// Compute degeneracy: how many states lead to the same outcome.
///
/// deg = H_max - H(marginal_output)
pub fn degeneracy(tpm: &TransitionMatrix) -> f64 {
    let n = tpm.n;
    let h_max = (n as f64).ln();

    // Marginal output distribution (average of all rows).
    let mut marginal = vec![0.0f64; n];
    for state in 0..n {
        for j in 0..n {
            marginal[j] += tpm.get(state, j);
        }
    }
    let inv_n = 1.0 / n as f64;
    for m in &mut marginal {
        *m *= inv_n;
    }

    h_max - entropy(&marginal)
}

// ---------------------------------------------------------------------------
// Coarse-graining
// ---------------------------------------------------------------------------

/// Apply a coarse-graining mapping to a TPM.
///
/// `mapping[i]` gives the macro-state index for micro-state i.
/// The resulting macro-TPM has size max(mapping) + 1.
pub fn coarse_grain(tpm: &TransitionMatrix, mapping: &[usize]) -> TransitionMatrix {
    assert_eq!(mapping.len(), tpm.n);

    let n_macro = mapping.iter().copied().max().unwrap_or(0) + 1;
    let mut macro_tpm = vec![0.0f64; n_macro * n_macro];
    let mut counts = vec![0usize; n_macro]; // micro-states per macro-state

    for micro_from in 0..tpm.n {
        let macro_from = mapping[micro_from];
        counts[macro_from] += 1;

        for micro_to in 0..tpm.n {
            let macro_to = mapping[micro_to];
            macro_tpm[macro_from * n_macro + macro_to] += tpm.get(micro_from, micro_to);
        }
    }

    // Normalize by the number of micro-states in each macro-state.
    for macro_from in 0..n_macro {
        if counts[macro_from] > 0 {
            let inv = 1.0 / counts[macro_from] as f64;
            for macro_to in 0..n_macro {
                macro_tpm[macro_from * n_macro + macro_to] *= inv;
            }
        }
    }

    TransitionMatrix::new(n_macro, macro_tpm)
}

// ---------------------------------------------------------------------------
// Causal Emergence Engine
// ---------------------------------------------------------------------------

/// Causal emergence engine that searches for the coarse-graining
/// that maximizes effective information.
pub struct CausalEmergenceEngine {
    max_macro_states: usize,
}

impl CausalEmergenceEngine {
    pub fn new(max_macro_states: usize) -> Self {
        Self { max_macro_states }
    }
}

impl Default for CausalEmergenceEngine {
    fn default() -> Self {
        Self {
            max_macro_states: 16,
        }
    }
}

impl EmergenceEngine for CausalEmergenceEngine {
    fn compute_emergence(
        &self,
        tpm: &TransitionMatrix,
        budget: &ComputeBudget,
    ) -> Result<EmergenceResult, ConsciousnessError> {
        let start = Instant::now();
        let n = tpm.n;

        if n < 2 {
            return Err(ValidationError::EmptySystem.into());
        }

        // Micro-level EI.
        let ei_micro = effective_information(tpm)?;
        let det_micro = determinism(tpm);
        let deg_micro = degeneracy(tpm);

        // Search coarse-grainings: try merging pairs greedily.
        let mut best_ei_macro = ei_micro;
        let mut best_mapping: Vec<usize> = (0..n).collect(); // identity = no coarse-graining

        // Greedy merge: try reducing to k macro-states for k = n-1, n-2, ..., 2.
        let min_k = 2.max(self.max_macro_states.min(n));
        for target_k in (2..n).rev() {
            if target_k < min_k && best_ei_macro > ei_micro {
                break; // Found improvement, stop searching
            }
            if start.elapsed() > budget.max_time {
                break;
            }

            // Greedy: merge the two states with most similar output distributions.
            let mapping = greedy_merge(tpm, target_k);
            let macro_tpm = coarse_grain(tpm, &mapping);

            if let Ok(ei) = effective_information(&macro_tpm) {
                if ei > best_ei_macro {
                    best_ei_macro = ei;
                    best_mapping = mapping;
                }
            }
        }

        Ok(EmergenceResult {
            ei_micro,
            ei_macro: best_ei_macro,
            causal_emergence: best_ei_macro - ei_micro,
            coarse_graining: best_mapping,
            determinism: det_micro,
            degeneracy: deg_micro,
            elapsed: start.elapsed(),
        })
    }

    fn effective_information(
        &self,
        tpm: &TransitionMatrix,
    ) -> Result<f64, ConsciousnessError> {
        effective_information(tpm)
    }
}

/// Greedy merge: iteratively merge the two most similar states until
/// we reach the target number of macro-states.
fn greedy_merge(tpm: &TransitionMatrix, target_k: usize) -> Vec<usize> {
    let n = tpm.n;
    let mut mapping: Vec<usize> = (0..n).collect();
    let mut current_k = n;

    while current_k > target_k {
        // Find the pair of macro-states with minimum distribution distance.
        let mut best_dist = f64::MAX;
        let mut best_i = 0;
        let mut best_j = 1;

        let macro_ids: Vec<usize> = {
            let mut ids: Vec<usize> = mapping.iter().copied().collect();
            ids.sort_unstable();
            ids.dedup();
            ids
        };

        for (ai, &mi) in macro_ids.iter().enumerate() {
            for &mj in macro_ids[ai + 1..].iter() {
                // Average distribution distance.
                let dist = state_distribution_distance(tpm, &mapping, mi, mj);
                if dist < best_dist {
                    best_dist = dist;
                    best_i = mi;
                    best_j = mj;
                }
            }
        }

        // Merge: map all occurrences of best_j to best_i.
        for m in &mut mapping {
            if *m == best_j {
                *m = best_i;
            }
        }

        // Re-index to be contiguous.
        let mut unique: Vec<usize> = mapping.iter().copied().collect();
        unique.sort_unstable();
        unique.dedup();
        for m in &mut mapping {
            *m = unique.iter().position(|&u| u == *m).unwrap();
        }

        current_k = unique.len();
    }

    mapping
}

/// L2 distance between average output distributions of two macro-states.
fn state_distribution_distance(
    tpm: &TransitionMatrix,
    mapping: &[usize],
    macro_a: usize,
    macro_b: usize,
) -> f64 {
    let n = tpm.n;
    let mut avg_a = vec![0.0f64; n];
    let mut avg_b = vec![0.0f64; n];
    let mut count_a = 0usize;
    let mut count_b = 0usize;

    for micro in 0..n {
        if mapping[micro] == macro_a {
            for j in 0..n {
                avg_a[j] += tpm.get(micro, j);
            }
            count_a += 1;
        } else if mapping[micro] == macro_b {
            for j in 0..n {
                avg_b[j] += tpm.get(micro, j);
            }
            count_b += 1;
        }
    }

    if count_a > 0 {
        let inv = 1.0 / count_a as f64;
        for a in &mut avg_a {
            *a *= inv;
        }
    }
    if count_b > 0 {
        let inv = 1.0 / count_b as f64;
        for b in &mut avg_b {
            *b *= inv;
        }
    }

    // L2 distance.
    let mut dist = 0.0f64;
    for j in 0..n {
        let d = avg_a[j] - avg_b[j];
        dist += d * d;
    }
    dist.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_tpm(n: usize) -> TransitionMatrix {
        TransitionMatrix::identity(n)
    }

    fn uniform_tpm(n: usize) -> TransitionMatrix {
        let val = 1.0 / n as f64;
        let data = vec![val; n * n];
        TransitionMatrix::new(n, data)
    }

    #[test]
    fn ei_identity_is_max() {
        // Identity TPM: each state deterministically maps to itself = max EI.
        let tpm = identity_tpm(4);
        let ei = effective_information(&tpm).unwrap();
        let h_max = (4.0f64).ln();
        assert!(
            (ei - h_max).abs() < 1e-6,
            "identity TPM should have EI = log(n), got {ei}"
        );
    }

    #[test]
    fn ei_uniform_is_zero() {
        // Uniform TPM: every state maps uniformly = zero EI.
        let tpm = uniform_tpm(4);
        let ei = effective_information(&tpm).unwrap();
        assert!(ei.abs() < 1e-6, "uniform TPM should have EI ≈ 0, got {ei}");
    }

    #[test]
    fn determinism_identity_is_max() {
        let tpm = identity_tpm(4);
        let det = determinism(&tpm);
        let h_max = (4.0f64).ln();
        assert!((det - h_max).abs() < 1e-6);
    }

    #[test]
    fn degeneracy_identity_is_zero() {
        // Identity: marginal is uniform, so degeneracy = 0.
        let tpm = identity_tpm(4);
        let deg = degeneracy(&tpm);
        assert!(deg.abs() < 1e-6, "identity TPM degeneracy should be 0, got {deg}");
    }

    #[test]
    fn coarse_grain_identity() {
        let tpm = identity_tpm(4);
        let mapping = vec![0, 0, 1, 1]; // merge 0+1 and 2+3
        let macro_tpm = coarse_grain(&tpm, &mapping);
        assert_eq!(macro_tpm.n, 2);
        // Identity: each micro-state maps to itself. After merging 0+1 into macro 0,
        // both micro-states transition within macro 0, so P(macro 0 -> macro 0) = 1.0.
        assert!((macro_tpm.get(0, 0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn causal_emergence_runs() {
        let tpm = identity_tpm(4);
        let budget = ComputeBudget::fast();
        let engine = CausalEmergenceEngine::default();
        let result = engine.compute_emergence(&tpm, &budget).unwrap();
        assert!(result.ei_micro >= 0.0);
        assert!(result.causal_emergence.is_finite());
    }
}
