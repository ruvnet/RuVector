//! Partial Information Decomposition (PID).
//!
//! Implements the Williams & Beer (2010) framework for decomposing
//! the mutual information that multiple sources carry about a target
//! into redundant, unique, and synergistic components.
//!
//! PID answers: "What kind of information does each source provide?"
//! - Redundancy: information that any source alone can provide
//! - Unique: information only available from one specific source
//! - Synergy: information only available from combining sources
//!
//! Uses the I_min (minimum specific information) measure for redundancy.

use crate::error::ConsciousnessError;
use crate::simd::marginal_distribution;
use crate::types::{PidResult, TransitionMatrix};

use std::time::Instant;

// ---------------------------------------------------------------------------
// PID computation
// ---------------------------------------------------------------------------

/// Compute PID for a system with given source subsets and a target.
///
/// `sources`: list of element index sets (each source is a subset of elements).
/// `target`: element indices for the target variable.
///
/// For a TPM-based system, "sources" are subsets of elements at time t,
/// and "target" is the system state at time t+1.
pub fn compute_pid(
    tpm: &TransitionMatrix,
    sources: &[Vec<usize>],
    target: &[usize],
) -> Result<PidResult, ConsciousnessError> {
    let n = tpm.n;
    if n < 2 {
        return Err(crate::error::ValidationError::EmptySystem.into());
    }
    if sources.is_empty() {
        return Err(crate::error::ValidationError::DimensionMismatch(
            "need at least one source".into(),
        )
        .into());
    }
    let start = Instant::now();

    let marginal = marginal_distribution(tpm.as_slice(), n);

    // Compute I(S_i; T) for each source.
    let mut source_mis: Vec<f64> = Vec::with_capacity(sources.len());
    for source in sources {
        let mi = source_target_mi(tpm, n, source, target, &marginal);
        source_mis.push(mi);
    }

    // Total MI: I(all_sources; T)
    let all_sources: Vec<usize> = sources.iter().flat_map(|s| s.iter().copied()).collect();
    let total_mi = source_target_mi(tpm, n, &all_sources, target, &marginal);

    // Redundancy: I_min = min_i I(S_i; T)
    // Williams & Beer I_min: the minimum specific information any source
    // provides about each target state.
    let redundancy = williams_beer_imin(tpm, n, sources, target, &marginal);

    // Unique information per source.
    let mut unique: Vec<f64> = Vec::with_capacity(sources.len());
    for &mi in &source_mis {
        unique.push((mi - redundancy).max(0.0));
    }

    // Synergy.
    let unique_sum: f64 = unique.iter().sum();
    let synergy = (total_mi - redundancy - unique_sum).max(0.0);

    Ok(PidResult {
        redundancy,
        unique,
        synergy,
        total_mi,
        num_sources: sources.len(),
        elapsed: start.elapsed(),
    })
}

/// Williams & Beer I_min redundancy measure.
///
/// I_min(S1, S2, ...; T) = Σ_t p(t) min_i I_spec(S_i; t)
///
/// where I_spec(S; t) = Σ_s p(s|t) log(p(t|s) / p(t)) is the
/// specific information source S provides about target outcome t.
fn williams_beer_imin(
    tpm: &TransitionMatrix,
    n: usize,
    sources: &[Vec<usize>],
    target: &[usize],
    marginal: &[f64],
) -> f64 {
    let target_marginal = compute_target_marginal(tpm, n, target);
    let target_size = target_marginal.len();

    // Pre-compute source marginals once (avoids recomputation per target state).
    let source_marginals: Vec<Vec<f64>> = sources
        .iter()
        .map(|s| compute_source_marginal(tpm, n, s))
        .collect();

    let mut imin = 0.0f64;

    for t_state in 0..target_size {
        let p_t = target_marginal[t_state];
        if p_t < 1e-15 {
            continue;
        }

        let mut min_spec = f64::MAX;
        for (source, source_marginal) in sources.iter().zip(source_marginals.iter()) {
            let spec = specific_information_cached(
                tpm,
                n,
                source,
                target,
                t_state,
                &target_marginal,
                source_marginal,
            );
            min_spec = min_spec.min(spec);
        }

        if min_spec < f64::MAX {
            imin += p_t * min_spec;
        }
    }

    imin.max(0.0)
}

/// Specific information with pre-computed source marginal (avoids reallocation).
fn specific_information_cached(
    tpm: &TransitionMatrix,
    n: usize,
    source: &[usize],
    target: &[usize],
    target_state: usize,
    target_marginal: &[f64],
    source_marginal: &[f64],
) -> f64 {
    let source_size = source_marginal.len();
    let p_t = target_marginal[target_state];

    if p_t < 1e-15 {
        return 0.0;
    }

    let mut p_s_given_t = vec![0.0f64; source_size];
    let inv_n = 1.0 / n as f64;
    for global_state in 0..n {
        let s_state = extract_substate(global_state, source);
        if s_state < source_size {
            let mut p_target_given_global = 0.0;
            for future in 0..n {
                if extract_substate(future, target) == target_state {
                    p_target_given_global += tpm.get(global_state, future);
                }
            }
            p_s_given_t[s_state] += inv_n * p_target_given_global;
        }
    }

    let sum: f64 = p_s_given_t.iter().sum();
    if sum > 1e-15 {
        let inv = 1.0 / sum;
        for p in &mut p_s_given_t {
            *p *= inv;
        }
    }

    let mut dkl = 0.0f64;
    for i in 0..source_size {
        let p = p_s_given_t[i];
        let q = source_marginal[i];
        if p > 1e-15 && q > 1e-15 {
            dkl += p * (p / q).ln();
        }
    }
    dkl.max(0.0)
}

/// Specific information: I_spec(S; t) = D_KL(P(S|T=t) || P(S))
///
/// How much knowing outcome t updates our belief about source S.
fn specific_information(
    tpm: &TransitionMatrix,
    n: usize,
    source: &[usize],
    target: &[usize],
    target_state: usize,
    target_marginal: &[f64],
) -> f64 {
    let source_marginal = compute_source_marginal(tpm, n, source);
    let source_size = source_marginal.len();
    let p_t = target_marginal[target_state];

    if p_t < 1e-15 {
        return 0.0;
    }

    // P(source | target = t_state): Bayes' rule.
    let mut p_s_given_t = vec![0.0f64; source_size];

    let inv_n = 1.0 / n as f64;
    for global_state in 0..n {
        let s_state = extract_substate(global_state, source);
        let t_state_actual = extract_substate(global_state, target);

        if s_state < source_size {
            // P(target_state | global_state) from TPM, marginalized.
            let mut p_target_given_global = 0.0;
            for future in 0..n {
                if extract_substate(future, target) == target_state {
                    p_target_given_global += tpm.get(global_state, future);
                }
            }
            p_s_given_t[s_state] += inv_n * p_target_given_global;
        }
    }

    // Normalize.
    let sum: f64 = p_s_given_t.iter().sum();
    if sum > 1e-15 {
        let inv = 1.0 / sum;
        for p in &mut p_s_given_t {
            *p *= inv;
        }
    }

    // D_KL(P(S|T=t) || P(S))
    let mut dkl = 0.0f64;
    for i in 0..source_size {
        let p = p_s_given_t[i];
        let q = source_marginal[i];
        if p > 1e-15 && q > 1e-15 {
            dkl += p * (p / q).ln();
        }
    }
    dkl.max(0.0)
}

/// MI between source and target subsets.
fn source_target_mi(
    tpm: &TransitionMatrix,
    n: usize,
    source: &[usize],
    target: &[usize],
    _marginal: &[f64],
) -> f64 {
    let source_marginal = compute_source_marginal(tpm, n, source);
    let target_marginal = compute_target_marginal(tpm, n, target);
    let joint = compute_joint_distribution(tpm, n, source, target);

    let source_size = source_marginal.len();
    let target_size = target_marginal.len();

    let mut mi = 0.0f64;
    for s in 0..source_size {
        for t in 0..target_size {
            let pst = joint[s * target_size + t];
            let ps = source_marginal[s];
            let pt = target_marginal[t];
            if pst > 1e-15 && ps > 1e-15 && pt > 1e-15 {
                mi += pst * (pst / (ps * pt)).ln();
            }
        }
    }
    mi.max(0.0)
}

/// Marginal distribution over source subset states.
fn compute_source_marginal(tpm: &TransitionMatrix, n: usize, source: &[usize]) -> Vec<f64> {
    let size = 1usize << source.len();
    let mut dist = vec![0.0f64; size];
    let inv_n = 1.0 / n as f64;
    for state in 0..n {
        let sub = extract_substate(state, source);
        if sub < size {
            dist[sub] += inv_n;
        }
    }
    dist
}

/// Marginal distribution over target subset states (in the future).
fn compute_target_marginal(tpm: &TransitionMatrix, n: usize, target: &[usize]) -> Vec<f64> {
    let size = 1usize << target.len();
    let mut dist = vec![0.0f64; size];
    let inv_n = 1.0 / n as f64;
    for state in 0..n {
        for future in 0..n {
            let t_sub = extract_substate(future, target);
            if t_sub < size {
                dist[t_sub] += inv_n * tpm.get(state, future);
            }
        }
    }
    dist
}

/// Joint distribution P(source_past, target_future).
fn compute_joint_distribution(
    tpm: &TransitionMatrix,
    n: usize,
    source: &[usize],
    target: &[usize],
) -> Vec<f64> {
    let source_size = 1usize << source.len();
    let target_size = 1usize << target.len();
    let mut joint = vec![0.0f64; source_size * target_size];
    let inv_n = 1.0 / n as f64;

    for state in 0..n {
        let s_sub = extract_substate(state, source);
        for future in 0..n {
            let t_sub = extract_substate(future, target);
            if s_sub < source_size && t_sub < target_size {
                joint[s_sub * target_size + t_sub] += inv_n * tpm.get(state, future);
            }
        }
    }
    joint
}

#[inline]
fn extract_substate(global_state: usize, indices: &[usize]) -> usize {
    let mut sub = 0usize;
    for (bit, &idx) in indices.iter().enumerate() {
        sub |= ((global_state >> idx) & 1) << bit;
    }
    sub
}

#[cfg(test)]
mod tests {
    use super::*;

    fn and_gate_tpm() -> TransitionMatrix {
        #[rustfmt::skip]
        let data = vec![
            0.5, 0.25, 0.25, 0.0,
            0.5, 0.25, 0.25, 0.0,
            0.5, 0.25, 0.25, 0.0,
            0.0, 0.0,  0.0,  1.0,
        ];
        TransitionMatrix::new(4, data)
    }

    #[test]
    fn pid_two_sources() {
        let tpm = and_gate_tpm();
        let sources = vec![vec![0, 1], vec![2, 3]];
        let target = vec![0, 1];
        let result = compute_pid(&tpm, &sources, &target).unwrap();
        assert!(result.total_mi >= 0.0);
        assert!(result.redundancy >= 0.0);
        assert!(result.synergy >= 0.0);
        assert_eq!(result.num_sources, 2);
    }

    #[test]
    fn pid_decomposition_sums() {
        let tpm = and_gate_tpm();
        let sources = vec![vec![0], vec![1]];
        let target = vec![0, 1];
        let result = compute_pid(&tpm, &sources, &target).unwrap();
        let sum = result.redundancy + result.unique.iter().sum::<f64>() + result.synergy;
        assert!(
            (sum - result.total_mi).abs() < 1e-6,
            "PID sum {} should equal total MI {}",
            sum,
            result.total_mi
        );
    }

    #[test]
    fn pid_rejects_empty() {
        let tpm = and_gate_tpm();
        assert!(compute_pid(&tpm, &[], &[0]).is_err());
    }
}
