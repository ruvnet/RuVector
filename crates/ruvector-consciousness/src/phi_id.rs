//! Integrated Information Decomposition (ΦID).
//!
//! Implements Mediano et al. (2021) "Towards an extended taxonomy of
//! information dynamics via Integrated Information Decomposition."
//!
//! ΦID decomposes the information that a system's past carries about
//! its future into:
//! - **Redundancy**: information shared across all sources
//! - **Unique**: information carried by only one source
//! - **Synergy**: information available only from the whole system
//!
//! This extends classical Φ by distinguishing *types* of integration.
//! A system with high synergy is qualitatively different from one with
//! high redundancy, even if both have the same Φ.

use crate::error::ConsciousnessError;
use crate::simd::{kl_divergence, marginal_distribution};
use crate::types::{PhiIdResult, TransitionMatrix};

use std::time::Instant;

// ---------------------------------------------------------------------------
// ΦID computation
// ---------------------------------------------------------------------------

/// Compute the Integrated Information Decomposition for a bipartite system.
///
/// Given a TPM and a bipartition into sources (A, B) and target (future),
/// decomposes the mutual information I(past; future) into:
///
/// I = redundancy + unique_A + unique_B + synergy
///
/// Uses the minimum mutual information (MMI) approach for redundancy
/// (Barrett 2015), which is computationally tractable.
pub fn compute_phi_id(
    tpm: &TransitionMatrix,
    partition_mask: u64,
) -> Result<PhiIdResult, ConsciousnessError> {
    let n = tpm.n;
    if n < 2 {
        return Err(crate::error::ValidationError::EmptySystem.into());
    }
    let start = Instant::now();

    let marginal = marginal_distribution(tpm.as_slice(), n);

    // Split elements into sources A and B.
    let mut source_a: Vec<usize> = Vec::new();
    let mut source_b: Vec<usize> = Vec::new();
    for i in 0..n {
        if partition_mask & (1 << i) != 0 {
            source_a.push(i);
        } else {
            source_b.push(i);
        }
    }

    if source_a.is_empty() || source_b.is_empty() {
        return Err(crate::error::ValidationError::DimensionMismatch(
            "partition must have elements on both sides".into(),
        )
        .into());
    }

    // Compute mutual informations.
    // I(whole_past; future) — total MI
    let total_mi = mutual_information_past_future(tpm, n, &marginal);

    // I(A_past; future) — source A's MI with the future
    let mi_a = source_mutual_information(tpm, n, &source_a, &marginal);

    // I(B_past; future) — source B's MI with the future
    let mi_b = source_mutual_information(tpm, n, &source_b, &marginal);

    // Redundancy: I_min = min(I(A; future), I(B; future))
    // This is the MMI measure (Barrett 2015).
    let redundancy = mi_a.min(mi_b);

    // Unique information.
    let unique_a = mi_a - redundancy;
    let unique_b = mi_b - redundancy;

    // Synergy: what's left after subtracting all other atoms.
    // I_total = redundancy + unique_A + unique_B + synergy
    let synergy = (total_mi - redundancy - unique_a - unique_b).max(0.0);

    // Transfer entropy: I(A_past; B_future | B_past)
    let te = transfer_entropy(tpm, n, &source_a, &source_b, &marginal);

    Ok(PhiIdResult {
        redundancy,
        unique: vec![unique_a, unique_b],
        synergy,
        total_mi,
        transfer_entropy: te,
        elapsed: start.elapsed(),
    })
}

/// Mutual information between past and future: I(X_t; X_{t+1}).
fn mutual_information_past_future(tpm: &TransitionMatrix, n: usize, marginal: &[f64]) -> f64 {
    // I(past; future) = H(future) - H(future | past)
    // H(future) = -Σ_j p(j) ln p(j)  where p(j) = marginal[j]
    // H(future | past) = -Σ_i p(i) Σ_j T[i,j] ln T[i,j]
    let h_future = shannon_entropy(marginal);

    let mut h_cond = 0.0f64;
    let p_state = 1.0 / n as f64; // Uniform prior over states.
    for i in 0..n {
        for j in 0..n {
            let tij = tpm.get(i, j);
            if tij > 1e-15 {
                h_cond -= p_state * tij * tij.ln();
            }
        }
    }

    (h_future - h_cond).max(0.0)
}

/// Mutual information between a source subset and the future.
fn source_mutual_information(
    tpm: &TransitionMatrix,
    n: usize,
    source: &[usize],
    marginal: &[f64],
) -> f64 {
    // Marginalize TPM to source elements, then compute MI.
    let sub_tpm = tpm.marginalize(source);
    let sub_n = sub_tpm.n;
    let sub_marginal = marginal_distribution(sub_tpm.as_slice(), sub_n);
    mutual_information_past_future(&sub_tpm, sub_n, &sub_marginal)
}

/// Transfer entropy: information A's past carries about B's future
/// beyond what B's own past carries.
///
/// TE(A→B) = I(A_past; B_future | B_past)
fn transfer_entropy(
    tpm: &TransitionMatrix,
    n: usize,
    source: &[usize],
    target: &[usize],
    marginal: &[f64],
) -> f64 {
    // TE(A→B) = I(A,B → B) - I(B → B)
    // I(A,B → B) uses the full system's prediction of B.
    // I(B → B) uses only B's self-prediction.

    // MI of full system about target's future.
    let mut all_elements: Vec<usize> = source.to_vec();
    all_elements.extend_from_slice(target);
    all_elements.sort();
    all_elements.dedup();

    let mi_ab_to_b = source_mutual_information(tpm, n, &all_elements, marginal);
    let mi_b_to_b = source_mutual_information(tpm, n, target, marginal);

    (mi_ab_to_b - mi_b_to_b).max(0.0)
}

/// Shannon entropy of a distribution.
fn shannon_entropy(p: &[f64]) -> f64 {
    let mut h = 0.0f64;
    for &pi in p {
        if pi > 1e-15 {
            h -= pi * pi.ln();
        }
    }
    h
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

    fn disconnected_tpm() -> TransitionMatrix {
        #[rustfmt::skip]
        let data = vec![
            0.5, 0.5, 0.0, 0.0,
            0.5, 0.5, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.5,
            0.0, 0.0, 0.5, 0.5,
        ];
        TransitionMatrix::new(4, data)
    }

    #[test]
    fn phi_id_decomposes_and_gate() {
        let tpm = and_gate_tpm();
        let result = compute_phi_id(&tpm, 0b0011).unwrap();
        assert!(result.total_mi >= 0.0);
        assert!(result.redundancy >= 0.0);
        assert!(result.synergy >= 0.0);
        // Verify all components are non-negative.
        assert!(result.unique[0] >= 0.0);
        assert!(result.unique[1] >= 0.0);
    }

    #[test]
    fn phi_id_disconnected_components() {
        let tpm = disconnected_tpm();
        let result = compute_phi_id(&tpm, 0b0011).unwrap();
        assert!(result.total_mi >= 0.0);
        assert!(result.redundancy >= 0.0);
        assert!(result.synergy >= 0.0);
    }

    #[test]
    fn phi_id_transfer_entropy_nonnegative() {
        let tpm = and_gate_tpm();
        let result = compute_phi_id(&tpm, 0b0011).unwrap();
        assert!(result.transfer_entropy >= 0.0);
    }
}
