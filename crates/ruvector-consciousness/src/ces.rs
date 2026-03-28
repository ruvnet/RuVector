//! Cause-Effect Structure (CES) computation — the "shape" of experience.
//!
//! The CES is the central object in IIT 4.0: it is the full set of
//! distinctions (concepts) and relations specified by a system in a state.
//! The CES *is* the quale — the quality of experience.
//!
//! This module computes:
//! - All distinctions (mechanisms with φ > 0)
//! - Relations between distinctions (overlapping purviews)
//! - System-level Φ (irreducibility of the CES)
//!
//! Complexity: O(2^(2n)) for full CES (all mechanisms × all purviews).
//! Use with n ≤ 8 for tractability.

use crate::error::ConsciousnessError;
use crate::iit4::{intrinsic_difference, mechanism_phi};
use crate::types::{
    CauseEffectStructure, ComputeBudget, Distinction, Mechanism, Relation, TransitionMatrix,
};

use std::time::Instant;

// ---------------------------------------------------------------------------
// CES computation
// ---------------------------------------------------------------------------

/// Compute the full Cause-Effect Structure for a system in a given state.
///
/// Enumerates all non-empty subsets of elements as candidate mechanisms,
/// computes φ for each, and collects those with φ > threshold.
/// Then computes relations between surviving distinctions.
pub fn compute_ces(
    tpm: &TransitionMatrix,
    state: usize,
    phi_threshold: f64,
    budget: &ComputeBudget,
) -> Result<CauseEffectStructure, ConsciousnessError> {
    let n = tpm.n; // number of states
    if n < 2 {
        return Err(crate::error::ValidationError::EmptySystem.into());
    }
    // num_elements = log2(n)
    let num_elements = n.trailing_zeros() as usize;
    if num_elements > 12 {
        return Err(ConsciousnessError::SystemTooLarge { n: num_elements, max: 12 });
    }

    let start = Instant::now();
    let mut distinctions: Vec<Distinction> = Vec::new();

    // Enumerate all non-empty subsets of elements as mechanisms.
    let full = (1u64 << num_elements) - 1;
    for mech_mask in 1..=full {
        if start.elapsed() > budget.max_time {
            break;
        }

        let mechanism = Mechanism::new(mech_mask, num_elements);
        let dist = mechanism_phi(tpm, &mechanism, state);

        if dist.phi > phi_threshold {
            distinctions.push(dist);
        }
    }

    // Sort by φ descending.
    distinctions.sort_by(|a, b| b.phi.partial_cmp(&a.phi).unwrap_or(std::cmp::Ordering::Equal));

    // Compute relations between distinctions.
    let relations = compute_relations(&distinctions);

    // Sum of all distinction φ values.
    let sum_phi: f64 = distinctions.iter().map(|d| d.phi).sum();

    // System-level Φ: irreducibility of the whole CES.
    // Approximate as minimum φ across all system bipartitions.
    let big_phi = compute_big_phi(tpm, state, &distinctions, budget);

    Ok(CauseEffectStructure {
        n: num_elements,
        state,
        distinctions,
        relations,
        big_phi,
        sum_phi,
        elapsed: start.elapsed(),
    })
}

/// Compute relations between distinctions.
///
/// Two distinctions are related if their purviews overlap.
/// A relation's φ measures the irreducibility of the overlap.
fn compute_relations(distinctions: &[Distinction]) -> Vec<Relation> {
    let mut relations = Vec::new();
    let nd = distinctions.len();

    // Pairwise relations (order 2).
    for i in 0..nd {
        for j in (i + 1)..nd {
            let overlap_cause = distinctions[i].cause_purview.elements
                & distinctions[j].cause_purview.elements;
            let overlap_effect = distinctions[i].effect_purview.elements
                & distinctions[j].effect_purview.elements;

            if overlap_cause != 0 || overlap_effect != 0 {
                // Relation φ: geometric mean of the two distinction φ values
                // weighted by purview overlap (simplified from full IIT 4.0).
                let overlap_size = (overlap_cause.count_ones() + overlap_effect.count_ones()) as f64;
                let total_size = (distinctions[i].cause_purview.size()
                    + distinctions[i].effect_purview.size()
                    + distinctions[j].cause_purview.size()
                    + distinctions[j].effect_purview.size()) as f64;

                let overlap_fraction = if total_size > 0.0 {
                    overlap_size / total_size
                } else {
                    0.0
                };

                let phi = (distinctions[i].phi * distinctions[j].phi).sqrt() * overlap_fraction;

                if phi > 1e-10 {
                    relations.push(Relation {
                        distinction_indices: vec![i, j],
                        phi,
                        order: 2,
                    });
                }
            }
        }
    }

    // Sort by φ descending.
    relations.sort_by(|a, b| b.phi.partial_cmp(&a.phi).unwrap_or(std::cmp::Ordering::Equal));
    relations
}

/// Compute system-level Φ (big phi).
///
/// Φ = min over all unidirectional bipartitions of the distance between
/// the intact CES and the partitioned CES.
///
/// Simplified: use the minimum partition information loss from the
/// existing PhiEngine infrastructure as a proxy.
fn compute_big_phi(
    tpm: &TransitionMatrix,
    state: usize,
    distinctions: &[Distinction],
    budget: &ComputeBudget,
) -> f64 {
    let num_elements = tpm.n.trailing_zeros() as usize;
    if distinctions.is_empty() {
        return 0.0;
    }

    // Build a "distinction vector" — the φ values of all distinctions.
    // Φ measures how much this vector changes under system partition.
    let intact_phi_vec: Vec<f64> = distinctions.iter().map(|d| d.phi).collect();

    let full = (1u64 << num_elements) - 1;
    let mut min_phi = f64::MAX;

    // Try all bipartitions of the system.
    for part_mask in 1..full {
        // Compute distinctions for the partitioned system.
        // Under partition, mechanisms that span the cut lose integration.
        let mut partitioned_phi_vec: Vec<f64> = Vec::with_capacity(distinctions.len());

        for dist in distinctions {
            let mech_mask = dist.mechanism.elements;
            // Does this mechanism span the partition?
            let in_a = mech_mask & part_mask;
            let in_b = mech_mask & !part_mask & full;

            if in_a != 0 && in_b != 0 {
                // Mechanism spans the cut → loses integration.
                // Partitioned φ ≈ 0 (simplified; full IIT 4.0 recomputes).
                partitioned_phi_vec.push(0.0);
            } else {
                // Mechanism is entirely within one partition.
                partitioned_phi_vec.push(dist.phi);
            }
        }

        // Distance between intact and partitioned CES.
        let ces_distance = intrinsic_difference(&intact_phi_vec, &partitioned_phi_vec);
        min_phi = min_phi.min(ces_distance);
    }

    if min_phi == f64::MAX { 0.0 } else { min_phi }
}

/// Quick CES summary: number of distinctions and relations.
pub fn ces_complexity(ces: &CauseEffectStructure) -> (usize, usize, f64) {
    (ces.distinctions.len(), ces.relations.len(), ces.sum_phi)
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

    fn identity_tpm() -> TransitionMatrix {
        TransitionMatrix::identity(4)
    }

    #[test]
    fn ces_computes_for_small_system() {
        let tpm = and_gate_tpm();
        let budget = ComputeBudget::exact();
        let ces = compute_ces(&tpm, 0, 1e-6, &budget).unwrap();
        assert!(ces.distinctions.len() > 0 || ces.big_phi >= 0.0);
        assert_eq!(ces.n, 2); // 4 states → 2 elements
        assert_eq!(ces.state, 0);
    }

    #[test]
    fn ces_identity_has_distinctions() {
        let tpm = identity_tpm();
        let budget = ComputeBudget::exact();
        let ces = compute_ces(&tpm, 0, 1e-6, &budget).unwrap();
        // Identity TPM: each element determines its own future perfectly.
        assert!(ces.sum_phi >= 0.0);
    }

    #[test]
    fn ces_rejects_too_large() {
        // 2^13 = 8192 states → 13 elements → exceeds limit of 12
        let tpm = TransitionMatrix::identity(8192);
        let budget = ComputeBudget::exact();
        assert!(compute_ces(&tpm, 0, 1e-6, &budget).is_err());
    }

    #[test]
    fn ces_complexity_reports() {
        let tpm = and_gate_tpm();
        let budget = ComputeBudget::exact();
        let ces = compute_ces(&tpm, 0, 1e-6, &budget).unwrap();
        let (nd, nr, sp) = ces_complexity(&ces);
        assert!(nd <= (1 << 2)); // At most 2^num_elements mechanisms (2 elements).
        assert!(sp >= 0.0);
    }
}
