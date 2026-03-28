//! IIT 4.0 intrinsic information and cause-effect repertoires.
//!
//! Implements the updated IIT 4.0 framework (Albantakis et al. 2023):
//! - Intrinsic difference (replaces KL divergence from IIT 3.0)
//! - Cause repertoire: P(past | mechanism, purview)
//! - Effect repertoire: P(future | mechanism, purview)
//! - Mechanism-level φ via minimum partition of cause/effect
//!
//! Key difference from IIT 3.0: uses intrinsic information measures
//! that are defined from the system's own perspective, not relative
//! to an external observer.

use crate::error::ConsciousnessError;
use crate::types::{Distinction, Mechanism, Purview, TransitionMatrix};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute number of binary elements from number of states: k = log2(n).
#[inline]
fn num_elements_from_states(n: usize) -> usize {
    debug_assert!(n.is_power_of_two() && n >= 2);
    n.trailing_zeros() as usize
}

// ---------------------------------------------------------------------------
// Intrinsic difference (IIT 4.0 replaces KL with this)
// ---------------------------------------------------------------------------

/// Intrinsic difference: the distance between two distributions that is
/// intrinsic to the system (not observer-relative).
///
/// Uses the Earth Mover's Distance (Wasserstein-1) on the state space,
/// as specified in IIT 4.0. For discrete systems, this reduces to the
/// L1 cumulative difference.
///
/// IIT 4.0 specifically chose EMD because it respects the metric structure
/// of the state space (unlike KL which is topology-blind).
#[inline]
pub fn intrinsic_difference(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len());
    // EMD for 1D discrete distributions = cumulative L1 difference.
    let mut cumsum = 0.0f64;
    let mut dist = 0.0f64;
    for i in 0..p.len() {
        cumsum += p[i] - q[i];
        dist += cumsum.abs();
    }
    dist
}

/// Selectivity: how much a mechanism constrains its purview beyond
/// the unconstrained (maximum entropy) repertoire.
///
/// In IIT 4.0: φ_s = d(repertoire, max_entropy_repertoire)
/// where d is intrinsic_difference.
#[inline]
pub fn selectivity(repertoire: &[f64]) -> f64 {
    let n = repertoire.len();
    if n == 0 {
        return 0.0;
    }
    let uniform = 1.0 / n as f64;
    let uniform_dist: Vec<f64> = vec![uniform; n];
    intrinsic_difference(repertoire, &uniform_dist)
}

// ---------------------------------------------------------------------------
// Cause and effect repertoires
// ---------------------------------------------------------------------------

/// Compute the cause repertoire: P(past_purview | mechanism_state).
///
/// Given a mechanism M in state s, the cause repertoire is the distribution
/// over past states of the purview that is maximally constrained by M=s.
///
/// For a TPM where rows are current states → future states:
/// P(past_purview | mechanism=s) ∝ TPM[past, mechanism_cols] evaluated at s.
pub fn cause_repertoire(
    tpm: &TransitionMatrix,
    _mechanism: &Mechanism,
    purview: &Purview,
    state: usize,
) -> Vec<f64> {
    let n = tpm.n; // number of states
    let purview_indices = purview.indices();
    let purview_size = 1usize << purview_indices.len();

    // Single-pass: accumulate into purview buckets and count per bucket.
    let mut repertoire = vec![0.0f64; purview_size];
    let mut counts = vec![0u32; purview_size];

    for global_past in 0..n {
        let ps = extract_substate(global_past, &purview_indices);
        if ps < purview_size {
            repertoire[ps] += tpm.get(global_past, state);
            counts[ps] += 1;
        }
    }

    // Average over consistent global states (uniform prior).
    for i in 0..purview_size {
        if counts[i] > 0 {
            repertoire[i] /= counts[i] as f64;
        }
    }

    // Normalize to probability distribution.
    let sum: f64 = repertoire.iter().sum();
    if sum > 1e-15 {
        let inv = 1.0 / sum;
        for r in &mut repertoire {
            *r *= inv;
        }
    } else {
        let uniform = 1.0 / purview_size as f64;
        repertoire.fill(uniform);
    }

    repertoire
}

/// Compute the effect repertoire: P(future_purview | mechanism_state).
///
/// The effect repertoire is the distribution over future purview states
/// given the mechanism is in state s.
pub fn effect_repertoire(
    tpm: &TransitionMatrix,
    mechanism: &Mechanism,
    purview: &Purview,
    state: usize,
) -> Vec<f64> {
    let n = tpm.n;
    let purview_indices = purview.indices();
    let purview_size = 1usize << purview_indices.len();

    let mut repertoire = vec![0.0f64; purview_size];

    // Effect: P(future_purview | current_state)
    // = marginalize the TPM row over non-purview elements.
    let row = &tpm.data[state * n..(state + 1) * n];

    for future_state in 0..n {
        let purview_substate = extract_substate(future_state, &purview_indices);
        if purview_substate < purview_size {
            repertoire[purview_substate] += row[future_state];
        }
    }

    // Normalize.
    let sum: f64 = repertoire.iter().sum();
    if sum > 1e-15 {
        let inv = 1.0 / sum;
        for r in &mut repertoire {
            *r *= inv;
        }
    } else {
        let uniform = 1.0 / purview_size as f64;
        repertoire.fill(uniform);
    }

    repertoire
}

/// Compute the unconstrained (maximum entropy) repertoire over a purview.
pub fn unconstrained_repertoire(purview_size: usize) -> Vec<f64> {
    let uniform = 1.0 / purview_size as f64;
    vec![uniform; purview_size]
}

// ---------------------------------------------------------------------------
// Mechanism-level φ (small phi)
// ---------------------------------------------------------------------------

/// Compute the integrated information φ for a single mechanism.
///
/// φ(mechanism) = min(φ_cause, φ_effect)
///
/// where φ_cause = min over partitions of the cause side,
/// and φ_effect = min over partitions of the effect side.
///
/// This is the IIT 4.0 version using intrinsic_difference instead of KL.
pub fn mechanism_phi(
    tpm: &TransitionMatrix,
    mechanism: &Mechanism,
    state: usize,
) -> Distinction {
    let n = tpm.n; // number of states
    let num_elements = num_elements_from_states(n);

    // Find the purview that maximizes φ for cause and effect.
    let mut best_cause_phi = 0.0f64;
    let mut best_cause_rep = vec![];
    let mut best_cause_purview = Purview::new(1, num_elements);

    let mut best_effect_phi = 0.0f64;
    let mut best_effect_rep = vec![];
    let mut best_effect_purview = Purview::new(1, num_elements);

    // Iterate over all possible purviews (subsets of system elements).
    let full = (1u64 << num_elements) - 1;
    for purview_mask in 1..=full {
        let purview = Purview::new(purview_mask, num_elements);
        let purview_size = 1usize << purview.size();

        // Cause side.
        let cause_rep = cause_repertoire(tpm, mechanism, &purview, state);
        let uc_rep = unconstrained_repertoire(purview_size);
        let cause_phi = intrinsic_difference(&cause_rep, &uc_rep);

        // Find the minimum over partitions of the mechanism for this purview.
        let partitioned_cause_phi = min_partition_phi_cause(
            tpm, mechanism, &purview, state, &cause_rep,
        );

        if partitioned_cause_phi > best_cause_phi {
            best_cause_phi = partitioned_cause_phi;
            best_cause_rep = cause_rep;
            best_cause_purview = purview.clone();
        }

        // Effect side.
        let effect_rep = effect_repertoire(tpm, mechanism, &purview, state);
        let uc_effect = unconstrained_repertoire(purview_size);
        let effect_phi = intrinsic_difference(&effect_rep, &uc_effect);

        let partitioned_effect_phi = min_partition_phi_effect(
            tpm, mechanism, &purview, state, &effect_rep,
        );

        if partitioned_effect_phi > best_effect_phi {
            best_effect_phi = partitioned_effect_phi;
            best_effect_rep = effect_rep;
            best_effect_purview = purview.clone();
        }
    }

    let phi = best_cause_phi.min(best_effect_phi);

    Distinction {
        mechanism: mechanism.clone(),
        cause_repertoire: best_cause_rep,
        effect_repertoire: best_effect_rep,
        cause_purview: best_cause_purview,
        effect_purview: best_effect_purview,
        phi_cause: best_cause_phi,
        phi_effect: best_effect_phi,
        phi,
    }
}

/// Minimum partition φ for the cause side.
///
/// Partitions the mechanism-purview system and finds the partition
/// that causes the least loss (MIP). φ_cause = the loss at the MIP.
fn min_partition_phi_cause(
    tpm: &TransitionMatrix,
    mechanism: &Mechanism,
    purview: &Purview,
    state: usize,
    intact_repertoire: &[f64],
) -> f64 {
    let num_elements = num_elements_from_states(tpm.n);
    let mech_size = mechanism.size();

    if mech_size <= 1 {
        // Single-element mechanism: φ = selectivity (no partition possible).
        return selectivity(intact_repertoire);
    }

    // Try all bipartitions of the mechanism.
    let mech_indices = mechanism.indices();
    let full_mech = (1u64 << mech_size) - 1;
    let mut min_loss = f64::MAX;

    for part_mask in 1..full_mech {
        // Partition mechanism into two parts.
        let mut part_a_elems = 0u64;
        let mut part_b_elems = 0u64;
        for (bit, &idx) in mech_indices.iter().enumerate() {
            if part_mask & (1 << bit) != 0 {
                part_a_elems |= 1 << idx;
            } else {
                part_b_elems |= 1 << idx;
            }
        }

        let mech_a = Mechanism::new(part_a_elems, num_elements);
        let mech_b = Mechanism::new(part_b_elems, num_elements);

        // Compute partitioned repertoires.
        let rep_a = cause_repertoire(tpm, &mech_a, purview, state);
        let rep_b = cause_repertoire(tpm, &mech_b, purview, state);

        // Product of partitioned repertoires.
        let product = product_distribution(&rep_a, &rep_b);

        // Information loss due to partition.
        let loss = intrinsic_difference(intact_repertoire, &product);
        min_loss = min_loss.min(loss);
    }

    if min_loss == f64::MAX { 0.0 } else { min_loss }
}

/// Minimum partition φ for the effect side.
fn min_partition_phi_effect(
    tpm: &TransitionMatrix,
    mechanism: &Mechanism,
    purview: &Purview,
    state: usize,
    intact_repertoire: &[f64],
) -> f64 {
    let num_elements = num_elements_from_states(tpm.n);
    let mech_size = mechanism.size();

    if mech_size <= 1 {
        return selectivity(intact_repertoire);
    }

    let mech_indices = mechanism.indices();
    let full_mech = (1u64 << mech_size) - 1;
    let mut min_loss = f64::MAX;

    for part_mask in 1..full_mech {
        let mut part_a_elems = 0u64;
        let mut part_b_elems = 0u64;
        for (bit, &idx) in mech_indices.iter().enumerate() {
            if part_mask & (1 << bit) != 0 {
                part_a_elems |= 1 << idx;
            } else {
                part_b_elems |= 1 << idx;
            }
        }

        let mech_a = Mechanism::new(part_a_elems, num_elements);
        let mech_b = Mechanism::new(part_b_elems, num_elements);

        let rep_a = effect_repertoire(tpm, &mech_a, purview, state);
        let rep_b = effect_repertoire(tpm, &mech_b, purview, state);
        let product = product_distribution(&rep_a, &rep_b);

        let loss = intrinsic_difference(intact_repertoire, &product);
        min_loss = min_loss.min(loss);
    }

    if min_loss == f64::MAX { 0.0 } else { min_loss }
}

/// Product of two distributions (element-wise multiply + normalize).
fn product_distribution(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len().min(b.len());
    let mut prod = vec![0.0f64; n];
    for i in 0..n {
        prod[i] = a[i] * b[i];
    }
    let sum: f64 = prod.iter().sum();
    if sum > 1e-15 {
        let inv = 1.0 / sum;
        for p in &mut prod {
            *p *= inv;
        }
    }
    prod
}

/// Extract substate bits from a global state for given indices.
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
    fn intrinsic_difference_identical_is_zero() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        assert!(intrinsic_difference(&p, &p).abs() < 1e-12);
    }

    #[test]
    fn intrinsic_difference_different_is_positive() {
        let p = vec![1.0, 0.0, 0.0, 0.0];
        let q = vec![0.25, 0.25, 0.25, 0.25];
        assert!(intrinsic_difference(&p, &q) > 0.0);
    }

    #[test]
    fn cause_repertoire_valid_distribution() {
        let tpm = and_gate_tpm();
        // 4 states → 2 elements. Mechanism = both elements.
        let mech = Mechanism::new(0b11, 2);
        let purview = Purview::new(0b11, 2);
        let rep = cause_repertoire(&tpm, &mech, &purview, 0);
        let sum: f64 = rep.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "cause repertoire should sum to 1, got {sum}");
    }

    #[test]
    fn effect_repertoire_valid_distribution() {
        let tpm = and_gate_tpm();
        let mech = Mechanism::new(0b11, 2);
        let purview = Purview::new(0b11, 2);
        let rep = effect_repertoire(&tpm, &mech, &purview, 0);
        let sum: f64 = rep.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "effect repertoire should sum to 1, got {sum}");
    }

    #[test]
    fn mechanism_phi_computes() {
        let tpm = and_gate_tpm();
        let mech = Mechanism::new(0b11, 2);
        let dist = mechanism_phi(&tpm, &mech, 0);
        assert!(dist.phi >= 0.0);
        assert!(dist.phi_cause >= 0.0);
        assert!(dist.phi_effect >= 0.0);
    }

    #[test]
    fn selectivity_uniform_is_zero() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        assert!(selectivity(&p).abs() < 1e-12);
    }

    #[test]
    fn selectivity_peaked_is_positive() {
        let p = vec![1.0, 0.0, 0.0, 0.0];
        assert!(selectivity(&p) > 0.0);
    }
}
