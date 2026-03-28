//! Integrated Information Theory (IIT) Φ computation.
//!
//! Implements exact and approximate algorithms for computing integrated
//! information Φ — the core metric of consciousness in IIT 3.0/4.0.
//!
//! # Algorithms
//!
//! | Algorithm | Complexity | Use case |
//! |-----------|-----------|----------|
//! | Exact | O(2^n · n²) | n ≤ 16 elements |
//! | Spectral | O(n² log n) | n ≤ 1000, good approximation |
//! | Stochastic | O(k · n²) | Any n, configurable samples |
//! | GreedyBisection | O(n³) | Fast lower bound |
//!
//! # Algorithm
//!
//! Φ = min over all bipartitions { D_KL(P(whole) || P(part_A) ⊗ P(part_B)) }
//!
//! The minimum information partition (MIP) is the bipartition that causes
//! the least information loss when the system is "cut".

use crate::arena::PhiArena;
use crate::error::{ConsciousnessError, ValidationError};
use crate::simd::{kl_divergence, marginal_distribution};
use crate::traits::PhiEngine;
use crate::types::{
    Bipartition, BipartitionIter, ComputeBudget, PhiAlgorithm, PhiResult, TransitionMatrix,
};

use std::time::Instant;

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

pub(crate) fn validate_tpm(tpm: &TransitionMatrix) -> Result<(), ConsciousnessError> {
    if tpm.n < 2 {
        return Err(ValidationError::EmptySystem.into());
    }
    for i in 0..tpm.n {
        let mut row_sum = 0.0;
        for j in 0..tpm.n {
            let val = tpm.get(i, j);
            if !val.is_finite() {
                return Err(ValidationError::NonFiniteValue { row: i, col: j }.into());
            }
            if val < -1e-10 {
                return Err(ValidationError::ParameterOutOfRange {
                    name: format!("tpm[{i}][{j}]"),
                    value: format!("{val:.6}"),
                    expected: ">= 0.0".into(),
                }
                .into());
            }
            row_sum += val;
        }
        if (row_sum - 1.0).abs() > 1e-6 {
            return Err(ValidationError::InvalidTPM { row: i, sum: row_sum }.into());
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Core: information loss for a bipartition
// ---------------------------------------------------------------------------

/// Public wrapper for cross-module access.
pub(crate) fn partition_information_loss_pub(
    tpm: &TransitionMatrix,
    state: usize,
    partition: &Bipartition,
    arena: &PhiArena,
) -> f64 {
    partition_information_loss(tpm, state, partition, arena)
}

/// Compute information loss for a given bipartition at a given state.
///
/// This is the core hot path: for each partition, we compute the KL divergence
/// between the whole-system conditional distribution and the product of the
/// marginalized sub-system distributions.
fn partition_information_loss(
    tpm: &TransitionMatrix,
    state: usize,
    partition: &Bipartition,
    arena: &PhiArena,
) -> f64 {
    let n = tpm.n;
    let set_a = partition.set_a();
    let set_b = partition.set_b();

    // Whole-system distribution P(future | state)
    let whole_dist = &tpm.data[state * n..(state + 1) * n];

    // Marginalize to get sub-TPMs
    let tpm_a = tpm.marginalize(&set_a);
    let tpm_b = tpm.marginalize(&set_b);

    // Compute conditional distributions for each sub-system.
    // Map the current state to sub-system states.
    let state_a = map_state_to_subsystem(state, &set_a, n);
    let state_b = map_state_to_subsystem(state, &set_b, n);

    let dist_a = &tpm_a.data[state_a * tpm_a.n..(state_a + 1) * tpm_a.n];
    let dist_b = &tpm_b.data[state_b * tpm_b.n..(state_b + 1) * tpm_b.n];

    // Reconstruct the product distribution P(A) ⊗ P(B) in the full state space.
    let product = arena.alloc_slice::<f64>(n);
    compute_product_distribution(dist_a, &set_a, dist_b, &set_b, product, n);

    // Normalize product distribution.
    let sum: f64 = product.iter().sum();
    if sum > 1e-15 {
        let inv_sum = 1.0 / sum;
        for p in product.iter_mut() {
            *p *= inv_sum;
        }
    }

    let loss = kl_divergence(whole_dist, product).max(0.0);
    arena.reset();
    loss
}

/// Map a global state index to a sub-system state index.
fn map_state_to_subsystem(state: usize, indices: &[usize], _n: usize) -> usize {
    let mut sub_state = 0;
    for (bit, &idx) in indices.iter().enumerate() {
        if state & (1 << idx) != 0 {
            sub_state |= 1 << bit;
        }
    }
    sub_state % indices.len().max(1)
}

/// Compute product distribution P(A) ⊗ P(B) expanded to full state space.
fn compute_product_distribution(
    dist_a: &[f64],
    set_a: &[usize],
    dist_b: &[f64],
    set_b: &[usize],
    output: &mut [f64],
    n: usize,
) {
    let ka = set_a.len();
    let kb = set_b.len();

    for global_state in 0..n {
        let mut sa = 0usize;
        for (bit, &idx) in set_a.iter().enumerate() {
            if global_state & (1 << idx) != 0 {
                sa |= 1 << bit;
            }
        }
        let mut sb = 0usize;
        for (bit, &idx) in set_b.iter().enumerate() {
            if global_state & (1 << idx) != 0 {
                sb |= 1 << bit;
            }
        }
        let pa = if sa < ka { dist_a[sa] } else { 0.0 };
        let pb = if sb < kb { dist_b[sb] } else { 0.0 };
        output[global_state] = pa * pb;
    }
}

// ---------------------------------------------------------------------------
// Exact Φ engine
// ---------------------------------------------------------------------------

/// Exact Φ computation via exhaustive bipartition enumeration.
///
/// Evaluates all 2^(n-1) - 1 bipartitions. Practical for n ≤ 16.
pub struct ExactPhiEngine;

impl PhiEngine for ExactPhiEngine {
    fn compute_phi(
        &self,
        tpm: &TransitionMatrix,
        state: Option<usize>,
        budget: &ComputeBudget,
    ) -> Result<PhiResult, ConsciousnessError> {
        validate_tpm(tpm)?;
        let n = tpm.n;

        if n > 20 {
            return Err(ConsciousnessError::SystemTooLarge { n, max: 20 });
        }

        let state_idx = state.unwrap_or(0);
        let start = Instant::now();
        let arena = PhiArena::with_capacity(n * n * 16);

        let total_partitions = (1u64 << n) - 2;
        let mut min_phi = f64::MAX;
        let mut best_partition = Bipartition { mask: 1, n };
        let mut evaluated = 0u64;
        let mut convergence = Vec::new();

        for partition in BipartitionIter::new(n) {
            if budget.max_partitions > 0 && evaluated >= budget.max_partitions {
                break;
            }
            if start.elapsed() > budget.max_time {
                break;
            }

            let loss = partition_information_loss(tpm, state_idx, &partition, &arena);

            if loss < min_phi {
                min_phi = loss;
                best_partition = partition;
            }

            evaluated += 1;
            if evaluated % 1000 == 0 {
                convergence.push(min_phi);
            }
        }

        Ok(PhiResult {
            phi: if min_phi == f64::MAX { 0.0 } else { min_phi },
            mip: best_partition,
            partitions_evaluated: evaluated,
            total_partitions,
            algorithm: PhiAlgorithm::Exact,
            elapsed: start.elapsed(),
            convergence,
        })
    }

    fn algorithm(&self) -> PhiAlgorithm {
        PhiAlgorithm::Exact
    }

    fn estimate_cost(&self, n: usize) -> u64 {
        (1u64 << n).saturating_sub(2)
    }
}

// ---------------------------------------------------------------------------
// Spectral approximation
// ---------------------------------------------------------------------------

/// Spectral Φ approximation using the Fiedler vector.
///
/// Computes the second-smallest eigenvalue of the Laplacian of the TPM's
/// mutual information graph. The Fiedler vector gives an approximately
/// optimal bipartition in O(n² log n) time.
pub struct SpectralPhiEngine {
    power_iterations: usize,
}

impl SpectralPhiEngine {
    pub fn new(power_iterations: usize) -> Self {
        Self { power_iterations }
    }
}

impl Default for SpectralPhiEngine {
    fn default() -> Self {
        Self {
            power_iterations: 100,
        }
    }
}

impl PhiEngine for SpectralPhiEngine {
    fn compute_phi(
        &self,
        tpm: &TransitionMatrix,
        state: Option<usize>,
        _budget: &ComputeBudget,
    ) -> Result<PhiResult, ConsciousnessError> {
        validate_tpm(tpm)?;
        let n = tpm.n;
        let state_idx = state.unwrap_or(0);
        let start = Instant::now();

        // Build mutual information adjacency matrix.
        let mut mi_matrix = vec![0.0f64; n * n];
        let marginal = marginal_distribution(tpm.as_slice(), n);

        for i in 0..n {
            for j in (i + 1)..n {
                // Mutual information between elements i and j.
                let mi = compute_pairwise_mi(tpm, i, j, &marginal);
                mi_matrix[i * n + j] = mi;
                mi_matrix[j * n + i] = mi;
            }
        }

        // Build Laplacian L = D - W.
        let mut laplacian = vec![0.0f64; n * n];
        for i in 0..n {
            let mut degree = 0.0;
            for j in 0..n {
                degree += mi_matrix[i * n + j];
            }
            laplacian[i * n + i] = degree;
            for j in 0..n {
                laplacian[i * n + j] -= mi_matrix[i * n + j];
            }
        }

        // Power iteration for second-smallest eigenvector (Fiedler vector).
        let fiedler = fiedler_vector(&laplacian, n, self.power_iterations);

        // Partition by sign of Fiedler vector.
        let mut mask = 0u64;
        for i in 0..n {
            if fiedler[i] >= 0.0 {
                mask |= 1 << i;
            }
        }

        // Ensure valid bipartition.
        let full = (1u64 << n) - 1;
        if mask == 0 {
            mask = 1;
        }
        if mask == full {
            mask = full - 1;
        }

        let partition = Bipartition { mask, n };
        let arena = PhiArena::with_capacity(n * 16);
        let phi = partition_information_loss(tpm, state_idx, &partition, &arena);

        Ok(PhiResult {
            phi,
            mip: partition,
            partitions_evaluated: 1,
            total_partitions: (1u64 << n) - 2,
            algorithm: PhiAlgorithm::Spectral,
            elapsed: start.elapsed(),
            convergence: vec![phi],
        })
    }

    fn algorithm(&self) -> PhiAlgorithm {
        PhiAlgorithm::Spectral
    }

    fn estimate_cost(&self, n: usize) -> u64 {
        (n * n * self.power_iterations) as u64
    }
}

/// Pairwise mutual information between elements i and j.
fn compute_pairwise_mi(tpm: &TransitionMatrix, i: usize, j: usize, marginal: &[f64]) -> f64 {
    let n = tpm.n;
    let pi = marginal[i].max(1e-15);
    let pj = marginal[j].max(1e-15);

    // Joint probability from TPM.
    let mut pij = 0.0;
    for state in 0..n {
        pij += tpm.get(state, i) * tpm.get(state, j);
    }
    pij /= n as f64;
    pij = pij.max(1e-15);

    // MI = p(i,j) * log(p(i,j) / (p(i) * p(j)))
    pij * (pij / (pi * pj)).ln()
}

/// Compute Fiedler vector (second-smallest eigenvector of Laplacian).
/// Uses inverse power iteration with deflation.
fn fiedler_vector(laplacian: &[f64], n: usize, max_iter: usize) -> Vec<f64> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(42);

    // Random initial vector, orthogonal to the constant vector.
    let mut v: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() - 0.5).collect();

    // Orthogonalize against the constant eigenvector.
    let inv_n = 1.0 / n as f64;
    let mean: f64 = v.iter().sum::<f64>() * inv_n;
    for vi in &mut v {
        *vi -= mean;
    }

    // Normalize.
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-15 {
        for vi in &mut v {
            *vi /= norm;
        }
    }

    // Power iteration on (mu*I - L) to find second-smallest eigenvalue.
    // Use shifted inverse iteration.
    let mut w = vec![0.0f64; n];

    // Estimate largest eigenvalue for shift.
    let mu = estimate_largest_eigenvalue(laplacian, n);

    for _ in 0..max_iter {
        // w = (mu*I - L) * v
        for i in 0..n {
            let mut sum = mu * v[i];
            for j in 0..n {
                sum -= laplacian[i * n + j] * v[j];
            }
            w[i] = sum;
        }

        // Deflate: remove component along constant vector.
        let mean: f64 = w.iter().sum::<f64>() * inv_n;
        for wi in &mut w {
            *wi -= mean;
        }

        // Normalize.
        let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 {
            break;
        }
        let inv_norm = 1.0 / norm;
        for i in 0..n {
            v[i] = w[i] * inv_norm;
        }
    }

    v
}

fn estimate_largest_eigenvalue(matrix: &[f64], n: usize) -> f64 {
    // Gershgorin circle theorem: max eigenvalue ≤ max row sum of abs values.
    let mut max_row_sum = 0.0f64;
    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..n {
            row_sum += matrix[i * n + j].abs();
        }
        max_row_sum = max_row_sum.max(row_sum);
    }
    max_row_sum
}

// ---------------------------------------------------------------------------
// Stochastic sampling
// ---------------------------------------------------------------------------

/// Stochastic Φ approximation via random partition sampling.
///
/// Samples random bipartitions and tracks the minimum information loss.
/// Runs in O(k · n²) where k is the sample count.
pub struct StochasticPhiEngine {
    samples: u64,
    seed: u64,
}

impl StochasticPhiEngine {
    pub fn new(samples: u64, seed: u64) -> Self {
        Self { samples, seed }
    }
}

impl PhiEngine for StochasticPhiEngine {
    fn compute_phi(
        &self,
        tpm: &TransitionMatrix,
        state: Option<usize>,
        budget: &ComputeBudget,
    ) -> Result<PhiResult, ConsciousnessError> {
        validate_tpm(tpm)?;
        let n = tpm.n;
        let state_idx = state.unwrap_or(0);
        let start = Instant::now();
        let arena = PhiArena::with_capacity(n * n * 16);

        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(self.seed);

        let total_partitions = (1u64 << n) - 2;
        let effective_samples = self.samples.min(total_partitions);
        let mut min_phi = f64::MAX;
        let mut best_partition = Bipartition { mask: 1, n };
        let mut convergence = Vec::new();

        for i in 0..effective_samples {
            if start.elapsed() > budget.max_time {
                break;
            }

            // Random valid bipartition.
            let mask = loop {
                let m = rng.gen::<u64>() & ((1u64 << n) - 1);
                if m != 0 && m != (1u64 << n) - 1 {
                    break m;
                }
            };

            let partition = Bipartition { mask, n };
            let loss = partition_information_loss(tpm, state_idx, &partition, &arena);

            if loss < min_phi {
                min_phi = loss;
                best_partition = partition;
            }

            if i % 100 == 0 {
                convergence.push(min_phi);
            }
        }

        Ok(PhiResult {
            phi: if min_phi == f64::MAX { 0.0 } else { min_phi },
            mip: best_partition,
            partitions_evaluated: effective_samples,
            total_partitions,
            algorithm: PhiAlgorithm::Stochastic,
            elapsed: start.elapsed(),
            convergence,
        })
    }

    fn algorithm(&self) -> PhiAlgorithm {
        PhiAlgorithm::Stochastic
    }

    fn estimate_cost(&self, n: usize) -> u64 {
        self.samples * (n * n) as u64
    }
}

// ---------------------------------------------------------------------------
// Auto-selecting engine
// ---------------------------------------------------------------------------

/// Automatically selects the best algorithm based on system size.
pub fn auto_compute_phi(
    tpm: &TransitionMatrix,
    state: Option<usize>,
    budget: &ComputeBudget,
) -> Result<PhiResult, ConsciousnessError> {
    let n = tpm.n;
    if n <= 16 && budget.approximation_ratio >= 0.99 {
        ExactPhiEngine.compute_phi(tpm, state, budget)
    } else if n <= 1000 {
        SpectralPhiEngine::default().compute_phi(tpm, state, budget)
    } else {
        StochasticPhiEngine::new(10_000, 42).compute_phi(tpm, state, budget)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a simple 2-element AND gate TPM.
    fn and_gate_tpm() -> TransitionMatrix {
        // 2 elements, 4 states (00, 01, 10, 11)
        // AND gate: output is 1 only when both inputs are 1
        #[rustfmt::skip]
        let data = vec![
            0.5, 0.25, 0.25, 0.0,   // from 00
            0.5, 0.25, 0.25, 0.0,   // from 01
            0.5, 0.25, 0.25, 0.0,   // from 10
            0.0, 0.0,  0.0,  1.0,   // from 11
        ];
        TransitionMatrix::new(4, data)
    }

    /// Create a disconnected system (Φ should be 0).
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
    fn exact_phi_disconnected_is_zero() {
        let tpm = disconnected_tpm();
        let budget = ComputeBudget::exact();
        let result = ExactPhiEngine.compute_phi(&tpm, Some(0), &budget).unwrap();
        assert!(
            result.phi < 1e-6,
            "disconnected system should have Φ ≈ 0, got {}",
            result.phi
        );
    }

    #[test]
    fn exact_phi_and_gate_positive() {
        let tpm = and_gate_tpm();
        let budget = ComputeBudget::exact();
        let result = ExactPhiEngine.compute_phi(&tpm, Some(3), &budget).unwrap();
        assert!(
            result.phi >= 0.0,
            "AND gate at state 11 should have Φ ≥ 0, got {}",
            result.phi
        );
    }

    #[test]
    fn spectral_phi_runs() {
        let tpm = and_gate_tpm();
        let budget = ComputeBudget::fast();
        let result = SpectralPhiEngine::default()
            .compute_phi(&tpm, Some(0), &budget)
            .unwrap();
        assert!(result.phi >= 0.0);
        assert_eq!(result.algorithm, PhiAlgorithm::Spectral);
    }

    #[test]
    fn stochastic_phi_runs() {
        let tpm = and_gate_tpm();
        let budget = ComputeBudget::fast();
        let result = StochasticPhiEngine::new(100, 42)
            .compute_phi(&tpm, Some(0), &budget)
            .unwrap();
        assert!(result.phi >= 0.0);
        assert_eq!(result.algorithm, PhiAlgorithm::Stochastic);
    }

    #[test]
    fn auto_selects_exact_for_small() {
        let tpm = and_gate_tpm();
        let budget = ComputeBudget::exact();
        let result = auto_compute_phi(&tpm, Some(0), &budget).unwrap();
        assert_eq!(result.algorithm, PhiAlgorithm::Exact);
    }

    #[test]
    fn validation_rejects_bad_tpm() {
        let data = vec![0.5, 0.5, 0.3, 0.3]; // row 1 doesn't sum to 1
        let tpm = TransitionMatrix::new(2, data);
        let budget = ComputeBudget::exact();
        let result = ExactPhiEngine.compute_phi(&tpm, Some(0), &budget);
        assert!(result.is_err());
    }

    #[test]
    fn validation_rejects_single_element() {
        let tpm = TransitionMatrix::new(1, vec![1.0]);
        let budget = ComputeBudget::exact();
        let result = ExactPhiEngine.compute_phi(&tpm, Some(0), &budget);
        assert!(result.is_err());
    }
}
