//! Chebyshev polynomial spectral Φ approximation.
//!
//! Uses ruvector-math's ChebyshevExpansion to apply spectral filters
//! on the MI Laplacian without explicit eigendecomposition.
//! Achieves O(K·n²) where K is the polynomial degree (typically 20-50),
//! compared to O(n² · max_iter) for power iteration.
//!
//! Requires feature: `math-accel`

use crate::arena::PhiArena;
use crate::error::ConsciousnessError;
use crate::phi::partition_information_loss_pub;
use crate::simd::marginal_distribution;
use crate::traits::PhiEngine;
use crate::types::{Bipartition, ComputeBudget, PhiAlgorithm, PhiResult, TransitionMatrix};

use ruvector_math::spectral::{ChebyshevExpansion, ScaledLaplacian};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Chebyshev Spectral Φ Engine
// ---------------------------------------------------------------------------

/// Chebyshev-accelerated spectral Φ engine.
///
/// Instead of power iteration to find the Fiedler vector, applies a
/// low-pass Chebyshev filter to a random vector. The filtered result
/// approximates the Fiedler vector (projects onto low-frequency components
/// of the Laplacian spectrum).
///
/// Key advantage: no eigendecomposition needed. Filter application is
/// just K sparse matrix-vector products.
pub struct ChebyshevPhiEngine {
    /// Chebyshev polynomial degree (higher = better approximation).
    pub degree: usize,
    /// Low-pass filter cutoff (fraction of λ_max).
    pub cutoff: f64,
}

impl ChebyshevPhiEngine {
    pub fn new(degree: usize, cutoff: f64) -> Self {
        Self { degree, cutoff }
    }
}

impl Default for ChebyshevPhiEngine {
    fn default() -> Self {
        Self {
            degree: 30,
            cutoff: 0.1,
        }
    }
}

impl PhiEngine for ChebyshevPhiEngine {
    fn compute_phi(
        &self,
        tpm: &TransitionMatrix,
        state: Option<usize>,
        _budget: &ComputeBudget,
    ) -> Result<PhiResult, ConsciousnessError> {
        crate::phi::validate_tpm(tpm)?;
        let n = tpm.n;
        let state_idx = state.unwrap_or(0);
        let start = Instant::now();

        // Build MI adjacency as dense matrix for ScaledLaplacian.
        let marginal = marginal_distribution(tpm.as_slice(), n);
        let mut adj = vec![0.0f64; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                let mi = pairwise_mi_cheb(tpm, i, j, &marginal);
                adj[i * n + j] = mi;
                adj[j * n + i] = mi;
            }
        }

        // Build scaled Laplacian (normalizes eigenvalues to [-1, 1]).
        let scaled_lap = ScaledLaplacian::from_adjacency(&adj, n);

        // Create low-pass Chebyshev filter.
        // This amplifies the low-frequency components (near Fiedler eigenvalue).
        let filter = ChebyshevExpansion::low_pass(self.cutoff, self.degree);

        // Apply filter to a random vector to extract Fiedler-like component.
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(42);
        let mut signal: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() - 0.5).collect();

        // Remove DC component (project out constant eigenvector).
        let inv_n = 1.0 / n as f64;
        let mean: f64 = signal.iter().sum::<f64>() * inv_n;
        for s in &mut signal {
            *s -= mean;
        }

        // Apply Chebyshev filter: y = h(L̃) · x
        // This is done via K-step recurrence: T_0(L̃)x, T_1(L̃)x, ...
        let filtered = apply_chebyshev_filter(&scaled_lap, &signal, &filter, n);

        // Partition by sign of filtered vector (approximates Fiedler partition).
        let mut mask = 0u64;
        for i in 0..n {
            if filtered[i] >= 0.0 {
                mask |= 1 << i;
            }
        }
        let full = (1u64 << n) - 1;
        if mask == 0 { mask = 1; }
        if mask == full { mask = full - 1; }

        let partition = Bipartition { mask, n };
        let arena = PhiArena::with_capacity(n * 16);
        let phi = partition_information_loss_pub(tpm, state_idx, &partition, &arena);

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
        // K applications of L̃ · x, each O(n²) for dense or O(nnz) for sparse.
        self.degree as u64 * (n * n) as u64
    }
}

/// Apply Chebyshev filter to a signal via three-term recurrence.
///
/// y = Σ_k c_k T_k(L̃) x
/// where T_k is the k-th Chebyshev polynomial of the first kind.
fn apply_chebyshev_filter(
    lap: &ScaledLaplacian,
    signal: &[f64],
    filter: &ChebyshevExpansion,
    n: usize,
) -> Vec<f64> {
    let coeffs = &filter.coefficients;
    let k_max = coeffs.len();

    if k_max == 0 {
        return vec![0.0; n];
    }

    // T_0(L̃) x = x
    let t0 = signal.to_vec();

    // Output accumulator: y = c_0 * T_0(L̃) x
    let mut result = vec![0.0f64; n];
    for i in 0..n {
        result[i] = coeffs[0] * t0[i];
    }

    if k_max == 1 {
        return result;
    }

    // T_1(L̃) x = L̃ x
    let t1 = lap.apply(&t0);
    for i in 0..n {
        result[i] += coeffs[1] * t1[i];
    }

    if k_max == 2 {
        return result;
    }

    // Three-term recurrence: T_{k+1}(x) = 2x T_k(x) - T_{k-1}(x)
    let mut prev = t0;
    let mut curr = t1;

    for k in 2..k_max {
        let next_lap = lap.apply(&curr);
        let mut next = vec![0.0f64; n];
        for i in 0..n {
            next[i] = 2.0 * next_lap[i] - prev[i];
        }

        for i in 0..n {
            result[i] += coeffs[k] * next[i];
        }

        prev = curr;
        curr = next;
    }

    result
}

fn pairwise_mi_cheb(tpm: &TransitionMatrix, i: usize, j: usize, marginal: &[f64]) -> f64 {
    let n = tpm.n;
    let pi = marginal[i].max(1e-15);
    let pj = marginal[j].max(1e-15);
    let mut pij = 0.0;
    for state in 0..n {
        pij += tpm.get(state, i) * tpm.get(state, j);
    }
    pij /= n as f64;
    pij = pij.max(1e-15);
    (pij * (pij / (pi * pj)).ln()).max(0.0)
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
    fn chebyshev_phi_and_gate() {
        let tpm = and_gate_tpm();
        let budget = ComputeBudget::fast();
        let result = ChebyshevPhiEngine::default()
            .compute_phi(&tpm, Some(0), &budget)
            .unwrap();
        assert!(result.phi >= 0.0);
    }

    #[test]
    fn chebyshev_phi_disconnected() {
        let tpm = disconnected_tpm();
        let budget = ComputeBudget::exact();
        let result = ChebyshevPhiEngine::default()
            .compute_phi(&tpm, Some(0), &budget)
            .unwrap();
        assert!(result.phi < 1e-3, "chebyshev disconnected should be ~0, got {}", result.phi);
    }
}
