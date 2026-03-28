//! MinCut-accelerated partition search for Φ computation.
//!
//! Uses ruvector-mincut's graph partitioning algorithms to guide
//! the bipartition search instead of exhaustive enumeration.
//! The Minimum Information Partition is a form of graph cut on
//! the mutual information adjacency graph.
//!
//! Requires feature: `mincut-accel`

use crate::arena::PhiArena;
use crate::error::ConsciousnessError;
use crate::phi::partition_information_loss_pub;
use crate::simd::marginal_distribution;
use crate::traits::PhiEngine;
use crate::types::{Bipartition, ComputeBudget, PhiAlgorithm, PhiResult, TransitionMatrix};

use ruvector_mincut::MinCutBuilder;
use std::time::Instant;

// ---------------------------------------------------------------------------
// MinCut Φ Engine
// ---------------------------------------------------------------------------

/// MinCut-guided Φ engine.
///
/// Constructs a weighted graph from pairwise mutual information,
/// then uses ruvector-mincut's algorithms to find candidate
/// partitions. Evaluates information loss only for the top-k
/// candidate cuts, avoiding exhaustive enumeration.
pub struct MinCutPhiEngine {
    /// Number of candidate cuts to evaluate.
    pub max_candidates: usize,
}

impl MinCutPhiEngine {
    pub fn new(max_candidates: usize) -> Self {
        Self { max_candidates }
    }
}

impl Default for MinCutPhiEngine {
    fn default() -> Self {
        Self { max_candidates: 32 }
    }
}

impl PhiEngine for MinCutPhiEngine {
    fn compute_phi(
        &self,
        tpm: &TransitionMatrix,
        state: Option<usize>,
        budget: &ComputeBudget,
    ) -> Result<PhiResult, ConsciousnessError> {
        crate::phi::validate_tpm(tpm)?;
        let n = tpm.n;
        let state_idx = state.unwrap_or(0);
        let start = Instant::now();
        let arena = PhiArena::with_capacity(n * n * 16);

        // Build MI-weighted graph as edge list.
        let marginal = marginal_distribution(tpm.as_slice(), n);
        let mut edges: Vec<(u64, u64, f64)> = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                let mi = pairwise_mi_local(tpm, i, j, &marginal);
                if mi > 1e-10 {
                    // MinCut minimizes total cut weight → low MI edges are cheap to cut.
                    // We want MIP = partition that loses least info.
                    // Invert: cut weight = MI (cutting high-MI edges is costly).
                    edges.push((i as u64, j as u64, mi));
                }
            }
        }

        let total_partitions = (1u64 << n) - 2;
        let mut min_phi = f64::MAX;
        let mut best_partition = Bipartition { mask: 1, n };
        let mut evaluated = 0u64;
        let mut convergence = Vec::new();

        // Use MinCut builder pattern to find the minimum weight cut.
        let mincut_result = MinCutBuilder::new()
            .exact()
            .with_edges(edges)
            .build();

        if let Ok(mincut) = mincut_result {
            let result = mincut.min_cut();
            // Extract the partition from the cut result.
            if let Some((side_s, _side_t)) = &result.partition {
                let mut mask = 0u64;
                for &v in side_s {
                    if v < 64 {
                        mask |= 1 << v;
                    }
                }

                let full = (1u64 << n) - 1;
                if mask != 0 && mask != full {
                    let partition = Bipartition { mask, n };
                    let loss = partition_information_loss_pub(tpm, state_idx, &partition, &arena);
                    arena.reset();
                    evaluated += 1;

                    if loss < min_phi {
                        min_phi = loss;
                        best_partition = partition;
                    }
                    convergence.push(min_phi);
                }
            }
        }

        // Also try perturbations of the MinCut partition.
        let base_mask = best_partition.mask;
        for elem in 0..n.min(64) {
            if start.elapsed() > budget.max_time {
                break;
            }
            if evaluated >= self.max_candidates as u64 {
                break;
            }

            let new_mask = base_mask ^ (1 << elem);
            let full = (1u64 << n) - 1;
            if new_mask == 0 || new_mask == full {
                continue;
            }

            let partition = Bipartition { mask: new_mask, n };
            let loss = partition_information_loss_pub(tpm, state_idx, &partition, &arena);
            arena.reset();
            evaluated += 1;

            if loss < min_phi {
                min_phi = loss;
                best_partition = partition;
            }
        }

        convergence.push(if min_phi == f64::MAX { 0.0 } else { min_phi });

        Ok(PhiResult {
            phi: if min_phi == f64::MAX { 0.0 } else { min_phi },
            mip: best_partition,
            partitions_evaluated: evaluated,
            total_partitions,
            algorithm: PhiAlgorithm::GeoMIP, // MinCut-guided
            elapsed: start.elapsed(),
            convergence,
        })
    }

    fn algorithm(&self) -> PhiAlgorithm {
        PhiAlgorithm::GeoMIP
    }

    fn estimate_cost(&self, n: usize) -> u64 {
        // MinCut: ~O(n² log n) + candidates × n²
        (n * n) as u64 * (n as f64).log2() as u64 + self.max_candidates as u64 * (n * n) as u64
    }
}

fn pairwise_mi_local(tpm: &TransitionMatrix, i: usize, j: usize, marginal: &[f64]) -> f64 {
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
    fn mincut_phi_and_gate() {
        let tpm = and_gate_tpm();
        let budget = ComputeBudget::fast();
        let result = MinCutPhiEngine::default()
            .compute_phi(&tpm, Some(0), &budget)
            .unwrap();
        assert!(result.phi >= 0.0);
    }

    #[test]
    fn mincut_phi_disconnected() {
        let tpm = disconnected_tpm();
        let budget = ComputeBudget::exact();
        let result = MinCutPhiEngine::default()
            .compute_phi(&tpm, Some(0), &budget)
            .unwrap();
        assert!(result.phi < 1e-3, "disconnected should be ~0, got {}", result.phi);
    }
}
