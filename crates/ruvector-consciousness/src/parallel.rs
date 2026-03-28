//! Parallel partition search using rayon.
//!
//! Distributes bipartition evaluation across available CPU cores
//! for near-linear speedup on multi-core systems.
//!
//! Feature-gated behind `parallel` (requires `rayon` + `crossbeam`).

use crate::arena::PhiArena;
use crate::error::ConsciousnessError;
use crate::phi::{partition_information_loss_pub, validate_tpm};
use crate::traits::PhiEngine;
use crate::types::{
    Bipartition, BipartitionIter, ComputeBudget, PhiAlgorithm, PhiResult, TransitionMatrix,
};

use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Parallel Exact Engine
// ---------------------------------------------------------------------------

/// Parallel exact Φ computation.
///
/// Distributes bipartitions across rayon's thread pool. Each thread
/// maintains its own arena for zero-contention allocation.
pub struct ParallelPhiEngine {
    /// Chunk size for work distribution.
    pub chunk_size: usize,
}

impl ParallelPhiEngine {
    pub fn new(chunk_size: usize) -> Self {
        Self { chunk_size }
    }
}

impl Default for ParallelPhiEngine {
    fn default() -> Self {
        Self { chunk_size: 256 }
    }
}

impl PhiEngine for ParallelPhiEngine {
    fn compute_phi(
        &self,
        tpm: &TransitionMatrix,
        state: Option<usize>,
        budget: &ComputeBudget,
    ) -> Result<PhiResult, ConsciousnessError> {
        validate_tpm(tpm)?;
        let n = tpm.n;

        if n > 25 {
            return Err(ConsciousnessError::SystemTooLarge { n, max: 25 });
        }

        let state_idx = state.unwrap_or(0);
        let start = Instant::now();
        let total_partitions = (1u64 << n) - 2;
        let evaluated = AtomicU64::new(0);

        // Collect all valid bipartitions into chunks for parallel processing.
        let partitions: Vec<Bipartition> = BipartitionIter::new(n).collect();

        // Process in parallel chunks.
        let results: Vec<(f64, Bipartition)> = partitions
            .par_chunks(self.chunk_size)
            .filter_map(|chunk| {
                // Check time budget.
                if start.elapsed() > budget.max_time {
                    return None;
                }

                // Thread-local arena.
                let arena = PhiArena::with_capacity(n * n * 16);
                let mut local_min = f64::MAX;
                let mut local_best = chunk[0].clone();

                for partition in chunk {
                    if budget.max_partitions > 0
                        && evaluated.load(Ordering::Relaxed) >= budget.max_partitions
                    {
                        break;
                    }

                    let loss = partition_information_loss_pub(tpm, state_idx, partition, &arena);
                    arena.reset();
                    evaluated.fetch_add(1, Ordering::Relaxed);

                    if loss < local_min {
                        local_min = loss;
                        local_best = partition.clone();
                    }
                }

                Some((local_min, local_best))
            })
            .collect();

        // Reduce across chunks.
        let mut min_phi = f64::MAX;
        let mut best_partition = Bipartition { mask: 1, n };

        for (phi, partition) in results {
            if phi < min_phi {
                min_phi = phi;
                best_partition = partition;
            }
        }

        Ok(PhiResult {
            phi: if min_phi == f64::MAX { 0.0 } else { min_phi },
            mip: best_partition,
            partitions_evaluated: evaluated.load(Ordering::Relaxed),
            total_partitions,
            algorithm: PhiAlgorithm::Exact,
            elapsed: start.elapsed(),
            convergence: vec![min_phi],
        })
    }

    fn algorithm(&self) -> PhiAlgorithm {
        PhiAlgorithm::Exact
    }

    fn estimate_cost(&self, n: usize) -> u64 {
        // Same total work as exact, but wall-time is divided by core count.
        (1u64 << n).saturating_sub(2)
    }
}

// ---------------------------------------------------------------------------
// Parallel Stochastic Engine
// ---------------------------------------------------------------------------

/// Parallel stochastic Φ computation.
///
/// Distributes random partition samples across threads, each with
/// an independent RNG seed.
pub struct ParallelStochasticPhiEngine {
    /// Total samples across all threads.
    pub total_samples: u64,
    /// Base seed (each thread gets seed + thread_id).
    pub seed: u64,
}

impl ParallelStochasticPhiEngine {
    pub fn new(total_samples: u64, seed: u64) -> Self {
        Self { total_samples, seed }
    }
}

impl PhiEngine for ParallelStochasticPhiEngine {
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
        let total_partitions = (1u64 << n) - 2;

        let num_threads = rayon::current_num_threads();
        let samples_per_thread = (self.total_samples / num_threads as u64).max(1);

        let results: Vec<(f64, Bipartition, u64)> = (0..num_threads)
            .into_par_iter()
            .filter_map(|thread_id| {
                use rand::rngs::StdRng;
                use rand::{Rng, SeedableRng};

                if start.elapsed() > budget.max_time {
                    return None;
                }

                let mut rng = StdRng::seed_from_u64(self.seed + thread_id as u64);
                let arena = PhiArena::with_capacity(n * n * 16);
                let mut local_min = f64::MAX;
                let mut local_best = Bipartition { mask: 1, n };
                let mut count = 0u64;

                for _ in 0..samples_per_thread {
                    if start.elapsed() > budget.max_time {
                        break;
                    }

                    let mask = loop {
                        let m = rng.gen::<u64>() & ((1u64 << n) - 1);
                        if m != 0 && m != (1u64 << n) - 1 {
                            break m;
                        }
                    };

                    let partition = Bipartition { mask, n };
                    let loss = partition_information_loss_pub(tpm, state_idx, &partition, &arena);
                    arena.reset();
                    count += 1;

                    if loss < local_min {
                        local_min = loss;
                        local_best = partition;
                    }
                }

                Some((local_min, local_best, count))
            })
            .collect();

        let mut min_phi = f64::MAX;
        let mut best_partition = Bipartition { mask: 1, n };
        let mut total_evaluated = 0u64;

        for (phi, partition, count) in results {
            total_evaluated += count;
            if phi < min_phi {
                min_phi = phi;
                best_partition = partition;
            }
        }

        Ok(PhiResult {
            phi: if min_phi == f64::MAX { 0.0 } else { min_phi },
            mip: best_partition,
            partitions_evaluated: total_evaluated,
            total_partitions,
            algorithm: PhiAlgorithm::Stochastic,
            elapsed: start.elapsed(),
            convergence: vec![min_phi],
        })
    }

    fn algorithm(&self) -> PhiAlgorithm {
        PhiAlgorithm::Stochastic
    }

    fn estimate_cost(&self, n: usize) -> u64 {
        self.total_samples * (n * n) as u64
    }
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
    fn parallel_exact_disconnected_zero() {
        let tpm = disconnected_tpm();
        let budget = ComputeBudget::exact();
        let result = ParallelPhiEngine::default()
            .compute_phi(&tpm, Some(0), &budget)
            .unwrap();
        assert!(
            result.phi < 1e-6,
            "parallel disconnected should be ≈ 0, got {}",
            result.phi
        );
    }

    #[test]
    fn parallel_exact_and_gate() {
        let tpm = and_gate_tpm();
        let budget = ComputeBudget::exact();
        let result = ParallelPhiEngine::default()
            .compute_phi(&tpm, Some(3), &budget)
            .unwrap();
        assert!(result.phi >= 0.0);
    }

    #[test]
    fn parallel_stochastic_runs() {
        let tpm = and_gate_tpm();
        let budget = ComputeBudget::fast();
        let result = ParallelStochasticPhiEngine::new(500, 42)
            .compute_phi(&tpm, Some(0), &budget)
            .unwrap();
        assert!(result.phi >= 0.0);
        assert!(result.partitions_evaluated > 0);
    }

    #[test]
    fn parallel_matches_sequential() {
        let tpm = disconnected_tpm();
        let budget = ComputeBudget::exact();

        let seq = crate::phi::ExactPhiEngine
            .compute_phi(&tpm, Some(0), &budget)
            .unwrap();
        let par = ParallelPhiEngine::default()
            .compute_phi(&tpm, Some(0), &budget)
            .unwrap();

        assert!(
            (seq.phi - par.phi).abs() < 1e-10,
            "parallel ({}) should match sequential ({})",
            par.phi,
            seq.phi
        );
    }
}
