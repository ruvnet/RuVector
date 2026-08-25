//! # ruvector-dab-search
//!
//! Distance-Adaptive Beam (DAB) search: a per-query graph-traversal stopping
//! rule for approximate nearest-neighbour retrieval, replacing the fixed
//! `ef_search` budget used by standard HNSW-style beam search.
//!
//! ## Why this crate exists
//!
//! ADR-303 (`ruvector-entropy-ann`) tested whether the Shannon entropy of the
//! candidate-heap distance distribution could serve as a live, per-query
//! stopping signal. It measured a negative result: heap-distance entropy
//! saturates near `ln(n)` for every query on that PoC's data, so the
//! "adaptive" variant's apparent recall gain was entirely explained by a
//! larger effective search budget, not by any real per-query adaptivity.
//! That work's own prior-art table cited "Distance Adaptive Beam Search for
//! Provably Accurate Graph-Based Nearest Neighbor Search" (arXiv:2505.15636)
//! but did not implement it.
//!
//! This crate implements and measures that cited alternative honestly,
//! against the exact methodological trap that sank the entropy signal: a
//! **matched-budget control** is mandatory evidence here, not optional
//! commentary (see [`search`] docs and the research README).
//!
//! A second, unrelated trap surfaced during this crate's own development: an
//! exact per-node k-NN graph over well-separated clusters has few or no
//! edges *between* clusters, so a single fixed traversal entry point cannot
//! reach most of the corpus. [`graph::FlatGraph`] routes each query through a
//! small deterministic seed set instead (see its docs) — this is graph
//! plumbing, not part of the gamma hypothesis, but it is exactly the kind of
//! confound the attack pass in the research README is required to surface.
//!
//! ## The stopping rule
//!
//! Maintain the current top-k discovered results (`x_k` = the k-th best, the
//! worst of that set). Expand the frontier by nearest-first order. Stop as
//! soon as the closest unexpanded frontier candidate `x` satisfies:
//!
//! ```text
//! d(q, x) >= (1 + gamma) * d(q, x_k)      for gamma in (0, 2]
//! ```
//!
//! On a navigable graph this guarantees every undiscovered node is at least
//! `(gamma/2) * max_j d(q,j)` from the query — an approximation factor of
//! `2/gamma`, exact recovery at `gamma = 2`. This crate's flat k-NN graph is
//! not proven navigable, so that guarantee is not claimed to transfer
//! exactly; the benchmark measures recall empirically instead of relying on
//! the theorem. See the research README's attack pass for this distinction.
//!
//! ## Variants
//!
//! | Variant | Strategy | Description |
//! |---------|----------|-------------|
//! | [`search::FixedEf`] | Baseline | Fixed `ef_search` budget, result heap capacity `ef` |
//! | [`search::AdaptiveGamma`] (uncapped) | Candidate A | `(1+gamma)*d_k` stopping rule, result heap capacity `k` |
//! | [`search::AdaptiveGamma`] (capped) | Candidate B | Same rule plus a hard expansion-count safety bound |

pub mod dataset;
pub mod graph;
pub mod metrics;
pub mod search;

pub use graph::{FlatGraph, GraphConfig};
pub use search::{AdaptiveGamma, FixedEf, Hit, SearchOutcome, Searcher};

/// Recall@k: fraction of true top-k found in approximate results.
///
/// The denominator is `min(k, ground_truth.len())`. A searcher that returns
/// fewer than `k` results (e.g. via early termination) is penalised, not
/// rewarded: missing results count as misses.
pub fn recall_at_k(ground_truth: &[usize], results: &[Hit], k: usize) -> f32 {
    let k = k.min(ground_truth.len());
    if k == 0 {
        return 0.0;
    }
    let gt: std::collections::HashSet<usize> = ground_truth[..k].iter().cloned().collect();
    let found = results
        .iter()
        .take(k)
        .filter(|h| gt.contains(&h.id))
        .count();
    found as f32 / k as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_at_k_perfect() {
        let gt = vec![0usize, 1, 2, 3, 4];
        let results: Vec<Hit> = gt
            .iter()
            .enumerate()
            .map(|(i, &id)| Hit { id, dist: i as f32 })
            .collect();
        let r = recall_at_k(&gt, &results, 5);
        assert!((r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn recall_at_k_zero() {
        let gt = vec![0usize, 1, 2];
        let results: Vec<Hit> = vec![
            Hit { id: 10, dist: 0.1 },
            Hit { id: 11, dist: 0.2 },
            Hit { id: 12, dist: 0.3 },
        ];
        let r = recall_at_k(&gt, &results, 3);
        assert!(r.abs() < 1e-6);
    }

    #[test]
    fn recall_at_k_penalises_short_result_sets() {
        let gt = vec![0usize, 1, 2, 3, 4];
        // Only 2 results returned for k=5: should be scored out of 5, not 2.
        let results: Vec<Hit> = vec![Hit { id: 0, dist: 0.0 }, Hit { id: 1, dist: 0.1 }];
        let r = recall_at_k(&gt, &results, 5);
        assert!((r - 0.4).abs() < 1e-6, "r={r}");
    }
}
