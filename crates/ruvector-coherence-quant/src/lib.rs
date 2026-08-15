//! # ruvector-coherence-quant
//!
//! Coherence-adaptive quantization for approximate vector search.
//!
//! Uniform scalar/product quantization applies the same bit budget to every
//! vector in a corpus. But not every vector is equally sensitive to
//! precision loss: a vector deep in a dense semantic cluster is redundantly
//! described by its neighbours (losing a few bits of precision barely moves
//! its rank among nearby candidates), while a vector that sits on a cluster
//! boundary or bridges two topics has few redundant neighbours to fall back
//! on, so quantization error there is more likely to flip nearest-neighbour
//! rankings.
//!
//! This crate tests whether a **mutual k-NN coherence score**
//! ([`coherence::mutual_knn_coherence`]) — a lightweight, ground-truth-free
//! proxy for local graph conductance / cluster-boundary detection — can
//! drive a per-vector bit-width decision that recovers most of full-precision
//! recall at close to aggressive-quantization memory.
//!
//! **Hypothesised mechanism:** high mutual-kNN coherence (v's neighbours
//! agree that v is close to them too) marks a cluster core → quantize
//! aggressively (4-bit). Low coherence marks a boundary/bridge point →
//! keep more precision (8-bit).
//!
//! ## Variants
//!
//! | Variant | Bit allocation | Description |
//! |---------|----------------|-------------|
//! | Baseline | uniform 8-bit | Full scalar-quantization precision |
//! | Candidate A | uniform 4-bit | Aggressive uniform compression |
//! | Candidate B | adaptive 4/8-bit | Mutual-kNN coherence decides per vector |
//!
//! ## RuVector ecosystem fit
//!
//! Connects the coherence/graph-conductance theme (`ruvector-mincut`,
//! `ruvector-coherence`) with the quantization theme (`ruvector-rabitq`,
//! `ruvector-turboquant`, `ruvector-pq-search`) and agent-memory footprint
//! reduction (`ruvector-agent-memory`). This PoC keeps its own lightweight
//! mutual-kNN graph builder rather than depending on `ruvector-mincut`
//! directly, to stay evaluator-independent; see the research README for the
//! production integration path.

pub mod coherence;
pub mod dataset;
pub mod metrics;
pub mod quantize;
pub mod search;

pub use search::{Hit, QuantizedIndex};

/// Recall@k: fraction of true top-k found in approximate results.
///
/// The denominator is `min(k, ground_truth.len())` only. A searcher that
/// returns fewer than `k` results is penalised, not rewarded: missing
/// results count as misses.
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
    fn recall_at_k_partial() {
        let gt = vec![0usize, 1, 2, 3];
        let results: Vec<Hit> = vec![
            Hit { id: 0, dist: 0.1 },
            Hit { id: 99, dist: 0.2 },
            Hit { id: 2, dist: 0.3 },
            Hit { id: 98, dist: 0.4 },
        ];
        let r = recall_at_k(&gt, &results, 4);
        assert!((r - 0.5).abs() < 1e-6);
    }
}
