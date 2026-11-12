//! # ruvector-entropy-ann
//!
//! Entropy-adaptive beam search for approximate nearest-neighbour retrieval.
//!
//! Standard HNSW and graph-ANN systems use a fixed `ef_search` parameter that
//! must be tuned offline. Adaptive variants (arXiv:2505.15636, Ada-ef 2512.06636)
//! use per-query scalar distance thresholds. This crate goes one step further:
//! it computes the **Shannon entropy of the candidate-heap distance distribution**
//! at each traversal step and uses that live entropy signal to control beam width.
//!
//! **Hypothesised entropy semantics:**
//! - High H → flat distribution → query sits at the boundary of multiple clusters
//!   → ambiguous → expand more neighbours
//! - Low H → peaked distribution → neighbours cluster tightly → converged → stop
//!   early or reduce branching
//!
//! **Measured PoC outcome (negative result):** on the benchmark data the softmin
//! entropy of already-retrieved neighbour distances tracks local density, not
//! routing ambiguity, and has the wrong sign (hard queries show *lower* entropy
//! than easy ones). [`EntropyScaledEf`] saturates to the same `ef_actual` for
//! every query, so its recall gain over [`FixedEfSearch`] is entirely explained
//! by a larger ef budget. See ADR-303 and the research README for details.
//!
//! ## Variants
//!
//! | Variant | Strategy | Description |
//! |---------|----------|-------------|
//! | [`FixedEfSearch`] | Baseline | Fixed `ef` budget, stop when heap full |
//! | [`EntropyThresholdBeam`] | Reactive | Stop traversal when heap entropy < H_min |
//! | [`EntropyScaledEf`] | Predictive | Sample entropy at depth-5, scale ef for rest |
//!
//! ## RuVector fit
//!
//! Agent memory graphs frequently contain semantically ambiguous queries where
//! fixed-ef over-searches easy queries and under-searches hard ones. Entropy
//! provides a zero-calibration, runtime signal derived from the search frontier
//! itself.

pub mod dataset;
pub mod graph;
pub mod metrics;
pub mod search;

pub use graph::{FlatGraph, GraphConfig};
pub use search::{EntropyScaledEf, EntropyThresholdBeam, FixedEfSearch, Hit, Searcher};

// ─── shared types ────────────────────────────────────────────────────────────

/// Recall@k: fraction of true top-k found in approximate results.
///
/// The denominator is `min(k, ground_truth.len())` only. A searcher that
/// returns fewer than `k` results (e.g. via early termination) is penalised,
/// not rewarded: missing results count as misses.
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

/// Softmin-temperature Shannon entropy of a distance distribution.
///
/// Converts distances to a probability mass: `p_i = exp(-d_i / T) / Z`
/// where T is a temperature parameter. Returns H = -Σ p_i ln(p_i).
///
/// A temperature of ~0.1 makes the distribution sensitive to small distance
/// differences; temperature 1.0 gives a softer response.
pub fn heap_entropy(distances: &[f32], temperature: f32) -> f32 {
    if distances.is_empty() {
        return 0.0;
    }
    let t = temperature.max(1e-9);
    // Subtract max for numerical stability (log-sum-exp trick in forward direction)
    let max_neg = distances
        .iter()
        .fold(f32::NEG_INFINITY, |a, &d| a.max(-d / t));
    let scaled: Vec<f32> = distances
        .iter()
        .map(|&d| (-d / t - max_neg).exp())
        .collect();
    let z: f32 = scaled.iter().sum();
    if z <= 0.0 {
        return 0.0;
    }
    scaled
        .iter()
        .map(|&s| {
            let p = s / z;
            if p > 1e-12 {
                -p * p.ln()
            } else {
                0.0
            }
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropy_of_uniform_is_ln_n() {
        // Uniform distribution: H = ln(n)
        let distances = vec![1.0_f32; 8];
        let h = heap_entropy(&distances, 1.0);
        let expected = (8_f32).ln();
        assert!(
            (h - expected).abs() < 0.01,
            "uniform entropy {h:.4} expected {expected:.4}"
        );
    }

    #[test]
    fn entropy_of_singleton_is_zero() {
        let distances = vec![0.5_f32];
        let h = heap_entropy(&distances, 1.0);
        assert!(h.abs() < 1e-6, "singleton entropy should be 0, got {h}");
    }

    #[test]
    fn entropy_peaks_at_uniformity() {
        // A peaked distribution has lower entropy than uniform
        let uniform = vec![1.0_f32, 1.0, 1.0, 1.0];
        let peaked = vec![0.01_f32, 0.01, 0.01, 10.0]; // one very close neighbour
        let h_uniform = heap_entropy(&uniform, 1.0);
        let h_peaked = heap_entropy(&peaked, 1.0);
        assert!(
            h_uniform > h_peaked,
            "uniform H={h_uniform:.3} should exceed peaked H={h_peaked:.3}"
        );
    }

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
}
