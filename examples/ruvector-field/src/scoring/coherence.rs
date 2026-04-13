//! Coherence via a Laplacian-based effective-resistance proxy — spec 8.2.
//!
//! The real formula is `coherence = 1 / (1 + avg_effective_resistance)` over
//! a solver call. Since this crate is std-only we approximate effective
//! resistance with a local reciprocal-sum formulation:
//!
//! ```text
//! eff_resistance ≈ 1 / sum(w_i)     // parallel conductance model
//! coherence      ≈ 1 / (1 + eff_resistance)
//! ```
//!
//! where `w_i` are the positive cosine similarities of a node's k nearest
//! same-shell neighbors (soft-thresholded to `max(0, cos)`), scaled by an
//! edge-support term. The formula is diagonal-dominance-friendly and
//! monotone: adding a stronger neighbor can only increase coherence.
//!
//! # `solver` feature
//!
//! Under `--features solver` the [`local_coherence`] helper routes through a
//! real local Laplacian effective-resistance estimate — see
//! [`solver_backend::NeumannSolverBackend`]. The formula is:
//!
//! ```text
//! L        = D - W               // star-subgraph Laplacian around center
//! R(c, u)  = (e_c - e_u)^T L^+ (e_c - e_u)
//! coh      = 1 / (1 + mean_u R(c, u))
//! ```
//!
//! Both implementations are bounded in `[0, 1]` and monotone in neighbor
//! weight, so test expectations are consistent across feature configurations.

use crate::model::{Embedding, FieldNode};

#[cfg(feature = "solver")]
#[path = "coherence/solver_backend.rs"]
pub mod solver_backend;

/// Effective-resistance proxy for a single node given its `k` nearest
/// same-shell neighbors.
///
/// # Example
///
/// ```
/// use ruvector_field::scoring::effective_resistance_proxy;
/// let rs = effective_resistance_proxy(&[0.9, 0.8, 0.7]);
/// assert!(rs > 0.0 && rs <= 1.0);
/// let rs_empty = effective_resistance_proxy(&[]);
/// assert_eq!(rs_empty, 1.0);
/// ```
pub fn effective_resistance_proxy(conductances: &[f32]) -> f32 {
    // TODO(solver): replace with a call to ruvector-solver's local
    // effective-resistance routine once this layer graduates to a crate.
    if conductances.is_empty() {
        return 1.0;
    }
    let sum: f32 = conductances.iter().map(|w| w.max(0.0)).sum();
    if sum <= 1e-6 {
        return 1.0;
    }
    // Parallel conductance: r = 1 / sum(w_i). Unclamped so very strong
    // neighborhoods can still drive coherence close to 1.
    (1.0 / sum).max(0.0)
}

/// Compute a single node's local coherence given its same-shell neighbors.
///
/// `neighbors` is a slice of (embedding, support_weight) pairs; support_weight
/// biases the conductance so nodes with stronger `Supports`/`Refines` edges
/// contribute more.
///
/// # Example
///
/// ```
/// use ruvector_field::model::Embedding;
/// use ruvector_field::scoring::local_coherence;
/// let q = Embedding::new(vec![1.0, 0.0, 0.0]);
/// let n = Embedding::new(vec![0.9, 0.1, 0.0]);
/// let coh = local_coherence(&q, &[(&n, 1.0)], 4);
/// assert!(coh > 0.0 && coh <= 1.0);
/// ```
pub fn local_coherence(
    center: &Embedding,
    neighbors: &[(&Embedding, f32)],
    k: usize,
) -> f32 {
    if neighbors.is_empty() {
        return 0.5;
    }
    // Compute cosine similarity to each neighbor, soft-threshold to
    // non-negative values (positive conductance).
    let mut sims: Vec<f32> = neighbors
        .iter()
        .map(|(emb, weight)| {
            let s = center.cosine(emb).max(0.0);
            s * weight.clamp(0.0, 1.0)
        })
        .collect();
    sims.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    sims.truncate(k.max(1));

    #[cfg(feature = "solver")]
    {
        use solver_backend::{NeumannSolverBackend, SolverBackend};
        // Degenerate input: if the aggregated conductance is effectively
        // zero, fall back to the proxy's neutral value. This matches the
        // proxy's sign semantics and keeps downstream thresholds stable
        // across feature configurations.
        let total: f32 = sims.iter().sum();
        if total <= 1e-6 {
            let er = effective_resistance_proxy(&sims);
            return (1.0 / (1.0 + er)).clamp(0.0, 1.0);
        }
        let backend = NeumannSolverBackend::default();
        let er = backend.mean_effective_resistance(&sims);
        return (1.0 / (1.0 + er)).clamp(0.0, 1.0);
    }
    #[cfg(not(feature = "solver"))]
    {
        let er = effective_resistance_proxy(&sims);
        (1.0 / (1.0 + er)).clamp(0.0, 1.0)
    }
}

/// Batch helper: apply `local_coherence` over a set of nodes, using the
/// existing `FieldNode` embeddings.
pub fn coherence_for_node<'a, I>(center: &Embedding, neighbors: I, k: usize) -> f32
where
    I: IntoIterator<Item = (&'a FieldNode, &'a Embedding, f32)>,
{
    let collected: Vec<(&Embedding, f32)> = neighbors.into_iter().map(|(_, e, w)| (e, w)).collect();
    local_coherence(center, &collected, k)
}
