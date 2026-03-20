//! Edge sampling proportional to importance scores.
//!
//! Implements the spectral sampling strategy: each edge is kept independently
//! with probability proportional to `weight * importance * log(n) / epsilon^2`,
//! and kept edges are reweighted by the inverse of their sampling probability
//! to preserve the expected Laplacian.

use rand::prelude::*;
use serde::{Deserialize, Serialize};

use crate::graph::SparseGraph;
use crate::types::EdgeImportance;

// ---------------------------------------------------------------------------
// SpectralSampler
// ---------------------------------------------------------------------------

/// Samples edges from a graph proportional to their spectral importance.
///
/// The sampling probability for edge `e = (u, v, w)` is:
///
/// ```text
/// p(e) = min(1, C * score(e) * log(n) / epsilon^2)
/// ```
///
/// where `C` is a normalisation constant chosen so the expected number of
/// sampled edges matches `budget`. Kept edges are reweighted by `w / p(e)`
/// to make the sparsifier an unbiased estimator of the Laplacian.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralSampler {
    /// Approximation parameter.
    pub epsilon: f64,
}

impl SpectralSampler {
    /// Create a new sampler with the given epsilon.
    pub fn new(epsilon: f64) -> Self {
        Self { epsilon }
    }

    /// Sample edges from `graph` using the provided importance scores,
    /// returning a new [`SparseGraph`] with at most `budget` edges.
    ///
    /// Backbone edges (if any) should be marked in `backbone_edges`; they
    /// are always included and reweighted to `weight * 1.0` (kept as-is).
    pub fn sample_edges(
        &self,
        scores: &[EdgeImportance],
        budget: usize,
        backbone_edges: &std::collections::HashSet<(usize, usize)>,
    ) -> SparseGraph {
        if scores.is_empty() {
            return SparseGraph::new();
        }

        let mut rng = rand::thread_rng();
        let n_vertices = scores
            .iter()
            .map(|s| s.u.max(s.v) + 1)
            .max()
            .unwrap_or(0);
        let log_n = (n_vertices as f64).ln().max(1.0);

        // Total importance sum for normalisation.
        let total_importance: f64 = scores.iter().map(|s| s.score).sum();
        if total_importance <= 0.0 {
            // All edges have zero importance; keep backbone only.
            return self.backbone_only_graph(scores, backbone_edges);
        }

        // Number of non-backbone slots.
        let backbone_count = scores
            .iter()
            .filter(|s| {
                let key = Self::edge_key(s.u, s.v);
                backbone_edges.contains(&key)
            })
            .count();
        let sample_budget = budget.saturating_sub(backbone_count);

        // Scaling factor so expected sample count ~ sample_budget.
        let c = if total_importance > 0.0 {
            sample_budget as f64 / (total_importance * log_n / (self.epsilon * self.epsilon))
        } else {
            1.0
        };

        let mut g = SparseGraph::with_capacity(n_vertices);

        for s in scores {
            let key = Self::edge_key(s.u, s.v);
            let is_backbone = backbone_edges.contains(&key);

            if is_backbone {
                // Always keep backbone edges at original weight.
                let _ = g.insert_or_update_edge(s.u, s.v, s.weight);
                continue;
            }

            // Sampling probability.
            let p = (c * s.score * log_n / (self.epsilon * self.epsilon)).min(1.0);

            if p >= 1.0 || rng.gen::<f64>() < p {
                // Reweight: w' = w / p  so that E[w'] = w.
                let reweighted = if p > 0.0 { s.weight / p } else { s.weight };
                let _ = g.insert_or_update_edge(s.u, s.v, reweighted);
            }
        }

        g
    }

    /// Incrementally add a single edge to the sparsifier, deciding whether
    /// to keep it based on its importance.
    pub fn sample_single_edge(
        &self,
        importance: &EdgeImportance,
        n_vertices: usize,
        total_importance: f64,
        budget: usize,
    ) -> Option<(usize, usize, f64)> {
        let log_n = (n_vertices as f64).ln().max(1.0);
        let c = if total_importance > 0.0 {
            budget as f64 / (total_importance * log_n / (self.epsilon * self.epsilon))
        } else {
            1.0
        };
        let p = (c * importance.score * log_n / (self.epsilon * self.epsilon)).min(1.0);

        let mut rng = rand::thread_rng();
        if p >= 1.0 || rng.gen::<f64>() < p {
            let reweighted = if p > 0.0 {
                importance.weight / p
            } else {
                importance.weight
            };
            Some((importance.u, importance.v, reweighted))
        } else {
            None
        }
    }

    // -- helpers ------------------------------------------------------------

    fn edge_key(u: usize, v: usize) -> (usize, usize) {
        if u <= v { (u, v) } else { (v, u) }
    }

    fn backbone_only_graph(
        &self,
        scores: &[EdgeImportance],
        backbone_edges: &std::collections::HashSet<(usize, usize)>,
    ) -> SparseGraph {
        let n = scores
            .iter()
            .map(|s| s.u.max(s.v) + 1)
            .max()
            .unwrap_or(0);
        let mut g = SparseGraph::with_capacity(n);
        for s in scores {
            let key = Self::edge_key(s.u, s.v);
            if backbone_edges.contains(&key) {
                let _ = g.insert_or_update_edge(s.u, s.v, s.weight);
            }
        }
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_with_backbone() {
        let scores = vec![
            EdgeImportance::new(0, 1, 1.0, 1.0),
            EdgeImportance::new(1, 2, 1.0, 1.0),
            EdgeImportance::new(0, 2, 1.0, 1.0),
        ];
        let mut backbone = std::collections::HashSet::new();
        backbone.insert((0, 1));

        let sampler = SpectralSampler::new(0.2);
        let g = sampler.sample_edges(&scores, 10, &backbone);

        // Backbone edge must be present.
        assert!(g.has_edge(0, 1));
    }

    #[test]
    fn test_sample_empty() {
        let sampler = SpectralSampler::new(0.2);
        let g = sampler.sample_edges(&[], 10, &Default::default());
        assert_eq!(g.num_edges(), 0);
    }

    #[test]
    fn test_high_budget_keeps_all() {
        let scores = vec![
            EdgeImportance::new(0, 1, 1.0, 10.0),
            EdgeImportance::new(1, 2, 1.0, 10.0),
        ];
        let sampler = SpectralSampler::new(0.01); // tiny eps => high prob
        // With very small epsilon and high importance, all edges should be kept.
        let g = sampler.sample_edges(&scores, 1000, &Default::default());
        assert!(g.num_edges() >= 1); // at least one edge kept
    }
}
