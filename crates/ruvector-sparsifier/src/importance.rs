//! Edge importance scoring via random walks.
//!
//! Estimates effective resistance using short random walks (a practical
//! approximation to the Johnson-Lindenstrauss projection approach). The
//! importance score `w(e) * R_eff(e)` determines sampling probability.

use rand::prelude::*;
use serde::{Deserialize, Serialize};

use crate::graph::SparseGraph;
use crate::traits::ImportanceScorer;
use crate::types::EdgeImportance;

// ---------------------------------------------------------------------------
// EffectiveResistanceEstimator
// ---------------------------------------------------------------------------

/// Estimates effective resistance between two vertices via random walks.
///
/// Uses the commute-time identity: `R_eff(u,v) = commute_time(u,v) / (2m)`
/// where `m` is the total edge weight. The commute time is estimated by
/// running random walks from `u` until they hit `v` and back.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectiveResistanceEstimator {
    /// Maximum walk length before giving up.
    pub max_walk_length: usize,
    /// Number of walks to average over.
    pub num_walks: usize,
}

impl Default for EffectiveResistanceEstimator {
    fn default() -> Self {
        Self {
            max_walk_length: 100,
            num_walks: 10,
        }
    }
}

impl EffectiveResistanceEstimator {
    /// Create a new estimator with the given parameters.
    pub fn new(max_walk_length: usize, num_walks: usize) -> Self {
        Self {
            max_walk_length,
            num_walks,
        }
    }

    /// Estimate the effective resistance between `u` and `v` in `graph`.
    ///
    /// Returns a value in `[0, +inf)`. For disconnected pairs the
    /// estimate may be very large (capped at `max_walk_length / total_weight`).
    pub fn estimate(&self, graph: &SparseGraph, u: usize, v: usize) -> f64 {
        if u == v {
            return 0.0;
        }
        let total_w = graph.total_weight();
        if total_w <= 0.0 {
            return f64::MAX;
        }
        if graph.degree(u) == 0 || graph.degree(v) == 0 {
            return f64::MAX;
        }

        let mut rng = rand::thread_rng();
        let mut total_steps = 0u64;

        for _ in 0..self.num_walks {
            // Walk from u to v.
            total_steps += self.walk_to_target(graph, u, v, &mut rng) as u64;
            // Walk from v to u.
            total_steps += self.walk_to_target(graph, v, u, &mut rng) as u64;
        }

        let avg_commute = total_steps as f64 / self.num_walks as f64;
        // R_eff ~ commute_time / (2 * total_weight)
        avg_commute / (2.0 * total_w)
    }

    /// Random walk from `start` to `target`, returning the number of steps.
    fn walk_to_target<R: Rng>(
        &self,
        graph: &SparseGraph,
        start: usize,
        target: usize,
        rng: &mut R,
    ) -> usize {
        let mut current = start;
        for step in 1..=self.max_walk_length {
            current = self.random_neighbor(graph, current, rng);
            if current == target {
                return step;
            }
        }
        self.max_walk_length
    }

    /// Pick a random neighbour of `u` with probability proportional to weight.
    fn random_neighbor<R: Rng>(&self, graph: &SparseGraph, u: usize, rng: &mut R) -> usize {
        let w_deg = graph.weighted_degree(u);
        if w_deg <= 0.0 {
            return u; // isolated vertex stays put
        }
        let threshold = rng.gen::<f64>() * w_deg;
        let mut cumulative = 0.0;
        for (v, w) in graph.neighbors(u) {
            cumulative += w;
            if cumulative >= threshold {
                return v;
            }
        }
        // Fallback (numerical edge case).
        u
    }
}

// ---------------------------------------------------------------------------
// LocalImportanceScorer
// ---------------------------------------------------------------------------

/// Scores edge importance using localized random walks.
///
/// For each edge `(u, v, w)`, the score is `w * R_eff_estimate(u, v)`.
/// High-importance edges (bridges, cut edges) get high scores and are
/// more likely to be kept in the sparsifier.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LocalImportanceScorer {
    /// The underlying resistance estimator.
    pub estimator: EffectiveResistanceEstimator,
}

impl LocalImportanceScorer {
    /// Create a scorer with custom walk parameters.
    pub fn new(walk_length: usize, num_walks: usize) -> Self {
        Self {
            estimator: EffectiveResistanceEstimator::new(walk_length, num_walks),
        }
    }

    /// Compute the importance score for a single edge.
    pub fn importance_score(&self, graph: &SparseGraph, u: usize, v: usize, weight: f64) -> f64 {
        let r_eff = self.estimator.estimate(graph, u, v);
        weight * r_eff
    }
}

impl ImportanceScorer for LocalImportanceScorer {
    fn score(&self, graph: &SparseGraph, u: usize, v: usize, weight: f64) -> EdgeImportance {
        let r_eff = self.estimator.estimate(graph, u, v);
        EdgeImportance::new(u, v, weight, r_eff)
    }

    fn score_all(&self, graph: &SparseGraph) -> Vec<EdgeImportance> {
        // Collect edges first, then parallel-score them for large graphs.
        let edges: Vec<(usize, usize, f64)> = graph.edges().collect();

        if edges.len() > 100 {
            use rayon::prelude::*;
            edges
                .par_iter()
                .map(|&(u, v, w)| {
                    let r_eff = self.estimator.estimate(graph, u, v);
                    EdgeImportance::new(u, v, w, r_eff)
                })
                .collect()
        } else {
            edges
                .iter()
                .map(|&(u, v, w)| self.score(graph, u, v, w))
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_loop_resistance() {
        let g = SparseGraph::from_edges(&[(0, 1, 1.0)]);
        let est = EffectiveResistanceEstimator::new(50, 5);
        let r = est.estimate(&g, 0, 0);
        assert!((r - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_resistance_positive() {
        let g = SparseGraph::from_edges(&[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)]);
        let est = EffectiveResistanceEstimator::new(200, 20);
        let r = est.estimate(&g, 0, 2);
        assert!(r > 0.0);
    }

    #[test]
    fn test_scorer_all() {
        let g = SparseGraph::from_edges(&[(0, 1, 1.0), (1, 2, 2.0)]);
        let scorer = LocalImportanceScorer::new(50, 5);
        let scores = scorer.score_all(&g);
        assert_eq!(scores.len(), 2);
        for s in &scores {
            assert!(s.score > 0.0);
        }
    }
}
