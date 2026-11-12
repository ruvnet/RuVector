//! Flat k-NN proximity graph — equivalent to HNSW layer-0.

use crate::dataset::l2sq;

/// Configuration for graph construction.
#[derive(Clone, Debug)]
pub struct GraphConfig {
    /// Number of neighbours per node in the built graph.
    pub k_neighbours: usize,
}

impl Default for GraphConfig {
    fn default() -> Self {
        GraphConfig { k_neighbours: 16 }
    }
}

/// A single-layer k-NN proximity graph over f32 vectors.
///
/// Construction is O(n² · dim) — suitable for PoC datasets up to ~50 k vectors.
/// Larger datasets would use HNSW multi-layer construction; the entropy logic
/// applies unchanged to the per-layer graph walk.
pub struct FlatGraph {
    pub vectors: Vec<Vec<f32>>,
    /// adjacency[i] = sorted list of (dist², neighbour_id) for node i
    pub adjacency: Vec<Vec<(f32, usize)>>,
    pub config: GraphConfig,
}

impl FlatGraph {
    /// Build the graph from a vector corpus.
    pub fn build(vectors: Vec<Vec<f32>>, config: GraphConfig) -> Self {
        let n = vectors.len();
        let k = config.k_neighbours.min(n.saturating_sub(1));

        let mut adjacency: Vec<Vec<(f32, usize)>> = vec![Vec::with_capacity(k); n];

        for i in 0..n {
            let mut dists: Vec<(f32, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (l2sq(&vectors[i], &vectors[j]), j))
                .collect();
            dists.sort_unstable_by(|a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            dists.truncate(k);
            adjacency[i] = dists;
        }

        FlatGraph {
            vectors,
            adjacency,
            config,
        }
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.vectors.first().map(|v| v.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::random_unit_vectors;

    #[test]
    fn graph_has_k_neighbours() {
        let vecs = random_unit_vectors(30, 8, 7);
        let cfg = GraphConfig { k_neighbours: 6 };
        let g = FlatGraph::build(vecs, cfg);
        for adj in &g.adjacency {
            assert_eq!(adj.len(), 6);
        }
    }

    #[test]
    fn graph_neighbourhood_is_mostly_reciprocal_on_uniform() {
        // k-NN edges are not guaranteed to be symmetric, but on uniform data
        // a majority of directed edges should be reciprocated. Assert that.
        let vecs = random_unit_vectors(20, 4, 11);
        let cfg = GraphConfig { k_neighbours: 4 };
        let g = FlatGraph::build(vecs, cfg);
        let mut total = 0usize;
        let mut reciprocal = 0usize;
        for (i, adj) in g.adjacency.iter().enumerate() {
            for &(_, j) in adj {
                total += 1;
                if g.adjacency[j].iter().any(|&(_, k)| k == i) {
                    reciprocal += 1;
                }
            }
        }
        assert!(total > 0);
        let frac = reciprocal as f64 / total as f64;
        assert!(
            frac >= 0.5,
            "expected >= 50% reciprocated k-NN edges on uniform data, got {frac:.3}"
        );
    }
}
