//! kNN graph construction over a vector corpus.
//!
//! Builds a sparse undirected weighted graph where each node connects to its
//! `k` nearest neighbours (by cosine similarity). Edge weight = similarity.
//! Graph is symmetrised: if j ∈ kNN(i) or i ∈ kNN(j), both directed edges are kept.

use crate::distance::cosine_similarity;

/// Sparse adjacency list representation of a kNN graph.
pub struct KnnGraph {
    /// Number of nodes.
    pub n: usize,
    /// Adjacency list: `adj[i]` = list of (neighbour_index, edge_weight).
    pub adj: Vec<Vec<(usize, f32)>>,
    /// Weighted degree: `degree[i]` = sum of all edge weights incident on i.
    pub degree: Vec<f32>,
}

impl KnnGraph {
    /// Build kNN graph with cosine-similarity edge weights.
    pub fn build(vectors: &[Vec<f32>], k: usize) -> Self {
        Self::build_internal(vectors, k, false)
    }

    /// Build kNN graph with **squared** cosine-similarity weights (coherence emphasis:
    /// very similar pairs get disproportionately heavy edges, so the Fiedler cut
    /// prefers to sever weakly-related pairs).
    pub fn build_coherence_weighted(vectors: &[Vec<f32>], k: usize) -> Self {
        Self::build_internal(vectors, k, true)
    }

    fn build_internal(vectors: &[Vec<f32>], k: usize, squared: bool) -> Self {
        let n = vectors.len();
        let k = k.min(n.saturating_sub(1));
        let mut sym: Vec<Vec<(usize, f32)>> = vec![vec![]; n];

        for i in 0..n {
            // Compute similarities to all other nodes.
            let mut sims: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let s = cosine_similarity(&vectors[i], &vectors[j]);
                    let w = if squared { s * s } else { s };
                    (j, w.max(0.0)) // clamp negative similarities to 0
                })
                .collect();

            // Keep top-k by weight (descending).
            sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            sims.truncate(k);

            // Add directed edges i→j and j→i (symmetrise).
            for (j, w) in sims {
                sym[i].push((j, w));
                sym[j].push((i, w));
            }
        }

        // Deduplicate: keep the maximum weight for duplicate (i,j) edges.
        for adj in sym.iter_mut() {
            adj.sort_by_key(|&(j, _)| j);
            adj.dedup_by(|later, earlier| {
                if later.0 == earlier.0 {
                    earlier.1 = earlier.1.max(later.1);
                    true
                } else {
                    false
                }
            });
        }

        let degree: Vec<f32> = sym
            .iter()
            .map(|edges| {
                let s: f32 = edges.iter().map(|(_, w)| w).sum();
                s.max(1e-8) // avoid zero degree
            })
            .collect();

        KnnGraph {
            n,
            adj: sym,
            degree,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(d: usize, i: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; d];
        if i < d {
            v[i] = 1.0;
        }
        v
    }

    #[test]
    fn graph_is_symmetric() {
        let vecs: Vec<Vec<f32>> = (0..6).map(|i| unit(6, i)).collect();
        let g = KnnGraph::build(&vecs, 2);
        for i in 0..g.n {
            for (j, w) in &g.adj[i] {
                let j_has_i = g.adj[*j].iter().any(|(nb, _)| *nb == i);
                assert!(j_has_i, "edge {i}->{j} not symmetric (w={w})");
            }
        }
    }

    #[test]
    fn degree_is_positive() {
        let vecs: Vec<Vec<f32>> = (0..4)
            .map(|i| {
                let mut v = vec![0.1f32; 4];
                v[i] = 1.0;
                v
            })
            .collect();
        let g = KnnGraph::build(&vecs, 2);
        for d in &g.degree {
            assert!(*d > 0.0, "degree must be positive");
        }
    }
}
