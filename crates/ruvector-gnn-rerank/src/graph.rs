//! Candidate k-NN subgraph for GNN score diffusion.
//!
//! Given a small candidate set (typically 50–200 vectors returned by an
//! approximate first-stage retriever), this module builds a k-nearest-neighbour
//! graph over the candidates using cosine similarity between their full-precision
//! vectors.  The resulting graph is the propagation medium for score diffusion in
//! `GnnDiffusionReranker` and `GnnMincutReranker`.
//!
//! **Complexity:** O(n² × dim) — acceptable for n ≤ 200 and dim ≤ 2048.
//! At n=80, dim=128: ~820K multiply-adds, sub-millisecond on modern hardware.

use crate::reranker::Candidate;

/// k-NN graph over a set of ANN candidates.
///
/// `edges[i]` is a sorted list of `(neighbour_index, cosine_similarity)` for
/// candidate `i`, ordered by descending similarity.
pub struct CandidateGraph {
    pub edges: Vec<Vec<(usize, f32)>>,
}

impl CandidateGraph {
    /// Build a k-NN graph over `candidates` using cosine similarity.
    ///
    /// `k_graph` is the maximum degree per node.  Edges are undirected but
    /// stored as a directed adjacency list (each endpoint stores its own
    /// neighbourhood independently).
    pub fn build(candidates: &[Candidate], k_graph: usize) -> Self {
        let n = candidates.len();
        let k = k_graph.min(n.saturating_sub(1));

        // Pre-compute L2 norms to avoid recomputing in the inner loop.
        let norms: Vec<f32> = candidates.iter().map(|c| l2_norm(&c.vector)).collect();

        // Cosine similarity is symmetric: sim(i,j) == sim(j,i). Compute each
        // pair once (upper triangle) and push it into both neighbour lists,
        // halving the dot-product work vs. the naive O(n²) double computation.
        let mut sims: Vec<Vec<(usize, f32)>> =
            vec![Vec::with_capacity(n.saturating_sub(1)); n];
        for i in 0..n {
            let (vi, ni) = (&candidates[i].vector, norms[i]);
            for j in (i + 1)..n {
                let dot: f32 = vi
                    .iter()
                    .zip(candidates[j].vector.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let denom = ni * norms[j];
                let sim = if denom < 1e-9 { 0.0 } else { dot / denom };
                sims[i].push((j, sim));
                sims[j].push((i, sim));
            }
        }

        // Sort descending by similarity; keep top-k per row.
        let edges = sims
            .into_iter()
            .map(|mut row| {
                row.sort_unstable_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                row.truncate(k);
                row
            })
            .collect();

        Self { edges }
    }
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reranker::Candidate;

    fn unit_candidate(id: u32, v: Vec<f32>) -> Candidate {
        Candidate {
            id,
            vector: v,
            noisy_score: 0.5,
        }
    }

    #[test]
    fn self_not_in_neighbours() {
        let cands = vec![
            unit_candidate(0, vec![1.0, 0.0]),
            unit_candidate(1, vec![0.0, 1.0]),
            unit_candidate(2, vec![-1.0, 0.0]),
        ];
        let g = CandidateGraph::build(&cands, 2);
        for (i, nbrs) in g.edges.iter().enumerate() {
            assert!(!nbrs.iter().any(|(j, _)| *j == i), "node {i} found itself");
        }
    }

    #[test]
    fn degree_does_not_exceed_k_graph() {
        let cands: Vec<Candidate> = (0..15)
            .map(|i| unit_candidate(i, vec![(i as f32).sin(), (i as f32).cos()]))
            .collect();
        let g = CandidateGraph::build(&cands, 4);
        for nbrs in &g.edges {
            assert!(nbrs.len() <= 4);
        }
    }

    #[test]
    fn two_nodes_are_each_others_only_neighbour() {
        let cands = vec![
            unit_candidate(0, vec![1.0, 0.0]),
            unit_candidate(1, vec![0.5, 0.5]),
        ];
        let g = CandidateGraph::build(&cands, 5);
        assert_eq!(g.edges[0].len(), 1);
        assert_eq!(g.edges[0][0].0, 1);
        assert_eq!(g.edges[1].len(), 1);
        assert_eq!(g.edges[1][0].0, 0);
    }
}
