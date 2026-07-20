//! Greedy kNN graph operating on the first `d_fast` dimensions.
//!
//! Single-layer graph: each inserted vector connects to its M nearest
//! neighbours among previously inserted vectors, using D_FAST cosine.
//! Beam search then navigates this graph for approximate retrieval,
//! followed by full-dimensional exact rerank.

use std::collections::HashSet;

use crate::{dot, SearchResult};

/// Greedy single-layer kNN graph index.
///
/// Build: O(N² · D_FAST) — practical for N ≤ 10 K.
/// Search: O(ef · M · D_FAST) + O(k_over · D_FULL).
pub struct GreedyGraph {
    /// Full-dimensional vectors (used for reranking).
    vectors: Vec<Vec<f32>>,
    /// Adjacency list: adj[i] = neighbours of node i.
    adj: Vec<Vec<u32>>,
    /// Number of fast dimensions used for graph navigation.
    pub d_fast: usize,
    /// Number of full dimensions stored per vector.
    pub d_full: usize,
    /// Max neighbours per node.
    m: usize,
}

impl GreedyGraph {
    /// Create an empty graph.
    pub fn new(d_fast: usize, d_full: usize, m: usize) -> Self {
        assert!(d_fast <= d_full, "d_fast must not exceed d_full");
        GreedyGraph {
            vectors: Vec::new(),
            adj: Vec::new(),
            d_fast,
            d_full,
            m,
        }
    }

    /// Cosine similarity using only the first `d_fast` dimensions.
    #[inline]
    fn fast_score(&self, id: usize, q: &[f32]) -> f32 {
        dot(&self.vectors[id][..self.d_fast], &q[..self.d_fast])
    }

    /// Cosine similarity using all `d_full` dimensions.
    #[inline]
    fn full_score(&self, id: usize, q: &[f32]) -> f32 {
        dot(&self.vectors[id], &q[..self.d_full])
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Stage-1 insert: just store the vector without building edges.
    ///
    /// Call [`GreedyGraph::build_edges`] after all vectors are inserted to
    /// compute the full symmetric kNN adjacency in D_FAST space.
    pub fn insert(&mut self, vector: &[f32]) -> u32 {
        assert_eq!(vector.len(), self.d_full, "dimension mismatch");
        let id = self.vectors.len() as u32;
        self.vectors.push(vector.to_vec());
        self.adj.push(Vec::new());
        id
    }

    /// Stage-2: build the symmetric kNN graph in D_FAST space.
    ///
    /// For every node `i`, connects it to its M nearest neighbours (by D_FAST
    /// cosine) across the full dataset.  O(N² · D_FAST) — suited for N ≤ 20 K.
    pub fn build_edges(&mut self) {
        let n = self.vectors.len();
        // Clear any existing edges.
        for adj in &mut self.adj {
            adj.clear();
        }
        for i in 0..n {
            // Score all other nodes.
            let mut scored: Vec<(u32, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j as u32, self.fast_score(j, &self.vectors[i].clone())))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            // Take top-M as outgoing edges for node i.
            let take = self.m.min(scored.len());
            for (nb_id, _) in &scored[..take] {
                if !self.adj[i].contains(nb_id) {
                    self.adj[i].push(*nb_id);
                }
                // Symmetric: also add i as a neighbour of nb (cap at m).
                let nb = *nb_id as usize;
                let i_id = i as u32;
                if self.adj[nb].len() < self.m && !self.adj[nb].contains(&i_id) {
                    self.adj[nb].push(i_id);
                }
            }
        }
    }

    /// Beam search in D_FAST space (EFANNA-style greedy beam).
    ///
    /// Exploration set `W` drives expansion; candidate set `C` accumulates the
    /// top-ef results.  Prune when the best unexplored node scores below the
    /// worst entry in C (cosine similarity: higher = better).
    ///
    /// Returns up to `k_over` candidates sorted by fast-dim score (desc).
    pub fn beam_fast(&self, query: &[f32], k_over: usize, ef: usize) -> Vec<(u32, f32)> {
        let n = self.vectors.len();
        if n == 0 {
            return Vec::new();
        }

        let mut visited: HashSet<u32> = HashSet::new();

        // Use node 0 as entry point and seed both sets.
        let entry = 0u32;
        let entry_score = self.fast_score(entry as usize, query);
        visited.insert(entry);

        // W: exploration queue — max-heap by score (best first).
        // We simulate with a Vec; sort descending, pop from end = highest.
        let mut w: Vec<(f32, u32)> = vec![(entry_score, entry)];

        // C: candidate result set, size capped at ef.
        // We keep it sorted descending; the last element is the worst.
        let mut c: Vec<(f32, u32)> = vec![(entry_score, entry)];

        while !w.is_empty() {
            // Extract best from W.
            w.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let (best_w_score, best_w_id) = w.pop().unwrap();

            // Prune: if the best unexplored is worse than the worst in C, stop.
            let worst_c = c.last().map(|(s, _)| *s).unwrap_or(f32::NEG_INFINITY);
            if c.len() >= ef && best_w_score < worst_c {
                break;
            }

            // Expand neighbours.
            for &nb in &self.adj[best_w_id as usize] {
                if visited.insert(nb) {
                    let s = self.fast_score(nb as usize, query);
                    let worst_c = c.last().map(|(sc, _)| *sc).unwrap_or(f32::NEG_INFINITY);
                    if c.len() < ef || s > worst_c {
                        w.push((s, nb));
                        c.push((s, nb));
                        // Keep C sorted descending and trimmed to ef.
                        c.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                        if c.len() > ef {
                            c.pop(); // remove worst
                        }
                    }
                }
            }
            let _ = best_w_score; // used above
        }

        c.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        c.truncate(k_over);
        c.into_iter().map(|(s, id)| (id, s)).collect()
    }

    /// Rerank `candidates` using full-dimensional cosine, return top-k.
    pub fn rerank(&self, candidates: &[(u32, f32)], query: &[f32], k: usize) -> Vec<SearchResult> {
        let mut scored: Vec<SearchResult> = candidates
            .iter()
            .map(|&(id, _)| SearchResult {
                id,
                score: self.full_score(id as usize, query),
            })
            .collect();
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(k);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize;

    #[test]
    fn test_graph_insert_and_search() {
        let (d_fast, d_full, m) = (4, 8, 4);
        let mut g = GreedyGraph::new(d_fast, d_full, m);
        for i in 0..10u32 {
            let mut v: Vec<f32> = (0..d_full)
                .map(|j| if j == i as usize % d_full { 1.0 } else { 0.0 })
                .collect();
            normalize(&mut v);
            g.insert(&v);
        }
        g.build_edges();
        assert_eq!(g.len(), 10);

        // Query aligned with vector 0.
        let q: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let cands = g.beam_fast(&q, 5, 10);
        assert!(!cands.is_empty());
        // Rerank should return id=0 first.
        let results = g.rerank(&cands, &q, 1);
        assert_eq!(results[0].id, 0);
    }
}
