//! Navigable Small-World (NSW) graph for ANN search.
//!
//! A flat, single-layer NSW where each node stores its M nearest
//! neighbors found at insertion time. Beam search with a configurable
//! `ef` beam width is the core operation.
//!
//! This is an intentionally minimal implementation: no SIMD, no
//! parallelism, no external dependencies — just the algorithmic core
//! needed to study ef-adaptation in isolation.

/// A node in the NSW graph.
struct Node {
    vector: Vec<f32>,
    /// Indices of neighboring nodes.
    neighbors: Vec<usize>,
}

/// Navigable Small-World graph.
pub struct NswGraph {
    nodes: Vec<Node>,
    /// Max number of neighbors per node.
    m: usize,
    /// ef used during construction (controls index quality).
    ef_construct: usize,
}

impl NswGraph {
    pub fn new(m: usize, ef_construct: usize) -> Self {
        Self {
            nodes: Vec::new(),
            m,
            ef_construct,
        }
    }

    /// Insert a vector into the graph.
    pub fn insert(&mut self, vector: Vec<f32>) {
        let n = self.nodes.len();
        if n == 0 {
            self.nodes.push(Node {
                vector,
                neighbors: Vec::new(),
            });
            return;
        }

        // Find M nearest existing nodes via beam search.
        let ef = self.ef_construct.max(self.m);
        let candidates = self.beam_search_raw(&vector, self.m, ef);
        let neighbors: Vec<usize> = candidates
            .iter()
            .take(self.m)
            .map(|&(_, idx)| idx)
            .collect();

        // Bidirectional connections (with degree cap).
        let new_idx = n;
        for &nb_idx in &neighbors {
            if self.nodes[nb_idx].neighbors.len() < self.m * 2 {
                self.nodes[nb_idx].neighbors.push(new_idx);
            }
        }
        self.nodes.push(Node { vector, neighbors });
    }

    /// k-NN beam search with the given ef (beam width).
    ///
    /// Returns neighbor indices sorted by ascending squared distance.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(usize, f32)> {
        let ef = ef.max(k);
        let mut results = self.beam_search_raw(query, k, ef);
        results.truncate(k);
        results.iter().map(|&(d, i)| (i, d)).collect()
    }

    /// Number of nodes in the graph.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// True when graph is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Estimated memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.nodes
            .iter()
            .map(|n| n.vector.len() * 4 + n.neighbors.len() * 8 + 32)
            .sum::<usize>()
            + std::mem::size_of::<Self>()
    }

    // ── internal ──────────────────────────────────────────────────────────────

    /// Core beam search returning up to `k` results sorted by ascending dist².
    fn beam_search_raw(&self, query: &[f32], k: usize, ef: usize) -> Vec<(f32, usize)> {
        if self.nodes.is_empty() {
            return Vec::new();
        }
        let n = self.nodes.len();
        let mut visited = vec![false; n];

        let entry_dist = sq_dist(&self.nodes[0].vector, query);
        visited[0] = true;

        // candidates: (dist, idx) – we pull the closest each iteration.
        let mut cands: Vec<(f32, usize)> = vec![(entry_dist, 0)];
        // result set bounded to `ef`.
        let mut results: Vec<(f32, usize)> = vec![(entry_dist, 0)];

        loop {
            // Find closest unprocessed candidate (linear scan, ef is small).
            let best_pos = cands
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.0.partial_cmp(&b.0).unwrap())
                .map(|(i, _)| i);
            let (best_dist, best_idx) = match best_pos {
                Some(p) => cands.swap_remove(p),
                None => break,
            };

            // Termination: if the best candidate is further than the worst
            // result and we already have ef results, we're done.
            let worst_result = results
                .iter()
                .map(|r| r.0)
                .fold(f32::NEG_INFINITY, f32::max);
            if results.len() >= ef && best_dist > worst_result {
                break;
            }

            // Expand this node's neighbors.
            for &nb in &self.nodes[best_idx].neighbors {
                if visited[nb] {
                    continue;
                }
                visited[nb] = true;
                let d = sq_dist(&self.nodes[nb].vector, query);
                let wr = results
                    .iter()
                    .map(|r| r.0)
                    .fold(f32::NEG_INFINITY, f32::max);
                if results.len() < ef || d < wr {
                    cands.push((d, nb));
                    results.push((d, nb));
                    // Evict furthest if over budget.
                    if results.len() > ef {
                        let far = results
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.0.partial_cmp(&b.0).unwrap())
                            .map(|(i, _)| i)
                            .unwrap();
                        results.swap_remove(far);
                    }
                }
            }
        }

        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        results.truncate(k);
        results
    }
}

/// Squared Euclidean distance (avoids sqrt, monotone for comparison).
#[inline]
fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_line_graph(n: usize) -> NswGraph {
        // n vectors in R^1 arranged 0,1,2,...n-1
        let mut g = NswGraph::new(4, 20);
        for i in 0..n {
            g.insert(vec![i as f32]);
        }
        g
    }

    #[test]
    fn search_finds_nearest_on_line() {
        let g = build_line_graph(100);
        // Query at 50.1 → nearest should be 50 or 51.
        let results = g.search(&[50.1f32], 1, 20);
        assert!(!results.is_empty());
        let (idx, _) = results[0];
        assert!(
            idx == 50 || idx == 51,
            "nearest to 50.1 should be 50 or 51, got {idx}"
        );
    }

    #[test]
    fn search_returns_k_results() {
        let g = build_line_graph(50);
        let results = g.search(&[25.0f32], 10, 20);
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn memory_bytes_grows_with_nodes() {
        let mut g = NswGraph::new(8, 20);
        let m0 = g.memory_bytes();
        g.insert(vec![0.0; 64]);
        g.insert(vec![1.0; 64]);
        assert!(g.memory_bytes() > m0);
    }

    #[test]
    fn higher_ef_does_not_degrade_recall() {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(42);
        let mut g = NswGraph::new(8, 30);
        for _ in 0..500 {
            let v: Vec<f32> = (0..32).map(|_| rng.gen()).collect();
            g.insert(v);
        }
        let q: Vec<f32> = (0..32).map(|_| rng.gen()).collect();
        let r_lo = g.search(&q, 10, 10);
        let r_hi = g.search(&q, 10, 100);
        // high-ef result set should include ≥ as many unique indices as low-ef
        // (measured as: high-ef top-1 dist ≤ low-ef top-1 dist)
        if !r_lo.is_empty() && !r_hi.is_empty() {
            assert!(
                r_hi[0].1 <= r_lo[0].1 + 1e-4,
                "high ef should find at least as good a nearest neighbor"
            );
        }
    }
}
