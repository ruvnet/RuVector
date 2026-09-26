//! Incrementally-insertable navigable small-world graph.
//!
//! This is intentionally a single-layer NSW (navigable small world) —
//! the foundational layer beneath HNSW. The coherence-gated WAL flush
//! is the novel contribution; a battle-tested multi-layer HNSW would be
//! a drop-in replacement in production (see ADR-272 migration path).
//!
//! ## Incremental construction algorithm
//!
//! Each new vector is inserted by:
//!
//! 1. Running a beam search (width `ef_construction`) to find the M
//!    nearest current neighbours.
//! 2. Connecting the new node to those neighbours (forward edges).
//! 3. Adding a back-edge from each neighbour to the new node, pruning
//!    to keep at most `m_max` neighbours per node.
//!
//! This gives O(ef × M × D) per insert rather than O(N × D) for a
//! brute-force rebuild, making batch merges of WAL entries cheap.
//!
//! ## Long-jump edges
//!
//! When the graph contains fewer than `longjump_interval` nodes we skip
//! long-jump edges. Once the graph is large enough, each new node also
//! receives `m_longjump` randomly-sampled global connections to preserve
//! navigability across sparse regions.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BinaryHeap;

/// Configuration for the navigable graph.
#[derive(Clone, Debug)]
pub struct GraphConfig {
    /// Max neighbours per node (M in HNSW).
    pub m: usize,
    /// Max neighbours for the entry node (usually 2*m).
    pub m_max: usize,
    /// Beam width during construction (ef_construction).
    pub ef_construction: usize,
    /// Long-jump neighbours added per insert for global navigability.
    pub m_longjump: usize,
    /// Only add long-jump edges when graph has ≥ this many nodes.
    pub longjump_interval: usize,
    /// Vector dimensionality.
    pub dims: usize,
}

impl GraphConfig {
    pub fn new(dims: usize) -> Self {
        GraphConfig {
            m: 16,
            m_max: 32,
            ef_construction: 100,
            m_longjump: 6,
            longjump_interval: 16,
            dims,
        }
    }
}

/// Incrementally-insertable navigable small-world graph.
pub struct NavGraph {
    /// Row-major flat vector store: node i at [i*dims .. (i+1)*dims].
    pub vectors: Vec<f32>,
    /// Adjacency list: `adj[i]` = neighbour node indices.
    pub adj: Vec<Vec<u32>>,
    pub config: GraphConfig,
    /// Number of nodes currently in the graph.
    pub n: usize,
    rng: StdRng,
}

impl NavGraph {
    /// Create an empty graph with the given configuration.
    pub fn new(config: GraphConfig) -> Self {
        NavGraph {
            vectors: Vec::new(),
            adj: Vec::new(),
            config,
            n: 0,
            rng: StdRng::seed_from_u64(0xDEAD_BEEF_C0FFEEu64),
        }
    }

    /// Create a pre-populated graph from existing vectors.
    ///
    /// Builds incrementally (same as individual inserts) for full correctness.
    pub fn from_vecs(vectors: Vec<Vec<f32>>, config: GraphConfig) -> Self {
        let mut g = NavGraph::new(config);
        for v in vectors {
            g.insert(&v);
        }
        g
    }

    /// Insert a new vector, connecting it to its approximate nearest neighbours.
    ///
    /// Returns the index of the newly inserted node.
    pub fn insert(&mut self, vector: &[f32]) -> u32 {
        let new_idx = self.n as u32;
        let dims = self.config.dims;
        debug_assert_eq!(vector.len(), dims);

        // Store the vector.
        self.vectors.extend_from_slice(vector);
        self.adj.push(Vec::new());
        self.n += 1;

        if self.n == 1 {
            // First node — no neighbours possible.
            return new_idx;
        }

        // Beam search to find ef_construction candidate neighbours.
        let ef = self.config.ef_construction.min(self.n - 1);
        let candidates = self.beam_search_internal(vector, ef, new_idx);

        // Keep at most M nearest.
        let m = self.config.m.min(candidates.len());
        let chosen: Vec<u32> = candidates[..m].iter().map(|&(idx, _)| idx).collect();

        // Forward edges: new_idx → chosen.
        self.adj[new_idx as usize] = chosen.clone();

        // Back-edges: chosen → new_idx (prune to m_max).
        for &nb in &chosen {
            {
                let nb_adj = &mut self.adj[nb as usize];
                if nb_adj.len() < self.config.m_max {
                    nb_adj.push(new_idx);
                }
            }
            // If over m_max, prune to keep m_max nearest (separate borrow scope).
            if self.adj[nb as usize].len() > self.config.m_max {
                let m_max = self.config.m_max;
                let dims = self.config.dims;
                let nb_start = (nb as usize) * dims;
                let nb_vec: Vec<f32> = self.vectors[nb_start..nb_start + dims].to_vec();
                let neighbours = self.adj[nb as usize].clone();
                let mut with_dist: Vec<(u32, f32)> = neighbours
                    .iter()
                    .map(|&j| {
                        let js = (j as usize) * dims;
                        (j, l2_sq(&nb_vec, &self.vectors[js..js + dims]))
                    })
                    .collect();
                with_dist.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                with_dist.truncate(m_max);
                self.adj[nb as usize] = with_dist.into_iter().map(|(j, _)| j).collect();
            }
        }

        // Long-jump edges for global navigability.
        if self.n >= self.config.longjump_interval {
            let lj = self.config.m_longjump.min(self.n - 1);
            let mut added = 0;
            let mut attempts = 0;
            while added < lj && attempts < lj * 8 {
                attempts += 1;
                let j = self.rng.gen_range(0..self.n as u32);
                if j != new_idx && !self.adj[new_idx as usize].contains(&j) {
                    self.adj[new_idx as usize].push(j);
                    added += 1;
                }
            }
        }

        new_idx
    }

    /// Search for the `k` approximate nearest neighbours of `query`.
    ///
    /// Uses greedy beam search with width `ef` (≥ k).
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        if self.n == 0 {
            return Vec::new();
        }
        let ef = ef.max(k).min(self.n);
        let mut results = self.beam_search_internal(query, ef, u32::MAX);
        results.truncate(k);
        results
    }

    /// Internal beam search returning up to `ef` nearest nodes.
    ///
    /// `exclude` — node index to skip (used during insertion to avoid
    /// self-neighbour; pass `u32::MAX` to include all nodes).
    fn beam_search_internal(&self, query: &[f32], ef: usize, exclude: u32) -> Vec<(u32, f32)> {
        // Ordered set of candidates by distance (min-heap by negated distance).
        // BinaryHeap is a max-heap, so we negate distances for min-heap behavior.
        type Candidate = (u32, f32); // (node, sq_dist)

        // Entry point: sample N_ENTRY spread-out candidate nodes and pick the
        // one nearest to the query. Probing multiple diverse start points
        // dramatically improves recall on single-layer NSW vs. always starting
        // from node 0 (which may be in a different region of the space).
        const N_ENTRY: usize = 8;
        let entry = {
            let mut best_e = 0u32;
            let mut best_d = f32::MAX;
            let n = self.n as u32;
            for trial in 0..N_ENTRY.min(self.n) {
                // Spread deterministically across the node range.
                let j = (trial as u32 * n / N_ENTRY as u32).min(n - 1);
                if j == exclude {
                    continue;
                }
                let d = l2_sq(query, self.row(j as usize));
                if d < best_d {
                    best_d = d;
                    best_e = j;
                }
            }
            best_e
        };

        let entry_dist = l2_sq(query, self.row(entry as usize));
        let mut visited = vec![false; self.n];
        visited[entry as usize] = true;

        // Min-heap of (dist, idx) for candidates to expand.
        let mut frontier: BinaryHeap<std::cmp::Reverse<(OrderedF32, u32)>> = BinaryHeap::new();
        frontier.push(std::cmp::Reverse((OrderedF32(entry_dist), entry)));

        // Max-heap of (dist, idx) for the current best ef results.
        let mut results: BinaryHeap<(OrderedF32, u32)> = BinaryHeap::new();
        results.push((OrderedF32(entry_dist), entry));

        while let Some(std::cmp::Reverse((OrderedF32(cand_dist), cand_idx))) = frontier.pop() {
            // If the best candidate is farther than the worst current result, stop.
            if results.len() >= ef {
                if let Some(&(OrderedF32(worst), _)) = results.peek() {
                    if cand_dist > worst {
                        break;
                    }
                }
            }

            // Expand neighbours.
            for &nb in &self.adj[cand_idx as usize] {
                if nb == exclude || visited[nb as usize] {
                    continue;
                }
                visited[nb as usize] = true;
                let d = l2_sq(query, self.row(nb as usize));
                // Add to results if within top-ef.
                if results.len() < ef {
                    results.push((OrderedF32(d), nb));
                    frontier.push(std::cmp::Reverse((OrderedF32(d), nb)));
                } else if let Some(&(OrderedF32(worst), _)) = results.peek() {
                    if d < worst {
                        results.pop();
                        results.push((OrderedF32(d), nb));
                        frontier.push(std::cmp::Reverse((OrderedF32(d), nb)));
                    }
                }
            }
        }

        // Collect, sort ascending.
        let mut out: Vec<Candidate> = results
            .into_iter()
            .map(|(OrderedF32(d), i)| (i, d))
            .collect();
        out.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        out
    }

    /// Return the vector for node `i` as a slice.
    #[inline]
    pub fn row(&self, i: usize) -> &[f32] {
        let d = self.config.dims;
        &self.vectors[i * d..(i + 1) * d]
    }

    /// Number of nodes in the graph.
    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    /// True when no nodes have been inserted.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Estimated memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let vec_bytes = self.vectors.len() * std::mem::size_of::<f32>();
        let adj_bytes: usize = self.adj.iter().map(|v| v.len() * 4 + 24).sum();
        vec_bytes + adj_bytes
    }
}

// ── Ordered float wrapper (no NaN in our distances) ─────────────────────────

/// Total-order float wrapper for use in BinaryHeap.
#[derive(Clone, Copy, PartialEq)]
struct OrderedF32(f32);

impl Eq for OrderedF32 {}

impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Squared L2 distance (no sqrt).
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vecs(n: usize, d: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                let mut v = vec![0.0f32; d];
                v[i % d] = 1.0 + (i / d) as f32;
                v
            })
            .collect()
    }

    #[test]
    fn insert_builds_valid_graph() {
        let vecs = unit_vecs(50, 8);
        let cfg = GraphConfig::new(8);
        let g = NavGraph::from_vecs(vecs, cfg);
        assert_eq!(g.n, 50);
        for (i, adj) in g.adj.iter().enumerate() {
            for &nb in adj {
                assert!(
                    (nb as usize) < g.n,
                    "neighbour {nb} out of bounds at node {i}"
                );
                assert_ne!(nb as usize, i, "self-loop at node {i}");
            }
        }
    }

    #[test]
    fn search_returns_k_results() {
        let vecs = unit_vecs(100, 8);
        let cfg = GraphConfig::new(8);
        let g = NavGraph::from_vecs(vecs, cfg);
        let q = vec![1.0f32; 8];
        let results = g.search(&q, 10, 32);
        assert_eq!(results.len(), 10);
        // Results must be sorted ascending.
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1, "results not sorted");
        }
    }

    #[test]
    fn search_finds_exact_neighbour() {
        let vecs: Vec<Vec<f32>> = (0..20).map(|i| vec![i as f32; 4]).collect();
        let cfg = GraphConfig::new(4);
        let g = NavGraph::from_vecs(vecs.clone(), cfg);
        // Query exactly equal to node 10 → should be rank-1 result.
        let q = vec![10.0f32; 4];
        let results = g.search(&q, 5, 20);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 10, "expected node 10 as nearest");
        assert!(results[0].1 < 1e-6, "expected near-zero distance");
    }
}
