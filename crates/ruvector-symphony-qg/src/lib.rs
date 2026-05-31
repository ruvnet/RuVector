//! SymphonyQG: graph-coupled 4-bit FastScan neighbor scoring for ANN search.
//!
//! Implements the core ideas from:
//!   Gou et al., "SymphonyQG: Towards Symphonious Integration of Quantization
//!   and Graph for Approximate Nearest Neighbor Search", SIGMOD 2025.
//!   <https://arxiv.org/abs/2411.12229>
//!
//! ## Key insight
//!
//! Standard graph-based ANN (HNSW, DiskANN) stores raw f32 neighbor vectors
//! and computes exact distances during traversal — or scores candidates with
//! quantized codes but re-ranks with full precision in a separate step.
//! SymphonyQG co-locates packed 4-bit PQ codes *inside the graph edge list*,
//! then uses a SIMD lookup-table (FastScan) to score all neighbors in a single
//! cache-friendly pass, eliminating the re-rank stage entirely.
//!
//! ## Three search variants
//!
//! | Variant | Distance source | Re-rank? |
//! |---------|----------------|----------|
//! | `flat_exact` | full f32 brute force | — |
//! | `sqg_fastscan` | 4-bit PQ FastScan (graph) | no |
//! | `sqg_rerank` | 4-bit PQ FastScan + top-K f32 re-rank | yes |

pub mod error;
pub mod fastscan;
pub mod graph;
pub mod pq4;

use error::SqgError;
use graph::SqgGraph;
use pq4::Pq4;
use serde::{Deserialize, Serialize};

/// Top-level SymphonyQG index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SqgIndex {
    pub pq: Pq4,
    pub graph: SqgGraph,
    pub data: Vec<f32>,
    pub dim: usize,
    pub n: usize,
}

/// Build configuration.
#[derive(Debug, Clone)]
pub struct SqgConfig {
    /// Number of PQ subspaces (must divide `dim`). 8–16 for D=128.
    pub pq_subspaces: usize,
    /// k-means training iterations for PQ codebook.
    pub pq_iters: usize,
    /// Graph degree (neighbors per node).
    pub m_neighbors: usize,
}

impl Default for SqgConfig {
    fn default() -> Self {
        Self { pq_subspaces: 8, pq_iters: 20, m_neighbors: 16 }
    }
}

impl SqgIndex {
    /// Build index from `vectors` (row-major, shape [n, dim]).
    pub fn build(vectors: &[f32], dim: usize, cfg: SqgConfig) -> Result<Self, SqgError> {
        if vectors.len() % dim != 0 {
            return Err(SqgError::Config("vectors.len() not divisible by dim".into()));
        }
        let n = vectors.len() / dim;
        let pq = Pq4::train(vectors, dim, cfg.pq_subspaces, cfg.pq_iters)?;
        let graph = SqgGraph::build(vectors, dim, cfg.m_neighbors, &pq)?;
        Ok(Self { pq, graph, data: vectors.to_vec(), dim, n })
    }

    /// Variant A: brute-force exact L2 search (baseline).
    pub fn flat_exact(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let mut dists: Vec<(f32, u32)> = (0..self.n)
            .map(|i| {
                let v = &self.data[i * self.dim..(i + 1) * self.dim];
                let d: f32 = query.iter().zip(v).map(|(a, b)| (a - b).powi(2)).sum();
                (d, i as u32)
            })
            .collect();
        dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        dists.truncate(k);
        dists.iter().map(|&(d, id)| (id, d)).collect()
    }

    /// Variant B: SymphonyQG beam search with FastScan only (no f32 re-rank).
    pub fn sqg_fastscan(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        self.graph.search(query, &self.data, self.dim, &self.pq, k, ef, None)
    }

    /// Variant C: SymphonyQG beam search + exact f32 re-rank of all ef candidates.
    pub fn sqg_rerank(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        self.graph.search(query, &self.data, self.dim, &self.pq, k, ef, Some(&self.data))
    }
}

/// Recall@k: fraction of true top-k ids found in candidate ids.
pub fn recall_at_k(truth: &[u32], candidates: &[u32]) -> f64 {
    let k = truth.len();
    let found = candidates.iter().filter(|id| truth.contains(id)).count();
    found as f64 / k as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, Normal};

    fn random_vecs(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let dist = Normal::new(0.0f32, 1.0).unwrap();
        (0..n * d).map(|_| dist.sample(&mut rng)).collect()
    }

    #[test]
    fn build_and_search_exact_baseline() {
        let n = 200;
        let d = 32;
        let data = random_vecs(n, d, 42);
        let cfg = SqgConfig { pq_subspaces: 8, pq_iters: 10, m_neighbors: 8 };
        let idx = SqgIndex::build(&data, d, cfg).unwrap();
        let q = random_vecs(1, d, 1);
        let res = idx.flat_exact(&q, 5);
        assert_eq!(res.len(), 5);
        // Results should be ascending by distance.
        let dists: Vec<f32> = res.iter().map(|&(_, d)| d).collect();
        for w in dists.windows(2) {
            assert!(w[0] <= w[1] + 1e-5, "not sorted: {:?}", dists);
        }
    }

    #[test]
    fn fastscan_recall_at_k_reasonable() {
        let n = 300;
        let d = 32;
        let data = random_vecs(n, d, 55);
        let cfg = SqgConfig { pq_subspaces: 8, pq_iters: 15, m_neighbors: 16 };
        let idx = SqgIndex::build(&data, d, cfg).unwrap();
        let q = random_vecs(1, d, 77);

        let truth: Vec<u32> = idx.flat_exact(&q, 10).iter().map(|&(id, _)| id).collect();
        let cands: Vec<u32> = idx.sqg_fastscan(&q, 10, 50).iter().map(|&(id, _)| id).collect();
        let r = recall_at_k(&truth, &cands);
        // Low bound: even a naive graph should recall some true neighbors.
        assert!(r >= 0.1, "recall@10 too low: {r:.2}");
    }

    #[test]
    fn rerank_recall_geq_fastscan() {
        let n = 300;
        let d = 32;
        let data = random_vecs(n, d, 99);
        let cfg = SqgConfig { pq_subspaces: 8, pq_iters: 15, m_neighbors: 16 };
        let idx = SqgIndex::build(&data, d, cfg).unwrap();
        let q = random_vecs(1, d, 111);

        let truth: Vec<u32> = idx.flat_exact(&q, 10).iter().map(|&(id, _)| id).collect();
        let cands_fs: Vec<u32> = idx.sqg_fastscan(&q, 10, 50).iter().map(|&(id, _)| id).collect();
        let cands_rr: Vec<u32> = idx.sqg_rerank(&q, 10, 50).iter().map(|&(id, _)| id).collect();
        let r_fs = recall_at_k(&truth, &cands_fs);
        let r_rr = recall_at_k(&truth, &cands_rr);
        // Re-rank must not be worse than raw fastscan (it sees the same candidates and resorts exactly).
        assert!(r_rr >= r_fs - 0.05, "rerank recall {r_rr:.2} < fastscan {r_fs:.2}");
    }
}
