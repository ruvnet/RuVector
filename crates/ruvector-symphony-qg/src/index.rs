//! Index trait and three concrete implementations for benchmarking.
//!
//! | Variant | Build | Search | Memory |
//! |---|---|---|---|
//! | `FlatF32Index` | O(n) | O(n·D) exact L2 scan | n × D × 4 bytes |
//! | `GraphExact` | O(n²·D) | O(ef·R·D) beam, exact L2 | n × (D+R) × 4 bytes |
//! | `SymphonyIndex` | O(n²·D) | O(ef·R·D/64) beam, ADC | n × (D + R·(D/8+2)) × 4 bytes |
//!
//! All three share the `AnnIndex` trait so the benchmark harness is uniform.

use crate::error::{Result, SymphonyError};
use crate::graph::{l2_sq, GraphConfig, SymphonyGraph};
use crate::rotation::random_orthogonal;
use crate::search::{beam_search_exact, beam_search_symphony};

/// A single search result.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub id: usize,
    pub distance: f32,
}

/// Common interface for all ANN index variants.
pub trait AnnIndex {
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult>;
    fn len(&self) -> usize;
    fn memory_bytes(&self) -> usize;
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// FlatF32Index — brute-force exact L2 baseline
// ---------------------------------------------------------------------------

pub struct FlatF32Index {
    vectors: Vec<Vec<f32>>,
}

impl FlatF32Index {
    pub fn build(vectors: Vec<Vec<f32>>) -> Result<Self> {
        if vectors.is_empty() {
            return Err(SymphonyError::EmptyCorpus);
        }
        Ok(Self { vectors })
    }
}

impl AnnIndex for FlatF32Index {
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let mut dists: Vec<(f32, usize)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (l2_sq(query, v), i))
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        dists
            .into_iter()
            .take(k)
            .map(|(d, id)| SearchResult { id, distance: d })
            .collect()
    }

    fn len(&self) -> usize { self.vectors.len() }

    fn memory_bytes(&self) -> usize {
        self.vectors.iter().map(|v| v.len() * 4).sum()
    }

    fn name(&self) -> &'static str { "FlatF32" }
}

// ---------------------------------------------------------------------------
// GraphExact — graph traversal with exact L2 (no quantization)
// ---------------------------------------------------------------------------

pub struct GraphExact {
    graph: SymphonyGraph,
    n_starts: usize,
}

impl GraphExact {
    pub fn build(vectors: Vec<Vec<f32>>, config: GraphConfig) -> Result<Self> {
        if vectors.is_empty() {
            return Err(SymphonyError::EmptyCorpus);
        }
        let dim = config.dim;
        if vectors[0].len() != dim {
            return Err(SymphonyError::DimensionMismatch {
                expected: dim,
                actual: vectors[0].len(),
            });
        }
        let rot = random_orthogonal(dim, config.rotation_seed);
        let graph = SymphonyGraph::build(&vectors, config, &rot);
        Ok(Self { graph, n_starts: 4 })
    }
}

impl AnnIndex for GraphExact {
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let ef = self.graph.config.ef;
        beam_search_exact(&self.graph, query, k, ef, self.n_starts)
            .into_iter()
            .map(|(d, id)| SearchResult { id, distance: d })
            .collect()
    }

    fn len(&self) -> usize { self.graph.vertices.len() }

    fn memory_bytes(&self) -> usize { self.graph.memory_bytes() }

    fn name(&self) -> &'static str { "GraphExact" }
}

// ---------------------------------------------------------------------------
// SymphonyIndex — co-located codes + asymmetric batch distance
// ---------------------------------------------------------------------------

pub struct SymphonyIndex {
    graph: SymphonyGraph,
    n_starts: usize,
}

impl SymphonyIndex {
    pub fn build(vectors: Vec<Vec<f32>>, config: GraphConfig) -> Result<Self> {
        if vectors.is_empty() {
            return Err(SymphonyError::EmptyCorpus);
        }
        let dim = config.dim;
        if vectors[0].len() != dim {
            return Err(SymphonyError::DimensionMismatch {
                expected: dim,
                actual: vectors[0].len(),
            });
        }
        let rot = random_orthogonal(dim, config.rotation_seed);
        let graph = SymphonyGraph::build(&vectors, config, &rot);
        Ok(Self { graph, n_starts: 4 })
    }
}

impl AnnIndex for SymphonyIndex {
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let ef = self.graph.config.ef;
        beam_search_symphony(&self.graph, query, k, ef, self.n_starts)
            .into_iter()
            .map(|(d, id)| SearchResult { id, distance: d })
            .collect()
    }

    fn len(&self) -> usize { self.graph.vertices.len() }

    fn memory_bytes(&self) -> usize { self.graph.memory_bytes() }

    fn name(&self) -> &'static str { "SymphonyQG" }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gaussian_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0f32, 1.0).unwrap();
        (0..n)
            .map(|_| (0..dim).map(|_| normal.sample(&mut rng)).collect())
            .collect()
    }

    #[test]
    fn flat_nearest_is_self() {
        let vecs = gaussian_vecs(50, 16, 1);
        let idx = FlatF32Index::build(vecs.clone()).unwrap();
        let r = idx.search(&vecs[7], 1);
        assert_eq!(r[0].id, 7);
        assert!(r[0].distance < 1e-6);
    }

    #[test]
    fn graph_exact_returns_k_results() {
        let dim = 16;
        let vecs = gaussian_vecs(100, dim, 2);
        let cfg = GraphConfig::new(dim).with_r(8).with_ef(20);
        let idx = GraphExact::build(vecs.clone(), cfg).unwrap();
        let q = &vecs[0];
        let r = idx.search(q, 5);
        assert_eq!(r.len(), 5);
        assert_eq!(r[0].id, 0);
    }

    #[test]
    fn symphony_recall_reasonable() {
        let dim = 32;
        let n = 200;
        let vecs = gaussian_vecs(n, dim, 3);
        let cfg = GraphConfig::new(dim).with_r(16).with_ef(40);

        let flat = FlatF32Index::build(vecs.clone()).unwrap();
        let symphony = SymphonyIndex::build(vecs.clone(), cfg).unwrap();

        let mut total_recall = 0.0f64;
        let n_queries = 20;
        for qi in 0..n_queries {
            let q = &vecs[n - 1 - qi]; // Use held-out vectors as queries
            let truth: std::collections::HashSet<usize> = flat.search(q, 10)
                .into_iter().map(|r| r.id).collect();
            let found: std::collections::HashSet<usize> = symphony.search(q, 10)
                .into_iter().map(|r| r.id).collect();
            let hits = truth.intersection(&found).count();
            total_recall += hits as f64 / 10.0;
        }
        let recall = total_recall / n_queries as f64;
        assert!(recall >= 0.5, "recall@10={recall:.3} too low (expected ≥0.5)");
    }
}
