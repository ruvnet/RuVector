//! # ruvector-sq-hnsw
//!
//! Scalar-quantized HNSW with online calibration and approximate-then-rerank search.
//!
//! Three variants, all implementing [`AnnIndex`]:
//!
//! | Variant | Storage | Distance | Rerank | Use case |
//! |---------|---------|----------|--------|----------|
//! | [`F32Index`] | f32 | f32 L2 | — | Baseline exact-precision HNSW |
//! | [`Sq8Index`] | int8 | int8 L2 | — | Max speed, ~2-4% recall drop |
//! | [`Sq8RerankIndex`] | int8 + f32 | int8 → f32 rerank | ✓ | Balanced speed/recall |
//!
//! ## Example
//! ```rust
//! use ruvector_sq_hnsw::{AnnIndex, F32Index, HnswConfig};
//!
//! let config = HnswConfig { m: 16, ef_construction: 100, ef_search: 50 };
//! let mut idx = F32Index::new(config, 8);
//! idx.add(0, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
//! idx.add(1, vec![0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
//! let results = idx.search(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1);
//! assert_eq!(results[0].id, 0);
//! ```

pub mod hnsw;
pub mod index;
pub mod quantizer;

pub use index::{F32Index, Sq8Index, Sq8RerankIndex};
pub use quantizer::ScalarQuantizer;

/// Shared HNSW construction and search configuration.
#[derive(Debug, Clone, Copy)]
pub struct HnswConfig {
    /// Max connections per node per layer (except layer 0, which uses 2*m).
    pub m: usize,
    /// Beam width during construction.
    pub ef_construction: usize,
    /// Beam width during search.
    pub ef_search: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
        }
    }
}

/// Single ANN search result.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub id: usize,
    pub distance: f32,
}

/// Trait implemented by all three index variants.
pub trait AnnIndex {
    /// Add a vector with the given ID.
    fn add(&mut self, id: usize, vector: Vec<f32>);

    /// Add multiple vectors; all must be provided before the first search for
    /// calibrated variants — order is the same as individual `add` calls.
    fn add_batch(&mut self, vectors: Vec<(usize, Vec<f32>)>) {
        for (id, v) in vectors {
            self.add(id, v);
        }
    }

    /// Return the `k` approximate nearest neighbours for `query`.
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult>;

    /// Number of indexed vectors.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Heap bytes per stored vector (approximate).
    fn bytes_per_vector(&self) -> usize;
}

/// Brute-force exact search used to compute ground-truth recall.
pub fn exact_knn(corpus: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let mut dists: Vec<(f32, usize)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (l2_sq(v, query), i))
        .collect();
    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    dists.into_iter().take(k).map(|(_, i)| i).collect()
}

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}
