//! Product Quantization with Asymmetric Distance Computation (PQ-ADC)
//!
//! Decomposes D-dimensional vectors into M sub-vectors of d=D/M dimensions,
//! trains K centroids per sub-space with Lloyd's algorithm, and encodes each
//! vector as M bytes (K=256 → 8 bits per sub-space).
//!
//! Three search strategies:
//! - [`FlatPqIndex`] — linear ADC scan over all PQ codes (baseline)
//! - [`IvfPqIndex`] — coarse IVF + PQ per cell (alternative A)
//! - [`ResidualPqIndex`] — PQ + f32 residual correction for top-k (alternative B)
//!
//! All implement the [`PqSearch`] trait; no external service dependency.

pub mod codebook;
pub mod encoder;
pub mod flat;
pub mod ivf_pq;
pub mod residual;

pub use codebook::{PqCodebook, PqConfig};
pub use encoder::{decode_vector, encode_vector};
pub use flat::FlatPqIndex;
pub use ivf_pq::IvfPqIndex;
pub use residual::ResidualPqIndex;

/// Single search result returned by all variants.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// Original database vector index.
    pub id: usize,
    /// Approximate squared-L2 distance (ADC score).
    pub distance: f32,
}

/// Unified trait for all PQ-based search backends.
pub trait PqSearch {
    /// Insert a vector into the index (id assigned by insertion order).
    fn insert(&mut self, vector: &[f32]);

    /// Return the top-k approximate nearest neighbors for `query`.
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult>;

    /// Estimated heap memory used by this index, in bytes.
    fn memory_bytes(&self) -> usize;

    /// Human-readable name for reporting.
    fn name(&self) -> &'static str;
}

/// Brute-force exact L2 search; used for recall ground-truth only.
pub struct ExactSearch {
    vectors: Vec<Vec<f32>>,
}

impl ExactSearch {
    pub fn new() -> Self {
        Self {
            vectors: Vec::new(),
        }
    }

    pub fn insert(&mut self, v: &[f32]) {
        self.vectors.push(v.to_vec());
    }

    /// Returns the true top-k nearest neighbor ids by exact L2.
    pub fn search_exact(&self, query: &[f32], k: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, l2_sq(query, v)))
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        scored.into_iter().take(k).map(|(id, _)| id).collect()
    }
}

impl Default for ExactSearch {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute recall@k: fraction of true top-k found in approximate top-k.
pub fn recall_at_k(approx: &[SearchResult], truth: &[usize], k: usize) -> f32 {
    let k = k.min(approx.len()).min(truth.len());
    if k == 0 {
        return 0.0;
    }
    let approx_ids: std::collections::HashSet<usize> =
        approx.iter().take(k).map(|r| r.id).collect();
    let hits = truth
        .iter()
        .take(k)
        .filter(|&&id| approx_ids.contains(&id))
        .count();
    hits as f32 / k as f32
}

/// Squared Euclidean distance between two equal-length slices.
#[inline]
pub(crate) fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}
