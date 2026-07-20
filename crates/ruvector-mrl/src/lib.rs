//! # ruvector-mrl — Matryoshka Resolution Index
//!
//! Two-stage approximate nearest-neighbor search that exploits the **prefix
//! property** of Matryoshka Representation Learning (MRL) embeddings.
//!
//! Modern embedding models (OpenAI text-embedding-3, Cohere embed-v3,
//! Nomic embed v1.5) produce vectors where the first `d_fast` dimensions
//! are a meaningful approximation of the full `d_full`-dimensional vector.
//! This crate turns that prefix property into a search acceleration:
//!
//! 1. **Screen** — approximate search using only the first `d_fast` dimensions.
//! 2. **Rerank** — exact cosine on the full `d_full` vector for the top candidates.
//!
//! ## Variants
//!
//! | Variant | Stage 1 | Stage 2 | Notes |
//! |---------|---------|---------|-------|
//! | [`FlatIndex`] | brute-force, full D | none | exact baseline |
//! | [`MrlLinear`] | brute-force, D_FAST | exact rerank | shows pure dim reduction gain |
//! | [`MrlGraph`] | greedy kNN graph, D_FAST | exact rerank | full approach |
//!
//! ## Design choice
//!
//! The greedy kNN graph replaces HNSW hierarchy with a single-layer graph
//! built by connecting each inserted vector to its M nearest existing
//! neighbours in D_FAST space.  This is O(N²·D_FAST) to build — fine for
//! a PoC at N ≤ 10 K — and O(ef·D_FAST + k_over·D_FULL) to search.
//!
//! [`FlatIndex`]: crate::flat::FlatIndex
//! [`MrlLinear`]: crate::mrl::MrlLinear
//! [`MrlGraph`]: crate::mrl::MrlGraph

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod flat;
pub mod graph;
pub mod mrl;

/// Single nearest-neighbour result.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// Vector id as given at insert time.
    pub id: u32,
    /// Cosine similarity score (higher = more similar).
    pub score: f32,
}

/// Common trait for all index backends.
pub trait MrlSearch {
    /// Insert a full-dimensional vector with sequential id.
    fn insert(&mut self, id: u32, vector: &[f32]);
    /// Return the k approximate nearest neighbours of `query`.
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult>;
    /// Number of indexed vectors.
    fn len(&self) -> usize;
    /// True when the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Dot product of two equal-length slices.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Normalise `v` to unit length in-place.
pub fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Recall@k: fraction of ground-truth top-k ids present in `found`.
pub fn recall_at_k(ground_truth: &[u32], found: &[u32]) -> f32 {
    if ground_truth.is_empty() {
        return 1.0;
    }
    let k = ground_truth.len();
    let hits = found.iter().filter(|id| ground_truth.contains(id)).count();
    hits as f32 / k as f32
}
