//! Scalar-Quantized NSW Graph Search with Two-Stage Re-ranking.
//!
//! Three search backends sharing a common trait:
//!
//! | Variant         | Build      | Search               | Memory  |
//! |----------------|-----------|---------------------|---------|
//! | [`FlatExact`]  | O(1)/vec  | O(n·d) brute force  | n·d·4B  |
//! | [`FlatSq8`]    | O(d)/vec  | O(n·d) quantized    | n·d·1B  |
//! | [`GraphSq8`]   | O(M·ef·d) | O(ef·M·d) graph     | n·d·1B  |
//! | [`GraphSq4`]   | O(M·ef·d) | O(ef·M·d) graph     | n·d/2·B |
//!
//! All variants are no-dependency pure Rust, deterministic, and
//! independently measurable against ground-truth recall.

pub mod flat;
pub mod graph;
pub mod hnsw2;
pub mod quantizer;

pub use flat::{FlatExact, FlatSq8};
pub use graph::{GraphSq4, GraphSq8};
pub use hnsw2::SqHnsw2;
pub use quantizer::ScalarQuantizer;

/// Single ranked result returned by every variant.
#[derive(Debug, Clone)]
pub struct NnResult {
    /// Zero-based insertion-order ID.
    pub id: usize,
    /// Exact or approximate squared-L2 distance.
    pub distance: f32,
}

/// Unified nearest-neighbour search trait.
pub trait NnSearch {
    /// Insert one vector (assigned ID = current `len()`).
    fn insert(&mut self, vector: Vec<f32>);
    /// Return at most `k` nearest neighbours, unsorted or sorted ascending.
    fn search(&self, query: &[f32], k: usize) -> Vec<NnResult>;
    /// Number of indexed vectors.
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Estimated heap memory in bytes (excluding stack / OS overhead).
    fn memory_bytes(&self) -> usize;
}

/// Exact squared L2 distance, public for re-ranking helpers.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Compute recall@k: fraction of ground-truth IDs found in results.
pub fn recall_at_k(ground_truth: &[NnResult], results: &[NnResult], k: usize) -> f32 {
    let gt: std::collections::HashSet<usize> = ground_truth.iter().take(k).map(|r| r.id).collect();
    let found = results
        .iter()
        .take(k)
        .filter(|r| gt.contains(&r.id))
        .count();
    found as f32 / k.min(gt.len()) as f32
}
