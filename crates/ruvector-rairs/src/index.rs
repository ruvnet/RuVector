//! Shared ANN index trait and search result type.

use crate::error::RairsError;

/// A nearest-neighbor result from any index variant.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// Original vector ID (0-based insertion order).
    pub id: usize,
    /// Approximate L2 distance to the query.
    pub distance: f32,
}

/// Common interface for all three RAIRS index variants.
pub trait AnnIndex {
    /// Add a slice of f32 vectors to the index.
    fn add(&mut self, vectors: &[Vec<f32>]) -> Result<(), RairsError>;

    /// Search for the `k` approximate nearest neighbors of `query`.
    /// `nprobe` controls how many inverted lists are visited.
    fn search(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
    ) -> Result<Vec<SearchResult>, RairsError>;

    /// Return the number of indexed vectors.
    fn len(&self) -> usize;

    /// Return true if the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the number of inverted lists (clusters).
    fn num_lists(&self) -> usize;
}

// ─── shared distance helpers ─────────────────────────────────────────────────

/// Squared Euclidean distance between two equal-length f32 slices.
#[inline(always)]
pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Dot product of two equal-length f32 slices.
#[inline(always)]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
