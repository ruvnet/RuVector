//! Coherence-gated multi-tenant vector namespace router.
//!
//! Three variants with increasing semantic awareness:
//! 1. [`FlatIsolated`]     — strict per-namespace linear scan, zero cross-talk.
//! 2. [`CentroidRouted`]   — centroid index prunes distant namespaces before scan.
//! 3. [`CoherenceGated`]   — centroid routing + cross-namespace federation gated
//!    by a coherence threshold; every cross-boundary access is appended to a
//!    [`WitnessLog`].

pub mod centroid_routed;
pub mod coherence_gated;
pub mod flat_isolated;
pub mod witness;

pub use centroid_routed::CentroidRouted;
pub use coherence_gated::CoherenceGated;
pub use flat_isolated::FlatIsolated;
pub use witness::WitnessLog;

// ─── shared types ─────────────────────────────────────────────────────────────

/// An opaque namespace identifier.
pub type NamespaceId = u32;

/// An opaque vector identifier (unique within a namespace).
pub type VectorId = u64;

/// A single result returned by [`NamespaceIndex::search`].
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: VectorId,
    pub namespace: NamespaceId,
    /// Squared Euclidean distance to the query.
    pub distance: f32,
}

/// Common trait implemented by all three namespace backends.
pub trait NamespaceIndex {
    /// Insert a vector into `ns`. Returns an error string on failure.
    fn insert(&mut self, ns: NamespaceId, id: VectorId, vector: Vec<f32>) -> Result<(), String>;

    /// Return the `k` nearest vectors within `ns` (and optionally across
    /// coherent neighbours, depending on the implementation).
    fn search(&self, ns: NamespaceId, query: &[f32], k: usize) -> Vec<SearchResult>;

    /// Number of distinct namespaces currently held.
    fn namespace_count(&self) -> usize;

    /// Total number of vectors across all namespaces.
    fn total_vectors(&self) -> usize;

    /// Rough heap-allocated memory in bytes.
    fn memory_bytes(&self) -> usize;
}

// ─── distance helper ─────────────────────────────────────────────────────────

/// Squared Euclidean distance (avoids sqrt, monotone for ranking).
#[inline]
pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Centroid of a slice of equal-length vectors.
pub fn centroid(vectors: &[Vec<f32>]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }
    let dim = vectors[0].len();
    let n = vectors.len() as f32;
    let mut c = vec![0f32; dim];
    for v in vectors {
        for (acc, x) in c.iter_mut().zip(v.iter()) {
            *acc += x;
        }
    }
    c.iter_mut().for_each(|x| *x /= n);
    c
}
