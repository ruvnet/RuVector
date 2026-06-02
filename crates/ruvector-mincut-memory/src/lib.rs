//! MinCut-guided agent working memory compaction.
//!
//! An agent accumulates memory entries (vectors + metadata) over time.  As the
//! store grows it needs compaction — evicting stale or weakly-connected
//! entries so that future retrieval stays fast and accurate.
//!
//! Three strategies are provided:
//! - [`AgeEvict`]      — remove the oldest entries (simple baseline)
//! - [`CoherenceEvict`] — remove entries with the lowest average similarity to
//!                        graph neighbours (coherence-scored eviction)
//! - [`MinCutEvict`]   — remove entries whose weighted graph degree is lowest,
//!                        approximating the minimum-cut boundary of the memory
//!                        graph (graph-cut guided eviction)
//!
//! All three implement the [`Compactor`] trait so they can be swapped without
//! changing application code.

pub mod compaction;
pub mod metrics;
pub mod store;

pub use compaction::{AgeEvict, CoherenceEvict, MinCutEvict};
pub use metrics::CompactionResult;
pub use store::{Entry, MemoryStore};

/// The single trait every compaction strategy must satisfy.
pub trait Compactor {
    /// Compact `store` so that `store.len() <= target_size`.
    ///
    /// Returns a [`CompactionResult`] describing what happened and how long it
    /// took.  Implementations must not remove entries when the store is already
    /// within budget.
    fn compact(&self, store: &mut MemoryStore, target_size: usize) -> CompactionResult;
}

/// Cosine similarity in [-1, 1].  Returns 0.0 when either vector is zero.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-9 || nb < 1e-9 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

/// Squared L2 distance (no sqrt — monotone proxy for nearest-neighbour search).
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-5);
    }

    #[test]
    fn cosine_zero_vec() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }
}
