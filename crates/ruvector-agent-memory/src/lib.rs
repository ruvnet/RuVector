//! # ruvector-agent-memory
//!
//! Coherence-weighted agent memory compaction for ruvector.
//!
//! Agent memories decay in relevance over time.  This crate provides three
//! compaction policies that retain the most important entries when the memory
//! store exceeds a target capacity:
//!
//! | Policy | Signal | Novel? |
//! |--------|--------|--------|
//! | `LruPolicy` | Recency (`last_accessed_at`) | No — classical |
//! | `LfuPolicy` | Frequency (`access_count`) | No — classical |
//! | `CoherencePolicy` | Weighted score: recency + frequency + context coherence | **Yes** |
//!
//! The `CoherencePolicy` is the core research contribution: it scores each stored
//! memory vector against a *context window* — the embeddings of recent agent
//! queries — and preferentially retains memories that are semantically aligned
//! with the agent's current reasoning thread.
//!
//! ## References
//!
//! - Park et al. 2023, "Generative Agents" (arXiv:2304.03442)
//! - Zhong et al. 2023, "MemoryBank" (arXiv:2305.10250)
//! - Xu 2026, "Self-Aware Vector Embeddings for RAG" (arXiv:2604.20598)
//! - Karhade 2026, "Not All Memories Age the Same" (arXiv:2604.26970)
//! - Survey 2026, "From Storage to Experience" (arXiv:2605.06716)

pub mod compaction;
pub mod memory;
pub mod scoring;

pub use compaction::{CoherencePolicy, CoherenceWeights, CompactionPolicy, LfuPolicy, LruPolicy};
pub use memory::{MemoryEntry, MemoryStore, SearchResult};
pub use scoring::{coherence_score, cosine_sim, normalize};

/// Compact `store` in-place using `policy`, retaining `target_size` entries.
///
/// `context_window` is a slice of recent query embeddings used by
/// `CoherencePolicy` to score semantic alignment.  Pass an empty slice when
/// context is unavailable; `LruPolicy` and `LfuPolicy` ignore it.
///
/// # Panics
/// Panics if `target_size > store.len()`.
pub fn compact(
    store: &mut MemoryStore,
    policy: &dyn CompactionPolicy,
    target_size: usize,
    context_window: &[Vec<f32>],
) {
    assert!(
        target_size <= store.len(),
        "target_size ({}) must be ≤ store.len() ({})",
        target_size,
        store.len()
    );
    let entries = store.entries();
    let survivor_indices = policy.select_survivors(entries, target_size, context_window);
    let mut survivors: Vec<MemoryEntry> = survivor_indices
        .into_iter()
        .map(|i| entries[i].clone())
        .collect();
    survivors.sort_unstable_by_key(|e| e.id);
    store.replace_entries(survivors);
}

/// Recall@K: fraction of true top-K neighbors found in candidate set.
///
/// `truth` and `candidates` are sets of entry ids.  K = `truth.len()`.
pub fn recall_at_k(truth: &[u64], candidates: &[u64]) -> f32 {
    let k = truth.len();
    if k == 0 {
        return 1.0;
    }
    let hits = truth.iter().filter(|id| candidates.contains(id)).count();
    hits as f32 / k as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_reduces_store_size() {
        let mut store = MemoryStore::new(4);
        for _ in 0..20 {
            store.insert(vec![1.0, 0.0, 0.0, 0.0]);
        }
        compact(&mut store, &LruPolicy, 10, &[]);
        assert_eq!(store.len(), 10);
    }

    #[test]
    fn recall_perfect() {
        let truth = vec![0, 1, 2, 3, 4];
        assert!((recall_at_k(&truth, &truth) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn recall_zero() {
        let truth = vec![0, 1, 2];
        let cands = vec![5, 6, 7];
        assert!(recall_at_k(&truth, &cands) < 1e-6);
    }
}
