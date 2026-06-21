//! Temporal Coherence Decay for Agent Memory Retrieval.
//!
//! Three scored retrieval variants:
//! - `FlatSearch`: pure cosine similarity, no temporal awareness
//! - `TemporalSearch`: cosine × exponential time decay
//! - `CoherenceSearch`: cosine × decay × graph-coherence gate
//!
//! The coherence gate uses a lightweight adjacency graph where memory vectors
//! that are mutually similar (above `coherence_threshold`) form edges.
//! A memory's gate value is its normalised in-degree: highly connected
//! memories score higher because the graph has "voted" for their relevance.

// ── Public re-exports ────────────────────────────────────────────────────────
pub mod decay;
pub mod graph;
pub mod search;
pub mod store;

pub use decay::{DecayConfig, DecayKind};
pub use graph::CoherenceGraph;
pub use search::{CoherenceSearch, FlatSearch, SearchResult, TemporalSearch, VectorSearch};
pub use store::{MemoryId, MemoryMetadata, MemoryRecord, MemoryStore};

/// Build a populated `MemoryStore` for tests and benchmarks.
///
/// Generates `n` memories: vectors are drawn from a seeded PRNG in dimension
/// `dims`, timestamps are evenly spread over [0, time_span], cluster labels
/// control coherence topology (adjacent cluster members share high similarity).
pub fn generate_memory_corpus(
    n: usize,
    dims: usize,
    time_span: u64,
    num_clusters: usize,
    rng: &mut impl rand::Rng,
) -> MemoryStore {
    use rand::distributions::{Distribution, Uniform};
    let uni = Uniform::new(-1.0f32, 1.0);

    let mut store = MemoryStore::new(dims);
    for i in 0..n {
        let cluster = i % num_clusters;
        // Cluster centre is a fixed offset; individual vector adds noise.
        let centre_offset = cluster as f32 * 0.8;
        let vec: Vec<f32> = (0..dims)
            .map(|d| {
                let base = if d % num_clusters == cluster {
                    centre_offset
                } else {
                    0.0
                };
                base + uni.sample(rng) * 0.25
            })
            .collect();
        let ts = (i as u64 * time_span) / n as u64;
        store.insert(
            vec,
            MemoryMetadata {
                timestamp: ts,
                source: format!("agent-{}", cluster),
                tags: vec![format!("cluster-{}", cluster)],
            },
        );
    }
    store
}

/// Ground-truth recall@k for a query against the store (cosine only).
pub fn ground_truth_topk(query: &[f32], store: &MemoryStore, k: usize) -> Vec<MemoryId> {
    let mut scored: Vec<(MemoryId, f32)> = store
        .records()
        .map(|r| (r.id, cosine_sim(query, &r.vec)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

/// Fraction of `retrieved` ids that appear in `ground_truth`.
pub fn recall_at_k(retrieved: &[MemoryId], ground_truth: &[MemoryId]) -> f32 {
    if ground_truth.is_empty() {
        return 1.0;
    }
    let gt_set: std::collections::HashSet<MemoryId> = ground_truth.iter().copied().collect();
    let hits = retrieved.iter().filter(|id| gt_set.contains(id)).count();
    hits as f32 / ground_truth.len().min(retrieved.len()).max(1) as f32
}

/// Normalised cosine similarity in [-1, 1].
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na * nb).sqrt();
    if denom < 1e-9 {
        0.0
    } else {
        dot / denom
    }
}

/// Simple memory-usage estimate in bytes.
pub fn estimate_memory_bytes(n: usize, dims: usize) -> usize {
    // f32 vec + metadata (timestamps 8B, source string ~16B, id 8B overhead)
    n * (dims * 4 + 32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn cosine_sim_self_is_one() {
        let v = vec![1.0f32, 2.0, 3.0, 4.0];
        let s = cosine_sim(&v, &v);
        assert!((s - 1.0).abs() < 1e-5, "self-similarity={s}");
    }

    #[test]
    fn cosine_sim_orthogonal_is_zero() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let s = cosine_sim(&a, &b);
        assert!(s.abs() < 1e-5, "orthogonal sim={s}");
    }

    #[test]
    fn corpus_generation_count() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let store = generate_memory_corpus(100, 32, 1_000_000, 5, &mut rng);
        assert_eq!(store.len(), 100);
    }

    #[test]
    fn recall_perfect() {
        let ids: Vec<MemoryId> = (0..10).collect();
        assert!((recall_at_k(&ids, &ids) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn recall_zero() {
        let retrieved: Vec<MemoryId> = (0..5).collect();
        let truth: Vec<MemoryId> = (5..10).collect();
        assert!(recall_at_k(&retrieved, &truth).abs() < 1e-5);
    }
}
