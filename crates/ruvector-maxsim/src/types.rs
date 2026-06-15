//! Core types for multi-vector MaxSim late interaction search.

use serde::{Deserialize, Serialize};

/// Opaque document identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DocId(pub u64);

/// A single embedding vector stored as f32.
pub type Embedding = Vec<f32>;

/// A document represented by one or more token/chunk embeddings.
///
/// Each entry in `vecs` is a separate embedding: a sentence, a paragraph
/// chunk, or a ColBERT-style token projection. Similarity is computed with
/// MaxSim aggregation rather than averaging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiVecDoc {
    pub id: DocId,
    pub vecs: Vec<Embedding>,
}

/// A query likewise represented by one or more token embeddings.
#[derive(Debug, Clone)]
pub struct MultiVecQuery {
    pub vecs: Vec<Embedding>,
}

/// One ranked result returned from a MaxSim search.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub doc_id: DocId,
    /// Sum of per-query-token max cosine similarities over all document tokens.
    pub score: f32,
}

impl Eq for SearchResult {}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher score = better rank (reverse for BinaryHeap)
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Statistics from a benchmark or search run.
#[derive(Debug, Clone, Default)]
pub struct RunStats {
    pub variant: String,
    pub n_docs: usize,
    pub n_token_vecs: usize,
    pub dims: usize,
    pub n_queries: usize,
    pub mean_latency_us: f64,
    pub p50_latency_us: f64,
    pub p95_latency_us: f64,
    pub throughput_qps: f64,
    pub recall_at_k: f64,
    pub memory_bytes: usize,
}
