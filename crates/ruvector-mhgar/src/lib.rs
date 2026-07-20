//! Multi-Hop Graph-Anchored Retrieval (MHGAR)
//!
//! Combines approximate vector search with coherence-gated graph traversal.
//! Three retrieval variants enable head-to-head recall/latency comparison:
//!
//! - [`VectorOnlyRetriever`]: pure ANN baseline (no graph)
//! - [`OneHopExpander`]: ANN + single-hop graph expansion + rerank
//! - [`CoherenceGatedHopper`]: ANN + adaptive multi-hop with coherence stopping
//!
//! The fundamental hypothesis: for structured knowledge graphs, some true answers
//! to a query are graph-neighbors of vector-similar entities, not directly in
//! the vector neighborhood themselves. Graph traversal recovers these.

pub mod coherence;
pub mod graph;
pub mod retriever;
pub mod synth;
pub mod vector;

pub use retriever::{CoherenceGatedHopper, OneHopExpander, Retriever, VectorOnlyRetriever};

/// A retrieved result with entity id, distance, and traversal metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub id: u32,
    pub distance: f32,
    /// How many graph hops from the original vector-search candidate.
    pub hops: u32,
}

impl Hit {
    pub fn new(id: u32, distance: f32, hops: u32) -> Self {
        Self { id, distance, hops }
    }
}

/// Errors produced by MHGAR components.
#[derive(Debug, thiserror::Error)]
pub enum MhgarError {
    #[error("entity {0} not found in index")]
    EntityNotFound(u32),
    #[error("dimension mismatch: index has {index}D, query has {query}D")]
    DimensionMismatch { index: usize, query: usize },
    #[error("index is empty")]
    EmptyIndex,
}

/// Compute recall@k: fraction of ground-truth ids present in the top-k results.
pub fn recall_at_k(results: &[Hit], ground_truth: &[u32]) -> f32 {
    if ground_truth.is_empty() {
        return 1.0;
    }
    let k = ground_truth.len();
    let found: usize = results.iter().take(k).filter(|h| ground_truth.contains(&h.id)).count();
    found as f32 / k as f32
}
