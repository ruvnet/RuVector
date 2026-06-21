//! # ruvector-coherence-hnsw
//!
//! Coherence-gated beam search on a flat proximity graph.
//!
//! Standard beam search expands every candidate's neighbors unconditionally.
//! This crate adds a **traversal-direction coherence gate**: before expanding
//! a candidate's neighbors, we check whether the candidate lies roughly
//! *toward* the query from the search entry point. If not, we skip its
//! neighborhood while still considering it as a result.
//!
//! ## Variants
//!
//! | Type | Threshold | Description |
//! |------|-----------|-------------|
//! | [`BaselineSearch`] | N/A | Standard beam search — all neighbors expanded |
//! | [`CoherenceGatedSearch`] | fixed | Skip neighbors when coherence < threshold |
//! | [`AdaptiveCoherenceSearch`] | dynamic | Raise threshold as best result improves |
//!
//! ## Relationship to HNSW
//!
//! Full HNSW has multiple layers and an elaborate entry selection procedure.
//! This crate operates on a single-layer flat k-NN proximity graph — exactly
//! what HNSW's layer-0 looks like. The coherence gating innovation applies
//! unchanged to multi-layer HNSW; the flat graph keeps the PoC self-contained.
//!
//! ## Agent memory context
//!
//! Agent memory graphs embed memories as vectors with proximity edges.
//! Coherence gating skips memories that are "directionally off" from the
//! current query, reducing retrieval noise in long-lived agent memory stores.

pub mod coherence;
pub mod dataset;
pub mod graph;
pub mod metrics;
pub mod search;

pub use graph::{FlatGraph, GraphConfig};
pub use search::{
    AdaptiveCoherenceSearch, BaselineSearch, CoherenceGatedSearch, SearchResult, Searcher,
};
