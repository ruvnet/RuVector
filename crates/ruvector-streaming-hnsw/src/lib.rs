//! Streaming HNSW: concurrent-insert k-NN graph for ruvector nightly research.
//!
//! Implements three ANN index variants:
//!
//! | Struct | Strategy | Dynamic? | Notes |
//! |---|---|---|---|
//! | `FlatL2Index` | Brute force | Yes | O(n·D) per query; used as baseline |
//! | `StaticHnsw` | Offline greedy k-NN | No | Best recall; immutable after build |
//! | `StreamingHnsw` | Online RwLock-per-node | Yes | Concurrent inserts + queries |
//!
//! ## Motivation
//!
//! All existing ruvector HNSW variants (ruvector-core, ruvector-acorn, ruvector-rabitq)
//! are build-once-query-many. Production vector stores continuously ingest new
//! documents. `StreamingHnsw` solves this by wrapping each node's neighbor list in
//! an `Arc<parking_lot::RwLock>`, allowing inserts and queries to proceed concurrently
//! without a full rebuild.

pub mod dist;
pub mod error;
pub mod index;

pub use error::HnswError;
pub use index::{AnnIndex, FlatL2Index, StaticHnsw, StreamingHnsw, recall_at_k};
