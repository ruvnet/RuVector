//! # ruvector-slipstream
//!
//! Warm-start streaming HNSW insertions exploiting inter-arrival spatial locality.
//!
//! Standard HNSW insertion always begins graph traversal from a global entry
//! point.  For random batches that is correct, but **real agent-memory streams
//! have spatial locality**: consecutive observations, RAG chunks, and embeddings
//! from a single task tend to cluster.  Slipstream exploits this by reusing the
//! candidate set discovered during the previous insertion as the starting point
//! for the next one, dramatically reducing the traversal distance when the stream
//! stays in the same neighbourhood.
//!
//! ## Variants
//!
//! | Name | Strategy | Description |
//! |------|----------|-------------|
//! | [`EntryPoint`] | Baseline | Always traverse from node 0 |
//! | [`FixedCache`] | Warm-start | Seed from previous insert's discovered set |
//! | [`Adaptive`]   | Drift-aware | Reset cache when stream drifts; expand when stable |
//!
//! ## Algorithm (Slipstream PoC)
//!
//! 1. Insert vector v from seed candidates C (empty → use global entry node 0).
//! 2. Run ef-bounded beam search from C to find M nearest already-inserted nodes.
//! 3. Add bidirectional links between v and those M nodes (with degree pruning).
//! 4. Store the full ef-size discovered candidate set as C for the next insert.
//! 5. Adaptive variant: track cosine drift between consecutive vectors via an EMA.
//!    When drift exceeds θ the stream has shifted; reset C to avoid stale seeds.
//!
//! ## Connection to RuVector ecosystem
//!
//! - **Agent memory**: agents write observations in temporal bursts; Slipstream
//!   converts that locality into faster index builds without structural changes.
//! - **ruFlo**: a workflow driver can pass `stream_locality_hint` through the
//!   insert API to enable or disable caching per batch.
//! - **MCP tools**: a `vector_insert_stream` MCP tool can expose warm-start
//!   as a flag, enabling local-first inference loops to build indexes efficiently.
//! - **RVF packages**: a serialised `SlipstreamIndex` maps naturally onto the
//!   RVF portable cognitive package format.

pub mod dataset;
pub mod graph;
pub mod metrics;
pub mod slipstream;

pub use graph::{FlatGraph, GraphConfig};
pub use slipstream::{InsertStrategy, SlipstreamIndex, StreamStats};
