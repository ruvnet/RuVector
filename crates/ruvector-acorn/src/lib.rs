//! # ruvector-acorn
//!
//! **ACORN**: Predicate-Agnostic Filtered Approximate Nearest-Neighbour Search
//!
//! Based on the SIGMOD 2024 paper *"ACORN: Performant and Predicate-Agnostic
//! Search Over Vector Embeddings and Structured Data"* (Patel et al., 2024).
//!
//! ## Problem
//!
//! All modern vector databases support *metadata filters* (e.g. "find the 10
//! nearest products in category='electronics' with price < 50").  The naive
//! strategies both fail under selective filters:
//!
//! | Strategy | Issue |
//! |---|---|
//! | **PostFilter** — search all, then filter | most results are discarded; recall drops below k |
//! | **PreFilter** — filter first, then scan remainder | degenerates to brute-force at 1 % selectivity |
//! | **ACORN-γ** — navigate ALL nodes, count only filter-passing ones | remains connected; fast + high recall |
//!
//! ## Key Algorithm
//!
//! 1. **Build phase** — construct a Navigable Small-World graph (flat HNSW base
//!    layer) with `M` bidirectional edges per node.
//! 2. **Neighbour compression** — expand each node's adjacency list to
//!    `M × γ` edges by including neighbours-of-neighbours.  Guarantees that
//!    for any predicate, the induced subgraph of passing nodes remains
//!    navigable.
//! 3. **Query phase** — greedy beam search (`ef` candidates) that visits ALL
//!    neighbours for graph traversal but only counts predicate-passing nodes
//!    towards the result heap.
//!
//! ## Index types
//!
//! | Type | Use case |
//! |---|---|
//! | [`AcornIndex`] (γ=1, no compression) | Baseline — same recall as PostFilter but no wasted traversal |
//! | [`AcornIndex`] (γ=2) | Recommended — significant recall improvement at ≤10 % selectivity |
//!
//! ## Quick start
//!
//! ```rust
//! use ruvector_acorn::{AcornIndex, AcornConfig, SearchVariant};
//!
//! let cfg = AcornConfig { dim: 4, m: 8, gamma: 2, ef_construction: 32 };
//! let mut idx = AcornIndex::new(cfg);
//!
//! for i in 0u32..100 {
//!     let v = vec![i as f32, 0.0, 0.0, 0.0];
//!     idx.insert(i, v);
//! }
//!
//! idx.build_compression();
//!
//! // find 5 nearest with id < 50
//! let results = idx.search(
//!     &[10.0, 0.0, 0.0, 0.0],
//!     5,
//!     64,
//!     |id| id < 50,
//!     SearchVariant::AcornGamma,
//! ).unwrap();
//! assert!(!results.is_empty());
//! ```

pub mod error;
pub mod graph;
pub mod index;

pub use error::{AcornError, Result};
pub use index::{AcornConfig, AcornIndex, SearchResult, SearchVariant};
