//! Topology-Aware Quantization (TAQ) for graph-indexed vector search.
//!
//! Key insight: in a k-NN graph, high-degree hub nodes lie on many shortest
//! paths. Quantization error at hubs corrupts more search paths than equal
//! error at low-degree leaf nodes. TAQ allocates SQ8 precision to hubs and
//! SQ4 to leaves, achieving recall close to full SQ8 at memory between SQ4
//! and SQ8.

pub mod graph;
pub mod index;
pub mod metrics;
pub mod quantize;

pub use index::{FullPrecisionIndex, TaqIndex, UniformSq4Index, UniformSq8Index, VectorIndex};
pub use metrics::recall_at_k;
