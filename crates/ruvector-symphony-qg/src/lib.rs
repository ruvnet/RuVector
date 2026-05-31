//! SymphonyQG: Co-located RaBitQ codes + FastScan batch distance estimation
//! on graph-based approximate nearest-neighbor search.
//!
//! Based on: "SymphonyQG: Towards Symphonious Integration of Quantization
//! and Graph for Approximate Nearest Neighbor Search"
//! (Gou et al., SIGMOD 2025, arXiv:2411.12229)
//!
//! ## Key innovations over vanilla HNSW
//!
//! 1. **Co-located layout**: each vertex stores its R neighbors' RaBitQ codes
//!    in a single contiguous heap block alongside their IDs. One sequential
//!    read gives all R neighbor distances — no random pointer chasing.
//!
//! 2. **Batch asymmetric distance (FastScan)**: the R neighbor codes are
//!    processed in a single pass using u64 XNOR+popcount, yielding O(R·D/64)
//!    work per hop instead of O(R·D) for exact float computation.
//!
//! 3. **Reranking-free termination**: RaBitQ's unbiased estimator with bounded
//!    variance allows the beam search to terminate safely without a separate
//!    re-ranking pass over the top-ef candidates.
//!
//! ## Memory layout per vertex (D=128, R=16)
//!
//! ```text
//! [raw_f32: 512 B][neighbor_codes: 256 B][neighbor_norms: 64 B][ids: 64 B]
//!  ──── sequential ─────────────────────────────────────────────────────────
//!  Total: 896 B vs vanilla HNSW 512+64 B + R×512 B random reads = 8768 B
//! ```

pub mod codes;
pub mod error;
pub mod graph;
pub mod index;
pub mod rotation;
pub mod search;

pub use error::SymphonyError;
pub use graph::GraphConfig;
pub use index::{AnnIndex, FlatF32Index, GraphExact, SearchResult, SymphonyIndex};
