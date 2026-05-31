//! ruvector-codeq — CoDEQ: Consistent Dynamic Efficient Quantizer
//!
//! Implements the streaming quantization algorithm introduced in:
//! > Aden-Ali et al., "Quantization for Vector Search under Streaming Updates",
//! > arXiv:2512.18335, December 2025.
//!
//! ## The problem CoDEQ solves
//!
//! Existing quantization methods (PQ, OPQ, ScaNN) build their codebooks
//! offline via k-means. A single-point insert/delete invalidates cluster
//! centroids and, in the worst case, requires a full rebuild — O(n · k · iters)
//! work per update. Production vector stores continuously ingest data: a rebuild
//! on every insert is infeasible.
//!
//! ## CoDEQ's solution
//!
//! Replace k-means with **median-based kd-tree partitioning**:
//!
//! 1. Apply a random rotation to de-correlate dimensions.
//! 2. Recursively split on the dimension of maximum variance at the empirical
//!    median — creating 2^L leaf cells with L-bit codes.
//! 3. The tree *structure* (split dims + split values) is frozen at build time.
//!    Only leaf centroids change when points are inserted/deleted.
//! 4. Per the paper's Theorem 4.1, each insert/delete touches exactly ONE leaf,
//!    updating its mean via Welford's online formula in O(D) work — O(1) I/Os
//!    in the disk-backed model.
//!
//! ## This crate's three variants
//!
//! | Struct | Description |
//! |--------|-------------|
//! | `FlatL2Index` | Brute-force exact search — memory bandwidth baseline |
//! | `StaticPqIndex` | k-means PQ with frozen codebook — standard benchmark |
//! | `CoDEQIndex` | kd-tree quantizer — ADC search + streaming insert/delete |
//!
//! ## Measured results (x86_64, cargo --release, n=5K, D=128)
//!
//! See `src/main.rs` (`codeq-demo`) for reproducible numbers.

pub mod dist;
pub mod error;
pub mod kdquant;
pub mod pq_baseline;

pub use error::CoDEQError;
pub use kdquant::{CoDEQIndex, KdQuantizer, Rotation, recall_at_k};
pub use pq_baseline::{StaticPqIndex, FlatL2IndexCoDEQ};
