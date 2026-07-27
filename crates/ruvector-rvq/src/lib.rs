//! # ruvector-rvq — Residual Vector Quantization for ANN search
//!
//! Multi-stage codebook compression where each stage quantizes the residual
//! error from the previous stage.  Achieves higher recall at the same byte
//! budget compared to flat Product Quantization (PQ).
//!
//! ## Index types
//!
//! | Type | Description | Bytes/vec |
//! |---|---|---|
//! | `FlatF32Index` | Exact brute-force L2 | D × 4 |
//! | `PqIndex` | Standard product quantization | M × 1 |
//! | `RvqIndex` | Residual vector quantization | S × 1 |
//! | `RvqRerankIndex` | RVQ + exact rerank of top candidates | S × 1 + orig |
//!
//! All four implement [`AnnIndex`] for uniform benchmarking.
//!
//! ## Quick start
//!
//! ```no_run
//! use ruvector_rvq::{RvqConfig, index::{AnnIndex, RvqIndex}};
//!
//! let data: Vec<Vec<f32>> = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
//! let cfg = RvqConfig { dim: 3, num_stages: 2, codebook_size: 4, train_iters: 10, dropout_prob: 0.1 };
//! let idx = RvqIndex::build_with_config(cfg, data).unwrap();
//! let results = idx.search(&[1.0, 0.1, 0.0], 1);
//! ```

#![allow(clippy::needless_range_loop)]

pub mod codebook;
pub mod index;
pub mod rvq;

pub use index::AnnIndex;

/// Configuration for the RVQ encoder.
#[derive(Debug, Clone)]
pub struct RvqConfig {
    /// Dimensionality of input vectors.
    pub dim: usize,
    /// Number of residual stages (M).  Each stage adds 1 byte per vector.
    pub num_stages: usize,
    /// Centroids per stage (K).  Must be ≤ 256 so codes fit in u8.
    pub codebook_size: usize,
    /// K-means Lloyd iterations per stage.
    pub train_iters: usize,
    /// Probability of zeroing a stage's code during training to prevent
    /// codebook collapse (codebook dropout from DAC 2023, arXiv:2306.06546).
    pub dropout_prob: f32,
}

/// A scored candidate returned by ANN search.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub id: usize,
    pub distance: f32,
}
