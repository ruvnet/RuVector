//! Residual Vector Quantization (RVQ) for approximate nearest-neighbor search.
//!
//! RVQ encodes each vector as a cascade of M quantizers where each stage
//! quantizes the residual error from the previous stage. Unlike Product
//! Quantization (PQ), which partitions dimensions, RVQ operates on the
//! full-dimensional residual at every level, yielding lower distortion at
//! the same bit budget — especially for high-dimensional embeddings.
//!
//! ## Architecture
//!
//! | Variant            | Encoding        | Scoring        | Recall | QPS   |
//! |--------------------|-----------------|----------------|--------|-------|
//! | `RvqGreedyIndex`   | greedy (beam=1) | ADC table      | good   | fast  |
//! | `RvqBeamIndex`     | beam search     | ADC table      | better | fast  |
//! | `RvqRerankIndex`   | greedy + rerank | ADC + exact L2 | best   | med   |
//!
//! All three share the `AnnIndex` trait for transparent swapping.
//!
//! ## Usage
//!
//! ```rust
//! use ruvector_residual_vq::{RvqEncoder, RvqGreedyIndex, AnnIndex};
//!
//! let vecs: Vec<Vec<f32>> = vec![vec![0.1, 0.2, 0.3, 0.4]; 100];
//! let dim = 4;
//! let n_codebooks = 2;
//! let k = 16;
//! let encoder = RvqEncoder::train(&vecs, n_codebooks, k, 10, 42);
//! let mut idx = RvqGreedyIndex::build_with_encoder(encoder, &vecs);
//! let results = idx.search(&[0.1, 0.2, 0.3, 0.4], 5);
//! ```

pub mod codebook;
pub mod error;
pub mod rvq;

pub use codebook::Codebook;
pub use error::RvqError;
pub use rvq::{AnnIndex, RvqBeamIndex, RvqEncoder, RvqGreedyIndex, RvqRerankIndex, SearchResult};
