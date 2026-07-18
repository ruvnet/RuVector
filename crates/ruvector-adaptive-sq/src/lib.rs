//! # ruvector-adaptive-sq
//!
//! Adaptive scalar quantization with coherence-precision routing for
//! approximate nearest-neighbour (ANN) vector indices.
//!
//! ## Core idea
//!
//! Classic scalar quantization applies the same bit depth uniformly to all
//! stored vectors.  This crate routes each vector to either 8-bit or 16-bit
//! quantization based on its *local neighbourhood density*:
//!
//! - Vectors in **dense (contested) regions** — many close neighbours — are
//!   stored at 16-bit.  Quantization noise in these regions directly
//!   corrupts nearest-neighbour rank ordering.
//! - Vectors in **sparse regions** are stored at 8-bit.  Quantization noise
//!   here matters far less because true neighbours are already far apart.
//!
//! The routing decision is made once at index build time by computing the
//! mean kNN distance (density score) for each vector.  Vectors with a
//! density score below `mean × factor` are routed to the high-precision tier.
//!
//! ## Variants
//!
//! | Variant       | Tier         | Memory     | Purpose                |
//! |---------------|--------------|------------|------------------------|
//! | `Uniform8`    | 8-bit only   | 1× N·D B   | Baseline               |
//! | `Uniform16`   | 16-bit only  | 2× N·D B   | Quality upper bound    |
//! | `AdaptiveSQ`  | Mixed 8/16   | ~1.2–1.4× N·D B | Coherence-routed |
//!
//! ## Usage
//!
//! ```rust
//! use ruvector_adaptive_sq::{AdaptiveSqIndex, SqIndex};
//!
//! // Build an adaptive index on your f32 vectors.
//! let vectors: Vec<Vec<f32>> = (0..1000)
//!     .map(|i| vec![i as f32 * 0.001, -(i as f32) * 0.001])
//!     .collect();
//! let idx = AdaptiveSqIndex::build(&vectors, 2, 10, 0.6);
//!
//! // Query returns (id, approx_l2_sq) pairs, ascending.
//! let results = idx.search(&[0.5, -0.5], 5);
//! println!("HP ratio: {:.1}%", idx.hp_ratio() * 100.0);
//! ```

pub mod coherence;
pub mod dataset;
pub mod index;
pub mod quantizer;

pub use index::{AdaptiveSqIndex, SqIndex, Tier, Uniform16Index, Uniform8Index};
