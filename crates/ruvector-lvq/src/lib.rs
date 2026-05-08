//! Locally-Adaptive Vector Quantization (LVQ) for ruvector.
//!
//! LVQ is a per-vector scalar quantization scheme used by Intel's Scalable
//! Vector Search (Aguerrebere et al., VLDB 2024). Each database vector is
//! independently centered, then linearly mapped into a low-bit code with a
//! per-vector `(bias, scale)` pair. Queries stay in fp32 and distances are
//! computed *asymmetrically* against the decoded database vectors — yielding
//! ~4x memory reduction over fp32 with near-zero recall loss when paired
//! with a residual second level (LVQ-Bx8).
//!
//! This crate exposes:
//!   * [`Lvq8`]  — single-level 8-bit primary quantizer
//!   * [`Lvq8x8`] — two-level 8+8 bit (primary + residual) quantizer
//!   * [`FlatLvqIndex`] — brute-force index with reranking-friendly API
//!
//! All types are pure-Rust, `#![forbid(unsafe_code)]`, and produce identical
//! results across architectures (no platform-dependent SIMD intrinsics).

#![forbid(unsafe_code)]

pub mod distance;
pub mod error;
pub mod index;
pub mod quantize;
pub mod two_level;

pub use error::LvqError;
pub use index::{FlatF32, FlatLvqIndex, IndexKind, SearchHit};
pub use quantize::{Lvq8, Lvq8Code, Lvq8Stats};
pub use two_level::Lvq8x8;
