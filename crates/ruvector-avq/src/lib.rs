//! ruvector-avq — Anisotropic Vector Quantization (ScaNN-style).
//!
//! AVQ is a score-aware product quantizer. Standard PQ minimizes
//! reconstruction MSE; AVQ instead minimizes a weighted error that
//! penalizes the *parallel* component of the residual (along the
//! datapoint direction) more heavily than the orthogonal component.
//! For unit-norm queries, the parallel residual is exactly what
//! distorts the inner-product score, so anisotropic training yields
//! materially better recall at identical bit budgets.
//!
//! Reference: Guo et al., "Accelerating Large-Scale Inference with
//! Anisotropic Vector Quantization", ICML 2020 (ScaNN).
//!
//! This crate ships three swappable quantizer backends behind one
//! trait so callers can A/B them on real data:
//!   * `ScalarQuantizer`  — int8 per-dimension baseline.
//!   * `ProductQuantizer` — uniform-MSE PQ baseline.
//!   * `AnisotropicPq`    — score-aware PQ (this crate's contribution).

#![deny(unsafe_code)]

pub mod aniso;
pub mod error;
pub mod kmeans;
pub mod pq;
pub mod scalar;
pub mod traits;

pub use aniso::AnisotropicPq;
pub use error::AvqError;
pub use pq::ProductQuantizer;
pub use scalar::ScalarQuantizer;
pub use traits::{Encoder, Scorer};
