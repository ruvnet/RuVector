//! # ruvector-diskann
//!
//! DiskANN/Vamana implementation for billion-scale approximate nearest neighbor search.
//!
//! ## Algorithm
//! - **Vamana graph**: greedy search + α-robust pruning for bounded out-degree
//! - **Product Quantization (PQ)**: compressed distance for candidate filtering
//! - **Memory-mapped graph**: SSD-friendly access, only load neighbors on demand
//!
//! ## Reference
//! Subramanya et al., "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node" (NeurIPS 2019)

pub mod distance;
pub mod error;
pub mod graph;
pub mod index;
pub mod pq;
/// Fixed-topology reuse + periodic rebuild under metric drift (BET 1, ADR-200).
#[cfg(feature = "reuse-under-drift")]
pub mod reuse;

pub use error::{DiskAnnError, Result};
pub use index::{DiskAnnConfig, DiskAnnIndex};
pub use pq::ProductQuantizer;
#[cfg(feature = "reuse-under-drift")]
pub use reuse::{DriftingIndex, RebuildPolicy, RecallTrigger};
