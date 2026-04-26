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
pub mod quantize;

pub use error::{DiskAnnError, Result};
pub use index::{DiskAnnConfig, DiskAnnIndex, QuantizerKind};
pub use quantize::{ProductQuantizer, Quantizer};

#[cfg(feature = "rabitq")]
pub use quantize::RabitqQuantizer;

/// Backwards-compatible alias for the pre-quantize-module module path.
/// Existing callers that did `use ruvector_diskann::pq::ProductQuantizer;`
/// keep working without code changes. New code should prefer
/// `ruvector_diskann::quantize::ProductQuantizer`.
pub mod pq {
    pub use crate::quantize::pq::*;
}
