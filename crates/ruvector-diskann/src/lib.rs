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
pub mod graph;
pub mod pq;
pub mod index;
pub mod error;

pub use index::{DiskAnnIndex, DiskAnnConfig};
pub use error::{DiskAnnError, Result};
pub use pq::ProductQuantizer;
