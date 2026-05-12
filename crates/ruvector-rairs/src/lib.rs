//! # ruvector-rairs — IVF with Redundant Assignment + Amplified Inverse Residual
//!
//! Implements three variants from Yang & Chen, "RAIRS: Optimizing Redundant Assignment
//! and List Layout for IVF-Based ANN Search", SIGMOD 2026 (arXiv:2601.07183).
//!
//! ## Index family
//!
//! | Variant        | Assignment | Layout | Description                             |
//! |----------------|------------|--------|-----------------------------------------|
//! | `IvfFlat`      | single     | flat   | baseline — one list per vector          |
//! | `RairsStrict`  | dual RAIR  | flat   | secondary assignment, no dedup          |
//! | `RairsSeil`    | dual RAIR  | SEIL   | full RAIRS: blocks shared, dedup hash   |
//!
//! All three satisfy [`AnnIndex`].

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
pub mod index;
pub mod ivf;
pub mod kmeans;
pub mod rairs;
pub mod seil;

pub use error::RairsError;
pub use index::{AnnIndex, SearchResult};
pub use ivf::IvfFlat;
pub use rairs::RairsStrict;
pub use seil::RairsSeil;
