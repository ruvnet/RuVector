//! # ruvector-sparsifier
//!
//! Dynamic spectral graph sparsification for the RuVector ecosystem.
//!
//! This crate maintains a small weighted shadow graph **H** (the *sparsifier*)
//! that preserves the Laplacian energy of a full graph **G** within a factor
//! of `(1 +/- epsilon)`. It follows the ADKKP16 approach adapted for
//! practical real-time use:
//!
//! 1. **Backbone**: a spanning forest guaranteeing global connectivity.
//! 2. **Importance scoring**: random-walk-based effective-resistance estimation.
//! 3. **Spectral sampling**: edges kept proportional to `w * R_eff * log(n) / eps^2`.
//! 4. **Periodic audits**: random quadratic-form probes detect drift.
//!
//! ## Quick start
//!
//! ```rust
//! use ruvector_sparsifier::{AdaptiveGeoSpar, SparseGraph, SparsifierConfig, Sparsifier};
//!
//! // Build a graph.
//! let g = SparseGraph::from_edges(&[
//!     (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0),
//!     (3, 0, 1.0), (0, 2, 0.5),
//! ]);
//!
//! // Construct the sparsifier.
//! let mut spar = AdaptiveGeoSpar::build(&g, SparsifierConfig::default()).unwrap();
//!
//! // Dynamic updates.
//! spar.insert_edge(1, 3, 2.0).unwrap();
//! spar.delete_edge(0, 2).unwrap();
//!
//! // Audit quality.
//! let audit = spar.audit();
//! println!("Audit passed: {}, max error: {:.4}", audit.passed, audit.max_error);
//!
//! // Access the compressed graph.
//! let h = spar.sparsifier();
//! println!("Compression: {:.1}x ({} -> {} edges)",
//!     spar.compression_ratio(),
//!     spar.stats().full_edge_count,
//!     h.num_edges(),
//! );
//! ```
//!
//! ## Feature flags
//!
//! | Flag              | Default | Description                              |
//! |--------------------|---------|------------------------------------------|
//! | `static-sparsify` | yes     | One-shot static sparsification           |
//! | `dynamic`         | yes     | Dynamic insert/delete support            |
//! | `simd`            | no      | SIMD-accelerated distance operations     |
//! | `wasm`            | no      | WebAssembly-compatible paths             |
//! | `audit`           | no      | Extended audit & diagnostics             |
//! | `full`            | no      | All features                             |

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

pub mod audit;
pub mod backbone;
pub mod error;
pub mod graph;
pub mod importance;
pub mod sampler;
pub mod sparsifier;
pub mod traits;
pub mod types;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use audit::SpectralAuditor;
pub use backbone::Backbone;
pub use error::{Result, SparsifierError};
pub use graph::SparseGraph;
pub use importance::{EffectiveResistanceEstimator, LocalImportanceScorer};
pub use sampler::SpectralSampler;
pub use sparsifier::AdaptiveGeoSpar;
pub use traits::{BackboneStrategy, ImportanceScorer, Sparsifier};
pub use types::{AuditResult, EdgeImportance, SparsifierConfig, SparsifierStats};

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Crate name.
pub const NAME: &str = env!("CARGO_PKG_NAME");

/// Prelude for convenient imports.
///
/// Re-exports of the most commonly used types and traits.
pub mod prelude {
    pub use crate::{
        AdaptiveGeoSpar, AuditResult, Backbone, BackboneStrategy, EdgeImportance,
        EffectiveResistanceEstimator, ImportanceScorer, LocalImportanceScorer, SparseGraph,
        Sparsifier, SparsifierConfig, SparsifierError, SparsifierStats, SpectralAuditor,
        SpectralSampler,
    };
}
