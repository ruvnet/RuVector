//! EML-powered HNSW optimizations for ruvector.
//!
//! Six learned improvements using the EML function approximator:
//!
//! ## Distance & Search Optimizations
//!
//! - [`EmlDistanceModel`] — Cosine decomposition: learn which dimensions
//!   matter for fast approximate distance (10-30x distance speed).
//! - [`ProgressiveDistance`] — Layer-aware dimensionality: use fewer
//!   dimensions at higher HNSW layers (5-20x search speed).
//! - [`AdaptiveEfModel`] — Learn optimal beam width (ef) per query
//!   to avoid wasting work on easy queries (1.5-3x search speed).
//!
//! ## Structural Optimizations
//!
//! - [`SearchPathPredictor`] — Skip top-layer traversal by predicting
//!   entry points from query region (2-5x search speed).
//! - [`RebuildPredictor`] — Predict when an HNSW index rebuild is needed
//!   based on graph statistics (operational efficiency).
//! - [`PqDistanceCorrector`] — Correct PQ distance approximation error
//!   for DiskANN product quantization (improved recall).

pub mod adaptive_ef;
pub mod cosine_decomp;
pub mod path_predictor;
pub mod progressive_distance;
pub mod pq_corrector;
pub mod rebuild_predictor;

pub use adaptive_ef::AdaptiveEfModel;
pub use cosine_decomp::{cosine_distance_f32, EmlDistanceModel};
pub use path_predictor::SearchPathPredictor;
pub use progressive_distance::ProgressiveDistance;
pub use pq_corrector::PqDistanceCorrector;
pub use rebuild_predictor::{GraphStats, RebuildPredictor};
