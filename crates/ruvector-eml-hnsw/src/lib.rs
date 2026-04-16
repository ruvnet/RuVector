//! EML-powered HNSW optimizations for ruvector — v2 integrated pipeline.
//!
//! This crate ships a learned candidate-prefilter HNSW pipeline plus a small
//! catalog of supporting models. The accepted production path is:
//!
//! 1. **Train** a selector (Pearson *or* retention-objective) on a
//!    representative sample. See [`EmlDistanceModel::train`] (Pearson) and
//!    [`EmlDistanceModel::train_for_retention`] (recall-maximizing greedy
//!    forward selection — +10.5 pp recall@10 on SIFT1M vs Pearson).
//! 2. **Build** one of:
//!      - [`hnsw_integration::EmlHnsw`] — single reduced-dim HNSW + exact
//!        re-rank. The baseline pipeline.
//!      - [`progressive_hnsw::ProgressiveEmlHnsw`] — multi-level cascade
//!        (e.g. `[8, 32, 128]` dim schedule) + exact re-rank. Higher recall
//!        at matched latency, 5–6× build cost. For read-heavy workloads.
//!      - [`pq_hnsw::PqEmlHnsw`] — Product-Quantized codes + exact re-rank.
//!        64× memory reduction on SIFT1M at recall ≥ 0.95 after re-rank.
//! 3. **Search** with `search_with_rerank(query, k, fetch_k, ef)`. Default
//!    re-rank kernel is [`selected_distance::cosine_distance_simd`] via
//!    SimSIMD (5.6-6.2× faster than the scalar reference).
//!
//! See `docs/adr/ADR-151-eml-hnsw-selected-dims.md` for the acceptance
//! matrix and per-tier measurements on SIFT1M 50k / 200 queries.
//!
//! # Catalog of supporting models
//!
//! - [`EmlDistanceModel`] — dimension selector (Pearson or retention).
//! - [`ProgressiveDistance`] — per-layer dim schedule used by `ProgressiveEmlHnsw`.
//! - [`AdaptiveEfModel`] — per-query beam-width predictor. Not wired into any
//!   HNSW path in this crate — see ADR-151 §Rejected Surface.
//! - [`SearchPathPredictor`] — cached entry-point predictor. Reference impl only.
//! - [`RebuildPredictor`] — predicts recall degradation to trigger rebuild.
//! - [`PqDistanceCorrector`] — advisory-only PQ error corrector. Has a
//!   normalization design flaw under SIFT's distance scale (see ADR-151).

pub mod adaptive_ef;
pub mod cosine_decomp;
pub mod hnsw_integration;
pub mod path_predictor;
pub mod pq;
pub mod pq_corrector;
pub mod pq_hnsw;
pub mod progressive_distance;
pub mod progressive_hnsw;
pub mod rebuild_predictor;
pub mod selected_distance;

pub use adaptive_ef::AdaptiveEfModel;
pub use cosine_decomp::{cosine_distance_f32, EmlDistanceModel};
pub use hnsw_integration::{EmlHnsw, EmlMetric, EmlSearchResult};
pub use path_predictor::SearchPathPredictor;
pub use pq::PqCodebook;
pub use pq_corrector::PqDistanceCorrector;
pub use pq_hnsw::PqEmlHnsw;
pub use progressive_distance::ProgressiveDistance;
pub use progressive_hnsw::ProgressiveEmlHnsw;
pub use rebuild_predictor::{GraphStats, RebuildPredictor};
pub use selected_distance::{
    cosine_distance_selected, cosine_distance_simd, project_batch, project_vector,
    sq_euclidean_selected,
};
