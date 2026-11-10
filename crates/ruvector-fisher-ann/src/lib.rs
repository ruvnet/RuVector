//! Fisher-weighted ANN for EWC++ policy entries.
//!
//! ADR-002 ("ruvllm integration") sketches a `PolicyEntry` with an optional
//! `fisher_diagonal: Option<Vec<f32>>` living in a shared HNSW index
//! alongside session state and witness logs, but never specifies whether
//! that Fisher diagonal should change how the index *searches* — whether
//! "nearest prior task" lookup should be Fisher-weighted or plain
//! cosine/L2 with Fisher carried only as inert metadata.
//!
//! This crate is a nightly research experiment (2026-10-26,
//! `docs/research/nightly/2026-10-26-fisher-weighted-ewc-ann/`) that
//! answers that question: it generalizes `EwcPlusPlus`'s existing
//! `task_memory` linear scan (production code today, bounded to
//! `max_tasks`, typically 10) to large N, and measures whether an
//! approximate index built with a Fisher-weighted metric recovers the
//! brute-force-exact nearest task at a fraction of the latency, versus a
//! plain-L2 index over the same points.
//!
//! See [`baseline::brute_force_topk`] (ground truth + production-today
//! baseline), [`hnsw::HnswIndex`] (the two ANN candidates, parameterized
//! by [`types::TaskMetric`]), and [`dataset`] (deterministic synthetic
//! task/query generation).

pub mod baseline;
pub mod dataset;
pub mod hnsw;
pub mod types;

pub use baseline::brute_force_topk;
pub use hnsw::HnswIndex;
pub use types::{single_task_regularization_loss, FisherWeighted, PlainL2, TaskMetric, TaskRecord};
