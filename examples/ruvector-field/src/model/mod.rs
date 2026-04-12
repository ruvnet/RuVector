//! Shared data model for the RuVector field subsystem.
//!
//! Mirrors section 6 of `docs/research/ruvector-field/SPEC.md`. This module
//! re-exports the leaf types so call sites can `use ruvector_field::model::*`.

pub mod edge;
pub mod embedding;
pub mod ids;
pub mod node;
pub mod shell;

pub use edge::{EdgeKind, FieldEdge};
pub use embedding::{Embedding, EmbeddingId, EmbeddingStore};
pub use ids::{EdgeId, HintId, NodeId, WitnessCursor};
pub use node::{AxisScores, FieldNode, NodeKind};
pub use shell::Shell;

/// Clamp a float into `[0, 1]`.
pub fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}
