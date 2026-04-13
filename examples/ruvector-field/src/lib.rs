//! RuVector Field Subsystem — reference implementation.
//!
//! Implements the full specification at `docs/research/ruvector-field/SPEC.md`:
//! four logical shells, geometric and semantic antipodes, multiplicative
//! resonance scoring, four channel drift detection, policy aware retrieval,
//! shell promotion with hysteresis, routing hints with proof gating, and a
//! witness log of every committed mutation.
//!
//! The crate is `std`-only: no external dependencies, no `async`, no `unsafe`.
//! Everything is built to be read end-to-end.
//!
//! # Quick tour
//!
//! ```
//! use ruvector_field::prelude::*;
//!
//! let mut engine = FieldEngine::new();
//! let provider = HashEmbeddingProvider::new(16);
//! let embedding = provider.embed("user reports authentication timeout");
//! let axes = AxisScores::new(0.7, 0.6, 0.5, 0.8);
//! let id = engine
//!     .ingest(NodeKind::Interaction, "user reports timeout", embedding, axes, 0b0001)
//!     .unwrap();
//! assert!(engine.node(id).is_some());
//! ```

#![deny(unused_must_use)]
#![allow(clippy::too_many_arguments)]

pub mod clock;
pub mod embed;
#[cfg(feature = "onnx-embeddings")]
pub mod embed_onnx;
pub mod engine;
pub mod error;
pub mod model;
pub mod policy;
pub mod proof;
pub mod scoring;
pub mod storage;
pub mod witness;

/// Re-exports for the common case.
pub mod prelude {
    pub use crate::clock::{Clock, SystemClock, TestClock};
    pub use crate::embed::{EmbeddingProvider, HashEmbeddingProvider};
    pub use crate::engine::{FieldEngine, FieldEngineConfig, PromotionReason, PromotionRecord};
    pub use crate::error::FieldError;
    pub use crate::model::{
        AxisScores, EdgeId, EdgeKind, Embedding, EmbeddingId, EmbeddingStore, FieldEdge,
        FieldNode, HintId, NodeId, NodeKind, Shell, WitnessCursor,
    };
    pub use crate::policy::{AxisConstraint, AxisConstraints, Policy, PolicyRegistry};
    pub use crate::proof::{ManualProofGate, NoopProofGate, ProofError, ProofGate, ProofToken};
    pub use crate::scoring::{DriftSignal, RetrievalResult, RoutingHint};
    pub use crate::storage::{FieldSnapshot, LinearIndex, SemanticIndex, SnapshotDiff};
    #[cfg(feature = "hnsw")]
    pub use crate::storage::{HnswConfig, HnswIndex};
    #[cfg(feature = "onnx-embeddings")]
    pub use crate::embed_onnx::DeterministicEmbeddingProvider;
    pub use crate::witness::{WitnessEvent, WitnessLog};
}
