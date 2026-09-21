//! ruvector-kge — holographic knowledge-graph embeddings (ADR-001..006 under
//! `npm/packages/kge/docs/adr/`).
//!
//! Target-independent Rust: no filesystem, no network, no time (ADR-005). The
//! entity and relation tables live in [`tables`]; scoring functions implement
//! [`Scorer`]; training, evaluation and the self-optimization loop are built
//! on top and never reach around the trait.
//!
//! Module ownership during the initial build (one agent per group):
//! - `types`, `error`, `scorer` (trait), `tables`: skeleton (this commit)
//! - `scorer::{hole,rotate,complex}`, `ann`, `batch`: scorers + ANN glue
//! - `data`, `train`, `eval`: triples, negative sampling, losses, filtered ranking
//! - `optimize`: HPO/model-arm loop over typesafe-core's gate and receipts

pub mod error;
pub mod scorer;
pub mod tables;
pub mod types;

pub use error::{KgeError, Result};
pub use scorer::Scorer;
pub use tables::Tables;
pub use types::*;
