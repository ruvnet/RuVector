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

pub mod adversarial;
pub mod ann;
pub mod batch;
pub mod data;
pub mod error;
pub mod eval;
pub mod optimize;
pub mod scorer;
pub mod tables;
pub mod train;
pub mod types;

pub use ann::AnnIndex;
pub use batch::BatchScorer;
pub use data::{Split, Split4, TripleStore, Vocab};
pub use error::{KgeError, Result};
pub use eval::{evaluate, evaluate_ranks, EvalConfig, EvalReport, MetricSet, TieBreak};
pub use optimize::{
    ArmOutcome, Campaign, CampaignReport, CampaignSpec, ContinualUpdate, Evaluator,
};
pub use scorer::{HolE, RotatE, Scorer};
pub use tables::Tables;
pub use train::{Differentiable, LossKind, OptimKind, Progress, TrainConfig, Trainer};
pub use types::*;
