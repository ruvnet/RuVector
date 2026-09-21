//! ruvector-typesafe-core — a "System One" typed-decision engine.
//!
//! Contract (mirrors typesafe.ai, ADR-001): one `state` text plus a batch of
//! questions of type `choice` / `score` / `noul`; every answer carries a
//! calibrated `confidence`, an `abstain` mass and a receipt naming the head,
//! model and temperature that produced it (ADR-003).
//!
//! Everything in this crate is target-independent Rust: no embedding backend,
//! no filesystem, no network (ADR-002 §1, ADR-005). Backends implement
//! [`Embedder`]; `ruvector-embed-core` provides `ort` (native) and `tract`
//! (wasm) implementations behind features.
//!
//! Module ownership during the initial build (one agent per module):
//! - `types`, `limits`, `embedder`, `hash_embedder`: skeleton (this commit)
//! - `heads`, `calibration`, `engine`: decision heads + temperature scaling
//! - `bank`, `loop_gate`, `receipt`: example bank, promotion gate, receipts

pub mod bank;
pub mod calibration;
pub mod embedder;
pub mod engine;
pub mod heads;
pub mod limits;
pub mod loop_gate;
pub mod receipt;
pub mod types;

#[cfg(feature = "hash-embedder")]
pub mod hash_embedder;

pub use embedder::Embedder;
pub use types::*;

/// Errors are typed and never carry `state` text (ADR-005: no PII in errors/logs).
#[derive(Debug, thiserror::Error)]
pub enum TypesafeError {
    #[error("input limit exceeded: {0}")]
    Limit(&'static str),
    #[error("invalid request: {0}")]
    Invalid(String),
    #[error("embedder failure: {0}")]
    Embedder(String),
}

pub type Result<T> = std::result::Result<T, TypesafeError>;
