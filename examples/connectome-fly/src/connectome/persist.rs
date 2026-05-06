//! Binary serialization error type for `Connectome`.
//!
//! The round-trip is implemented on `Connectome` directly via
//! `bincode::{serialize, deserialize}`. This module owns only the
//! error alias so the other submodules can name it.

use thiserror::Error;

/// Errors surfaced by the connectome generator / serializer.
#[derive(Debug, Error)]
pub enum ConnectomeError {
    /// bincode / IO failure.
    #[error("serialization: {0}")]
    Serde(#[from] Box<bincode::ErrorKind>),
}
