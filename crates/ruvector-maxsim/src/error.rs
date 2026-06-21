//! Error types for ruvector-maxsim.

use thiserror::Error;

/// Errors that can occur in MaxSim index operations.
#[derive(Debug, Error)]
pub enum MaxSimError {
    #[error("dimension mismatch: index expects {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("empty document: at least one token vector is required")]
    EmptyDocument,

    #[error("index is empty: no documents have been added")]
    EmptyIndex,
}
