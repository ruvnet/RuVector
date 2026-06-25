//! Error types for ruvector-maxsim.

use thiserror::Error;

/// Errors that can occur in MaxSim index operations.
#[derive(Debug, Error)]
pub enum MaxSimError {
    /// An embedding's dimensionality did not match the index's expectation.
    #[error("dimension mismatch: index expects {expected}, got {got}")]
    DimensionMismatch {
        /// Dimensionality the index was configured with.
        expected: usize,
        /// Dimensionality of the offending input vector.
        got: usize,
    },

    /// A document was supplied with no token/chunk vectors.
    #[error("empty document: at least one token vector is required")]
    EmptyDocument,

    /// A search was issued against an index with no documents.
    #[error("index is empty: no documents have been added")]
    EmptyIndex,
}
