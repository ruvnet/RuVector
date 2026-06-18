//! Error types for graph condensation.

use thiserror::Error;

/// Errors that can occur during graph condensation.
#[derive(Debug, Error)]
pub enum CondenseError {
    /// The input graph has no vertices, so there is nothing to condense.
    #[error("empty graph: nothing to condense")]
    EmptyGraph,

    /// A feature vector did not match the configured embedding dimension.
    #[error("feature dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// The dimension the [`crate::NodeFeatures`] container was created with.
        expected: usize,
        /// The dimension of the offending vector.
        got: usize,
    },

    /// A vertex present in the graph had no associated feature vector.
    #[error("vertex {0} has no feature vector")]
    MissingFeature(u64),

    /// The configuration was internally inconsistent.
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// An error bubbled up from the underlying min-cut engine.
    #[error("min-cut engine error: {0}")]
    MinCut(#[from] ruvector_mincut::MinCutError),
}

/// Convenience result alias for condensation operations.
pub type Result<T> = std::result::Result<T, CondenseError>;
