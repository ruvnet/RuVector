//! Error types for the spectral graph sparsifier.
//!
//! All fallible operations in this crate return [`SparsifierError`] wrapped
//! in the crate-level [`Result`] alias.

use thiserror::Error;

/// Convenience result alias for sparsifier operations.
pub type Result<T> = std::result::Result<T, SparsifierError>;

/// Errors that can occur during sparsification.
#[derive(Error, Debug)]
pub enum SparsifierError {
    /// A vertex index was out of range for the current graph.
    #[error("vertex {0} is out of bounds (graph has {1} vertices)")]
    VertexOutOfBounds(usize, usize),

    /// An edge between the given vertices was not found.
    #[error("edge ({0}, {1}) not found")]
    EdgeNotFound(usize, usize),

    /// An edge between the given vertices already exists.
    #[error("edge ({0}, {1}) already exists")]
    EdgeAlreadyExists(usize, usize),

    /// An edge weight was non-positive or non-finite.
    #[error("invalid edge weight {0}: must be positive and finite")]
    InvalidWeight(f64),

    /// The epsilon parameter was out of the valid range `(0, 1)`.
    #[error("invalid epsilon {0}: must be in (0, 1)")]
    InvalidEpsilon(f64),

    /// The edge budget was too small for the given graph.
    #[error("edge budget {budget} is too small for {vertices} vertices (minimum {minimum})")]
    BudgetTooSmall {
        /// Requested budget.
        budget: usize,
        /// Number of vertices.
        vertices: usize,
        /// Minimum budget for this graph size.
        minimum: usize,
    },

    /// The graph has no vertices; sparsification is undefined.
    #[error("cannot sparsify an empty graph")]
    EmptyGraph,

    /// A spectral audit detected unacceptable distortion.
    #[error(
        "spectral audit failed: max relative error {max_error:.4} exceeds threshold {threshold:.4}"
    )]
    AuditFailed {
        /// Observed maximum relative error.
        max_error: f64,
        /// Acceptable threshold.
        threshold: f64,
    },

    /// An internal numerical error (e.g. NaN or overflow).
    #[error("numerical error: {0}")]
    Numerical(String),

    /// Catch-all for unexpected internal errors.
    #[error("internal error: {0}")]
    Internal(String),
}
