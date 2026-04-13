//! Fallible operation error type.
//!
//! All mutating APIs on [`crate::engine::FieldEngine`] return
//! `Result<T, FieldError>`. The enum is exhaustive and `Clone` so callers can
//! record failed operations in the witness log without consuming ownership.
//!
//! # Example
//!
//! ```
//! use ruvector_field::error::FieldError;
//! let err = FieldError::UnknownNode(42);
//! assert_eq!(format!("{}", err), "unknown node: 42");
//! ```

use core::fmt;

/// Errors produced by the field engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldError {
    /// Referenced node id does not exist in the engine.
    UnknownNode(u64),
    /// Referenced edge id does not exist.
    UnknownEdge(u64),
    /// Shell assignment was rejected (e.g. demotion below Event, promotion above Principle).
    ShellViolation(&'static str),
    /// A policy mask is incompatible with the requested mutation.
    PolicyConflict(&'static str),
    /// Operation requires a proof token and none was presented.
    ProofRequired,
    /// Proof gate denied the presented token.
    ProofDenied(&'static str),
    /// Embedding vector failed validation (zero length, NaN, mismatched dim).
    InvalidEmbedding(&'static str),
}

impl fmt::Display for FieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FieldError::UnknownNode(id) => write!(f, "unknown node: {}", id),
            FieldError::UnknownEdge(id) => write!(f, "unknown edge: {}", id),
            FieldError::ShellViolation(why) => write!(f, "shell violation: {}", why),
            FieldError::PolicyConflict(why) => write!(f, "policy conflict: {}", why),
            FieldError::ProofRequired => write!(f, "proof required for this hint"),
            FieldError::ProofDenied(why) => write!(f, "proof denied: {}", why),
            FieldError::InvalidEmbedding(why) => write!(f, "invalid embedding: {}", why),
        }
    }
}

impl std::error::Error for FieldError {}
