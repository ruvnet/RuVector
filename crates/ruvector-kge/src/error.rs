//! Typed errors; never carry entity or relation labels (ADR-005).

#[derive(Debug, thiserror::Error)]
pub enum KgeError {
    #[error("input limit exceeded: {0}")]
    Limit(&'static str),
    #[error("invalid input: {0}")]
    Invalid(String),
    #[error("dimension mismatch: expected {expected}, got {got}")]
    Dims { expected: usize, got: usize },
    #[error("unknown entity id {0}")]
    UnknownEntity(u32),
    #[error("unknown relation id {0}")]
    UnknownRelation(u32),
    #[error("scorer failure: {0}")]
    Scorer(String),
}

pub type Result<T> = std::result::Result<T, KgeError>;
