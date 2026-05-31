use thiserror::Error;

#[derive(Debug, Error)]
pub enum MuveraError {
    #[error("token set is empty")]
    EmptyTokenSet,

    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("invalid config: {0}")]
    InvalidConfig(String),
}
