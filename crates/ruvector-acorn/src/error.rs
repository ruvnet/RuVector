use thiserror::Error;

#[derive(Debug, Error)]
pub enum AcornError {
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("empty index — insert vectors before searching")]
    EmptyIndex,
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
}

pub type Result<T> = std::result::Result<T, AcornError>;
