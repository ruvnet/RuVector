use thiserror::Error;

#[derive(Debug, Error)]
pub enum RabitqError {
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("index is empty")]
    EmptyIndex,

    #[error("k ({k}) exceeds number of indexed vectors ({n})")]
    KTooLarge { k: usize, n: usize },

    #[error("invalid dimension {0}: must be > 0")]
    InvalidDimension(usize),

    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
}

pub type Result<T> = std::result::Result<T, RabitqError>;
