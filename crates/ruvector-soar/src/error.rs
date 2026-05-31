use thiserror::Error;

#[derive(Debug, Error)]
pub enum SoarError {
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("empty dataset")]
    Empty,

    #[error("invalid config: {0}")]
    InvalidConfig(String),

    #[error("index not trained")]
    NotTrained,
}

pub type Result<T> = std::result::Result<T, SoarError>;
