use thiserror::Error;

pub type Result<T> = std::result::Result<T, SymphonyError>;

#[derive(Debug, Error)]
pub enum SymphonyError {
    #[error("dimension mismatch: got {got}, expected {expected}")]
    DimensionMismatch { got: usize, expected: usize },

    #[error("index is empty")]
    Empty,

    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}
