use thiserror::Error;

#[derive(Debug, Error)]
pub enum SqgError {
    #[error("configuration error: {0}")]
    Config(String),
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}
