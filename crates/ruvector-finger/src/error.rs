use thiserror::Error;

#[derive(Debug, Error)]
pub enum FingerError {
    #[error("empty dataset")]
    EmptyDataset,
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
    #[error("k={k} exceeds corpus size n={n}")]
    KTooLarge { k: usize, n: usize },
    #[error("ef={ef} must be >= k={k}")]
    EfTooSmall { ef: usize, k: usize },
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
}
