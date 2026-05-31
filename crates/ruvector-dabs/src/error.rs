use thiserror::Error;

#[derive(Debug, Error)]
pub enum DabsError {
    #[error("empty dataset")]
    EmptyDataset,
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimMismatch { expected: usize, actual: usize },
    #[error("k={k} exceeds dataset size n={n}")]
    KExceedsDataset { k: usize, n: usize },
    #[error("gamma must be >= 0.0, got {gamma}")]
    InvalidGamma { gamma: f32 },
}
