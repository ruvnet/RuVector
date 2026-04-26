use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum AcornError {
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimMismatch { expected: usize, actual: usize },
    #[error("empty dataset: cannot build index over zero vectors")]
    EmptyDataset,
    #[error("k={k} exceeds dataset size={n}")]
    KTooLarge { k: usize, n: usize },
    #[error("gamma must be >= 1, got {gamma}")]
    InvalidGamma { gamma: usize },
}
