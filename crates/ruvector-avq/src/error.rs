use thiserror::Error;

#[derive(Debug, Error)]
pub enum AvqError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
    #[error("dim {dim} not divisible by m={m} subspaces")]
    BadSubspace { dim: usize, m: usize },
    #[error("k must be in 1..=256, got {0}")]
    BadK(usize),
    #[error("training set is empty")]
    EmptyTrain,
    #[error("eta must be >= 1.0, got {0}")]
    BadEta(f32),
}
