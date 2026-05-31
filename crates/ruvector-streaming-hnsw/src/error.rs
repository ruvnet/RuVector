use thiserror::Error;

#[derive(Debug, Error)]
pub enum HnswError {
    #[error("empty dataset")]
    EmptyDataset,
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimMismatch { expected: usize, actual: usize },
    #[error("k ({k}) exceeds collection size ({n})")]
    KTooLarge { k: usize, n: usize },
}
