use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoDEQError {
    #[error("empty dataset")]
    EmptyDataset,
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimMismatch { expected: usize, actual: usize },
    #[error("k ({k}) exceeds collection size ({n})")]
    KTooLarge { k: usize, n: usize },
    #[error("id {0} not found")]
    IdNotFound(u64),
    #[error("bits must be 1–8, got {0}")]
    InvalidBits(u8),
}
