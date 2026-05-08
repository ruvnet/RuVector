use thiserror::Error;

#[derive(Debug, Error)]
pub enum LvqError {
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimMismatch { expected: usize, actual: usize },

    #[error("empty input")]
    Empty,

    #[error("vector contains non-finite component at index {0}")]
    NonFinite(usize),

    #[error("index already finalized; cannot mutate after build")]
    AlreadyBuilt,

    #[error("k = {0} is larger than the dataset size {1}")]
    KTooLarge(usize, usize),
}
