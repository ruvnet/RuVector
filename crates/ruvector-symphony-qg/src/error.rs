use thiserror::Error;

#[derive(Debug, Error)]
pub enum SymphonyError {
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("empty corpus: cannot build index with zero vectors")]
    EmptyCorpus,

    #[error("k ({k}) exceeds corpus size ({n})")]
    KTooLarge { k: usize, n: usize },

    #[error("configuration error: {0}")]
    Config(String),
}

pub type Result<T> = std::result::Result<T, SymphonyError>;
