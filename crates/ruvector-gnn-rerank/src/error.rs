use thiserror::Error;

#[derive(Debug, Error)]
pub enum RerankerError {
    #[error("empty candidate set")]
    Empty,
    #[error("k={k} exceeds candidate count={n}")]
    KTooLarge { k: usize, n: usize },
    #[error("dimension mismatch: query has {query} dims, candidate has {candidate} dims")]
    DimMismatch { query: usize, candidate: usize },
}
