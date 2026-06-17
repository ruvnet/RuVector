use thiserror::Error;

#[derive(Debug, Error)]
pub enum RerankerError {
    #[error("empty candidate set")]
    Empty,
    #[error("k={k} exceeds candidate count={n}")]
    KTooLarge { k: usize, n: usize },
    #[error("dimension mismatch: query has {query} dims, candidate has {candidate} dims")]
    DimMismatch { query: usize, candidate: usize },
    /// A non-finite (NaN/±inf) value was found in untrusted input — rejected
    /// fail-fast rather than producing a silently-corrupted ranking (defends the
    /// poisoned-first-stage / MemoryGraft threat model).
    #[error("non-finite value in {what} (NaN/inf rejected)")]
    NonFinite { what: &'static str },
}
