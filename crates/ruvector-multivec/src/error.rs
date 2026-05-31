use thiserror::Error;

#[derive(Debug, Error)]
pub enum MultivecError {
    #[error("empty corpus")]
    EmptyCorpus,

    #[error("document {id} has no token vectors")]
    EmptyDocument { id: usize },

    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimMismatch { expected: usize, actual: usize },

    #[error("k ({k}) exceeds corpus size ({n})")]
    KTooLarge { k: usize, n: usize },

    #[error("FDE subspaces {m} must divide dimension {d}")]
    FdeSubspaceMismatch { m: usize, d: usize },

    #[error("MUVERA repetitions R must be ≥ 1")]
    InvalidRepetitions,

    #[error("index not yet built — call build() first")]
    NotBuilt,
}
