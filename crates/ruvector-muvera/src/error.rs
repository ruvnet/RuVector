use thiserror::Error;

#[derive(Debug, Error)]
pub enum MuveraError {
    #[error("empty dataset: need at least one document")]
    EmptyDataset,

    #[error("empty document: document {doc_idx} contains no token vectors")]
    EmptyDocument { doc_idx: usize },

    #[error("dimension mismatch: encoder expects {expected}D, got {actual}D in doc {doc_idx}")]
    DimMismatch {
        expected: usize,
        actual: usize,
        doc_idx: usize,
    },

    #[error("k={k} exceeds corpus size {n}")]
    KTooLarge { k: usize, n: usize },

    #[error("num_reps must be ≥ 1, got {num_reps}")]
    InvalidNumReps { num_reps: usize },

    #[error("orig_dim must be ≥ 1, got {orig_dim}")]
    InvalidDim { orig_dim: usize },
}
