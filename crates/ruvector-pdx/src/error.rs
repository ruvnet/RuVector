use thiserror::Error;

#[derive(Debug, Error)]
pub enum PdxError {
    #[error("dimension mismatch: index has {index_dim}D, vector has {vec_dim}D")]
    DimMismatch { index_dim: usize, vec_dim: usize },

    #[error("k={k} exceeds index size {size}")]
    KTooLarge { k: usize, size: usize },

    #[error("index is empty")]
    Empty,

    #[error("block size must be ≥ 1, got {0}")]
    BadBlockSize(usize),
}

pub type Result<T> = std::result::Result<T, PdxError>;
