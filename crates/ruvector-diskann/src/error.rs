use thiserror::Error;

pub type Result<T> = std::result::Result<T, DiskAnnError>;

#[derive(Error, Debug)]
pub enum DiskAnnError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Index not built — call build() first")]
    NotBuilt,

    #[error("Index is empty")]
    Empty,

    #[error("ID not found: {0}")]
    NotFound(String),

    #[error("PQ not trained — call train() first")]
    PqNotTrained,

    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    #[error("Serialization error: {0}")]
    Serialization(String),
}
