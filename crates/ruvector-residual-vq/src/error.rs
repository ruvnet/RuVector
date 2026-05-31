use thiserror::Error;

#[derive(Debug, Error)]
pub enum RvqError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
    #[error("empty dataset")]
    EmptyDataset,
    #[error("codebook size {0} exceeds u8 maximum of 256")]
    CodebookTooLarge(usize),
    #[error("invalid parameter: {0}")]
    InvalidParam(String),
}

pub type Result<T> = std::result::Result<T, RvqError>;
