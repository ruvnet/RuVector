//! Error type for the optical core.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PhotonError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("grid dimension {0} must be a non-zero power of two for FFT-based propagation")]
    NotPowerOfTwo(usize),

    #[error("invalid optical config: {0}")]
    InvalidConfig(String),

    #[error("invalid phase mask: {0}")]
    InvalidMask(String),

    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

pub type Result<T> = core::result::Result<T, PhotonError>;
