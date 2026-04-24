use thiserror::Error;

#[derive(Debug, Error)]
pub enum RuLakeError {
    #[error("unknown backend: {0}")]
    UnknownBackend(String),

    #[error("unknown collection: {backend}/{collection}")]
    UnknownCollection { backend: String, collection: String },

    #[error("dimension mismatch: collection dim={expected}, query dim={actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("backend error ({backend}): {detail}")]
    Backend { backend: String, detail: String },

    #[error("rabitq error: {0}")]
    Rabitq(#[from] ruvector_rabitq::RabitqError),

    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
}

pub type Result<T> = std::result::Result<T, RuLakeError>;
