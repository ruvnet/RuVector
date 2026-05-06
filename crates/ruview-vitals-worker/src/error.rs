//! Crate-wide error type.

use std::net::AddrParseError;

/// Convenience alias for `Result<T, Error>`.
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("config: {0}")]
    Config(String),

    #[error("invalid socket address {addr:?}: {source}")]
    Address {
        addr: String,
        #[source]
        source: AddrParseError,
    },

    #[error("ADR-018 frame parse error: {0}")]
    FrameParse(&'static str),

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("http: {0}")]
    Http(#[from] reqwest::Error),

    #[error("transport: {0}")]
    Transport(#[from] tonic::transport::Error),

    #[error("status: {0}")]
    Status(#[from] tonic::Status),

    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}
