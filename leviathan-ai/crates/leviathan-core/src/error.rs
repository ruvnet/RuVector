//! Error types for Leviathan

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Core error: {0}")]
    Core(String),
}

pub type Result<T> = std::result::Result<T, Error>;
