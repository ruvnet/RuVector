//! Error types for the decompiler crate.

use thiserror::Error;

/// Errors that can occur during decompilation.
#[derive(Debug, Error)]
pub enum DecompilerError {
    /// The input bundle is empty or contains no usable content.
    #[error("empty or invalid bundle: {0}")]
    EmptyBundle(String),

    /// The parser failed to extract any declarations from the bundle.
    #[error("no declarations found in bundle")]
    NoDeclarations,

    /// MinCut graph partitioning failed.
    #[error("partitioning failed: {0}")]
    PartitioningFailed(String),

    /// Source map generation failed.
    #[error("source map generation failed: {0}")]
    SourceMapError(String),

    /// Witness chain verification failed.
    #[error("witness chain verification failed: {0}")]
    WitnessError(String),

    /// Neural model loading or inference error (requires `neural` feature).
    #[error("model error: {0}")]
    ModelError(String),

    /// I/O error (file access, reading).
    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON serialization/deserialization error.
    #[error("json error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Regex compilation error.
    #[error("regex error: {0}")]
    RegexError(#[from] regex::Error),
}

/// Result type alias for decompiler operations.
pub type Result<T> = std::result::Result<T, DecompilerError>;
