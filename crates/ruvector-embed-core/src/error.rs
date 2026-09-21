//! Typed errors for the embedding backends.
//!
//! Errors never carry `state`/input text (ADR-005: no PII in errors or logs).
//! Only structural facts — a path, an expected vs actual hash, a stage name —
//! appear in messages.

use ruvector_typesafe_core::TypesafeError;

/// Everything that can go wrong loading or running an embedder.
#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    /// A manifest entry's `sha256` did not match the bytes on disk (fail closed).
    #[error("model hash mismatch for '{name}': manifest={expected}, actual={actual}")]
    HashMismatch {
        name: String,
        expected: String,
        actual: String,
    },

    /// The manifest JSON was malformed or a required field was missing.
    #[error("invalid manifest: {0}")]
    Manifest(String),

    /// A required file (model or tokenizer) was absent.
    #[error("file not found: {0}")]
    NotFound(String),

    /// I/O failure reading a model/tokenizer file (native only).
    #[error("io error: {0}")]
    Io(String),

    /// The tokenizer could not be constructed or failed to encode.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// The inference backend (ort/tract) failed to load, optimise or run.
    #[error("backend error: {0}")]
    Backend(String),

    /// The model produced an output shape the pooler cannot interpret.
    #[error("unexpected output shape: {0}")]
    Shape(String),

    /// An empty batch was passed to `embed`.
    #[error("empty input batch")]
    EmptyInput,
}

impl From<EmbedError> for TypesafeError {
    fn from(e: EmbedError) -> Self {
        // The engine's contract exposes embedder faults as a single variant;
        // the structured detail lives in the Display text (no input text).
        TypesafeError::Embedder(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, EmbedError>;
