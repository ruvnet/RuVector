//! Error types for RuvLLM
//!
//! This module defines the error hierarchy for the RuvLLM crate,
//! providing detailed error information for debugging and handling.

use std::path::PathBuf;
use thiserror::Error;

/// Result type alias for RuvLLM operations
pub type Result<T> = std::result::Result<T, RuvLLMError>;

/// Failures while managing an adapter applied to model generation.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum ModelAdapterError {
    /// The selected backend does not implement this adapter operation.
    #[error("Model adapter operation {operation} is unsupported by this backend")]
    UnsupportedBackend {
        /// Name of the requested adapter operation.
        operation: &'static str,
    },

    /// An adapter operation requires a loaded base model.
    #[error("No base model is loaded for adapter management")]
    ModelNotLoaded,

    /// A required adapter file or directory is absent.
    #[error("Model adapter path was not found: {path:?}")]
    NotFound {
        /// Missing file or directory.
        path: PathBuf,
    },

    /// The adapter data or configuration is invalid or unsupported.
    #[error("Invalid model adapter format: {message}")]
    InvalidFormat {
        /// Description of the format problem.
        message: String,
    },

    /// An adapter tensor does not match the target projection dimensions.
    #[error("Model adapter tensor {tensor} has shape {actual:?}, expected {expected:?}")]
    ShapeMismatch {
        /// Tensor whose dimensions do not match.
        tensor: String,
        /// Dimensions required by the base model and adapter rank.
        expected: Vec<usize>,
        /// Dimensions present in the adapter file.
        actual: Vec<usize>,
    },

    /// Adapter ranks are invalid or disagree across configuration and tensors.
    #[error("Model adapter rank mismatch: {message}")]
    RankMismatch {
        /// Description of the incompatible rank.
        message: String,
    },

    /// A requested layer or projection is unsupported by the backend.
    #[error("Unsupported model adapter target: {target}")]
    UnsupportedTarget {
        /// Unsupported target name.
        target: String,
    },

    /// The caller's multiplier cannot be applied safely.
    #[error("Invalid model adapter scale: {scale}")]
    InvalidScale {
        /// Rejected caller multiplier.
        scale: f32,
    },

    /// The inference backend could not complete an adapter operation.
    #[error("Model adapter backend failure: {message}")]
    Backend {
        /// Description of the backend failure.
        message: String,
    },

    /// An adapter file could not be read.
    #[error("Model adapter I/O failure at {path:?}: {source}")]
    Io {
        /// File or directory involved in the failure.
        path: PathBuf,
        /// Underlying filesystem error.
        #[source]
        source: std::io::Error,
    },
}

/// Main error type for RuvLLM
#[derive(Error, Debug)]
pub enum RuvLLMError {
    /// Storage-related errors
    #[error("Storage error: {0}")]
    Storage(String),

    /// Session management errors
    #[error("Session error: {0}")]
    Session(String),

    /// KV cache errors
    #[error("KV cache error: {0}")]
    KvCache(String),

    /// Paged attention errors
    #[error("Paged attention error: {0}")]
    PagedAttention(String),

    /// Adapter management errors
    #[error("Adapter error: {0}")]
    Adapter(String),

    /// Policy store errors
    #[error("Policy error: {0}")]
    Policy(String),

    /// Witness log errors
    #[error("Witness log error: {0}")]
    WitnessLog(String),

    /// SONA learning errors
    #[error("SONA error: {0}")]
    Sona(String),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Config(String),

    /// Resource exhaustion
    #[error("Out of memory: {0}")]
    OutOfMemory(String),

    /// Invalid operation
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    /// Not found
    #[error("Not found: {0}")]
    NotFound(String),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// IO errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Ruvector errors
    #[error("Ruvector error: {0}")]
    Ruvector(String),

    /// Backend inference errors
    #[error("Backend error: {0}")]
    Backend(String),

    /// Model loading errors
    #[error("Model error: {0}")]
    Model(String),

    /// Tokenization errors
    #[error("Tokenization error: {0}")]
    Tokenization(String),

    /// Generation errors
    #[error("Generation error: {0}")]
    Generation(String),

    /// Metal GPU errors (macOS only)
    #[error("Metal error: {0}")]
    Metal(String),

    /// Shader compilation errors
    #[error("Shader error: {0}")]
    Shader(String),

    /// GGUF format errors
    #[error("GGUF error: {0}")]
    Gguf(String),

    /// Quantization errors
    #[error("Quantization error: {0}")]
    Quantization(String),

    /// Not implemented errors
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// Hybrid pipeline errors
    #[error("Hybrid pipeline error: {0}")]
    HybridPipeline(String),

    /// Core ML errors (macOS only)
    #[error("Core ML error: {0}")]
    CoreML(String),
}

impl From<ruvector_core::RuvectorError> for RuvLLMError {
    fn from(err: ruvector_core::RuvectorError) -> Self {
        RuvLLMError::Ruvector(err.to_string())
    }
}

impl From<serde_json::Error> for RuvLLMError {
    fn from(err: serde_json::Error) -> Self {
        RuvLLMError::Serialization(err.to_string())
    }
}
