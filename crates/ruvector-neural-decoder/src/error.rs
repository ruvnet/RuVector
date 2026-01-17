//! Error types for the Neural Quantum Error Decoder
//!
//! This module provides error types for syndrome translation, graph encoding,
//! Mamba decoding, and feature fusion operations.

use thiserror::Error;

/// Result type for neural decoder operations
pub type Result<T> = std::result::Result<T, NeuralDecoderError>;

/// Errors that can occur in neural decoder operations
#[derive(Error, Debug)]
pub enum NeuralDecoderError {
    /// Invalid syndrome dimensions
    #[error("Invalid syndrome dimensions: expected {expected}x{expected}, got {actual_rows}x{actual_cols}")]
    InvalidSyndromeDimension {
        /// Expected dimension
        expected: usize,
        /// Actual row count
        actual_rows: usize,
        /// Actual column count
        actual_cols: usize,
    },

    /// Invalid embedding dimension
    #[error("Invalid embedding dimension: expected {expected}, got {actual}")]
    InvalidEmbeddingDimension {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        actual: usize,
    },

    /// Invalid hidden state dimension
    #[error("Invalid hidden state dimension: expected {expected}, got {actual}")]
    InvalidHiddenDimension {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        actual: usize,
    },

    /// Invalid attention heads configuration
    #[error("Embedding dimension {embed_dim} must be divisible by number of heads {num_heads}")]
    InvalidAttentionHeads {
        /// Embedding dimension
        embed_dim: usize,
        /// Number of heads
        num_heads: usize,
    },

    /// Empty graph
    #[error("Detector graph is empty")]
    EmptyGraph,

    /// Invalid detector index
    #[error("Invalid detector index: {0}")]
    InvalidDetector(usize),

    /// Invalid boundary type
    #[error("Invalid boundary type: {0}")]
    InvalidBoundary(String),

    /// Decoding failed
    #[error("Decoding failed: {0}")]
    DecodingFailed(String),

    /// Fusion error
    #[error("Feature fusion error: {0}")]
    FusionError(String),

    /// MinCut integration error
    #[error("MinCut integration error: {0}")]
    MinCutError(String),

    /// Shape mismatch
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Expected shape
        expected: Vec<usize>,
        /// Actual shape
        actual: Vec<usize>,
    },

    /// Numerical instability
    #[error("Numerical instability detected: {0}")]
    NumericalInstability(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Internal error
    #[error("Internal error: {0}")]
    InternalError(String),
}

impl NeuralDecoderError {
    /// Create a dimension mismatch error for syndromes
    pub fn syndrome_dim(expected: usize, rows: usize, cols: usize) -> Self {
        Self::InvalidSyndromeDimension {
            expected,
            actual_rows: rows,
            actual_cols: cols,
        }
    }

    /// Create an embedding dimension error
    pub fn embed_dim(expected: usize, actual: usize) -> Self {
        Self::InvalidEmbeddingDimension { expected, actual }
    }

    /// Create a hidden dimension error
    pub fn hidden_dim(expected: usize, actual: usize) -> Self {
        Self::InvalidHiddenDimension { expected, actual }
    }

    /// Create an attention heads error
    pub fn attention_heads(embed_dim: usize, num_heads: usize) -> Self {
        Self::InvalidAttentionHeads {
            embed_dim,
            num_heads,
        }
    }

    /// Create a shape mismatch error
    pub fn shape_mismatch(expected: Vec<usize>, actual: Vec<usize>) -> Self {
        Self::ShapeMismatch { expected, actual }
    }

    /// Check if the error is recoverable
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::InvalidDetector(_)
                | Self::InvalidBoundary(_)
                | Self::ConfigError(_)
        )
    }

    /// Check if the error is related to dimensions
    pub fn is_dimension_error(&self) -> bool {
        matches!(
            self,
            Self::InvalidSyndromeDimension { .. }
                | Self::InvalidEmbeddingDimension { .. }
                | Self::InvalidHiddenDimension { .. }
                | Self::InvalidAttentionHeads { .. }
                | Self::ShapeMismatch { .. }
        )
    }
}

impl From<ruvector_mincut::MinCutError> for NeuralDecoderError {
    fn from(err: ruvector_mincut::MinCutError) -> Self {
        Self::MinCutError(err.to_string())
    }
}

impl From<String> for NeuralDecoderError {
    fn from(msg: String) -> Self {
        Self::InternalError(msg)
    }
}

impl From<&str> for NeuralDecoderError {
    fn from(msg: &str) -> Self {
        Self::InternalError(msg.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = NeuralDecoderError::syndrome_dim(5, 3, 4);
        assert!(err.to_string().contains("5"));
        assert!(err.to_string().contains("3"));
        assert!(err.to_string().contains("4"));

        let err = NeuralDecoderError::embed_dim(128, 64);
        assert!(err.to_string().contains("128"));
        assert!(err.to_string().contains("64"));
    }

    #[test]
    fn test_is_recoverable() {
        assert!(NeuralDecoderError::InvalidDetector(0).is_recoverable());
        assert!(NeuralDecoderError::InvalidBoundary("test".to_string()).is_recoverable());
        assert!(!NeuralDecoderError::EmptyGraph.is_recoverable());
        assert!(!NeuralDecoderError::DecodingFailed("test".to_string()).is_recoverable());
    }

    #[test]
    fn test_is_dimension_error() {
        assert!(NeuralDecoderError::syndrome_dim(5, 3, 4).is_dimension_error());
        assert!(NeuralDecoderError::embed_dim(128, 64).is_dimension_error());
        assert!(NeuralDecoderError::attention_heads(128, 3).is_dimension_error());
        assert!(!NeuralDecoderError::EmptyGraph.is_dimension_error());
    }

    #[test]
    fn test_from_string() {
        let err: NeuralDecoderError = "test error".into();
        assert!(matches!(err, NeuralDecoderError::InternalError(_)));
        assert!(err.to_string().contains("test error"));
    }
}
