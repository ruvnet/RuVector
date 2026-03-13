//! Error types for ruvector-cnn.
//!
//! This module defines all error types that can occur during CNN operations,
//! including forward passes, configuration, and weight loading.

use thiserror::Error;

/// Result type for CNN operations.
pub type CnnResult<T> = Result<T, CnnError>;

/// Errors that can occur during CNN operations.
#[derive(Error, Debug, Clone)]
pub enum CnnError {
    /// Invalid input data.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Model loading error.
    #[error("Model error: {0}")]
    ModelError(String),

    /// Dimension mismatch (generic).
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// SIMD operation error.
    #[error("SIMD error: {0}")]
    SimdError(String),

    /// Quantization error.
    #[error("Quantization error: {0}")]
    QuantizationError(String),

    /// Invalid tensor shape for the operation.
    #[error("Invalid shape: expected {expected}, got {got}")]
    InvalidShape {
        /// Expected shape description
        expected: String,
        /// Actual shape description
        got: String,
    },

    /// Shape mismatch between tensors.
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),

    /// Invalid parameter value.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Memory allocation error.
    #[error("Memory allocation failed: {0}")]
    AllocationError(String),

    /// Invalid channel count.
    #[error("Invalid channel count: expected {expected}, got {actual}")]
    InvalidChannels {
        /// Expected channels
        expected: usize,
        /// Actual channels
        actual: usize,
    },

    /// Invalid convolution parameters.
    #[error("Invalid convolution parameters: {0}")]
    InvalidConvParams(String),

    /// Weight loading error.
    #[error("Weight loading error: {0}")]
    WeightLoadError(String),

    /// Empty input provided.
    #[error("Empty input: {0}")]
    EmptyInput(String),

    /// Numerical instability detected.
    #[error("Numerical instability: {0}")]
    NumericalInstability(String),

    /// Unsupported backbone type.
    #[error("Unsupported backbone: {0}")]
    UnsupportedBackbone(String),

    /// Batch processing error.
    #[error("Batch processing error: {0}")]
    BatchError(String),

    /// Error during convolution computation.
    #[error("Convolution error: {0}")]
    ConvolutionError(String),

    /// Error during pooling computation.
    #[error("Pooling error: {0}")]
    PoolingError(String),

    /// Error during normalization.
    #[error("Normalization error: {0}")]
    NormalizationError(String),

    /// Invalid kernel configuration.
    #[error(
        "Invalid kernel: kernel_size={kernel_size}, but input spatial dims are ({height}, {width})"
    )]
    InvalidKernel {
        /// Kernel size
        kernel_size: usize,
        /// Input height
        height: usize,
        /// Input width
        width: usize,
    },

    /// IO error (for model loading).
    #[error("IO error: {0}")]
    IoError(String),

    /// Image processing error.
    #[error("Image error: {0}")]
    ImageError(String),

    /// Index out of bounds.
    #[error("Index out of bounds: {index} >= {size}")]
    IndexOutOfBounds {
        /// The index that was accessed
        index: usize,
        /// The size of the container
        size: usize,
    },

    /// Unsupported operation.
    #[error("Unsupported operation: {0}")]
    Unsupported(String),
}

impl From<std::io::Error> for CnnError {
    fn from(err: std::io::Error) -> Self {
        CnnError::IoError(err.to_string())
    }
}

impl CnnError {
    /// Create a dimension mismatch error with expected and actual values.
    pub fn dim_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch(format!("expected {expected}, got {actual}"))
    }

    /// Create an invalid shape error.
    pub fn invalid_shape(expected: impl Into<String>, got: impl Into<String>) -> Self {
        Self::InvalidShape {
            expected: expected.into(),
            got: got.into(),
        }
    }

    /// Create a shape mismatch error.
    pub fn shape_mismatch(msg: impl Into<String>) -> Self {
        Self::ShapeMismatch(msg.into())
    }

    /// Create an invalid parameter error.
    pub fn invalid_parameter(msg: impl Into<String>) -> Self {
        Self::InvalidParameter(msg.into())
    }

    /// Create an invalid config error.
    pub fn invalid_config(msg: impl Into<String>) -> Self {
        Self::InvalidConfig(msg.into())
    }

    /// Create a convolution error.
    pub fn convolution_error(msg: impl Into<String>) -> Self {
        Self::ConvolutionError(msg.into())
    }

    /// Create a pooling error.
    pub fn pooling_error(msg: impl Into<String>) -> Self {
        Self::PoolingError(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = CnnError::DimensionMismatch("expected 64, got 32".to_string());
        assert!(err.to_string().contains("expected 64"));

        let err = CnnError::InvalidConfig("kernel_size must be positive".to_string());
        assert_eq!(
            err.to_string(),
            "Invalid configuration: kernel_size must be positive"
        );
    }

    #[test]
    fn test_error_clone() {
        let err = CnnError::ConvolutionError("test".to_string());
        let cloned = err.clone();
        assert_eq!(err.to_string(), cloned.to_string());
    }

    #[test]
    fn test_invalid_kernel_error() {
        let err = CnnError::InvalidKernel {
            kernel_size: 7,
            height: 3,
            width: 3,
        };
        assert!(err.to_string().contains("kernel_size=7"));
        assert!(err.to_string().contains("(3, 3)"));
    }

    #[test]
    fn test_invalid_channels_error() {
        let err = CnnError::InvalidChannels {
            expected: 3,
            actual: 1,
        };
        assert!(err.to_string().contains("expected 3"));
        assert!(err.to_string().contains("got 1"));
    }

    #[test]
    fn test_helper_methods() {
        let err = CnnError::invalid_shape("NCHW", "NHWC");
        assert!(err.to_string().contains("NCHW"));
        assert!(err.to_string().contains("NHWC"));

        let err = CnnError::invalid_config("dropout must be in [0, 1]");
        assert!(err.to_string().contains("dropout"));

        let err = CnnError::dim_mismatch(64, 32);
        assert!(err.to_string().contains("64"));
        assert!(err.to_string().contains("32"));
    }
}
