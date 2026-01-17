//! Error types for the quantum kernel coherence monitor.
//!
//! This module defines all error types that can occur during quantum kernel
//! computation, E-value testing, and drift monitoring.

use thiserror::Error;

/// Result type alias for quantum monitor operations.
pub type Result<T> = std::result::Result<T, QuantumMonitorError>;

/// Errors that can occur in the quantum kernel coherence monitor.
#[derive(Error, Debug, Clone)]
pub enum QuantumMonitorError {
    /// Dimension mismatch between vectors or matrices.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension encountered.
        actual: usize,
    },

    /// Sample size is too small for statistical validity.
    #[error("Insufficient samples: need at least {minimum}, got {actual}")]
    InsufficientSamples {
        /// Minimum required samples.
        minimum: usize,
        /// Actual number of samples.
        actual: usize,
    },

    /// Invalid parameter value provided.
    #[error("Invalid parameter '{name}': {reason}")]
    InvalidParameter {
        /// Parameter name.
        name: String,
        /// Reason why the parameter is invalid.
        reason: String,
    },

    /// Numerical computation error (overflow, underflow, NaN).
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// Kernel matrix is not positive semi-definite.
    #[error("Kernel matrix is not positive semi-definite")]
    NotPositiveSemiDefinite,

    /// E-value computation failed.
    #[error("E-value computation failed: {0}")]
    EValueError(String),

    /// Confidence sequence computation failed.
    #[error("Confidence sequence error: {0}")]
    ConfidenceSequenceError(String),

    /// Monitor is not initialized with baseline data.
    #[error("Monitor not initialized: {0}")]
    NotInitialized(String),

    /// Baseline distribution is empty or invalid.
    #[error("Invalid baseline: {0}")]
    InvalidBaseline(String),

    /// Feature map encoding failed.
    #[error("Feature map encoding failed: {0}")]
    FeatureMapError(String),
}

impl QuantumMonitorError {
    /// Create a dimension mismatch error.
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch { expected, actual }
    }

    /// Create an insufficient samples error.
    pub fn insufficient_samples(minimum: usize, actual: usize) -> Self {
        Self::InsufficientSamples { minimum, actual }
    }

    /// Create an invalid parameter error.
    pub fn invalid_parameter(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidParameter {
            name: name.into(),
            reason: reason.into(),
        }
    }

    /// Create a numerical error.
    pub fn numerical(msg: impl Into<String>) -> Self {
        Self::NumericalError(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = QuantumMonitorError::dimension_mismatch(128, 64);
        assert!(err.to_string().contains("128"));
        assert!(err.to_string().contains("64"));

        let err = QuantumMonitorError::insufficient_samples(100, 10);
        assert!(err.to_string().contains("100"));
        assert!(err.to_string().contains("10"));

        let err = QuantumMonitorError::invalid_parameter("sigma", "must be positive");
        assert!(err.to_string().contains("sigma"));
        assert!(err.to_string().contains("positive"));
    }

    #[test]
    fn test_error_clone() {
        let err = QuantumMonitorError::numerical("overflow");
        let cloned = err.clone();
        assert_eq!(err.to_string(), cloned.to_string());
    }
}
