//! Error types for consciousness computation.

use std::time::Duration;

/// Primary error type for consciousness computations.
#[derive(Debug, thiserror::Error)]
pub enum ConsciousnessError {
    /// Φ computation did not converge within budget.
    #[error("phi did not converge after {iterations} iterations (current={current:.6}, delta={delta:.2e})")]
    PhiNonConvergence {
        iterations: usize,
        current: f64,
        delta: f64,
    },

    /// Numerical instability (NaN/Inf in matrix operations).
    #[error("numerical instability at partition {partition}: {detail}")]
    NumericalInstability { partition: usize, detail: String },

    /// Compute budget exhausted.
    #[error("budget exhausted: {reason}")]
    BudgetExhausted { reason: String, elapsed: Duration },

    /// Invalid input.
    #[error("invalid input: {0}")]
    InvalidInput(#[from] ValidationError),

    /// System too large for exact computation.
    #[error("system size {n} exceeds exact limit {max} — use approximate mode")]
    SystemTooLarge { n: usize, max: usize },
}

/// Validation errors raised before computation.
#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("non-finite value at element ({row}, {col})")]
    NonFiniteValue { row: usize, col: usize },

    #[error("TPM rows must sum to 1.0 (row {row} sums to {sum:.6})")]
    InvalidTPM { row: usize, sum: f64 },

    #[error("parameter out of range: {name} = {value} (expected {expected})")]
    ParameterOutOfRange {
        name: String,
        value: String,
        expected: String,
    },

    #[error("empty system: need at least 2 elements")]
    EmptySystem,
}

pub type Result<T> = std::result::Result<T, ConsciousnessError>;
