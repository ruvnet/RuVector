//! Error types for ruvector-rairs.

use std::fmt;

/// Errors returned by RAIRS index operations.
#[derive(Debug, Clone, PartialEq)]
pub enum RairsError {
    /// Input vectors have inconsistent dimensionality.
    DimMismatch {
        /// Dimensionality the index was created with.
        expected: usize,
        /// Dimensionality of the offending vector.
        got: usize,
    },
    /// Index must be trained before search.
    NotTrained,
    /// Empty corpus passed to train.
    EmptyCorpus,
    /// k > n in top-k search.
    KTooLarge {
        /// Requested number of neighbours.
        k: usize,
        /// Number of vectors currently indexed.
        n: usize,
    },
    /// nprobe exceeds number of clusters.
    NprobeTooLarge {
        /// Requested number of lists to probe.
        nprobe: usize,
        /// Number of inverted lists in the index.
        nclusters: usize,
    },
    /// Invalid parameter value.
    InvalidParam(String),
}

impl fmt::Display for RairsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::NotTrained => write!(f, "index not trained"),
            Self::EmptyCorpus => write!(f, "corpus is empty"),
            Self::KTooLarge { k, n } => write!(f, "k={k} > n={n}"),
            Self::NprobeTooLarge { nprobe, nclusters } => {
                write!(f, "nprobe={nprobe} > nclusters={nclusters}")
            }
            Self::InvalidParam(msg) => write!(f, "invalid parameter: {msg}"),
        }
    }
}

impl std::error::Error for RairsError {}
