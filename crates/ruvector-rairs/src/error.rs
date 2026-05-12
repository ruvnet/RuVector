//! Error types for ruvector-rairs.

use std::fmt;

/// Errors returned by RAIRS index operations.
#[derive(Debug, Clone, PartialEq)]
pub enum RairsError {
    /// Input vectors have inconsistent dimensionality.
    DimMismatch { expected: usize, got: usize },
    /// Index must be trained before search.
    NotTrained,
    /// Empty corpus passed to train.
    EmptyCorpus,
    /// k > n in top-k search.
    KTooLarge { k: usize, n: usize },
    /// nprobe exceeds number of clusters.
    NprobeTooLarge { nprobe: usize, nclusters: usize },
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
