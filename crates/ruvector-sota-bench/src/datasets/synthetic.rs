//! Synthetic dataset generator matching ANN-Benchmarks distributions.
use crate::Dataset;

/// All 5 canonical ANN-Benchmarks synthetic datasets.
pub fn ann_benchmark_synthetic() -> Vec<Dataset> {
    crate::standard_synthetic_datasets()
}

/// Tiny smoke-test set for CI.
pub fn ci_smoke() -> Vec<Dataset> {
    crate::smoke_test_datasets()
}
