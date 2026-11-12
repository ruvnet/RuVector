//! Synthetic dataset generator matching ANN-Benchmarks distributions.
use crate::Dataset;

/// All 5 canonical ANN-Benchmarks synthetic datasets.
pub fn ann_benchmark_synthetic() -> Vec<Dataset> {
    crate::standard_synthetic_datasets()
}

pub fn ann_benchmark_synthetic_with_seed(seed: u64) -> Vec<Dataset> {
    crate::standard_synthetic_datasets_with_seed(seed)
}

/// Tiny smoke-test set for CI.
pub fn ci_smoke() -> Vec<Dataset> {
    crate::smoke_test_datasets()
}

pub fn ci_smoke_with_seed(seed: u64) -> Vec<Dataset> {
    crate::smoke_test_datasets_with_seed(seed)
}
