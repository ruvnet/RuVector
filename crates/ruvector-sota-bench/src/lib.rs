//! RuVector SOTA Benchmark Suite — ADR-265
//!
//! Proves RuVector against public leaderboards:
//! - ANN-Benchmarks (ann-benchmarks.com): recall@10 vs QPS
//! - VectorDBBench: commercial system comparison
//! - BEIR: zero-shot retrieval quality
//! - MTEB: embedding benchmark coverage
//!
//! # Score function (ADR-266)
//!
//! ```text
//! score = 0.40 × recall@10
//!       + 0.30 × log(QPS / baseline_QPS).clamp(0, 1)
//!       + 0.20 × (1 − memory_mb / baseline_memory_mb).max(0)
//!       + 0.10 × (1 − p99_ms / baseline_p99_ms).max(0)
//! ```
//!
//! Darwin Mode (MetaHarness) evolves the `scorePolicy` surface to
//! automatically maximize this score across all datasets.

pub mod datasets;
pub mod metrics;
pub mod report;
pub mod runners;

pub use metrics::{BenchScore, LatencyMetrics, RecallMetrics};
pub use report::{BenchReport, LeaderboardRow};

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Dataset descriptors
// ---------------------------------------------------------------------------

/// A benchmark dataset (synthetic or loaded from HDF5).
#[derive(Clone, Debug)]
pub struct Dataset {
    pub name: String,
    pub dims: usize,
    /// Corpus vectors — each is a slice of `dims` f32.
    pub corpus: Vec<Vec<f32>>,
    /// Query vectors.
    pub queries: Vec<Vec<f32>>,
    /// Ground-truth: for each query, the true top-100 nearest-neighbour ids.
    pub ground_truth: Vec<Vec<u64>>,
}

impl Dataset {
    /// Generate a synthetic Gaussian dataset (seeded, reproducible).
    pub fn synthetic(name: &str, n: usize, q: usize, dims: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::<f32>::new(0.0, 1.0).unwrap();

        let corpus: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dims).map(|_| normal.sample(&mut rng)).collect())
            .collect();
        let queries: Vec<Vec<f32>> = (0..q)
            .map(|_| (0..dims).map(|_| normal.sample(&mut rng)).collect())
            .collect();

        // Brute-force ground truth (top-100).
        let ground_truth: Vec<Vec<u64>> = queries
            .iter()
            .map(|q| brute_force_top_k(&corpus, q, 100))
            .collect();

        Dataset {
            name: name.to_string(),
            dims,
            corpus,
            queries,
            ground_truth,
        }
    }

    /// Recall@k between a result set and the ground truth for query `qi`.
    pub fn recall_at_k(&self, qi: usize, result_ids: &[u64], k: usize) -> f64 {
        let gt: std::collections::HashSet<u64> =
            self.ground_truth[qi].iter().take(k).cloned().collect();
        let res: std::collections::HashSet<u64> = result_ids.iter().take(k).cloned().collect();
        let hits = gt.intersection(&res).count();
        hits as f64 / k.min(gt.len()) as f64
    }
}

fn brute_force_top_k(corpus: &[Vec<f32>], query: &[f32], k: usize) -> Vec<u64> {
    let mut dists: Vec<(u64, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u64, sq_dist(v, query)))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.into_iter().take(k).map(|(id, _)| id).collect()
}

#[inline]
fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

// ---------------------------------------------------------------------------
// Configuration presets
// ---------------------------------------------------------------------------

/// Standard ANN-Benchmarks–compatible synthetic datasets.
pub fn standard_synthetic_datasets() -> Vec<Dataset> {
    vec![
        Dataset::synthetic("glove-25-angular", 100_000, 1_000, 25, 42),
        Dataset::synthetic("glove-100-angular", 100_000, 1_000, 100, 43),
        Dataset::synthetic("sift-128-euclidean", 100_000, 1_000, 128, 44),
        Dataset::synthetic("gist-960-euclidean", 5_000, 200, 960, 45),
        Dataset::synthetic("deep-image-96", 100_000, 1_000, 96, 46),
    ]
}

/// Minimal smoke-test datasets (fast, CI-safe).
pub fn smoke_test_datasets() -> Vec<Dataset> {
    vec![
        Dataset::synthetic("smoke-128", 10_000, 100, 128, 99),
        Dataset::synthetic("smoke-96", 5_000, 50, 96, 98),
    ]
}

// ---------------------------------------------------------------------------
// Scoring (ADR-266)
// ---------------------------------------------------------------------------

/// Compute the Darwin Mode / MetaHarness score for a benchmark run.
///
/// Higher is better. Typically in [0, 1].
pub fn darwin_score(
    recall_at_10: f64,
    qps: f64,
    baseline_qps: f64,
    mem_mb: f64,
    baseline_mem_mb: f64,
    p99_ms: f64,
    baseline_p99_ms: f64,
) -> f64 {
    let qps_term = ((qps / baseline_qps).ln().clamp(0.0, 1.0));
    let mem_term = (1.0 - mem_mb / baseline_mem_mb).max(0.0);
    let lat_term = (1.0 - p99_ms / baseline_p99_ms).max(0.0);
    0.40 * recall_at_10 + 0.30 * qps_term + 0.20 * mem_term + 0.10 * lat_term
}

// ---------------------------------------------------------------------------
// SOTA thresholds (ADR-267)
// ---------------------------------------------------------------------------

/// Minimum recall@10 to claim SOTA status on a dataset class.
pub const SOTA_RECALL_THRESHOLD: f64 = 0.95;

/// Minimum QPS ratio vs HNSWlib baseline to claim competitive throughput.
pub const SOTA_QPS_RATIO: f64 = 0.80;

/// Claim SOTA if both recall and QPS thresholds are met.
pub fn claim_sota(recall_at_10: f64, qps: f64, baseline_qps: f64) -> bool {
    recall_at_10 >= SOTA_RECALL_THRESHOLD && qps >= baseline_qps * SOTA_QPS_RATIO
}
