//! Adaptive ANN ef-search parameter tuning via multi-armed bandits.
//!
//! Most ANN indexes expose an `ef` (beam-width) parameter: larger ef yields
//! better recall at higher latency; smaller ef is faster but misses more
//! neighbors. In production the right ef depends on the query distribution,
//! the index structure, and the operator's recall/latency preference — and
//! that distribution shifts over time.
//!
//! This crate treats ef-selection as a **multi-armed bandit problem**.
//! Each arm corresponds to one candidate ef value. After every query the
//! arm receives a reward (recall@k vs. a reference set). Two policies are
//! implemented:
//!
//! - **UCB1**: Upper Confidence Bound, provably sub-linear regret.
//! - **ε-Greedy w/ decay**: simple, practical, converges in practice.
//!
//! A fixed-ef baseline is included for comparison.

pub mod bandit;
pub mod dataset;
pub mod graph;
pub mod metrics;
pub mod search;

/// Configuration shared across search variants.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Candidate ef values (one bandit arm each).
    pub ef_candidates: Vec<usize>,
    /// Top-k neighbors to return.
    pub k: usize,
    /// UCB1 exploration constant (√2 is the theoretically optimal default).
    pub exploration_c: f64,
    /// Initial ε for ε-greedy.
    pub epsilon_init: f64,
    /// Multiplicative decay applied after every query.
    pub epsilon_decay: f64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            ef_candidates: vec![10, 25, 50, 100],
            k: 10,
            exploration_c: 1.414,
            epsilon_init: 0.30,
            epsilon_decay: 0.999,
        }
    }
}

/// Result of a single ANN query.
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Neighbor indices sorted by ascending distance.
    pub indices: Vec<usize>,
    /// ef value selected for this query.
    pub ef_used: usize,
    /// Wall-clock latency in nanoseconds.
    pub latency_ns: u64,
}

/// Trait for adaptive ef-search strategies.
pub trait AdaptiveSearch: Send {
    /// Human-readable name of this variant.
    fn name(&self) -> &str;

    /// Run one query. `ground_truth` holds the exact k-NN indices and is used
    /// only for reward computation (it is NOT used to bias the search itself).
    fn query(&mut self, q: &[f32], ground_truth: &[usize]) -> QueryResult;

    /// The ef value the policy would pick right now (exploitation only).
    fn current_best_ef(&self) -> usize;

    /// Number of queries processed so far.
    fn query_count(&self) -> usize;

    /// Bytes consumed by bandit state alone (graph excluded).
    fn bandit_memory_bytes(&self) -> usize;
}

/// Aggregate statistics over a full run.
#[derive(Debug, Clone)]
pub struct RunStats {
    pub variant: String,
    pub n_vectors: usize,
    pub dims: usize,
    pub n_queries: usize,
    pub recall_at_k: f64,
    pub mean_latency_us: f64,
    pub p50_latency_us: f64,
    pub p95_latency_us: f64,
    pub throughput_qps: f64,
    /// Total memory: graph + bandit state.
    pub memory_bytes: usize,
    /// ef the policy settled on at the end.
    pub final_best_ef: usize,
}
