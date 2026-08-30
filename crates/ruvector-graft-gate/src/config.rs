//! Experiment constants for the mincut-gated-insertion nightly (ADR-340).
//!
//! Pre-registered before benchmarking per the pipeline's acceptance
//! rules: these values (thresholds included) were fixed before the
//! benchmark binary was run for the first time and were not adjusted
//! afterward. See the nightly README's "Hypothesis" section for the
//! human-readable statement these constants encode.

pub const DIM: usize = 64;
pub const NUM_CLUSTERS: usize = 20;
pub const CLUSTER_SIGMA: f32 = 0.15;
pub const N_CLEAN: usize = 5_000;
pub const N_TARGET_QUERIES: usize = 50;
pub const POISON_PER_TARGET: usize = 4; // 200 poison insertions total
pub const POISON_ALPHA: f32 = 0.7;
pub const N_ADDITIONAL_LEGIT: usize = 1_000;

pub const GRAPH_M: usize = 16;
pub const EF_CONSTRUCTION: usize = 64;
pub const EF_SEARCH: usize = 64;
pub const TOP_K: usize = 10;

pub const GATE_K: usize = 10;
pub const PEAKEDNESS_THRESHOLD: f32 = 1.35;
pub const MINCUT_EDGE_FACTOR: f32 = 0.85;
pub const MINCUT_REJECT_BELOW: usize = 2;
pub const BOOTSTRAP_MIN_INDEX_SIZE: usize = GATE_K;

pub const SEED_CENTROIDS: u64 = 0xC0FF_EE00_0000_0001;
pub const SEED_CLEAN_CORPUS: u64 = 0xC0FF_EE00_0000_0002;
pub const SEED_TARGET_QUERIES: u64 = 0xC0FF_EE00_0000_0003;
pub const SEED_POISON: u64 = 0xC0FF_EE00_0000_0004;
pub const SEED_ADDITIONAL_LEGIT: u64 = 0xC0FF_EE00_0000_0005;
pub const SEED_INTERLEAVE_SHUFFLE: u64 = 0xC0FF_EE00_0000_0006;

/// Acceptance thresholds (Given/When/Then/Subject-to clauses).
pub const LATENCY_BUDGET_NS: f64 = 500_000.0; // 500 microseconds, mean gate overhead
pub const RECALL_DROP_BUDGET_PP: f32 = 2.0;
pub const CATCH_RATE_GAP_BUDGET_PP: f32 = 10.0;
pub const ATTACK_SUCCESS_GAP_BUDGET_PP: f32 = 20.0;
pub const LEGIT_FALSE_REJECT_BUDGET_PCT: f32 = 5.0;
