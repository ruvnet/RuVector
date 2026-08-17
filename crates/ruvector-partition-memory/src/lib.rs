//! # ruvector-partition-memory
//!
//! Mincut-partitioned agent-memory consolidation.
//!
//! Nightly research (2026-08-17). Hypothesis: a global top-score compaction
//! policy (the winning `CoherencePolicy` from `ruvector-agent-memory`,
//! nightly 2026-06-14) is a session-context-biased scalar ranking, so a
//! semantic topic unrelated to the agent's most recent working context can
//! be evicted *in full* during compaction even at a favourable overall
//! compaction ratio. This crate tests whether partitioning the memory
//! similarity graph before applying a retention budget — guaranteeing each
//! partition a floor allocation — protects minority topics without
//! materially regressing overall recall.
//!
//! Three variants are benchmarked in `src/main.rs`:
//!
//! - `GlobalTopScore` (baseline): `ruvector_agent_memory::CoherencePolicy`
//!   applied directly, scored against a recency-biased context window.
//! - `MincutFixedK` (candidate A): `ruvector_mincut::GraphPartitioner`
//!   (existing, unweighted edge-count recursive bisection) with a
//!   size-heuristic `K`, then floor+proportional retention per partition.
//! - `MincutAdaptive` (candidate B): a new adaptive-depth recursive
//!   bisection that stops each branch once its normalized cut density
//!   indicates the component is already coherent, then applies the same
//!   floor+proportional retention.
//!
//! `mincut_exact.rs` exists because `ruvector_mincut::DynamicMinCut::partition()`
//! turned out, during this nightly's development, to return a vertex split
//! inconsistent with its own `min_cut_value()` (see that module's doc
//! comment and the nightly research doc for the repro). Candidate B's
//! splits are materialized by a from-scratch, tested Stoer–Wagner
//! implementation instead; `ruvector_mincut`'s value is still queried as an
//! independent cross-check, recorded in the witness chain.
//!
//! Every partition decision candidate B makes is committed to a SHA-256
//! witness chain (`witness.rs`) so a split cannot be silently re-run to
//! favour a result after the fact.

pub mod corpus;
pub mod graph;
pub mod metrics;
pub mod mincut_exact;
pub mod partition;
pub mod retention;
pub mod search;
pub mod witness;

pub use corpus::{Corpus, CorpusConfig, MemoryRecord, Query};
pub use graph::build_knn_edges;
pub use metrics::{evaluate, VariantReport};
pub use mincut_exact::global_min_cut;
pub use partition::{adaptive_partition, fixed_k_partition, AdaptiveConfig, PartitionResult};
pub use retention::{retain_global_top_score, retain_partitioned, RetentionPolicy};
pub use witness::{PartitionWitnessChain, SplitRecord};
