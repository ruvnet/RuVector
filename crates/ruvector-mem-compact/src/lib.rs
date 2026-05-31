//! Graph-Cut Memory Compaction for agent vector stores.
//!
//! Provides three compaction strategies benchmarked against each other:
//! - `AgeTtlCompactor`: drops oldest entries (baseline)
//! - `ThresholdCompactor`: pointwise cosine-threshold deduplication
//! - `GraphCutCompactor`: k-NN graph clustering with centroid representatives

pub mod age_ttl;
pub mod compactor;
pub mod dataset;
pub mod graph_cut;
pub mod metrics;
pub mod threshold;

pub use age_ttl::AgeTtlCompactor;
pub use compactor::{CompactionResult, MemoryCompactor, MemoryEntry, MemoryStore};
pub use graph_cut::GraphCutCompactor;
pub use metrics::{cluster_coverage_recall, recall_at_k, CompactionMetrics};
pub use threshold::ThresholdCompactor;
