//! # ruvector-memcompact — Agent Memory Compaction via Graph-Cut Clustering
//!
//! Implements three memory compaction strategies for AI agent memory stores:
//!
//! | Strategy            | Description                                        |
//! |---------------------|----------------------------------------------------|
//! | `AgeEviction`       | Keep the N most recently timestamped entries       |
//! | `ImportanceEviction`| Keep the N highest-importance entries              |
//! | `GraphCutCompaction`| Cluster by cosine similarity, merge to centroids   |
//!
//! All strategies implement [`CompactionStrategy`].  Design rationale and
//! benchmark evidence are in `docs/adr/ADR-194-agent-memory-graph-compaction.md`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod compaction;
pub mod graph;
pub mod memory;
pub mod metrics;

pub use compaction::{AgeEviction, CompactionStrategy, GraphCutCompaction, ImportanceEviction};
pub use memory::{MemId, MemoryEntry, MemoryStore};
pub use metrics::{mean_recall_at_k, recall_at_k};
