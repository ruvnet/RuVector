//! BET 2 ⊗ BET 4 — Region-pruned filtered ANN vs tuned ACORN.
//!
//! Pre-registered head-to-head (see `docs/plans/bet2-filtered-ann/PRE-REGISTRATION.md`):
//! does IVF **cluster-skip** pruning beat predicate-agnostic ACORN on *correlated*
//! predicates at low selectivity, by ≥5× distance-evals/query at equal (±2%) recall?
//!
//! This crate is **self-contained**: it depends only on `ruvector-acorn` (the incumbent and
//! the `exact_filtered_knn` oracle) and `ruvector-rairs` (the IVF substrate). It has no
//! dependency on `ruvector-seprag` (PR #535), so it ships as an independent PR.
//!
//! ## Module map (filled across milestones)
//! - `data` (M0)      — load ogbn-arxiv features / labels / years.
//! - `predicate` (M0) — predicate families + ρ-correlation knob + selectivity targeting.
//! - `prune` (M2)     — contender A: region-pruned IVF filtered search + eval counters.

pub mod contenders;
pub mod data;
pub mod predicate;
pub mod prune;

// Re-export the substrate + incumbent + oracle so the head-to-head harness has one entry
// point and the dependency graph is exercised at build time.
pub use ruvector_acorn::{recall_at_k, AcornIndexGamma, FilteredIndex, FlatFilteredIndex};
pub use ruvector_rairs::ivf::IvfFlat;

/// Exact filtered k-NN oracle (brute force) — ground truth for every contender.
/// Thin re-export of the in-repo incumbent's oracle to keep one source of truth.
pub use ruvector_acorn::graph::exact_filtered_knn;
