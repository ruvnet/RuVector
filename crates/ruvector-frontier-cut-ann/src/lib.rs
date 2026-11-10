//! # ruvector-frontier-cut-ann
//!
//! Frontier-cut conductance as a per-query adaptive `ef_search` signal for
//! graph-ANN beam search.
//!
//! ## Why this crate exists
//!
//! The 2026-08-13 `ruvector-entropy-ann` nightly (ADR-303) tested whether the
//! Shannon entropy of the *result heap's distance distribution* could drive
//! per-query beam-width control. It was refuted by measurement: softmin
//! entropy over already-retrieved neighbour distances tracks local point
//! density, not routing ambiguity, and has the wrong sign (hard queries
//! showed *lower* entropy than easy ones on that benchmark).
//!
//! That failure is specific to a *distance-based* signal computed over the
//! result heap. This crate tests a different, purely *topological* signal
//! computed over the **search frontier's induced cut** in the proximity
//! graph — the same family of object (sparsest cut / conductance) that
//! `ruvector-mincut`'s local k-cut tooling already computes for graph
//! partitioning elsewhere in this repository, applied here as a live
//! traversal-control signal instead of an offline partitioning tool.
//!
//! **Hypothesised cut semantics:**
//! - As the beam visits nodes, each expanded node's k-NN edges either point
//!   back into the already-visited set (*internal* edges) or out to
//!   unvisited territory (*cut* edges).
//! - `phi = cut_edges / (cut_edges + internal_edges)` is the fraction of
//!   edges crossing the visited/unvisited boundary — a cheap, incremental
//!   conductance proxy for the local region the search is currently in.
//! - Inside a tight, well-separated cluster, repeated k-NN edges fold back
//!   onto already-visited neighbours as the frontier grows (low `phi`):
//!   the query is "easy", the local region is nearly closed off.
//! - Near a cluster boundary or in a sparse region, the frontier keeps
//!   discovering new, previously-unseen territory (high `phi`): the query
//!   is "hard" and needs a larger beam before it stabilises.
//!
//! This is unrelated to inter-point distance and is computed from the graph
//! topology alone, so it does not inherit entropy's wrong-sign failure mode
//! — though whether it has the *right* sign, and whether it is *more* than a
//! constant, is an empirical question this crate's benchmark answers rather
//! than assumes. See the research README and ADR-353 for the measured
//! outcome; this module does not claim a result in advance.
//!
//! ## Variants
//!
//! | Variant | Strategy | Description |
//! |---------|----------|-------------|
//! | `FixedEfSearch` (from `ruvector-entropy-ann`) | Baseline | Fixed `ef` budget |
//! | [`cut::CutThresholdBeam`] | Reactive | Stop expanding when running `phi` < `phi_stop` |
//! | [`cut::FrontierCutScaledEf`] | Predictive | Probe `phi` at depth-5, scale `ef` for the rest |
//!
//! ## RuVector fit
//!
//! Connects HNSW/ANN beam search (`ruvector-core`), graph-cut/conductance
//! concepts already used in `ruvector-mincut` for partitioning, and the
//! agent-memory adaptive-retrieval use case documented in the
//! `ruvector-entropy-ann` and `ruvector-coherence-hnsw` nightlies.

pub mod cut;

pub use cut::{CutProbeStats, CutThresholdBeam, FrontierCutScaledEf};

// Re-export the shared benchmark primitives from the entropy-ann crate so
// downstream code (and this crate's own benchmark binary) has one import
// path for dataset generation, the flat graph, the baseline searcher, and
// recall measurement.
pub use ruvector_entropy_ann::{
    dataset, graph,
    metrics::LatencyStats,
    recall_at_k,
    search::{FixedEfSearch, Hit, Searcher},
};
