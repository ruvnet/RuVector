//! # ruvector-voi-router
//!
//! Value-of-information (VoI) routing between [`BaselineSearch`] and
//! [`CoherenceGatedSearch`] (both from `ruvector-coherence-hnsw`), decided
//! **per query** instead of picking one policy for the whole workload.
//!
//! ## Motivation
//!
//! `ruvector-coherence-hnsw`'s coherence gate (accepted 2026-06-16) prunes
//! off-path neighbor expansions when the search entry point is far from the
//! query — the fixed layer-0 entry has to traverse the graph before it gets
//! near the query, and the gate skips branches that visibly point away from
//! it. When the entry is already near the query, the beam converges in a
//! handful of hops regardless of gating: there is little left to prune, so
//! the gate's per-pop coherence check (a dot product) is pure overhead, and
//! its fixed threshold is one more way to occasionally mis-prune a genuinely
//! useful branch for no benefit.
//!
//! This crate tests whether a **free** per-query signal — the squared L2
//! distance from the entry point to the query, `d0`, which every beam search
//! computes as its very first step anyway — predicts which regime a query is
//! in well enough to route it to the cheaper policy without materially
//! changing recall. The threshold on `d0` is calibrated once (from a
//! calibration query set, never the evaluation set) via a small bounded
//! search (see [`calibrate`]).
//!
//! [`BaselineSearch`]: ruvector_coherence_hnsw::search::BaselineSearch
//! [`CoherenceGatedSearch`]: ruvector_coherence_hnsw::search::CoherenceGatedSearch

pub mod calibrate;
pub mod router;

pub use router::{RoutedResult, VoiRoutedSearch};
