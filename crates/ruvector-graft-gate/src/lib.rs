//! `ruvector-graft-gate`: mincut-gated proximity-graph insertion.
//!
//! A write-time structural defense against RAG corpus-poisoning
//! insertions into a graph-based ANN index. Extends the "is this write
//! honest" question `ruvector-proof-gate` (ADR-227) answers cryptographically
//! with a distinct "is this write locally coherent" structural question,
//! evaluated at insertion time against the candidate's would-be graph
//! neighborhood.
//!
//! See `docs/research/nightly/2026-08-30-mincut-gated-insertion/README.md`
//! and `docs/adr/ADR-340-mincut-gated-insertion.md` in the repository root
//! for the hypothesis, attack model, benchmark methodology, and results.

pub mod config;
pub mod data;
pub mod gate;
pub mod graph_index;
pub mod rng;
pub mod vector;

pub use gate::{evaluate as evaluate_gate, GateConfig, GateDecision, GateVariant};
pub use graph_index::{brute_force_top_k, GraphIndex};
pub use rng::Xorshift64;
pub use vector::Vector;
