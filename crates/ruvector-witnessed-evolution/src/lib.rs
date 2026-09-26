//! Witness-chained provenance for evolutionary ANN parameter search.
//!
//! `ruvector-coherence-hnsw`'s coherence threshold and beam width are
//! query-time knobs with no principled default — production systems tune
//! them by hand or by an unaudited search loop, and once tuned there is no
//! record of *why* the chosen values were promoted over the alternatives
//! that were tried and rejected.
//!
//! This crate runs a (1+1)-evolution strategy over that genome and commits
//! every generation — genome, fitness, accept/reject decision — to a
//! `ruvector-proof-gate` hash chain via [`witness::WitnessedLineage`]. An
//! independent replayer ([`witness::WitnessedLineage::replay_verify`]) can
//! later recompute the entire lineage from the raw workload and confirm it
//! matches what was committed, byte for byte, without trusting the search
//! process that produced it.
//!
//! # Modules
//!
//! * [`genome`] — the two-parameter search-time genome and its mutation.
//! * [`fitness`] — deterministic fitness evaluation against a fixed
//!   `ruvector-coherence-hnsw` workload.
//! * [`witness`] — hash-chain commitment and replay verification.
//! * [`evolve`] — the (1+1)-ES loop, witnessed and unwitnessed variants.

pub mod evolve;
pub mod fitness;
pub mod genome;
pub mod witness;

pub use evolve::{run_unwitnessed, run_witnessed, EsOutcome};
pub use fitness::{Fitness, Workload, WorkloadConfig};
pub use genome::{Genome, DEFAULT_GENOME};
pub use witness::{GenerationRecord, ReplayReport, WitnessedLineage};
