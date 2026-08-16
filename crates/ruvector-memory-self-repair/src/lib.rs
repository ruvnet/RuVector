//! Spectral drift detection and witness-gated targeted repair for agent
//! memory graphs.
//!
//! Reuses `ruvector_coherence::spectral` (Fiedler value, spectral gap,
//! effective resistance, degree regularity — the same primitives built for
//! HNSW index health monitoring) as the trigger signal for detecting
//! structural drift in an evolving agent-memory association graph, and adds
//! a bounded, hash-chain-witnessed repair action that reconnects the
//! structurally weakest nodes to their current semantic neighbors.
//!
//! See `docs/research/nightly/2026-08-16_spectral-memory-self-repair/README.md`
//! for the full hypothesis, methodology, and benchmark results.

pub mod drift;
pub mod embedding;
pub mod graph;
pub mod repair;
pub mod retrieval;
pub mod scoring;
pub mod witness;

pub use drift::{
    apply_arrival, apply_compaction, apply_drift_wave, build_arrivals, build_drift_schedule,
    DriftEvent, SimConfig,
};
pub use graph::MemoryGraph;
pub use repair::{RepairConfig, Runner, Variant};
pub use retrieval::mean_recall_at_k;
pub use witness::WitnessLog;
