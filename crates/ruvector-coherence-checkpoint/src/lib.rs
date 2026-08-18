//! Coherence-drift-triggered snapshot scheduling for portable agent memory.
//!
//! `ruvector-agent-memory` already scores *which* memories to keep during
//! compaction; `ruvector-temporal-coherence` scores *how* a query should
//! weight memories by recency and graph coherence. Neither answers a third
//! scheduling question that every durable agent-memory deployment has to
//! solve: **when should the running store's state be checkpointed** into a
//! signed, portable snapshot (the RVF use case) so a crash or migration can
//! recover without replaying the entire write history from scratch?
//!
//! This crate implements and benchmarks three snapshot-scheduling policies
//! against that question:
//!
//! - [`policy::FixedInterval`] — baseline, snapshot every N writes.
//! - [`policy::DriftTriggered`] — snapshot when the store's running centroid
//!   has drifted (cosine distance) past a threshold since the last snapshot.
//! - [`policy::DriftTriggeredCapped`] — drift-triggered with a hard maximum
//!   interval, bounding worst-case replay gap during long calm periods.
//!
//! Every snapshot is witness-chained through `ruvector-proof-gate`'s
//! [`ruvector_proof_gate::HashChainGate`], and every policy's recovery
//! correctness is verified by exact vector-for-vector replay
//! reconstruction, not just digest comparison (see [`replay`]).

pub mod checkpoint;
pub mod drift;
pub mod policy;
pub mod replay;
pub mod workload;

pub use checkpoint::{run_checkpoint_policy, CheckpointRun, Snapshot};
pub use drift::{drift, RunningCentroid};
pub use policy::{CheckpointPolicy, DriftTriggered, DriftTriggeredCapped, FixedInterval};
pub use replay::verify_exact_replay;
pub use workload::{generate_workload, WorkloadConfig, WorkloadEvent};
