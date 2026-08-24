//! Structural-time agent memory decay.
//!
//! Agent memory compaction (e.g. `ruvector-agent-memory`, nightly 2026-06-14)
//! scores a stored memory's "recency" against **wall-clock time**: the number
//! of turns/steps since it was written. This crate isolates that one variable
//! and asks whether `emergent-time`'s [`emergent_time::StructuralProperTime`]
//! — internal time defined as accumulated *embedding-arc-length* (and,
//! optionally, entropy) rather than step count — makes a better recency clock
//! for compaction retention scoring.
//!
//! The mechanism is simple: during a long, low-drift stretch of a session
//! (the agent is heads-down on one topic), a structural clock accumulates
//! almost no internal time, so memories written early and late in that
//! stretch end up at nearly the same structural age even though many wall
//! steps separate them. A wall clock cannot make that distinction — it ages
//! every memory at a constant rate regardless of whether anything actually
//! changed. See `src/main.rs` for the benchmark that measures the
//! consequence: compaction recall against an oracle nearest-neighbour set.
//!
//! Three clocks are compared, all literal instances of `emergent-time`
//! types — no new clock math is introduced by this crate:
//!
//! 1. [`emergent_time::WallClock`] — the baseline (today's production
//!    convention).
//! 2. `StructuralProperTime` with only the embedding channel weighted
//!    ([`clocks::structural_embedding_clock`]) — pure accumulated context
//!    drift.
//! 3. `StructuralProperTime` with embedding + entropy channels weighted
//!    ([`clocks::structural_full_clock`]) — drift plus a genuine derived
//!    "topic uncertainty" signal (Shannon entropy of the softmax over
//!    cosine similarities to the session's topic centroids, via
//!    [`emergent_time::entropy::entropy_from_spectrum`]).
//!
//! The graph and prediction-error channels of `StructuralProperTime` are left
//! at zero weight throughout: this harness has no honest signal source for
//! them (no dependency graph, no forward model). Wiring those channels to
//! real RuVector primitives (`ruvector-mincut` for `ΔG`, a task-success
//! predictor for `ΔE`) is noted as future work, not simulated here.

pub mod clocks;
pub mod compaction;
pub mod scenario;

pub use clocks::{structural_embedding_clock, structural_full_clock};
pub use compaction::{compact, oracle_top_k, recall_at_k, CompactionWeights};
pub use scenario::{generate_session, ScenarioConfig, Session};
