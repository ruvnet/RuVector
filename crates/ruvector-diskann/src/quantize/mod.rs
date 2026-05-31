//! Pluggable quantizer abstraction for DiskANN.
//!
//! DiskANN's hot paths (graph traversal + candidate distance estimation) only
//! need three things from a quantizer:
//!
//! 1. **Train** on a slice of training vectors so codebooks / rotations /
//!    centroids are fitted to the data.
//! 2. **Encode** an arbitrary input vector into a compact byte slice.
//! 3. **Estimate distance** from a prepared query handle (the fast path) to a
//!    stored code, without touching the original f32 vector.
//!
//! Everything else (codebook size, internal layout, on-disk format) is private
//! to the implementation. Two concrete impls ship here:
//!
//! | Impl | Compression | Distance estimator | Feature |
//! |------|-------------|--------------------|---------|
//! | [`ProductQuantizer`] | M bytes / vec (≈ 8–16×) | PQ asymmetric LUT | always on |
//! | [`RabitqQuantizer`] | ⌈D/8⌉ bytes / vec (≈ 32×) | RaBitQ angular | `rabitq` |
//!
//! ## Pattern 1 — direct embed (per `docs/research/nightly/2026-04-23-rabitq`)
//!
//! `RabitqQuantizer` is implemented in this crate by taking a path dependency
//! on `ruvector-rabitq` and using `RabitqIndex` directly for encoding /
//! distance. We deliberately do **not** route through the `VectorKernel` trait
//! at this stage — that is reserved for ruLake's kernel registry (see ADR-154
//! and the integration roadmap).
//!
//! ## Determinism
//!
//! ADR-154 requires `(seed, dim, vectors) → bit-identical codes`. Both impls
//! honour this: PQ via `rand::thread_rng()` is **non-deterministic** today
//! (pre-existing behaviour of this crate), but the new RaBitQ quantizer takes
//! an explicit seed and forwards it to the rotation matrix, so the RaBitQ path
//! is fully reproducible. Closing the determinism gap on PQ is out of scope
//! for this PR.

use crate::error::Result;

pub mod pq;

#[cfg(feature = "rabitq")]
pub mod rabitq;

pub use pq::ProductQuantizer;

#[cfg(feature = "rabitq")]
pub use rabitq::RabitqQuantizer;

/// Minimal interface DiskANN needs from a quantizer.
///
/// The trait is split into a build-time half (`train`, `encode`) and a
/// query-time half (`prepare_query`, `distance`). The query handle is an
/// associated type so each impl can ship whatever shape it needs (PQ uses a
/// flat lookup table; RaBitQ uses a rotated unit query plus its norm).
pub trait Quantizer: Send + Sync {
    /// Per-query precomputed state used by [`Self::distance`].
    type Query;

    /// Vector dimensionality this quantizer is configured for.
    fn dim(&self) -> usize;

    /// Bytes produced by a single call to [`Self::encode`]. Constant for the
    /// lifetime of a trained quantizer.
    fn code_bytes(&self) -> usize;

    /// Whether [`Self::train`] has been called and the quantizer is ready to
    /// encode.
    fn is_trained(&self) -> bool;

    /// Fit codebooks / rotations on a set of training vectors. Idempotent
    /// failure: returning `Err` leaves the quantizer in an untrained state.
    fn train(&mut self, vectors: &[Vec<f32>], iterations: usize) -> Result<()>;

    /// Encode a single vector into the impl-defined compact form.
    fn encode(&self, vector: &[f32]) -> Result<Vec<u8>>;

    /// Build a per-query handle. Done **once per search** and reused across
    /// every candidate.
    fn prepare_query(&self, query: &[f32]) -> Result<Self::Query>;

    /// Estimated squared-L2 distance between the prepared query and a stored
    /// code. Hot path — must not allocate.
    fn distance(&self, query: &Self::Query, code: &[u8]) -> f32;
}
