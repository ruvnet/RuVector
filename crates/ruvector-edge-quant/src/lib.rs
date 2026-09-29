//! # ruvector-edge-quant
//!
//! The rv-quant core of ruvector edge (ADR-351 §3, milestone M4): one
//! tenant shard's RaBitQ index, built on `ruvector-rabitq`'s seeded
//! rotations, sign quantiser and popcount scan kernel.
//!
//! - [`QuantShard`]: SoA codes keyed by the store's `u64` row keys, with
//!   all-or-nothing upsert, delete, and top-k query in the collection
//!   metric (cosine / L2 / dot, estimated from the 1-bit cosine).
//! - [`persist`]: the v2 snapshot (`rbqx0002`): packed codes, norms, keys
//!   and rotation kind/seed/fingerprint, chunked into ≤ 1 MiB CRC-checked
//!   frames. Cold load never re-encodes from f32.
//! - [`budget`]: resident-byte and work-unit accounting; every refusal is
//!   [`QuantError::BudgetExceeded`] → `413 budget_exceeded`, checked before
//!   the work or allocation it guards.
//! - [`query`]: optional exact rerank of the scan's candidates through a
//!   [`RerankSource`] callback, so f32 originals stay in the store.
//!
//! Pure Rust and `wasm32-unknown-unknown` clean: no `unsafe` in this crate,
//! no clock, no threads, no OS entropy (rotations are seeded).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod budget;
pub mod error;
pub mod persist;
pub mod query;
pub mod shard;

pub use budget::Budget;
pub use error::{BudgetResource, CorruptKind, QuantError, Result};
pub use query::{QueryHit, QueryOptions, QueryOutcome, QueryStats, RerankSource};
pub use ruvector_edge_store::Metric;
pub use ruvector_rabitq::RandomRotationKind;
pub use shard::{QuantConfig, QuantShard, UpsertStats};
