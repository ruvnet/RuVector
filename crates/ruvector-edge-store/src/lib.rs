//! # ruvector-edge-store
//!
//! Storage core for ruvector edge (ADR-351 §6, §16), to be wired into the
//! gateway's Durable Objects:
//!
//! - [`ports`]: the DO-SQLite-shaped [`SqlStore`] port (plus the clock and
//!   entropy ports); [`mem_store`] is its in-memory implementation.
//! - [`schema`]: every SQL statement, as constants.
//! - [`shard`]: `VectorShard` — M2 int8 `flat` (default) or `hnsw` index
//!   resident, exact f32 rerank from SQLite, metadata filter, chunked index
//!   persistence with lazy load and replay fallback, alarm maintenance,
//!   op log, per-shard resident cap.
//! - [`ledger`]: `TenantLedger` — memberships (default-deny), collection
//!   catalog with never-reused `collection_uid`s, checked quota counters,
//!   `op_id` idempotency.
//! - [`ops`]: the `/v1/ops` dispatcher (§16.3) with every error code.
//! - [`resident`]: the isolate-wide resident-set registry (LRU eviction).
//! - [`distance`], [`filter`]: kernels and metadata filters.
//!
//! Pure Rust, `wasm32-unknown-unknown` clean: no `std::time`, no threads,
//! no `getrandom`; time and randomness come in through [`ports::Clock`] and
//! [`ports::EntropySource`].

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod context;
pub mod distance;
pub mod error;
pub mod filter;
pub mod ledger;
pub mod mem_store;
pub mod ops;
pub mod ports;
pub mod resident;
pub mod schema;
pub mod shard;

pub use context::{ledger_meta_for, shard_meta_for, CallerContext};
pub use distance::Metric;
pub use error::{ErrorCode, OpError};
pub use ledger::{CatalogEntry, CreateCollection, TenantLedger};
pub use mem_store::MemSqlStore;
pub use ops::{Dispatcher, LocalCluster, Op, OpReply, OpRequest, OpResponse};
pub use ports::{Clock, EntropySource, Row, SqlStore, StoreError, Value};
pub use resident::ResidentRegistry;
pub use shard::{IndexConfig, QueryRequest, ShardConfig, UpsertRow, VectorShard};
