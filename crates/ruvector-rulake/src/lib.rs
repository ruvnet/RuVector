//! ruLake — vector-native federation intermediary
//!
//! Implements the MVP shape from ADR-155: a `BackendAdapter` trait that
//! each data-lake backend implements, a RaBitQ-compressed cache that sits
//! in front of the backends, and a router that fans out a query across
//! backends and merges their top-k by score.
//!
//! ## Flow
//!
//! ```text
//!   caller
//!     │  RuLake::search(collection, query, k)
//!     ▼
//!   router ─── cache hit? ──────────────────┐
//!     │ miss                                 │
//!     ▼                                      │
//!   BackendAdapter::pull_vectors             │
//!     │                                      │
//!     ▼                                      │
//!   cache.prime (RaBitQ compress)            │
//!     │                                      │
//!     └──── search the cache ────────────────┤
//!                                             ▼
//!                                        SearchResult
//! ```
//!
//! ## What v1 ships
//!
//! - `BackendAdapter` trait — minimum surface: `id`, `list_collections`,
//!   `pull_vectors`, `generation` (coherence token).
//! - `LocalBackend` — in-memory adapter for tests and demos.
//! - `VectorCache` — wraps `ruvector_rabitq::RabitqPlusIndex` and keeps a
//!   per-collection generation so staleness is checkable.
//! - `RuLake` — the public entry: register backends, run search.
//!
//! ## What v1 explicitly does not ship
//!
//! - Push-down to backend-native vector ops (v1.1 inside each adapter).
//! - Parquet / BigQuery / Snowflake / Iceberg / Delta adapters
//!   (follow-up crates; M2–M5 in `docs/research/ruLake/07-implementation-plan.md`).
//! - RBAC / PII / lineage / audit — M4 in the plan.
//! - Persistent cache — current cache is RAM-only.

#![allow(clippy::needless_range_loop)]

pub mod backend;
pub mod bundle;
pub mod cache;
pub mod error;
pub mod fs_backend;
pub mod lake;

pub use backend::{BackendAdapter, BackendId, CollectionId, LocalBackend, PulledBatch};
pub use bundle::{Generation, RuLakeBundle};
pub use cache::{CacheStats, PerBackendStats, VectorCache};
pub use error::{Result, RuLakeError};
pub use fs_backend::FsBackend;
pub use lake::{RefreshResult, RuLake, SearchResult};
