//! ruvector edge analytics core: the rv-mincut service (ADR-351 §3, §10,
//! §15 M4).
//!
//! * [`graph`] — a tenant graph is a validated, canonical undirected edge
//!   list; `ruvector-mincut`'s `DynamicGraph` is not serializable, so the
//!   solver is rebuilt from the list through `MinCutBuilder` on load.
//! * [`format`] — the persisted form: versioned, sha256-checksummed,
//!   chunked (each chunk one Durable Object BLOB), graph-bound.
//! * [`cost`] — a-priori work/memory estimate that predicts the solver
//!   branch, so budgets are enforced before the solver runs.
//! * [`service`] — exact queries (approximate requests are answered
//!   exactly; the heuristic solver is not served) under limits and a
//!   budget; `BudgetExceeded`/`LimitExceeded` map to HTTP 413.
//! * [`job`] — async job descriptor (`queued -> running -> done|failed`).
//! * [`runtime`] — rayon behaviour on wasm32 (single thread, no panic).
//!
//! Pure Rust; compiles for `wasm32-unknown-unknown`.

#![warn(missing_docs)]

mod codec;
pub mod cost;
pub mod error;
pub mod format;
pub mod graph;
pub mod job;
pub mod runtime;
pub mod service;

pub use cost::{CostEstimate, Precheck, SolverPath};
pub use error::{AnalyticsError, CorruptKind, LimitKind, Result};
pub use format::{decode_graph, encode_graph, EncodedGraph, Manifest, MAX_CHUNK_EDGES};
pub use graph::{EdgeRecord, GraphLimits, GraphUid, TenantGraph};
pub use job::{route, JobDescriptor, JobState, Route};
pub use service::{plan, query, Budget, CutReport, Profile, QueryMode};
