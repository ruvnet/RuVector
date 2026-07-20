//! Adaptive ef-search control for HNSW-style approximate nearest-neighbour indexes.
//!
//! Production HNSW deployments require a fixed `ef_search` parameter: low values yield fast
//! but imprecise results; high values improve recall but spike latency. This crate implements
//! three self-tuning policies that observe query latency and estimated recall in real time and
//! adjust `ef` automatically to meet a caller-specified latency budget while defending a recall
//! floor.
//!
//! # Policies
//!
//! | Policy | Mechanism | Best for |
//! |--------|-----------|----------|
//! | `FixedPolicy` | Constant ef (baseline) | Stable workloads, manual tuning |
//! | `EwmaGreedy` | EWMA latency + hill-climb | Smooth workloads with slow drift |
//! | `BanditPolicy` | ε-greedy multi-armed bandit | Bursty / mixed workloads |
//! | `PidController` | PID control on latency error | Strict latency SLA environments |
//!
//! # Quick start
//!
//! ```rust
//! use ruvector_adaptive_ef::{SearchPolicy, EwmaGreedy};
//!
//! let mut policy = EwmaGreedy::new(64, 0.1);
//! let ef = policy.recommend_ef(200); // budget 200 µs
//! // … run search …
//! policy.observe(180, 0.92, ef);     // latency_us, recall, ef actually used
//! ```

pub mod hnsw_sim;
pub mod metrics;
pub mod policy;

pub use hnsw_sim::HnswSim;
pub use metrics::LatencyWindow;
pub use policy::{BanditPolicy, EwmaGreedy, FixedPolicy, PidController, SearchPolicy};
