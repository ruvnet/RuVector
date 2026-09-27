//! `ruvector-kge-bench` — the native benchmark binary of ADR-007 (plan M3).
//!
//! - [`datasets`]: pinned, hash-verified loaders identical to the JS harness.
//! - [`config`]: run config, the C1–C8 grid, `config_hash`.
//! - [`runner`]: valid-only training with early stop, best-on-valid, receipts.
//! - [`checkpoint`]: atomic checkpoints; bit-identical resume.
//! - [`export`]: tables-only weights (no triples, no labels).
//! - [`final_mode`] / [`scoring`] / [`gitops`] / [`ledger`]: the ledger-gated
//!   one-time test scoring and its verification.
//!
//! The core crate stays fs-free; all I/O is here.

pub mod canon;
pub mod checkpoint;
pub mod config;
pub mod datasets;
pub mod export;
pub mod final_mode;
pub mod gitops;
pub mod hpo;
pub mod ledger;
pub mod metrics;
pub mod receipt;
pub mod runner;
pub mod scoring;
pub mod selection;
