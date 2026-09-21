//! `ruvector-embed-core` — sentence-embedding backends for `@ruvector/typesafe`.
//!
//! Two implementations of the [`ruvector_typesafe_core::Embedder`] trait, each
//! behind a feature (ADR-002):
//!
//! * `native` — [`OrtEmbedder`] on `ort` 2.0.0-rc.13 (onnxruntime CPU).
//! * `wasm` — [`TractEmbedder`] on `tract-onnx` 0.23 (pure-Rust, wasm-safe).
//!
//! The default build (`--features` none) compiles **no** inference engine and
//! downloads nothing: only the manifest, pooling and error types are present,
//! so `cargo build --workspace` is cheap and network-free.
//!
//! Model weights are pinned by a content-hash [`ModelManifest`] and verified at
//! load; a mismatch is a typed error and nothing is loaded (fail closed,
//! ADR-005). There is no network at runtime — native reads a local path, wasm
//! is handed the bytes.
//!
//! Every embedding is mean- or CLS-pooled (per the manifest) and L2-normalised,
//! so cosine similarity is a dot product. Inputs are truncated at
//! `max_tokens` (256 by default); longer `state` is rejected upstream at 16 KB
//! (ADR-005), so this truncation is a backstop, not the primary limit.

pub mod error;
pub mod manifest;
pub mod pooling;

#[cfg(any(feature = "native", feature = "wasm"))]
pub mod tokenize;

#[cfg(feature = "native")]
pub mod ort_backend;

#[cfg(feature = "wasm")]
pub mod tract_backend;

pub use error::{EmbedError, Result};
pub use manifest::{ManifestFile, ModelManifest, Pooling};

#[cfg(feature = "native")]
pub use ort_backend::OrtEmbedder;

#[cfg(feature = "wasm")]
pub use tract_backend::{diagnose_load, LoadOutcome, TractEmbedder};

// Re-export the trait so callers need only this crate.
pub use ruvector_typesafe_core::Embedder;
