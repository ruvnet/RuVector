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

/// Largest number of texts sent to the model in one forward pass.
///
/// The ORT backend pads a whole batch to its longest sequence and runs it as
/// one tensor, so memory grows with `batch × seq²` in attention. Embedding
/// thousands of texts in one call (e.g. `train()` with a large labelled set)
/// asked onnxruntime for a 10 GB buffer. Chunking keeps peak memory bounded
/// and does not change the results: each text's vector depends only on that
/// text (padding is masked out).
pub const MAX_EMBED_BATCH: usize = 32;

/// Run `f` over `texts` in chunks of at most `max` (0 is treated as 1) and
/// concatenate the outputs in order.
pub fn embed_in_chunks<T, E>(
    texts: &[&str],
    max: usize,
    mut f: impl FnMut(&[&str]) -> std::result::Result<Vec<T>, E>,
) -> std::result::Result<Vec<T>, E> {
    let mut out = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(max.max(1)) {
        out.extend(f(chunk)?);
    }
    Ok(out)
}
pub use manifest::{ManifestFile, ModelManifest, Pooling};

#[cfg(feature = "native")]
pub use ort_backend::OrtEmbedder;

#[cfg(feature = "wasm")]
pub use tract_backend::{diagnose_load, LoadOutcome, TractEmbedder};

// Re-export the trait so callers need only this crate.
pub use ruvector_typesafe_core::Embedder;

#[cfg(test)]
mod chunk_tests {
    use super::*;

    #[test]
    fn chunks_preserve_order_and_bound_size() {
        let texts: Vec<String> = (0..70).map(|i| format!("t{i}")).collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let mut sizes = Vec::new();
        let out: Vec<String> = embed_in_chunks(&refs, MAX_EMBED_BATCH, |c| {
            sizes.push(c.len());
            Ok::<_, ()>(c.iter().map(|s| s.to_string()).collect())
        })
        .unwrap();
        assert_eq!(out, texts, "order preserved");
        assert_eq!(sizes, vec![32, 32, 6], "no chunk exceeds the cap");
    }

    #[test]
    fn empty_and_zero_cap() {
        let out: Vec<u8> = embed_in_chunks(&[], 0, |_| Ok::<_, ()>(vec![1])).unwrap();
        assert!(out.is_empty());
        let out: Vec<usize> =
            embed_in_chunks(&["a", "b"], 0, |c| Ok::<_, ()>(vec![c.len()])).unwrap();
        assert_eq!(out, vec![1, 1], "cap 0 behaves as 1");
    }

    #[test]
    fn first_error_stops() {
        let mut calls = 0;
        let r: std::result::Result<Vec<u8>, &str> = embed_in_chunks(&["a"; 5], 2, |_| {
            calls += 1;
            if calls == 2 {
                Err("boom")
            } else {
                Ok(vec![0, 0])
            }
        });
        assert_eq!(r, Err("boom"));
        assert_eq!(calls, 2);
    }
}
