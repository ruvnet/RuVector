//! Verifies the `onnx-embeddings` provider is deterministic and dimensionally
//! correct.
//!
//! Build/run with:
//!
//! ```text
//! cargo test --features onnx-embeddings --test feature_onnx
//! ```

#![cfg(feature = "onnx-embeddings")]

use ruvector_field::embed::EmbeddingProvider;
use ruvector_field::embed_onnx::{DeterministicEmbeddingProvider, DEFAULT_DIM};

#[test]
fn deterministic() {
    let p = DeterministicEmbeddingProvider::new();
    let a = p.embed("user reports authentication timeout");
    let b = p.embed("user reports authentication timeout");
    assert_eq!(a.values, b.values);
}

#[test]
fn correct_dim() {
    let p = DeterministicEmbeddingProvider::new();
    assert_eq!(p.dim(), DEFAULT_DIM);
    let v = p.embed("hello world");
    assert_eq!(v.values.len(), DEFAULT_DIM);
}

#[test]
fn unit_norm() {
    let p = DeterministicEmbeddingProvider::new();
    let v = p.embed("some reasonably long sentence for testing");
    let norm: f32 = v.values.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5 || norm == 0.0, "norm = {}", norm);
}

#[test]
fn similar_texts_more_similar_than_unrelated() {
    let p = DeterministicEmbeddingProvider::new();
    let a = p.embed("authentication timeout detected");
    let b = p.embed("authentication timeouts detected");
    let c = p.embed("random unrelated lunar mission");
    assert!(a.cosine(&b) > a.cosine(&c));
}
