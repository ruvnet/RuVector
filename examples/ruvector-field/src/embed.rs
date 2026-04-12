//! Embedding provider trait.
//!
//! The demo uses [`HashEmbeddingProvider`] — a deterministic, std-only bucketed
//! sparse hash so text inputs become stable embeddings without pulling in a
//! real model. Production builds would swap in an ONNX or remote provider.
//!
//! # Example
//!
//! ```
//! use ruvector_field::embed::{EmbeddingProvider, HashEmbeddingProvider};
//! let p = HashEmbeddingProvider::new(16);
//! let a = p.embed("user reports timeout");
//! let b = p.embed("user reports timeout");
//! assert_eq!(a.values, b.values);
//! assert_eq!(p.dim(), 16);
//! ```

use crate::model::Embedding;

/// Text-to-embedding provider.
pub trait EmbeddingProvider {
    /// Produce an embedding for `text`.
    fn embed(&self, text: &str) -> Embedding;
    /// Dimension of produced embeddings.
    fn dim(&self) -> usize;
}

/// Deterministic bucketed-hash embedding.
///
/// Each lowercased whitespace token is hashed with FNV-1a, mapped into
/// `[0, dim)`, and incremented by a small value. The result is L2-normalized.
/// Two inputs that share most tokens come out close; opposites stay apart.
#[derive(Debug, Clone, Copy)]
pub struct HashEmbeddingProvider {
    dim: usize,
}

impl HashEmbeddingProvider {
    /// Create a provider with the given embedding dimension.
    pub fn new(dim: usize) -> Self {
        Self { dim: dim.max(4) }
    }
}

impl EmbeddingProvider for HashEmbeddingProvider {
    fn embed(&self, text: &str) -> Embedding {
        let mut buckets = vec![0.0_f32; self.dim];
        for tok in tokens(text) {
            let h1 = fnv1a(tok.as_bytes());
            let sign_bit = h1 & 1;
            let bucket = (h1 >> 1) as usize % self.dim;
            let step = if sign_bit == 0 { 1.0 } else { -1.0 };
            buckets[bucket] += step;
            // Also deposit a weaker signal in a second bucket so nearby words
            // interfere constructively with shared suffixes.
            let h2 = fnv1a_with_seed(tok.as_bytes(), 0x9e3779b97f4a7c15);
            let bucket2 = h2 as usize % self.dim;
            buckets[bucket2] += step * 0.5;
        }
        Embedding::new(buckets)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

fn tokens(s: &str) -> impl Iterator<Item = String> + '_ {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_ascii_lowercase())
}

fn fnv1a(bytes: &[u8]) -> u64 {
    fnv1a_with_seed(bytes, 0xcbf29ce484222325)
}

fn fnv1a_with_seed(bytes: &[u8], seed: u64) -> u64 {
    let mut h: u64 = seed;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
