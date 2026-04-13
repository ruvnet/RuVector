//! Higher-quality deterministic embedding provider used under
//! `--features onnx-embeddings`.
//!
//! Ideally this module would host an `ort`-backed MiniLM provider, but
//! because the example crate is an isolated workspace with no external
//! dependencies we ship a much-improved **deterministic** backend instead:
//! character n-gram hashing (n=3, n=4) into a 384-dim bucketed vector with
//! L2 normalization. This is the same shape a real MiniLM embedding would
//! produce (384 dims, unit norm), so downstream code paths — including the
//! HNSW index, the coherence solver, and the drift detector — exercise
//! exactly the same numerical ranges the production backend would produce.
//!
//! The provider is:
//!
//! * **Deterministic**: the same text always maps to the same vector.
//! * **Semantically informed**: shared character n-grams yield high cosine
//!   similarity, unlike the default token-bucketed [`HashEmbeddingProvider`].
//! * **Zero-dep**: runs under pure `std`, so `cargo build
//!   --features onnx-embeddings` works out of the box.
//!
//! Swapping in a real ONNX Runtime backend is a drop-in replacement: the
//! provider only has to implement [`crate::embed::EmbeddingProvider`].

use crate::embed::EmbeddingProvider;
use crate::model::Embedding;

/// Dimension for the deterministic MiniLM-shaped provider.
pub const DEFAULT_DIM: usize = 384;

/// Char n-gram hashing embedding provider.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "onnx-embeddings")] {
/// use ruvector_field::embed::EmbeddingProvider;
/// use ruvector_field::embed_onnx::DeterministicEmbeddingProvider;
/// let p = DeterministicEmbeddingProvider::new();
/// let a = p.embed("timeout in authentication");
/// let b = p.embed("timeout in authentication");
/// assert_eq!(a.values, b.values);
/// assert_eq!(p.dim(), 384);
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DeterministicEmbeddingProvider {
    dim: usize,
    /// Min n-gram size (inclusive).
    n_min: usize,
    /// Max n-gram size (inclusive).
    n_max: usize,
}

impl DeterministicEmbeddingProvider {
    /// 384-dim, char n-gram sizes 3..=4.
    pub fn new() -> Self {
        Self {
            dim: DEFAULT_DIM,
            n_min: 3,
            n_max: 4,
        }
    }

    /// Custom configuration.
    pub fn with_config(dim: usize, n_min: usize, n_max: usize) -> Self {
        Self {
            dim: dim.max(16),
            n_min: n_min.max(1),
            n_max: n_max.max(n_min.max(1)),
        }
    }
}

impl Default for DeterministicEmbeddingProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingProvider for DeterministicEmbeddingProvider {
    fn embed(&self, text: &str) -> Embedding {
        let mut buckets = vec![0.0_f32; self.dim];
        // Pad with a leading/trailing marker so boundary n-grams are
        // distinguishable from interior ones.
        let padded: String = format!(" {} ", text.to_ascii_lowercase());
        let chars: Vec<char> = padded.chars().collect();
        for n in self.n_min..=self.n_max {
            if n > chars.len() {
                continue;
            }
            for i in 0..=chars.len() - n {
                let gram: String = chars[i..i + n].iter().collect();
                let h = fnv1a(gram.as_bytes());
                // Two-bucket deposit with opposite signs from separate
                // hash seeds — reduces collision noise.
                let b1 = (h >> 1) as usize % self.dim;
                let sign = if h & 1 == 0 { 1.0 } else { -1.0 };
                buckets[b1] += sign;
                let h2 = fnv1a_with_seed(gram.as_bytes(), 0x9e3779b97f4a7c15);
                let b2 = h2 as usize % self.dim;
                buckets[b2] += sign * 0.5;
            }
        }
        Embedding::new(buckets)
    }

    fn dim(&self) -> usize {
        self.dim
    }
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

#[cfg(test)]
mod tests {
    use super::*;

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
        let v = p.embed("hello");
        assert_eq!(v.values.len(), 384);
    }

    #[test]
    fn similar_texts_have_high_cosine() {
        let p = DeterministicEmbeddingProvider::new();
        let a = p.embed("authentication timeout");
        let b = p.embed("authentication timeouts");
        let c = p.embed("completely unrelated lunar cartography");
        let sim_ab = a.cosine(&b);
        let sim_ac = a.cosine(&c);
        assert!(sim_ab > sim_ac, "{} vs {}", sim_ab, sim_ac);
    }
}
