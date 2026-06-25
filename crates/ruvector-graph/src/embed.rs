//! Pluggable text embedding for inline `embed()`-at-insert/at-query
//! (HelixDB-inspired, ADR-252 P3).
//!
//! HelixQL's built-in `Embed(text)` vectorizes inline so a caller never has to
//! marshal a separate embedding service into the query. This module gives
//! `TypedGraph` the same ergonomic via a pluggable [`Embedder`] trait: attach a
//! model once, then create nodes from text or search by text and the binding's
//! dimension is validated against the schema's vector type.
//!
//! A real model is supplied by implementing [`Embedder`]. A dependency-free
//! [`HashEmbedder`] is included for offline/dev/test use — it is **not**
//! semantic (lexical token overlap only) and must be opted into explicitly;
//! consistent with ADR-194, the typed graph never silently falls back to it.

use crate::error::Result;

/// A text → vector embedding model.
pub trait Embedder: Send + Sync {
    /// Output dimension; must match the bound vector type's `dimensions`.
    fn dimensions(&self) -> usize;

    /// Embed a single text.
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Embed a batch. Default loops `embed`; implementors may override with a
    /// vectorized path.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
}

/// Deterministic, dependency-free feature-hashing embedder.
///
/// Captures **lexical token overlap only** — it is not a semantic model. It
/// exists for offline/dev/test use and as an *explicit* opt-in (never a silent
/// fallback, per ADR-194). Identical text always yields an identical vector, so
/// it is useful for deterministic tests. For semantic search, supply a real
/// model via [`Embedder`].
#[derive(Debug, Clone)]
pub struct HashEmbedder {
    dims: usize,
}

impl HashEmbedder {
    /// Create a hashing embedder of the given output dimension.
    pub fn new(dims: usize) -> Self {
        assert!(dims > 0, "HashEmbedder dimension must be > 0");
        Self { dims }
    }

    /// FNV-1a hash of a token.
    #[inline]
    fn token_hash(token: &str) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in token.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }
}

impl Embedder for HashEmbedder {
    fn dimensions(&self) -> usize {
        self.dims
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut v = vec![0.0f32; self.dims];
        for raw in text.split_whitespace() {
            // Case-fold so "Cat" and "cat" collide.
            let token = raw.to_ascii_lowercase();
            let h = Self::token_hash(&token);
            let idx = (h % self.dims as u64) as usize;
            // Signed feature hashing reduces collision bias.
            let sign = if (h >> 32) & 1 == 0 { 1.0 } else { -1.0 };
            v[idx] += sign;
        }
        // L2-normalize so cosine of identical text is exactly 1.0.
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_normalized() {
        let e = HashEmbedder::new(64);
        let a = e.embed("the quick brown fox").unwrap();
        let b = e.embed("the quick brown fox").unwrap();
        assert_eq!(a, b); // deterministic
        assert_eq!(a.len(), 64);
        let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5); // unit length
    }

    #[test]
    fn overlap_scores_higher_than_disjoint() {
        let e = HashEmbedder::new(256);
        let cos = |x: &[f32], y: &[f32]| -> f32 { x.iter().zip(y).map(|(a, b)| a * b).sum() };
        let base = e.embed("machine learning vector database").unwrap();
        let near = e.embed("machine learning vector search").unwrap();
        let far = e.embed("unrelated cooking recipe content").unwrap();
        assert!(cos(&base, &near) > cos(&base, &far));
    }

    #[test]
    fn case_insensitive_tokens() {
        let e = HashEmbedder::new(64);
        assert_eq!(
            e.embed("Hello World").unwrap(),
            e.embed("hello world").unwrap()
        );
    }

    #[test]
    fn empty_text_is_zero_vector() {
        let e = HashEmbedder::new(32);
        assert_eq!(e.embed("   ").unwrap(), vec![0.0f32; 32]);
    }

    #[test]
    fn batch_matches_single() {
        let e = HashEmbedder::new(48);
        let batch = e.embed_batch(&["alpha beta", "gamma"]).unwrap();
        assert_eq!(batch[0], e.embed("alpha beta").unwrap());
        assert_eq!(batch[1], e.embed("gamma").unwrap());
    }
}
