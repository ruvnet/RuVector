//! Embeddings, embedding store, and embedding ids.
//!
//! [`FieldNode`](super::FieldNode) holds an [`EmbeddingId`] and never owns its
//! vector directly — the [`EmbeddingStore`] interns every distinct vector so
//! cloning a node is cheap even with large embeddings.
//!
//! # Example
//!
//! ```
//! use ruvector_field::model::{Embedding, EmbeddingStore};
//! let mut store = EmbeddingStore::new();
//! let id = store.intern(Embedding::new(vec![0.1, 0.2, 0.3, 0.4]));
//! let back = store.get(id).unwrap();
//! assert_eq!(back.values.len(), 4);
//! ```

use core::fmt;
use std::collections::HashMap;

/// Dense embedding vector, L2 normalized at construction time.
#[derive(Debug, Clone, PartialEq)]
pub struct Embedding {
    /// L2-normalized values.
    pub values: Vec<f32>,
}

impl Embedding {
    /// Build an embedding, L2-normalizing the input. Zero vectors stay zero.
    pub fn new(values: Vec<f32>) -> Self {
        Self {
            values: l2_normalize(values),
        }
    }

    /// Dimension of the embedding.
    pub fn dim(&self) -> usize {
        self.values.len()
    }

    /// Geometric antipode — the L2-normalized negation. Cheap and separate from
    /// semantic opposition per spec section 5.2.
    pub fn geometric_antipode(&self) -> Embedding {
        Embedding {
            values: self.values.iter().map(|v| -v).collect(),
        }
    }

    /// Cosine similarity in `[-1, 1]`. Assumes both embeddings are normalized.
    pub fn cosine(&self, other: &Embedding) -> f32 {
        let n = self.values.len().min(other.values.len());
        let mut acc = 0.0_f32;
        for i in 0..n {
            acc += self.values[i] * other.values[i];
        }
        acc
    }

    /// Cosine mapped into `[0, 1]`.
    pub fn cosine01(&self, other: &Embedding) -> f32 {
        ((self.cosine(other) + 1.0) / 2.0).clamp(0.0, 1.0)
    }

    /// 64-bit content hash — used by [`EmbeddingStore`] for interning.
    pub fn content_hash(&self) -> u64 {
        // FxHash-style mix, no external dep. Deterministic.
        let mut h: u64 = 0xcbf29ce484222325;
        for v in &self.values {
            h ^= (v.to_bits() as u64).wrapping_mul(0x100000001b3);
            h = h.rotate_left(13).wrapping_mul(0x9e3779b97f4a7c15);
        }
        h
    }
}

impl fmt::Display for Embedding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Embedding(dim={})", self.values.len())
    }
}

/// Strongly typed identifier for an interned embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EmbeddingId(pub u64);

impl fmt::Display for EmbeddingId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "emb#{}", self.0)
    }
}

/// Intern embeddings by content hash.
///
/// Two nodes with the same underlying vector share a single [`EmbeddingId`].
/// Geometric antipodes live in the same store.
#[derive(Debug, Clone, Default)]
pub struct EmbeddingStore {
    /// Indexed by `EmbeddingId.0 - 1`; id 0 is reserved for "empty".
    vectors: Vec<Embedding>,
    by_hash: HashMap<u64, EmbeddingId>,
}

impl EmbeddingStore {
    /// Empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern the embedding, returning its stable id.
    pub fn intern(&mut self, emb: Embedding) -> EmbeddingId {
        let h = emb.content_hash();
        if let Some(id) = self.by_hash.get(&h) {
            return *id;
        }
        let id = EmbeddingId((self.vectors.len() as u64) + 1);
        self.vectors.push(emb);
        self.by_hash.insert(h, id);
        id
    }

    /// Fetch by id. `None` if the id is out of range.
    pub fn get(&self, id: EmbeddingId) -> Option<&Embedding> {
        if id.0 == 0 {
            return None;
        }
        self.vectors.get((id.0 - 1) as usize)
    }

    /// Number of distinct interned embeddings.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// `true` if no embeddings have been interned.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

/// L2-normalize a vector. Leaves the zero vector unchanged.
pub fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}
