//! Shell-segmented semantic index trait.
//!
//! The default implementation is a linear scan over an interned embedding
//! store. A future HNSW / DiskANN / Vamana backend would plug in here by
//! implementing [`SemanticIndex`] — see the `TODO(hnsw)` comment below.
//!
//! # Example
//!
//! ```
//! use ruvector_field::model::{Embedding, EmbeddingId, EmbeddingStore, NodeId, Shell};
//! use ruvector_field::storage::{LinearIndex, SemanticIndex};
//! let mut store = EmbeddingStore::new();
//! let mut idx = LinearIndex::new();
//! let e = store.intern(Embedding::new(vec![1.0, 0.0, 0.0]));
//! idx.upsert(NodeId(1), e, Shell::Event);
//! let hits = idx.search(&store, &Embedding::new(vec![1.0, 0.0, 0.0]), &[Shell::Event], 1);
//! assert_eq!(hits[0].0, NodeId(1));
//! ```

use crate::model::{Embedding, EmbeddingId, EmbeddingStore, NodeId, Shell};

/// Shell-segmented ANN-style search interface.
pub trait SemanticIndex {
    /// Search for up to `k` nearest neighbors in `shells`, returning
    /// `(node, cosine_similarity_in_[-1,1])` pairs sorted descending.
    fn search(
        &self,
        store: &EmbeddingStore,
        query: &Embedding,
        shells: &[Shell],
        k: usize,
    ) -> Vec<(NodeId, f32)>;
}

/// Default linear-scan index. O(n) search, zero setup.
///
/// TODO(hnsw): replace with a hierarchical proximity graph. The seam is the
/// [`SemanticIndex`] trait; swap this struct for an HNSW-backed one and the
/// engine picks up the new backend without any call-site changes.
#[derive(Debug, Clone, Default)]
pub struct LinearIndex {
    entries: Vec<(NodeId, EmbeddingId, Shell)>,
}

impl LinearIndex {
    /// Empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or update a node's embedding id and shell.
    pub fn upsert(&mut self, node: NodeId, embedding: EmbeddingId, shell: Shell) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.0 == node) {
            e.1 = embedding;
            e.2 = shell;
        } else {
            self.entries.push((node, embedding, shell));
        }
    }

    /// Remove a node from the index.
    pub fn remove(&mut self, node: NodeId) {
        self.entries.retain(|e| e.0 != node);
    }

    /// Number of indexed nodes.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl SemanticIndex for LinearIndex {
    fn search(
        &self,
        store: &EmbeddingStore,
        query: &Embedding,
        shells: &[Shell],
        k: usize,
    ) -> Vec<(NodeId, f32)> {
        let mut scored: Vec<(NodeId, f32)> = Vec::new();
        for (node, eid, shell) in &self.entries {
            if !shells.is_empty() && !shells.contains(shell) {
                continue;
            }
            let Some(emb) = store.get(*eid) else { continue };
            let sim = query.cosine(emb);
            scored.push((*node, sim));
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}
