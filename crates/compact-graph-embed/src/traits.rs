use crate::error::EmbedError;

/// A single anchor-distance token representing a node's relationship to an anchor node.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub anchor_idx: u32,
    pub dist: u8,
}

/// Trait for tokenizing graph nodes into anchor-distance token sequences.
pub trait NodeTokenizer: Send + Sync {
    /// Returns the list of tokens for the given node.
    fn tokenize(&self, node_id: usize) -> Result<Vec<Token>, EmbedError>;
    /// Total number of nodes in the graph.
    fn num_nodes(&self) -> usize;
    /// Number of anchor nodes selected.
    fn num_anchors(&self) -> usize;
    /// Maximum BFS distance used during tokenization.
    fn max_dist(&self) -> u8;
}

/// Trait for storing and retrieving token embeddings.
pub trait TokenStorage: Send + Sync {
    /// Embedding dimension.
    fn dim(&self) -> usize;
    /// Total number of (anchor, dist) entries = num_anchors * max_dist.
    fn num_entries(&self) -> usize;
    /// Look up the embedding vector for a given anchor and distance.
    fn embed_token(&self, anchor_idx: u32, dist: u8) -> Result<Vec<f32>, EmbedError>;
    /// Total bytes used by this storage.
    fn byte_size(&self) -> usize;

    /// Accumulate the token embedding into `acc` in-place (zero allocation).
    ///
    /// Default implementation allocates; concrete types should override this.
    fn accumulate_token(
        &self,
        anchor_idx: u32,
        dist: u8,
        acc: &mut [f32],
    ) -> Result<(), EmbedError> {
        let emb = self.embed_token(anchor_idx, dist)?;
        for (a, e) in acc.iter_mut().zip(emb.iter()) {
            *a += e;
        }
        Ok(())
    }
}

/// Trait for producing node embeddings by composing tokenizer and storage.
pub trait NodeEmbedder: Send + Sync {
    /// Embed the given node and return a float vector of length `dim()`.
    fn embed(&self, node_id: usize) -> Result<Vec<f32>, EmbedError>;
    /// Embedding dimension.
    fn dim(&self) -> usize;
}
