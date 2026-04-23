use crate::anchor::AnchorTokenizer;
use crate::error::EmbedError;
use crate::traits::{NodeEmbedder, NodeTokenizer, TokenStorage};

/// Composes a [`NodeTokenizer`] and a [`TokenStorage`] to produce node embeddings
/// via mean-pooling over anchor-distance token embeddings.
pub struct MeanEmbedder<T: NodeTokenizer, S: TokenStorage> {
    tokenizer: T,
    storage: S,
}

impl<T: NodeTokenizer, S: TokenStorage> MeanEmbedder<T, S> {
    /// Create a new embedder from the given tokenizer and storage.
    pub fn new(tokenizer: T, storage: S) -> Self {
        MeanEmbedder { tokenizer, storage }
    }

    /// Estimated RAM footprint of the embedding table in bytes.
    ///
    /// To include the tokenizer's token storage, use `ram_bytes_with_token_size`.
    pub fn ram_bytes(&self) -> usize {
        self.storage.byte_size()
    }

    /// RAM footprint including an explicit token byte count.
    pub fn ram_bytes_with_token_size(&self, token_bytes: usize) -> usize {
        token_bytes + self.storage.byte_size()
    }

    /// Expose a reference to the inner tokenizer.
    pub fn tokenizer(&self) -> &T {
        &self.tokenizer
    }

    /// Expose a reference to the inner storage.
    pub fn storage(&self) -> &S {
        &self.storage
    }
}

/// Fast zero-allocation embed path for `AnchorTokenizer`.
///
/// Reads packed tokens directly as (u8, u8) pairs and accumulates into a
/// pre-allocated buffer, avoiding any intermediate Vec allocation.
impl<S: TokenStorage> MeanEmbedder<AnchorTokenizer, S> {
    /// Embed a node using the compact packed-token fast path.
    pub fn embed_fast(&self, node_id: usize) -> Result<Vec<f32>, EmbedError> {
        let packed = self
            .tokenizer
            .tokens_packed(node_id)
            .ok_or(EmbedError::NodeOutOfRange(node_id))?;

        if packed.is_empty() {
            return Err(EmbedError::EmptyTokens(node_id));
        }

        let dim = self.storage.dim();
        let mut acc = vec![0.0_f32; dim];
        let n = packed.len() as f32;

        for &(anchor_idx, dist) in packed {
            self.storage
                .accumulate_token(anchor_idx as u32, dist, &mut acc)?;
        }

        for a in acc.iter_mut() {
            *a /= n;
        }

        Ok(acc)
    }
}

impl<S: TokenStorage> NodeEmbedder for MeanEmbedder<AnchorTokenizer, S> {
    /// Embed a node via the zero-allocation fast path.
    #[inline]
    fn embed(&self, node_id: usize) -> Result<Vec<f32>, EmbedError> {
        self.embed_fast(node_id)
    }

    fn dim(&self) -> usize {
        self.storage.dim()
    }
}

/// Generic implementation for any `NodeTokenizer` (uses trait object path with allocation).
///
/// This is a blanket impl — but Rust doesn't allow conflicting impls, so we use
/// a separate wrapper type for non-AnchorTokenizer cases.
/// For the `AnchorTokenizer` specialization above, the concrete impl takes priority.
///
/// Note: The generic `NodeEmbedder` impl below covers `T != AnchorTokenizer`.
/// However, since Rust doesn't have specialization in stable, we provide
/// `embed_generic` as an associated method accessible via the trait when T is
/// a generic NodeTokenizer.

/// A generic embedder wrapping any NodeTokenizer — used when the tokenizer is
/// not AnchorTokenizer (e.g., in tests or future tokenizer types).
pub struct GenericMeanEmbedder<T: NodeTokenizer, S: TokenStorage> {
    tokenizer: T,
    storage: S,
}

impl<T: NodeTokenizer, S: TokenStorage> GenericMeanEmbedder<T, S> {
    pub fn new(tokenizer: T, storage: S) -> Self {
        GenericMeanEmbedder { tokenizer, storage }
    }

    pub fn ram_bytes(&self) -> usize {
        self.storage.byte_size()
    }
}

impl<T: NodeTokenizer, S: TokenStorage> NodeEmbedder for GenericMeanEmbedder<T, S> {
    fn embed(&self, node_id: usize) -> Result<Vec<f32>, EmbedError> {
        let tokens = self.tokenizer.tokenize(node_id)?;
        if tokens.is_empty() {
            return Err(EmbedError::EmptyTokens(node_id));
        }

        let dim = self.storage.dim();
        let mut acc = vec![0.0_f32; dim];
        let n = tokens.len() as f32;

        for token in &tokens {
            self.storage
                .accumulate_token(token.anchor_idx, token.dist, &mut acc)?;
        }

        for a in acc.iter_mut() {
            *a /= n;
        }

        Ok(acc)
    }

    fn dim(&self) -> usize {
        self.storage.dim()
    }
}
