//! Shared tokenizer wrapper (native + wasm).
//!
//! Both backends use the HuggingFace `tokenizers` crate; only the regex engine
//! differs by target (`onig` native, `unstable_wasm` wasm), selected by the
//! crate feature. Truncation is done by the tokenizer itself (max 256 tokens by
//! default, per the manifest) so `[CLS]`/`[SEP]` survive on long inputs, then
//! each batch is padded to its own longest row.

#![cfg(any(feature = "native", feature = "wasm"))]

use crate::error::{EmbedError, Result};
use tokenizers::tokenizer::Tokenizer as HfTokenizer;
use tokenizers::TruncationParams;

/// A batch tokenised and padded to the batch's longest row.
pub struct EncodedBatch {
    /// `[batch][seq_len]` token ids.
    pub input_ids: Vec<Vec<i64>>,
    /// `[batch][seq_len]` attention mask (1 real, 0 pad).
    pub attention_mask: Vec<Vec<i64>>,
    /// `[batch][seq_len]` token type ids (all 0 for single-segment inputs).
    pub token_type_ids: Vec<Vec<i64>>,
}

impl EncodedBatch {
    pub fn batch_size(&self) -> usize {
        self.input_ids.len()
    }
    pub fn seq_len(&self) -> usize {
        self.input_ids.first().map(|r| r.len()).unwrap_or(0)
    }
}

/// Tokenizer with a fixed truncation length and known pad id.
pub struct SharedTokenizer {
    inner: HfTokenizer,
    pad_id: i64,
}

impl SharedTokenizer {
    /// Build from raw `tokenizer.json` bytes with token-level truncation.
    pub fn from_bytes(bytes: &[u8], max_tokens: usize) -> Result<Self> {
        let mut inner =
            HfTokenizer::from_bytes(bytes).map_err(|e| EmbedError::Tokenizer(e.to_string()))?;
        inner
            .with_truncation(Some(TruncationParams {
                max_length: max_tokens,
                ..Default::default()
            }))
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;
        let pad_id = find_pad_id(&inner);
        Ok(Self { inner, pad_id })
    }

    /// Encode a batch, padding to the batch's longest row.
    pub fn encode_batch(&self, texts: &[&str]) -> Result<EncodedBatch> {
        if texts.is_empty() {
            return Err(EmbedError::EmptyInput);
        }

        let encodings = texts
            .iter()
            .map(|t| self.inner.encode(*t, true))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0)
            .max(1);

        let mut input_ids = Vec::with_capacity(texts.len());
        let mut attention_mask = Vec::with_capacity(texts.len());
        let mut token_type_ids = Vec::with_capacity(texts.len());

        for enc in &encodings {
            let ids = enc.get_ids();
            let types = enc.get_type_ids();
            let len = ids.len();

            let mut id_row: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
            let mut mask_row: Vec<i64> = vec![1; len];
            let mut type_row: Vec<i64> = types.iter().map(|&x| x as i64).collect();

            let pad = max_len - len;
            if pad > 0 {
                id_row.extend(std::iter::repeat_n(self.pad_id, pad));
                mask_row.extend(std::iter::repeat_n(0i64, pad));
                type_row.extend(std::iter::repeat_n(0i64, pad));
            }

            input_ids.push(id_row);
            attention_mask.push(mask_row);
            token_type_ids.push(type_row);
        }

        Ok(EncodedBatch {
            input_ids,
            attention_mask,
            token_type_ids,
        })
    }
}

/// Find the `[PAD]`/`<pad>` id, defaulting to 0 (BERT/MiniLM/bge all use 0).
fn find_pad_id(tok: &HfTokenizer) -> i64 {
    let vocab = tok.get_vocab(true);
    vocab
        .get("[PAD]")
        .or_else(|| vocab.get("<pad>"))
        .or_else(|| vocab.get("<|pad|>"))
        .map(|&v| v as i64)
        .unwrap_or(0)
}
