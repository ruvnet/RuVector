//! Tokenizer wrapper + tokenized dataset adapter.
//!
//! Wraps `tokenizers::Tokenizer` and produces a `TokenizedDataset` that
//! implements `crate::training::DatasetSource` so `Trainer` can consume it.

use super::DataError;
use super::wiki::WikiCorpus;
use std::collections::HashMap;
use std::path::Path;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::tokenizer::Tokenizer;

/// Thin wrapper around `tokenizers::Tokenizer`.
pub struct TokenizerWrapper {
    inner: Tokenizer,
    pad_token_id: u32,
}

impl TokenizerWrapper {
    /// Load a pretrained tokenizer from the HuggingFace Hub by name
    /// (e.g. `"bert-base-uncased"`). Requires the `tokenizers` crate to be
    /// built with the `http` feature; if not present, callers should fall
    /// back to [`from_file`] or [`from_vocab`].
    ///
    /// In the current build the `http` feature is disabled, so this is a
    /// shim that always returns an error. We keep the API for forward
    /// compatibility — `pretrain.rs` falls back gracefully.
    pub fn from_pretrained(name: &str) -> Result<Self, DataError> {
        let _ = name;
        Err(DataError::Tokenizer(
            "from_pretrained: `tokenizers` http feature not enabled in this build; \
             use TokenizerWrapper::from_file or from_vocab instead"
                .into(),
        ))
    }

    /// Load a tokenizer from a local `tokenizer.json` file.
    pub fn from_file(path: &Path) -> Result<Self, DataError> {
        let inner = Tokenizer::from_file(path)
            .map_err(|e| DataError::Tokenizer(format!("from_file({}): {e}", path.display())))?;
        let pad_token_id = inner
            .token_to_id("[PAD]")
            .or_else(|| inner.token_to_id("<pad>"))
            .unwrap_or(0);
        Ok(Self {
            inner,
            pad_token_id,
        })
    }

    /// Build a minimal whitespace WordLevel tokenizer from an explicit vocab.
    /// Useful for tests and offline fixtures (no network, no Hub fetch).
    ///
    /// The vocab MUST contain `"[UNK]"` and `"[PAD]"`. Token IDs should be
    /// contiguous starting at 0 for best behavior, but this is not enforced.
    pub fn from_vocab(vocab: HashMap<String, u32>) -> Result<Self, DataError> {
        let pad_token_id = *vocab
            .get("[PAD]")
            .ok_or_else(|| DataError::Tokenizer("vocab missing [PAD]".into()))?;
        if !vocab.contains_key("[UNK]") {
            return Err(DataError::Tokenizer("vocab missing [UNK]".into()));
        }

        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .map_err(|e| DataError::Tokenizer(format!("WordLevel build: {e}")))?;

        let mut inner = Tokenizer::new(model);
        inner.with_pre_tokenizer(Some(Whitespace {}));

        Ok(Self {
            inner,
            pad_token_id,
        })
    }

    /// Encode text into token ids (no special tokens added).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, DataError> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| DataError::Tokenizer(format!("encode: {e}")))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Vocabulary size including added tokens.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Pad token id (for padding short sequences).
    pub fn pad_token_id(&self) -> u32 {
        self.pad_token_id
    }
}

/// Tokenized dataset built from a `WikiCorpus`.
///
/// Implements [`crate::training::DatasetSource`] so the existing `Trainer`
/// can consume it identically to the synthetic dataset.
pub struct TokenizedDataset {
    sequences: Vec<Vec<u32>>,
    vocab_size: usize,
    seq_length: usize,
}

impl TokenizedDataset {
    /// Build a tokenized dataset by streaming over the corpus.
    ///
    /// Articles are tokenized then chunked into fixed `seq_length` sequences
    /// with stride `seq_length` (no overlap). `max_articles` caps how many
    /// articles to ingest (None = all).
    pub fn from_corpus(
        corpus: &WikiCorpus,
        tokenizer: &TokenizerWrapper,
        seq_length: usize,
        max_articles: Option<usize>,
    ) -> Result<Self, DataError> {
        if seq_length < 2 {
            return Err(DataError::Corpus(
                "seq_length must be >= 2 for next-token training".into(),
            ));
        }

        let mut buffer: Vec<u32> = Vec::with_capacity(seq_length * 16);
        let mut sequences: Vec<Vec<u32>> = Vec::new();

        let limit = max_articles.unwrap_or(usize::MAX);
        for (i, article) in corpus.iter_articles().enumerate() {
            if i >= limit {
                break;
            }
            let ids = tokenizer.encode(&article)?;
            buffer.extend_from_slice(&ids);

            // Drain whole `seq_length` chunks.
            while buffer.len() >= seq_length {
                let chunk: Vec<u32> = buffer.drain(..seq_length).collect();
                sequences.push(chunk);
            }
        }

        // Pad-and-keep any leftover that has at least 2 tokens (so input/target
        // both exist).
        if buffer.len() >= 2 {
            let pad = tokenizer.pad_token_id();
            while buffer.len() < seq_length {
                buffer.push(pad);
            }
            sequences.push(buffer.clone());
        }

        Ok(Self {
            sequences,
            vocab_size: tokenizer.vocab_size(),
            seq_length,
        })
    }

    /// Build a dataset directly from a list of pre-tokenized sequences. Useful in tests.
    pub fn from_token_sequences(
        sequences: Vec<Vec<u32>>,
        vocab_size: usize,
        seq_length: usize,
    ) -> Self {
        Self {
            sequences,
            vocab_size,
            seq_length,
        }
    }

    /// Number of sequences.
    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    /// Whether the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    /// Configured vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Sequence length.
    pub fn seq_length(&self) -> usize {
        self.seq_length
    }

    /// Get an (input, target) pair for a sequence index, mirroring
    /// `TrainingDataset::get_batch`'s shift-by-one convention.
    pub fn get_batch(&self, indices: &[usize]) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
        let inputs: Vec<Vec<u32>> = indices
            .iter()
            .map(|&i| {
                let seq = &self.sequences[i % self.sequences.len()];
                seq[..seq.len().saturating_sub(1)].to_vec()
            })
            .collect();
        let targets: Vec<Vec<u32>> = indices
            .iter()
            .map(|&i| {
                let seq = &self.sequences[i % self.sequences.len()];
                seq[1..].to_vec()
            })
            .collect();
        (inputs, targets)
    }

    /// Borrow the raw sequences (read-only).
    pub fn sequences(&self) -> &[Vec<u32>] {
        &self.sequences
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_vocab() -> HashMap<String, u32> {
        let mut v = HashMap::new();
        v.insert("[PAD]".to_string(), 0);
        v.insert("[UNK]".to_string(), 1);
        for (i, w) in ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
            .iter()
            .enumerate()
        {
            v.insert((*w).to_string(), (i as u32) + 2);
        }
        v
    }

    #[test]
    fn test_from_vocab_and_encode() {
        let tok = TokenizerWrapper::from_vocab(small_vocab()).unwrap();
        let ids = tok.encode("the quick brown fox").unwrap();
        assert_eq!(ids.len(), 4);
        assert!(tok.vocab_size() >= 10);
        assert_eq!(tok.pad_token_id(), 0);
    }
}
