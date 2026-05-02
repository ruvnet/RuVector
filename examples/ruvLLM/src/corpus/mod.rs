//! Wikipedia-corpus pretraining data pipeline (Patch P4).
//!
//! Provides:
//! - `wiki::WikiCorpus` — streaming reader over already-extracted Simple-English-Wiki shards
//! - `tokenize::TokenizerWrapper` — thin wrapper over `tokenizers::Tokenizer`
//! - `tokenize::TokenizedDataset` — `DatasetSource`-compatible token stream
//!
//! The whole module is gated behind the `real-inference` feature because
//! it depends on the `tokenizers` crate.

pub mod tokenize;
pub mod wiki;

pub use tokenize::{TokenizedDataset, TokenizerWrapper};
pub use wiki::{WikiArticleIter, WikiCorpus};

/// Errors produced by the wiki/data pipeline.
#[derive(Debug, thiserror::Error)]
pub enum DataError {
    /// I/O error reading corpus files.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Tokenizer error (load/encode/etc).
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// Corpus directory missing or empty.
    #[error("corpus error: {0}")]
    Corpus(String),

    /// Bincode (de)serialization error.
    #[error("serialization error: {0}")]
    Serialization(String),
}
