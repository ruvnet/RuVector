//! rv-embed: the embedding port and a pure batch planner for Workers AI
//! `@cf/baai/bge-small-en-v1.5` (384 dims). No network here: the Worker
//! implements [`EmbeddingPort`] over its `AI` binding.
//!
//! The planner refuses (with the offending input index) empty texts, texts
//! over the byte limit and texts whose token estimate exceeds the model's
//! 512-token window, then packs texts in order into batches bounded by text
//! count and estimated tokens. [`EmbeddingBatcher::accept`] checks each
//! response: one vector per text, exactly `dim` finite values.

use crate::error::EmbedError;
use core::future::Future;
use core::ops::Range;

/// Workers AI model id.
pub const BGE_SMALL_MODEL: &str = "@cf/baai/bge-small-en-v1.5";
/// Output dimension of bge-small-en-v1.5.
pub const BGE_SMALL_DIM: usize = 384;

/// Input limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbedLimits {
    /// Texts per model call.
    pub max_texts_per_batch: usize,
    /// Estimated tokens per text (model window, incl. `[CLS]`/`[SEP]`).
    pub max_tokens_per_text: usize,
    /// Estimated tokens per model call.
    pub max_tokens_per_batch: usize,
    /// UTF-8 bytes per text (checked before estimating).
    pub max_text_bytes: usize,
    /// Texts per request.
    pub max_texts: usize,
}

impl Default for EmbedLimits {
    fn default() -> Self {
        EmbedLimits {
            max_texts_per_batch: 100,
            max_tokens_per_text: 512,
            max_tokens_per_batch: 100 * 512,
            max_text_bytes: 8 * 1024,
            max_texts: 500,
        }
    }
}

/// Conservative WordPiece (bert-uncased) token estimate, plus 2 for
/// `[CLS]`/`[SEP]`:
///
/// * every ASCII punctuation / symbol and every non-ASCII character counts 1;
/// * an ASCII alphanumeric run that contains a digit or no vowel (hex,
///   base64, UUIDs, codes, digit runs, identifiers like `xkcd`) counts
///   **1 per character** — WordPiece splits such runs into 1–2-character
///   pieces, so `ceil(len / 4)` would under-count them 2–4×;
/// * any other alphabetic run counts `ceil(len / 3)` (a long rare word like
///   "antidisestablishmentarianism" is ~8 pieces; `/4` gave 7, `/3` gives 10).
///
/// This over-counts ordinary English by ~1.5×, so a text the planner accepts
/// fits the 512-token window in practice. It remains an estimate: a port
/// error is still possible and typed, and [`EmbedOptions::truncate_inputs`]
/// lets the Worker ask the model to truncate instead of failing the batch.
pub fn estimate_tokens(text: &str) -> usize {
    fn run_cost(run: &[u8]) -> usize {
        let has_digit = run.iter().any(u8::is_ascii_digit);
        let has_vowel = run.iter().any(|b| {
            matches!(
                b.to_ascii_lowercase(),
                b'a' | b'e' | b'i' | b'o' | b'u' | b'y'
            )
        });
        if has_digit || !has_vowel {
            run.len()
        } else {
            run.len().div_ceil(3)
        }
    }
    let bytes = text.as_bytes();
    let mut n = 2;
    let mut start = None;
    for (i, c) in text.char_indices() {
        if c.is_ascii_alphanumeric() {
            start.get_or_insert(i);
            continue;
        }
        if let Some(s) = start.take() {
            n += run_cost(&bytes[s..i]);
        }
        if !c.is_whitespace() {
            n += 1;
        }
    }
    if let Some(s) = start {
        n += run_cost(&bytes[s..]);
    }
    n
}

/// One planned model call: the input range and its token estimate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedBatch {
    /// Indices into the input.
    pub range: Range<usize>,
    /// Sum of token estimates.
    pub est_tokens: usize,
}

/// Per-call options passed through to the port.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EmbedOptions {
    /// Ask the model to truncate inputs longer than its window instead of
    /// failing the whole call (Workers AI `truncate_inputs`). Off by default:
    /// the planner refuses over-long texts, and silently embedding a prefix
    /// is a caller decision. The crate only forwards the flag; its meaning
    /// is the model's.
    pub truncate_inputs: bool,
}

/// Batch planner / response validator.
#[derive(Debug, Clone)]
pub struct EmbeddingBatcher {
    /// Limits.
    pub limits: EmbedLimits,
    /// Options forwarded to every port call.
    pub options: EmbedOptions,
    /// Model id passed to the port.
    pub model: &'static str,
    /// Expected output dimension.
    pub dim: usize,
}

impl Default for EmbeddingBatcher {
    fn default() -> Self {
        EmbeddingBatcher {
            limits: EmbedLimits::default(),
            options: EmbedOptions::default(),
            model: BGE_SMALL_MODEL,
            dim: BGE_SMALL_DIM,
        }
    }
}

impl EmbeddingBatcher {
    /// Validate every text and pack them, in order, into batches.
    pub fn plan(&self, texts: &[&str]) -> Result<Vec<EmbedBatch>, EmbedError> {
        let l = &self.limits;
        if texts.len() > l.max_texts {
            return Err(EmbedError::TooManyTexts(texts.len()));
        }
        let mut out: Vec<EmbedBatch> = Vec::new();
        let mut cur = EmbedBatch {
            range: 0..0,
            est_tokens: 0,
        };
        for (index, t) in texts.iter().enumerate() {
            if t.trim().is_empty() {
                return Err(EmbedError::EmptyText { index });
            }
            if t.len() > l.max_text_bytes {
                return Err(EmbedError::TextTooLong { index });
            }
            let tok = estimate_tokens(t);
            if tok > l.max_tokens_per_text || tok > l.max_tokens_per_batch {
                return Err(EmbedError::TooManyTokens { index });
            }
            let full = cur.range.len() >= l.max_texts_per_batch
                || cur.est_tokens + tok > l.max_tokens_per_batch;
            if !cur.range.is_empty() && full {
                let next = EmbedBatch {
                    range: index..index,
                    est_tokens: 0,
                };
                out.push(core::mem::replace(&mut cur, next));
            }
            cur.range.end = index + 1;
            cur.est_tokens += tok;
        }
        if !cur.range.is_empty() {
            out.push(cur);
        }
        Ok(out)
    }

    /// Validate a model response for `batch`.
    pub fn accept(
        &self,
        batch: &EmbedBatch,
        vectors: Vec<Vec<f32>>,
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        if vectors.len() != batch.range.len() {
            return Err(EmbedError::ResponseCount);
        }
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != self.dim || !v.iter().all(|x| x.is_finite()) {
                return Err(EmbedError::BadVector {
                    index: batch.range.start + i,
                });
            }
        }
        Ok(vectors)
    }
}

/// The embedding port (Workers AI binding in production, a mock in tests).
/// Futures are not required to be `Send` (Workers are single-threaded).
pub trait EmbeddingPort {
    /// Embed `texts` with `model`, one vector per text, in order.
    fn embed(
        &self,
        model: &str,
        texts: &[&str],
        options: EmbedOptions,
    ) -> impl Future<Output = Result<Vec<Vec<f32>>, EmbedError>>;
}

/// Plan, call the port batch by batch, and validate: one vector per input.
pub async fn embed_all<P: EmbeddingPort>(
    port: &P,
    batcher: &EmbeddingBatcher,
    texts: &[&str],
) -> Result<Vec<Vec<f32>>, EmbedError> {
    let plan = batcher.plan(texts)?;
    let mut out = Vec::with_capacity(texts.len());
    for b in &plan {
        let got = port
            .embed(batcher.model, &texts[b.range.clone()], batcher.options)
            .await?;
        out.extend(batcher.accept(b, got)?);
    }
    Ok(out)
}
