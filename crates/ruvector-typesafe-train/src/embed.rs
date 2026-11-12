//! Tokenization → tensors, and a candle-backed `Embedder` for the engine.
//!
//! Tokenization always goes through `ruvector_embed_core::tokenize::SharedTokenizer`
//! so padding, truncation and special tokens are the engine's own.

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use ruvector_embed_core::tokenize::{EncodedBatch, SharedTokenizer};
use ruvector_typesafe_core::{Embedder, Result as CoreResult};

use crate::model::Bert;

pub fn load_tokenizer(path: &Path, max_tokens: usize) -> Result<SharedTokenizer> {
    let bytes = crate::pins::read_verified(path, crate::pins::TOKENIZER.sha256)?;
    SharedTokenizer::from_bytes(&bytes, max_tokens).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))
}

/// Pre-tokenized batch as device tensors: (ids u32, types u32, mask f32).
pub struct Batch {
    pub ids: Tensor,
    pub types: Tensor,
    pub mask: Tensor,
}

pub fn to_tensors(enc: &EncodedBatch, device: &Device) -> Result<Batch> {
    let b = enc.input_ids.len();
    let s = enc.input_ids.first().map(|r| r.len()).unwrap_or(0);
    let flat_u32 =
        |rows: &Vec<Vec<i64>>| -> Vec<u32> { rows.iter().flatten().map(|&v| v as u32).collect() };
    let mask: Vec<f32> = enc
        .attention_mask
        .iter()
        .flatten()
        .map(|&v| v as f32)
        .collect();
    Ok(Batch {
        ids: Tensor::from_vec(flat_u32(&enc.input_ids), (b, s), device)?,
        types: Tensor::from_vec(flat_u32(&enc.token_type_ids), (b, s), device)?,
        mask: Tensor::from_vec(mask, (b, s), device)?,
    })
}

/// Pad a set of pre-tokenized rows (each `ids` without padding) to one batch.
pub fn pad_rows(rows: &[&[u32]], pad_id: u32, device: &Device) -> Result<Batch> {
    let s = rows.iter().map(|r| r.len()).max().unwrap_or(1).max(1);
    let b = rows.len();
    let mut ids = Vec::with_capacity(b * s);
    let mut mask = Vec::with_capacity(b * s);
    for r in rows {
        ids.extend_from_slice(r);
        mask.extend(std::iter::repeat_n(1f32, r.len()));
        ids.extend(std::iter::repeat_n(pad_id, s - r.len()));
        mask.extend(std::iter::repeat_n(0f32, s - r.len()));
    }
    let ids = Tensor::from_vec(ids, (b, s), device)?;
    Ok(Batch {
        types: ids.zeros_like()?,
        ids,
        mask: Tensor::from_vec(mask, (b, s), device)?,
    })
}

/// Tokenize each text once (no padding) — the training-time cache.
pub fn pretokenize(tok: &SharedTokenizer, texts: &[&str]) -> Result<Vec<Vec<u32>>> {
    let mut out = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(256) {
        let enc = tok
            .encode_batch(chunk)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        for (ids, m) in enc.input_ids.iter().zip(enc.attention_mask.iter()) {
            let n = m.iter().filter(|&&v| v == 1).count();
            out.push(ids[..n].iter().map(|&v| v as u32).collect());
        }
    }
    Ok(out)
}

/// Embed texts with the candle model in batches; rows are L2-normalized.
pub fn embed_texts(
    bert: &Bert,
    tok: &SharedTokenizer,
    texts: &[&str],
    batch: usize,
) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(batch.max(1)) {
        let enc = tok
            .encode_batch(chunk)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let t = to_tensors(&enc, &bert.device)?;
        let e = bert
            .embed(&t.ids, &t.types, &t.mask)
            .context("candle forward")?;
        out.extend(e.to_vec2::<f32>()?);
    }
    Ok(out)
}

/// The engine-facing wrapper: lets `Engine::new` run on the candle forward.
pub struct CandleEmbedder {
    pub bert: Bert,
    pub tok: SharedTokenizer,
    pub id: String,
    pub dims: usize,
}

impl Embedder for CandleEmbedder {
    fn embed(&self, texts: &[&str]) -> CoreResult<Vec<Vec<f32>>> {
        embed_texts(&self.bert, &self.tok, texts, 32)
            .map_err(|e| ruvector_typesafe_core::TypesafeError::Embedder(e.to_string()))
    }
    fn dims(&self) -> usize {
        self.dims
    }
    fn id(&self) -> &str {
        &self.id
    }
}
