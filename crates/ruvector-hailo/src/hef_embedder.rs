//! End-to-end NPU-accelerated embedder.
//!
//! ADR-176 P3 (`hailo-backend`, iter 161). Composes:
//!
//!   tokenizer (HF tokenizer.json)         — text → input_ids + mask
//!   HostEmbeddings (P2, iter 160)         — input_ids → [seq, hidden] FP32
//!   HefPipeline (P1, iter 158-159)        — [seq, hidden] FP32 → [seq, hidden] FP32 via NPU
//!   inference::mean_pool                  — [seq, hidden] + mask → [hidden]
//!   inference::l2_normalize               — [hidden] → unit vector
//!
//! Public API mirrors `CpuEmbedder::embed` so `HailoEmbedder` can
//! route to either (P4, iter 162) without callers caring.
//!
//! Feature gates: requires both `hailo` (for HefPipeline) and
//! `cpu-fallback` (for HostEmbeddings + HF tokenizers + safetensors
//! loader). Builds only with `--features hailo,cpu-fallback`.

#![cfg(all(feature = "hailo", feature = "cpu-fallback"))]

use crate::device::HailoDevice;
use crate::error::HailoError;
use crate::hef_pipeline::HefPipeline;
use crate::host_embeddings::HostEmbeddings;
use crate::inference::{l2_normalize, mean_pool};
use std::path::Path;
use std::sync::Mutex;
use tokenizers::Tokenizer;

/// NPU-accelerated end-to-end embedder.
///
/// `embed()` is `&self` (interior mutability via `Mutex`) so it
/// matches `EmbeddingProvider`'s shape and slots into `HailoEmbedder`
/// the same way `CpuEmbedder` does.
pub struct HefEmbedder {
    inner: Mutex<Inner>,
    output_dim: usize,
    /// Sequence length the HEF was compiled for. Iter 156b: 128.
    /// We pad shorter inputs with `[PAD]` and truncate longer ones.
    max_seq: usize,
}

struct Inner {
    pipeline: HefPipeline,
    embeddings: HostEmbeddings,
    tokenizer: Tokenizer,
    /// Iter 175 — pooled buffer for the NPU output. Sized once at
    /// construct time to `seq_len * hidden`. Reused across embed()
    /// calls to avoid the ~196 KB allocation per call.
    last_hidden_buf: Vec<f32>,
    /// Iter 176 — pooled buffer for the host-side embedding lookup
    /// output. Same shape as last_hidden_buf, both reused across
    /// embed() calls. Together iter 175+176 eliminate ~393 KB of
    /// allocator churn per request.
    embeds_buf: Vec<f32>,
}

impl HefEmbedder {
    /// Open the HEF + load the host-side embedding tables + tokenizer.
    /// `model_dir` must contain:
    ///   * `model.hef`              (compiled by deploy/compile-encoder-hef.py)
    ///   * `model.safetensors`      (HF weights — for the embedding tables)
    ///   * `tokenizer.json`         (HF fast tokenizer)
    ///   * `config.json`            (BERT config — vocab/hidden sizes)
    pub fn open(device: &HailoDevice, model_dir: &Path) -> Result<Self, HailoError> {
        let hef_path = model_dir.join("model.hef");
        let tokenizer_path = model_dir.join("tokenizer.json");

        if !hef_path.exists() {
            return Err(HailoError::BadModelDir {
                path: model_dir.display().to_string(),
                what: "model.hef",
            });
        }
        if !tokenizer_path.exists() {
            return Err(HailoError::BadModelDir {
                path: model_dir.display().to_string(),
                what: "tokenizer.json",
            });
        }

        let pipeline = HefPipeline::open(device, &hef_path)?;
        let embeddings = HostEmbeddings::open(model_dir)?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| HailoError::Tokenizer(format!("Tokenizer::from_file: {}", e)))?;

        // Pull the HEF's seq + hidden from the pipeline shape so we
        // pad / truncate consistently. iter-156b HEF: [1, 128, 384].
        let in_shape = pipeline.input_shape();
        let max_seq = in_shape[1];
        let output_dim = pipeline.output_shape()[2];
        if in_shape[2] != output_dim {
            return Err(HailoError::Shape {
                expected: output_dim,
                actual: in_shape[2],
            });
        }

        Ok(Self {
            inner: Mutex::new(Inner {
                pipeline,
                embeddings,
                tokenizer,
                last_hidden_buf: vec![0.0; max_seq * output_dim],
                embeds_buf: Vec::with_capacity(max_seq * output_dim),
            }),
            output_dim,
            max_seq,
        })
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// End-to-end embed: `text → unit-norm Vec<f32>`. Bit-equivalent
    /// shape contract to `CpuEmbedder::embed` so the cluster's
    /// dispatch can mix the two safely (iter-143 fingerprint will
    /// distinguish them anyway, since the safetensors-only worker
    /// and the safetensors+hef worker hash to different fingerprints).
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, HailoError> {
        let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        let inner = &mut *g;

        // 1. Tokenize. Pad/truncate to max_seq so the NPU input
        // matches the HEF's compiled shape.
        let mut encoding = inner
            .tokenizer
            .encode(text, true)
            .map_err(|e| HailoError::Tokenizer(format!("encode: {}", e)))?;
        encoding.truncate(self.max_seq, 1, tokenizers::TruncationDirection::Right);
        encoding.pad(
            self.max_seq,
            0,
            0,
            "[PAD]",
            tokenizers::PaddingDirection::Right,
        );

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<u32> = encoding.get_attention_mask().to_vec();

        if input_ids.len() != self.max_seq || attention_mask.len() != self.max_seq {
            return Err(HailoError::Shape {
                expected: self.max_seq,
                actual: input_ids.len(),
            });
        }

        // 2. Host-side embedding lookup → [seq, hidden] FP32 row-major.
        // Iter 176 — write into pooled embeds_buf via
        // HostEmbeddings::forward_into, skipping the per-call
        // ~196 KB Vec allocation.
        // 3. NPU forward pass — iter 175 — write into pooled
        //    last_hidden_buf, skipping another ~196 KB allocation.
        // We destructure Inner so the borrow checker accepts
        // simultaneous &mut on the three fields.
        let Inner {
            pipeline,
            embeddings,
            last_hidden_buf,
            embeds_buf,
            ..
        } = inner;
        embeddings.forward_into(&input_ids, embeds_buf)?;
        if embeds_buf.len() != self.max_seq * self.output_dim {
            return Err(HailoError::Shape {
                expected: self.max_seq * self.output_dim,
                actual: embeds_buf.len(),
            });
        }
        pipeline.forward_into(embeds_buf, last_hidden_buf)?;
        let expected_out = self.max_seq * self.output_dim;
        if last_hidden_buf.len() < expected_out {
            return Err(HailoError::Shape {
                expected: expected_out,
                actual: last_hidden_buf.len(),
            });
        }

        // 4. Mean-pool over the seq dim with attention mask.
        let mut pooled = mean_pool(
            &last_hidden_buf[..expected_out],
            &attention_mask,
            self.max_seq,
            self.output_dim,
        );

        // 5. L2-normalize — matches sentence-transformers convention
        // and is what cpu_embedder produces too, so cluster fingerprint
        // post-iter-143 distinguishes the encoders but the dispatch
        // contract holds.
        l2_normalize(&mut pooled);
        Ok(pooled)
    }
}

// SAFETY: the inner Mutex serializes all NPU writes/reads + candle
// tensor mutations. The HEF + embedding-table mmaps are shared,
// immutable.
unsafe impl Send for HefEmbedder {}
unsafe impl Sync for HefEmbedder {}
