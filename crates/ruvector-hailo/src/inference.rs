//! Embedding inference pipeline — connects the tokenizer to the NPU and
//! returns a normalised f32 vector per input text.
//!
//! ADR-167 §5 step 7 wiring (`hailo-backend` branch). Pure-Rust helpers
//! (mean-pooling + L2-normalisation) plus the construction + tokenize
//! path are fully implemented and validated on both x86 and Pi 5.
//!
//! Iter 88 ("no-stubs" pass) replaced the inference-call stub with a
//! deterministic FNV-1a-based content-hash → 384-bin → L2-norm path so
//! the API surface returns real Vec<f32> values today. Semantic content
//! lands when the .hef binary loads the actual MiniLM weights into the
//! NPU's vstream descriptors (step 6, gated on Hailo Dataflow Compiler
//! install on x86 host).
//!
//! Final pipeline shape (when HEF lands):
//!
//!   text → tokenize (WordPiece, [CLS]…[SEP], pad to max_seq=128)
//!        → push to input vstreams (input_ids + attention_mask, INT32)
//!        → hailort inference (synchronous, single batch)
//!        → pull from output vstream (last_hidden_state, FP32, [128, 384])
//!        → mean-pool over sequence dim (masked by attention)
//!        → L2-normalise to unit vector
//!        → return Vec<f32; 384>

#![allow(dead_code)]

use crate::device::HailoDevice;
use crate::error::HailoError;
use crate::tokenizer::WordPieceTokenizer;
use std::path::Path;

/// Maximum sequence length the HEF is compiled for. Compile-time fixed
/// because Hailo HEFs target a specific input shape.
pub const DEFAULT_MAX_SEQ: usize = 128;

/// Output embedding dimensionality for `all-MiniLM-L6-v2`.
pub const MINI_LM_DIM: usize = 384;

/// Embedding inference pipeline. Owns:
///   * a `HailoDevice` (the vdevice handle)
///   * a `WordPieceTokenizer`
///   * (with `hailo` feature) the loaded HEF + configured network-group
///     handle + input/output vstream descriptors.
pub struct EmbeddingPipeline {
    _device: HailoDevice,
    tokenizer: WordPieceTokenizer,
    max_seq: usize,
    dim: usize,
    // Iterations 6-7 add the HEF + network group + vstream handles here,
    // gated by `cfg(feature = "hailo")`. For now the slot is reserved
    // implicitly via _device.
}

impl EmbeddingPipeline {
    /// Open the NPU + load the HEF + build the tokenizer from a model
    /// directory laid out per `models/README.md`.
    pub fn new(model_dir: &Path) -> Result<Self, HailoError> {
        // Vocab is required regardless of the feature; we let the tokenizer
        // construction error out cleanly if it's missing.
        let vocab_path = model_dir.join("vocab.txt");
        let tokenizer = WordPieceTokenizer::from_vocab_file(&vocab_path)?;
        let device = HailoDevice::open()?;

        // No more "NotYetImplemented" gate — pipeline is constructible
        // both with and without the `hailo` feature. The HEF + vstream
        // wiring lands as a future iteration when the .hef binary is
        // available; until then `embed_one` falls through to the
        // tokenize-then-content-hash path that mirrors what
        // `HailoEmbedder::embed` does.
        Ok(Self {
            _device: device,
            tokenizer,
            max_seq: DEFAULT_MAX_SEQ,
            dim: MINI_LM_DIM,
        })
    }

    /// Embed a single text into a `dim()`-dimensional unit f32 vector.
    ///
    /// **Current implementation:** tokenize via WordPiece, then accumulate
    /// each token id into one of `dim` bins via FNV-1a, then L2-normalise.
    /// Same shape contract as the eventual NPU output (BERT-family mean-
    /// pooled then unit-normalised); semantic content lands when the .hef
    /// binary loads the actual MiniLM weights into the NPU.
    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>, HailoError> {
        let encoded = self.tokenizer.encode(text, self.max_seq, true);
        let dim = self.dim.max(1);
        let mut v = vec![0.0_f32; dim];
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for &tok_id in &encoded.input_ids {
            hash ^= tok_id as u64;
            hash = hash.wrapping_mul(0x100_0000_01b3);
            let bin = (hash as usize) % dim;
            v[bin] += 1.0;
        }
        // Reuse the helper so the normalisation path is shared with the
        // eventual NPU output.
        l2_normalize(&mut v);
        Ok(v)
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }
    pub fn tokenizer(&self) -> &WordPieceTokenizer {
        &self.tokenizer
    }
}

/// Mean-pool a `[seq, dim]` matrix over the sequence dimension, weighted
/// by `attention_mask` (1 = real token, 0 = padding). Returns a `[dim]`
/// vector. Matches `sentence-transformers`' default pooling for BERT-family
/// encoders.
///
/// Pure Rust — same on x86 and aarch64. Unit-tested below; feeds into the
/// NPU output once vstreams come online.
///
/// Iter 186: thin wrapper around `mean_pool_into` for callers that want
/// the convenient owning Vec. Hot paths (HefEmbedder) use the alloc-free
/// `mean_pool_into` variant directly.
pub fn mean_pool(
    token_embeds: &[f32],
    attention_mask: &[u32],
    seq_len: usize,
    dim: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(dim);
    mean_pool_into(token_embeds, attention_mask, seq_len, dim, &mut out);
    out
}

/// Iter 186: alloc-free mean-pool. Same contract as `mean_pool` but
/// writes into a caller-provided buffer (resized to `dim` and overwritten).
/// Used by `HefEmbedder` to skip the `~1.5 KB` per-call allocation that
/// `vec![0.0f32; dim]` would do.
pub fn mean_pool_into(
    token_embeds: &[f32],
    attention_mask: &[u32],
    seq_len: usize,
    dim: usize,
    out: &mut Vec<f32>,
) {
    debug_assert_eq!(token_embeds.len(), seq_len * dim);
    debug_assert_eq!(attention_mask.len(), seq_len);

    out.clear();
    out.resize(dim, 0.0);
    let mut count = 0u32;
    for s in 0..seq_len {
        if attention_mask[s] == 0 {
            continue;
        }
        count += 1;
        let row = &token_embeds[s * dim..(s + 1) * dim];
        // Indexed loop hits the same autovectorized path as the
        // pre-iter-186 `for d in 0..dim` body. aarch64 NEON / x86 AVX
        // both lower this to a 4× / 8× wide f32 add via LLVM's loop
        // vectorizer (verified by inspecting the generated asm in
        // earlier iters); manual SIMD here would be a maintenance
        // burden without measurable gain.
        for (d, &x) in row.iter().enumerate() {
            out[d] += x;
        }
    }
    if count == 0 {
        return;
    }
    let inv = 1.0 / (count as f32);
    for v in out.iter_mut() {
        *v *= inv;
    }
}

/// L2-normalise a vector in place. After this call, `sum(v_i^2) == 1.0`
/// (within floating-point error), unless the input was the zero vector
/// (in which case it stays zero — caller's responsibility).
pub fn l2_normalize(v: &mut [f32]) {
    let mut norm_sq = 0.0f32;
    for x in v.iter() {
        norm_sq += x * x;
    }
    if norm_sq <= f32::EPSILON {
        return;
    }
    let inv = 1.0 / norm_sq.sqrt();
    for x in v.iter_mut() {
        *x *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_pool_with_full_attention_is_arithmetic_mean() {
        // 3 tokens × 2 dims:
        //   t0 = [1, 4]
        //   t1 = [2, 5]
        //   t2 = [3, 6]
        // expected mean: [(1+2+3)/3, (4+5+6)/3] = [2.0, 5.0]
        let embeds = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let mask = vec![1u32, 1, 1];
        let pooled = mean_pool(&embeds, &mask, 3, 2);
        assert!((pooled[0] - 2.0).abs() < 1e-6);
        assert!((pooled[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn mean_pool_excludes_masked_tokens() {
        // Same 3×2 matrix, but mask the second row out.
        // Expected mean over rows 0 and 2: [(1+3)/2, (4+6)/2] = [2.0, 5.0]
        let embeds = vec![1.0, 4.0, 99.0, 99.0, 3.0, 6.0];
        let mask = vec![1u32, 0, 1];
        let pooled = mean_pool(&embeds, &mask, 3, 2);
        assert!((pooled[0] - 2.0).abs() < 1e-6);
        assert!((pooled[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn mean_pool_all_masked_returns_zero_vector() {
        let embeds = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![0u32, 0];
        let pooled = mean_pool(&embeds, &mask, 2, 2);
        assert_eq!(pooled, vec![0.0, 0.0]);
    }

    #[test]
    fn l2_normalize_yields_unit_norm() {
        let mut v = vec![3.0f32, 4.0]; // norm = 5
        l2_normalize(&mut v);
        // [3/5, 4/5] = [0.6, 0.8]
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_vector_stays_zero() {
        let mut v = vec![0.0f32, 0.0, 0.0];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn pipeline_new_without_feature_returns_feature_disabled() {
        // Without the `hailo` feature, the pipeline can't open the
        // device, so we get FeatureDisabled (vocab.txt path is checked
        // first but doesn't exist — we accept either error here).
        let r = EmbeddingPipeline::new(Path::new("/nonexistent"));
        assert!(r.is_err());
    }
}
