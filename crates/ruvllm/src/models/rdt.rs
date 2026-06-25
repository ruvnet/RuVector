//! Recurrent-Depth Transformer (RDT) execution (ADR-latest)
//!
//! Standard transformers map depth linearly to parameter count: a 32-layer
//! model carries 32 distinct weight sets. A **Recurrent-Depth Transformer**
//! shares a single block of weights (or a small set of blocks) and routes the
//! hidden state through them repeatedly, forming a deep computational loop in
//! latent space. This lets the model reason *deeper* without growing *larger*.
//!
//! Looping a fixed number of times wastes compute, so this module implements an
//! **Adaptive Halting Mechanism** (PonderNet / Universal-Transformer style). A
//! lightweight linear probe evaluates the hidden state after every loop and
//! decides, per token, whether that token is "cooked" enough to exit. Easy
//! tokens (`"the"`) resolve in a couple of loops; hard tokens (a logic gate in
//! code) may run to the `max_loops` ceiling.
//!
//! # The Honest Boundary
//!
//! This path is **only** valid for weights natively trained for weight-sharing
//! (ALBERT-style cross-layer sharing or an explicit RDT fine-tune). Running
//! standard Llama/Qwen weights through a shared block emits garbage tokens
//! because those weights were trained for distinct per-layer transforms. The
//! loader therefore validates GGUF metadata on initialization and refuses
//! incompatible weights via [`validate_rdt_metadata`] rather than silently
//! producing nonsense. See [`RdtCompatibilityError`].
//!
//! # Telemetry
//!
//! Inference latency is no longer deterministic per token. The realized loop
//! depth is recorded in [`DepthTelemetry`] (mean / max / per-call history) so an
//! audit dashboard or evolutionary harness can track `mean_inference_depth` and
//! penalize agents that pick overly expensive RDT configs for trivial work.
//!
//! # Substrate status
//!
//! The execution graph is fully implemented and unit-tested with synthetic
//! weights. End-to-end generation is dormant until a compatible RDT GGUF is
//! supplied; see [`validate_rdt_metadata`].

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::error::{Result, RuvLLMError};

// ---------------------------------------------------------------------------
// Configuration (always available, no candle dependency)
// ---------------------------------------------------------------------------

/// Configuration for a Recurrent-Depth Transformer.
///
/// Mirrors the subset of GGUF metadata needed to build the shared-block
/// execution graph plus the recurrent-loop controls (`max_loops`,
/// `halt_threshold`).
#[derive(Debug, Clone, PartialEq)]
pub struct RdtConfig {
    /// Hidden / embedding dimension.
    pub hidden_size: usize,
    /// Feed-forward intermediate dimension.
    pub intermediate_size: usize,
    /// Number of attention query heads.
    pub num_heads: usize,
    /// Number of key/value heads (GQA). Equal to `num_heads` for MHA.
    pub num_kv_heads: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length (for RoPE table sizing).
    pub max_position_embeddings: usize,
    /// RoPE base frequency (theta).
    pub rope_theta: f32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f64,
    /// Number of *distinct* shared blocks. `1` is the canonical RDT; a small
    /// value (e.g. 2) supports ALBERT-style grouped sharing.
    pub num_shared_blocks: usize,
    /// Maximum recurrent loop iterations (the depth ceiling).
    pub max_loops: usize,
    /// Halting probability threshold in `(0, 1]`. A token exits once its
    /// `p_halt` reaches this value.
    pub halt_threshold: f32,
}

impl Default for RdtConfig {
    fn default() -> Self {
        // A small, valid config suitable for tests and smoke runs.
        Self {
            hidden_size: 256,
            intermediate_size: 688,
            num_heads: 8,
            num_kv_heads: 8,
            vocab_size: 1024,
            max_position_embeddings: 2048,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            num_shared_blocks: 1,
            max_loops: 16,
            halt_threshold: 0.9,
        }
    }
}

impl RdtConfig {
    /// Head dimension (`hidden_size / num_heads`).
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// GQA grouping ratio (`num_heads / num_kv_heads`).
    pub fn gqa_ratio(&self) -> usize {
        self.num_heads / self.num_kv_heads.max(1)
    }

    /// Validate structural invariants of the configuration.
    ///
    /// This guards the *shape* contract; semantic RDT-compatibility of weights
    /// is enforced separately by [`validate_rdt_metadata`].
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 || self.num_heads == 0 {
            return Err(RuvLLMError::Config(
                "RDT: hidden_size and num_heads must be non-zero".into(),
            ));
        }
        if self.hidden_size % self.num_heads != 0 {
            return Err(RuvLLMError::Config(format!(
                "RDT: hidden_size ({}) must be divisible by num_heads ({})",
                self.hidden_size, self.num_heads
            )));
        }
        if self.num_kv_heads == 0 || self.num_heads % self.num_kv_heads != 0 {
            return Err(RuvLLMError::Config(format!(
                "RDT: num_heads ({}) must be divisible by num_kv_heads ({})",
                self.num_heads, self.num_kv_heads
            )));
        }
        if self.num_shared_blocks == 0 {
            return Err(RuvLLMError::Config(
                "RDT: num_shared_blocks must be >= 1".into(),
            ));
        }
        if self.max_loops == 0 {
            return Err(RuvLLMError::Config("RDT: max_loops must be >= 1".into()));
        }
        if !(self.halt_threshold > 0.0 && self.halt_threshold <= 1.0) {
            return Err(RuvLLMError::Config(format!(
                "RDT: halt_threshold ({}) must be in (0, 1]",
                self.halt_threshold
            )));
        }
        if self.max_loops < self.num_shared_blocks {
            return Err(RuvLLMError::Config(format!(
                "RDT: max_loops ({}) must be >= num_shared_blocks ({})",
                self.max_loops, self.num_shared_blocks
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The Honest Boundary: weight-sharing compatibility validation
// ---------------------------------------------------------------------------

/// Error returned when a model is asked to run through the RDT path but its
/// weights were not trained for weight-sharing.
///
/// Surfacing this as a hard error (rather than emitting garbage tokens) is the
/// "honest boundary" mandated by the ADR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RdtCompatibilityError {
    /// The GGUF `general.architecture` we observed.
    pub detected_architecture: String,
    /// Human-readable reason the weights were rejected.
    pub reason: String,
}

impl std::fmt::Display for RdtCompatibilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "non-RDT GGUF rejected (architecture='{}'): {}. \
             The RDT execution path requires weights natively trained for \
             weight-sharing (ALBERT-style cross-layer sharing or an explicit \
             RDT fine-tune). Running standard weights through a shared block \
             produces garbage tokens.",
            self.detected_architecture, self.reason
        )
    }
}

impl std::error::Error for RdtCompatibilityError {}

impl From<RdtCompatibilityError> for RuvLLMError {
    fn from(e: RdtCompatibilityError) -> Self {
        RuvLLMError::Model(e.to_string())
    }
}

/// GGUF metadata keys that mark a checkpoint as weight-sharing / RDT.
///
/// A compatible export must set `general.architecture` to one of
/// [`RDT_ARCHITECTURES`] *or* declare an explicit recurrence flag via one of
/// these keys.
pub const RDT_RECURRENCE_KEYS: &[&str] = &[
    "rdt.recurrent",
    "rdt.weight_sharing",
    "general.weight_sharing",
    "recurrent_depth.enabled",
];

/// `general.architecture` values recognized as natively weight-sharing.
pub const RDT_ARCHITECTURES: &[&str] =
    &["rdt", "recurrent_depth", "albert", "universal_transformer"];

/// Validate that GGUF metadata describes a weight-sharing (RDT) checkpoint.
///
/// This is the load-time gate enforcing the honest boundary. It accepts the
/// metadata as a string map (the architecture-agnostic projection of GGUF
/// metadata) so it can be unit-tested without a real GGUF file and reused from
/// the candle loader.
///
/// A checkpoint is accepted iff **either**:
/// - `general.architecture` is one of [`RDT_ARCHITECTURES`], or
/// - one of [`RDT_RECURRENCE_KEYS`] is present and truthy
///   (`true` / `1` / `yes`, case-insensitive).
///
/// # Errors
///
/// Returns [`RdtCompatibilityError`] for any non-RDT architecture (e.g.
/// `llama`, `qwen2`). Per the ADR the caller should treat this as fatal at
/// initialization rather than proceeding to emit garbage tokens.
pub fn validate_rdt_metadata(
    metadata: &BTreeMap<String, String>,
) -> std::result::Result<(), RdtCompatibilityError> {
    let arch = metadata
        .get("general.architecture")
        .map(|s| s.trim().to_lowercase())
        .unwrap_or_default();

    if RDT_ARCHITECTURES.contains(&arch.as_str()) {
        return Ok(());
    }

    for key in RDT_RECURRENCE_KEYS {
        if let Some(raw) = metadata.get(*key) {
            if is_truthy(raw) {
                return Ok(());
            }
        }
    }

    let reason = if arch.is_empty() {
        "no 'general.architecture' and no recurrence flag found".to_string()
    } else {
        format!(
            "architecture '{}' is not weight-sharing and no recurrence flag \
             ({:?}) was set",
            arch, RDT_RECURRENCE_KEYS
        )
    };

    Err(RdtCompatibilityError {
        detected_architecture: if arch.is_empty() {
            "<unknown>".to_string()
        } else {
            arch
        },
        reason,
    })
}

fn is_truthy(raw: &str) -> bool {
    matches!(
        raw.trim().to_lowercase().as_str(),
        "true" | "1" | "yes" | "on"
    )
}

// ---------------------------------------------------------------------------
// Depth telemetry (always available)
// ---------------------------------------------------------------------------

/// Snapshot of recurrent-depth statistics over recorded forward passes.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DepthStats {
    /// Number of recorded forward passes.
    pub samples: usize,
    /// Mean realized loop depth across all recorded tokens.
    pub mean_inference_depth: f32,
    /// Maximum loop depth observed in any single pass.
    pub max_inference_depth: usize,
    /// Minimum loop depth observed in any single pass.
    pub min_inference_depth: usize,
}

impl Default for DepthStats {
    fn default() -> Self {
        Self {
            samples: 0,
            mean_inference_depth: 0.0,
            max_inference_depth: 0,
            min_inference_depth: 0,
        }
    }
}

#[derive(Debug, Default)]
struct DepthInner {
    /// Per-call mean token depth.
    means: Vec<f32>,
    /// Per-call maximum token depth.
    maxes: Vec<usize>,
    /// Per-call minimum token depth.
    mins: Vec<usize>,
}

/// Thread-safe recorder for per-token recurrent depth.
///
/// `bench/system-audit.mjs` consumes `mean_inference_depth` from
/// [`DepthStats`] to track token-efficiency over time.
#[derive(Debug, Default)]
pub struct DepthTelemetry {
    inner: Mutex<DepthInner>,
}

impl DepthTelemetry {
    /// Create an empty telemetry recorder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one forward pass given the per-token realized depths.
    pub fn record(&self, token_depths: &[usize]) {
        if token_depths.is_empty() {
            return;
        }
        let sum: usize = token_depths.iter().sum();
        let mean = sum as f32 / token_depths.len() as f32;
        let max = *token_depths.iter().max().unwrap();
        let min = *token_depths.iter().min().unwrap();

        let mut inner = self.inner.lock().unwrap();
        inner.means.push(mean);
        inner.maxes.push(max);
        inner.mins.push(min);
    }

    /// Aggregate statistics over all recorded passes.
    pub fn stats(&self) -> DepthStats {
        let inner = self.inner.lock().unwrap();
        let samples = inner.means.len();
        if samples == 0 {
            return DepthStats::default();
        }
        let mean = inner.means.iter().sum::<f32>() / samples as f32;
        DepthStats {
            samples,
            mean_inference_depth: mean,
            max_inference_depth: inner.maxes.iter().copied().max().unwrap_or(0),
            min_inference_depth: inner.mins.iter().copied().min().unwrap_or(0),
        }
    }

    /// Reset all recorded telemetry.
    pub fn reset(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.means.clear();
        inner.maxes.clear();
        inner.mins.clear();
    }

    /// Serialize the aggregate [`DepthStats`] to a JSON string for audit
    /// dashboards (`bench/system-audit.mjs` tracks `mean_inference_depth`).
    pub fn report_json(&self) -> String {
        serde_json::to_string(&self.stats()).unwrap_or_else(|_| "{}".to_string())
    }
}

// ---------------------------------------------------------------------------
// Candle execution graph
// ---------------------------------------------------------------------------

#[cfg(feature = "candle")]
pub use candle_impl::{HaltingRouter, RdtCache, RdtModel, SharedBlock};

#[cfg(feature = "candle")]
mod candle_impl {
    use super::*;
    use candle_core::{DType, Device, IndexOp, Tensor, D};
    use candle_nn::{ops, Embedding, Linear, Module, RmsNorm, VarBuilder};

    /// The routing head that decides whether a hidden state is "cooked" enough
    /// to halt. Projects the hidden state to a per-token scalar and applies a
    /// sigmoid to obtain `p_halt`.
    pub struct HaltingRouter {
        proj: Linear,
        threshold: f32,
    }

    impl HaltingRouter {
        /// Build from a projection layer and halting threshold.
        pub fn new(proj: Linear, threshold: f32) -> Self {
            Self { proj, threshold }
        }

        /// Load the routing head from a [`VarBuilder`].
        pub fn load(vb: VarBuilder, hidden_size: usize, threshold: f32) -> Result<Self> {
            // [hidden_size] -> [1]; bias lets the probe learn a base halt rate.
            let proj = candle_nn::linear(hidden_size, 1, vb.pp("proj")).map_err(cand)?;
            Ok(Self::new(proj, threshold))
        }

        /// Compute halting probability for each token.
        ///
        /// Returns `p_halt` shaped `[batch, seq, 1]`.
        pub fn p_halt(&self, hidden_state: &Tensor) -> Result<Tensor> {
            let logits = self.proj.forward(hidden_state).map_err(cand)?;
            ops::sigmoid(&logits).map_err(cand)
        }

        /// Convenience: `(p_halt, should_stop)` using the batch-max policy from
        /// the ADR. The full per-token policy lives in
        /// [`RdtModel::forward`]; this mirrors the ADR's reference signature and
        /// is handy for diagnostics.
        pub fn compute_halt(&self, hidden_state: &Tensor) -> Result<(Tensor, bool)> {
            let p_halt = self.p_halt(hidden_state)?;
            let max_p = p_halt
                .max_all()
                .map_err(cand)?
                .to_scalar::<f32>()
                .map_err(cand)?;
            Ok((p_halt, max_p >= self.threshold))
        }
    }

    /// A single shared transformer block: pre-norm attention + pre-norm SwiGLU
    /// MLP, each with a residual connection. The *same* block instance is
    /// applied repeatedly by [`RdtModel`].
    pub struct SharedBlock {
        input_norm: RmsNorm,
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        post_attn_norm: RmsNorm,
        gate_proj: Linear,
        up_proj: Linear,
        down_proj: Linear,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    }

    impl SharedBlock {
        /// Load a shared block from a [`VarBuilder`].
        pub fn load(vb: VarBuilder, cfg: &RdtConfig) -> Result<Self> {
            let h = cfg.hidden_size;
            let head_dim = cfg.head_dim();
            let q_out = cfg.num_heads * head_dim;
            let kv_out = cfg.num_kv_heads * head_dim;

            let input_norm =
                candle_nn::rms_norm(h, cfg.rms_norm_eps, vb.pp("input_layernorm")).map_err(cand)?;
            let attn = vb.pp("self_attn");
            let q_proj = candle_nn::linear_no_bias(h, q_out, attn.pp("q_proj")).map_err(cand)?;
            let k_proj = candle_nn::linear_no_bias(h, kv_out, attn.pp("k_proj")).map_err(cand)?;
            let v_proj = candle_nn::linear_no_bias(h, kv_out, attn.pp("v_proj")).map_err(cand)?;
            let o_proj = candle_nn::linear_no_bias(q_out, h, attn.pp("o_proj")).map_err(cand)?;

            let post_attn_norm =
                candle_nn::rms_norm(h, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))
                    .map_err(cand)?;
            let mlp = vb.pp("mlp");
            let gate_proj =
                candle_nn::linear_no_bias(h, cfg.intermediate_size, mlp.pp("gate_proj"))
                    .map_err(cand)?;
            let up_proj = candle_nn::linear_no_bias(h, cfg.intermediate_size, mlp.pp("up_proj"))
                .map_err(cand)?;
            let down_proj =
                candle_nn::linear_no_bias(cfg.intermediate_size, h, mlp.pp("down_proj"))
                    .map_err(cand)?;

            Ok(Self {
                input_norm,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                post_attn_norm,
                gate_proj,
                up_proj,
                down_proj,
                num_heads: cfg.num_heads,
                num_kv_heads: cfg.num_kv_heads,
                head_dim,
            })
        }

        /// One pass of the shared block over `[batch, seq, hidden]`.
        ///
        /// `cos`/`sin` are the RoPE tables for the current positions, and
        /// `mask` is the additive causal mask `[seq, seq]`.
        pub fn forward(
            &self,
            xs: &Tensor,
            cos: &Tensor,
            sin: &Tensor,
            mask: &Tensor,
        ) -> Result<Tensor> {
            let (out, _kv) = self.forward_cached(xs, cos, sin, mask, None)?;
            Ok(out)
        }

        /// Cached pass returning the updated `RdtKvCache` for the caller to store.
        pub fn forward_cached(
            &self,
            xs: &Tensor,
            cos: &Tensor,
            sin: &Tensor,
            mask: &Tensor,
            past: Option<&RdtKvCache>,
        ) -> Result<(Tensor, RdtKvCache)> {
            let (b, seq, _h) = xs.dims3().map_err(cand)?;

            // --- Attention sub-layer (pre-norm + residual) ---
            let normed = self.input_norm.forward(xs).map_err(cand)?;
            let (attn_out, kv) = self.attention(&normed, cos, sin, mask, past, b, seq)?;
            let xs = (xs + attn_out).map_err(cand)?;

            // --- MLP sub-layer (pre-norm + residual) ---
            let normed = self.post_attn_norm.forward(&xs).map_err(cand)?;
            let mlp_out = self.mlp(&normed)?;
            let out = (xs + mlp_out).map_err(cand)?;
            Ok((out, kv))
        }

        fn attention(
            &self,
            xs: &Tensor,
            cos: &Tensor,
            sin: &Tensor,
            mask: &Tensor,
            past: Option<&RdtKvCache>,
            b: usize,
            seq: usize,
        ) -> Result<(Tensor, RdtKvCache)> {
            let q = self.q_proj.forward(xs).map_err(cand)?;
            let k = self.k_proj.forward(xs).map_err(cand)?;
            let v = self.v_proj.forward(xs).map_err(cand)?;

            // [b, seq, n*hd] -> [b, n, seq, hd]
            let q = q
                .reshape((b, seq, self.num_heads, self.head_dim))
                .map_err(cand)?
                .transpose(1, 2)
                .map_err(cand)?
                .contiguous()
                .map_err(cand)?;
            let k = k
                .reshape((b, seq, self.num_kv_heads, self.head_dim))
                .map_err(cand)?
                .transpose(1, 2)
                .map_err(cand)?
                .contiguous()
                .map_err(cand)?;
            let v = v
                .reshape((b, seq, self.num_kv_heads, self.head_dim))
                .map_err(cand)?
                .transpose(1, 2)
                .map_err(cand)?
                .contiguous()
                .map_err(cand)?;

            let q = apply_rope(&q, cos, sin)?;
            let k_cur = apply_rope(&k, cos, sin)?;

            let n_rep = self.num_heads / self.num_kv_heads;
            // Accumulate KV — two paths matching OpenMythos GqaPrealloc/Gqa.
            let (k, v, new_kv) = match past {
                // Pre-allocated: scatter_set + skip repeat_kv on full history.
                Some(RdtKvCache::Prealloc {
                    k: buf_k,
                    v: buf_v,
                    seq_len,
                    max_seq,
                }) => {
                    let k_cur_rep = repeat_kv(&k_cur, n_rep)?; // [b, n_heads, seq, hd]
                    let v_rep_new = repeat_kv(&v, n_rep)?;
                    let idx = Tensor::full(*seq_len as u32, k_cur_rep.shape(), k_cur_rep.device())
                        .map_err(cand)?;
                    buf_k.scatter_set(&idx, &k_cur_rep, 2).map_err(cand)?;
                    buf_v.scatter_set(&idx, &v_rep_new, 2).map_err(cand)?;
                    let new_seq = seq_len + seq;
                    let k_v = buf_k.narrow(2, 0, new_seq).map_err(cand)?;
                    let v_v = buf_v.narrow(2, 0, new_seq).map_err(cand)?;
                    let cache = RdtKvCache::Prealloc {
                        k: buf_k.clone(),
                        v: buf_v.clone(),
                        seq_len: new_seq,
                        max_seq: *max_seq,
                    };
                    (k_v, v_v, cache)
                }
                // Legacy cat path.
                Some(RdtKvCache::Cat(pk, pv)) => {
                    let k_f = Tensor::cat(&[pk, &k_cur], 2).map_err(cand)?;
                    let v_f = Tensor::cat(&[pv, &v], 2).map_err(cand)?;
                    let k_r = repeat_kv(&k_f, n_rep)?;
                    let v_r = repeat_kv(&v_f, n_rep)?;
                    (k_r, v_r, RdtKvCache::Cat(k_f, v_f))
                }
                None => {
                    let k_r = repeat_kv(&k_cur, n_rep)?;
                    let v_r = repeat_kv(&v, n_rep)?;
                    (k_r, v_r, RdtKvCache::Cat(k_cur, v))
                }
            };

            let scale = 1.0 / (self.head_dim as f64).sqrt();
            let scores = (q.matmul(&k.transpose(2, 3).map_err(cand)?).map_err(cand)? * scale)
                .map_err(cand)?;
            // Additive causal mask broadcast over [b, n, seq, kv_len].
            let scores = scores.broadcast_add(mask).map_err(cand)?;
            let probs = ops::softmax_last_dim(&scores).map_err(cand)?;

            let ctx = probs.matmul(&v).map_err(cand)?; // [b, n, seq, hd]
            let ctx = ctx
                .transpose(1, 2)
                .map_err(cand)?
                .contiguous()
                .map_err(cand)?
                .reshape((b, seq, self.num_heads * self.head_dim))
                .map_err(cand)?;
            let out = self.o_proj.forward(&ctx).map_err(cand)?;
            Ok((out, new_kv))
        }

        fn mlp(&self, xs: &Tensor) -> Result<Tensor> {
            let gate = self.gate_proj.forward(xs).map_err(cand)?;
            let gate = ops::silu(&gate).map_err(cand)?;
            let up = self.up_proj.forward(xs).map_err(cand)?;
            let hidden = (gate * up).map_err(cand)?;
            self.down_proj.forward(&hidden).map_err(cand)
        }
    }

    /// The Recurrent-Depth Transformer model.
    pub struct RdtModel {
        embed_tokens: Embedding,
        /// One or more shared blocks, cycled through across loop iterations.
        shared_blocks: Vec<SharedBlock>,
        halting_router: HaltingRouter,
        ln_f: RmsNorm,
        lm_head: Linear,
        cfg: RdtConfig,
        device: Device,
        dtype: DType,
        /// Precomputed RoPE tables `[max_position_embeddings, head_dim]` in model dtype.
        rope_cos: Tensor,
        rope_sin: Tensor,
        /// Precomputed lower-triangular additive causal mask
        /// `[max_position_embeddings, max_position_embeddings]` in model dtype.
        causal_mask_cache: Tensor,
        /// Recurrent-depth telemetry (see [`DepthTelemetry`]).
        pub telemetry: DepthTelemetry,
    }

    impl RdtModel {
        /// Build an RDT model from a [`VarBuilder`].
        ///
        /// Validates [`RdtConfig`] structurally. Weight-sharing compatibility of
        /// the *source checkpoint* must be enforced separately by the loader via
        /// [`validate_rdt_metadata`] — that is the honest boundary.
        pub fn load(vb: VarBuilder, cfg: RdtConfig) -> Result<Self> {
            cfg.validate()?;
            let device = vb.device().clone();
            let dtype = vb.dtype();

            let embed_tokens =
                candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))
                    .map_err(cand)?;

            let mut shared_blocks = Vec::with_capacity(cfg.num_shared_blocks);
            let blocks_vb = vb.pp("shared_blocks");
            for i in 0..cfg.num_shared_blocks {
                shared_blocks.push(SharedBlock::load(blocks_vb.pp(i), &cfg)?);
            }

            let halting_router =
                HaltingRouter::load(vb.pp("halting_router"), cfg.hidden_size, cfg.halt_threshold)?;
            let ln_f = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))
                .map_err(cand)?;
            let lm_head =
                candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))
                    .map_err(cand)?;

            // Precompute RoPE tables and causal mask for all positions up to
            // max_position_embeddings — eliminates per-forward-pass from_vec + H2D upload.
            let head_dim = cfg.head_dim();
            let max_pos = cfg.max_position_embeddings;
            let (rope_cos, rope_sin) = {
                let half = head_dim / 2;
                let theta = cfg.rope_theta as f64;
                let inv_freq: Vec<f32> = (0..half)
                    .map(|i| (1.0 / theta.powf(2.0 * i as f64 / head_dim as f64)) as f32)
                    .collect();
                let inv_freq = Tensor::from_vec(inv_freq, (1, half), &device).map_err(cand)?;
                let positions: Vec<f32> = (0..max_pos).map(|p| p as f32).collect();
                let positions = Tensor::from_vec(positions, (max_pos, 1), &device).map_err(cand)?;
                let freqs = positions.matmul(&inv_freq).map_err(cand)?;
                let freqs = Tensor::cat(&[&freqs, &freqs], D::Minus1).map_err(cand)?;
                let cos = freqs.cos().map_err(cand)?.to_dtype(dtype).map_err(cand)?;
                let sin = freqs.sin().map_err(cand)?.to_dtype(dtype).map_err(cand)?;
                (cos, sin)
            };
            let causal_mask_cache = {
                let n = max_pos;
                let mut data = vec![0f32; n * n];
                for i in 0..n {
                    for j in (i + 1)..n {
                        data[i * n + j] = f32::NEG_INFINITY;
                    }
                }
                Tensor::from_vec(data, (n, n), &device)
                    .map_err(cand)?
                    .to_dtype(dtype)
                    .map_err(cand)?
            };

            Ok(Self {
                embed_tokens,
                shared_blocks,
                halting_router,
                ln_f,
                lm_head,
                cfg,
                device,
                dtype,
                rope_cos,
                rope_sin,
                causal_mask_cache,
                telemetry: DepthTelemetry::new(),
            })
        }

        /// Access the model configuration.
        pub fn config(&self) -> &RdtConfig {
            &self.cfg
        }

        /// Run a forward pass over `input_ids` (`[batch, seq]`, dtype u32),
        /// returning logits `[batch, seq, vocab]`.
        ///
        /// Internally embeds the tokens and drives the recurrent loop with
        /// per-token adaptive halting before the final norm + LM head.
        pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
            let mut cache = RdtCache::new();
            self.forward_cached(input_ids, &mut cache)
        }

        /// Forward that reads and updates `cache` for incremental decode.
        /// Processes the `seq` new positions in `input_ids` against the cached
        /// prefix and returns logits `[batch, seq, vocab]`.
        pub fn forward_cached(&self, input_ids: &Tensor, cache: &mut RdtCache) -> Result<Tensor> {
            let (b, seq) = input_ids.dims2().map_err(cand)?;
            let offset = cache.seq_len;
            let xs = self.embed_tokens.forward(input_ids).map_err(cand)?;
            let xs = xs.to_dtype(self.dtype).map_err(cand)?;

            // Slice precomputed RoPE tables for positions offset..offset+seq.
            let cos = self.rope_cos.narrow(0, offset, seq).map_err(cand)?;
            let sin = self.rope_sin.narrow(0, offset, seq).map_err(cand)?;
            // Slice precomputed causal mask: rows offset..offset+seq, cols 0..offset+seq.
            let mask = self
                .causal_mask_cache
                .narrow(0, offset, seq)
                .map_err(cand)?
                .narrow(1, 0, offset + seq)
                .map_err(cand)?;

            let (hidden, kv) =
                self.recurrent_loop(&xs, &cos, &sin, &mask, cache.kv.as_ref(), b, seq)?;
            cache.kv = Some(kv);
            cache.seq_len += seq;

            let normed = self.ln_f.forward(&hidden).map_err(cand)?;
            self.lm_head.forward(&normed).map_err(cand)
        }

        /// Greedy autoregressive generation from a single-sequence prompt.
        /// Returns the newly generated token ids; stops early on `eos`.
        pub fn generate(
            &self,
            prompt_ids: &[u32],
            max_new_tokens: usize,
            eos: Option<u32>,
        ) -> Result<Vec<u32>> {
            if prompt_ids.is_empty() {
                return Err(RuvLLMError::Generation("empty prompt".into()));
            }
            let mut cache = RdtCache::with_prealloc(&self.cfg, 1, &self.device, self.dtype)
                .unwrap_or_else(|_| RdtCache::new());
            let prompt = Tensor::from_slice(prompt_ids, (1, prompt_ids.len()), &self.device)
                .map_err(cand)?;
            let logits = self.forward_cached(&prompt, &mut cache)?;
            let mut next = last_argmax(&logits)?;

            let mut out = Vec::with_capacity(max_new_tokens);
            for _ in 0..max_new_tokens {
                out.push(next);
                if Some(next) == eos {
                    break;
                }
                let step = Tensor::from_slice(&[next], (1, 1), &self.device).map_err(cand)?;
                let logits = self.forward_cached(&step, &mut cache)?;
                next = last_argmax(&logits)?;
            }
            Ok(out)
        }

        /// Autoregressive generation with configurable sampling. Uses GPU top-k
        /// sort + on-device argmax — same optimised path as `OpenMythos`.
        pub fn generate_sampled(
            &self,
            prompt_ids: &[u32],
            max_new_tokens: usize,
            eos: Option<u32>,
            sampling: crate::models::sampling::SamplingConfig,
        ) -> Result<Vec<u32>> {
            if prompt_ids.is_empty() {
                return Err(RuvLLMError::Generation("empty prompt".into()));
            }
            let top_k_transfer = if sampling.top_k > 0 {
                sampling.top_k
            } else {
                512.min(self.cfg.vocab_size)
            };
            let mut sampler = crate::models::sampling::Sampler::new(sampling);
            let mut cache = RdtCache::with_prealloc(&self.cfg, 1, &self.device, self.dtype)
                .unwrap_or_else(|_| RdtCache::new());
            let mut history: Vec<u32> = prompt_ids.to_vec();

            let prompt = Tensor::from_slice(prompt_ids, (1, prompt_ids.len()), &self.device)
                .map_err(cand)?;
            let logits = self.forward_cached(&prompt, &mut cache)?;
            let (vals, idxs) = last_logits_topk(&logits, top_k_transfer, &self.device)?;
            let mut next = sampler.sample_topk(&vals, &idxs, &history);

            let mut out = Vec::with_capacity(max_new_tokens);
            for _ in 0..max_new_tokens {
                out.push(next);
                history.push(next);
                if Some(next) == eos {
                    break;
                }
                let step = Tensor::from_slice(&[next], (1, 1), &self.device).map_err(cand)?;
                let logits = self.forward_cached(&step, &mut cache)?;
                let (vals, idxs) = last_logits_topk(&logits, top_k_transfer, &self.device)?;
                next = sampler.sample_topk(&vals, &idxs, &history);
            }
            Ok(out)
        }

        /// Token-by-token streaming generation via callback (same pattern as
        /// `OpenMythos::generate_stream_sampled`).
        pub fn generate_stream_sampled(
            &self,
            prompt_ids: &[u32],
            max_new_tokens: usize,
            eos: Option<u32>,
            sampling: crate::models::sampling::SamplingConfig,
            mut on_token: impl FnMut(u32) -> bool,
        ) -> Result<()> {
            if prompt_ids.is_empty() {
                return Err(RuvLLMError::Generation("empty prompt".into()));
            }
            let top_k_transfer = if sampling.top_k > 0 {
                sampling.top_k
            } else {
                512.min(self.cfg.vocab_size)
            };
            let mut sampler = crate::models::sampling::Sampler::new(sampling);
            let mut cache = RdtCache::with_prealloc(&self.cfg, 1, &self.device, self.dtype)
                .unwrap_or_else(|_| RdtCache::new());
            let mut history: Vec<u32> = prompt_ids.to_vec();

            let prompt = Tensor::from_slice(prompt_ids, (1, prompt_ids.len()), &self.device)
                .map_err(cand)?;
            let logits = self.forward_cached(&prompt, &mut cache)?;
            let (vals, idxs) = last_logits_topk(&logits, top_k_transfer, &self.device)?;
            let mut next = sampler.sample_topk(&vals, &idxs, &history);

            for _ in 0..max_new_tokens {
                if !on_token(next) {
                    break;
                }
                history.push(next);
                if Some(next) == eos {
                    break;
                }
                let step = Tensor::from_slice(&[next], (1, 1), &self.device).map_err(cand)?;
                let logits = self.forward_cached(&step, &mut cache)?;
                let (vals, idxs) = last_logits_topk(&logits, top_k_transfer, &self.device)?;
                next = sampler.sample_topk(&vals, &idxs, &history);
            }
            Ok(())
        }

        /// The deep recurrent loop with per-token adaptive halting.
        ///
        /// Each iteration applies a shared block, then the routing head emits a
        /// per-token `p_halt`. Tokens crossing `halt_threshold` halt: their
        /// hidden state is frozen for subsequent iterations while still-running
        /// tokens keep computing. The loop ends when every token has halted or
        /// `max_loops` is reached. Returns the final hidden state and the
        /// final-iteration K/V (for the decode cache); records depth in telemetry.
        fn recurrent_loop(
            &self,
            xs: &Tensor,
            cos: &Tensor,
            sin: &Tensor,
            mask: &Tensor,
            past: Option<&RdtKvCache>,
            b: usize,
            seq: usize,
        ) -> Result<(Tensor, RdtKvCache)> {
            let n = b * seq;
            let mut hidden = xs.clone();

            // GPU-resident ACT state — eliminates per-iteration to_vec1()/from_vec() transfers.
            // `running_f32`:  1.0 for still-running tokens, 0.0 for halted, [b, seq, 1]
            // `depth_f32`:    count of iterations each token was running, [b, seq, 1]
            let mut running_f32 =
                Tensor::ones((b, seq, 1), DType::F32, &self.device).map_err(cand)?;
            let mut depth_f32 =
                Tensor::zeros((b, seq, 1), DType::F32, &self.device).map_err(cand)?;
            let mut last_kv: Option<RdtKvCache> = None;

            let max_loops = self.cfg.max_loops;
            for step in 0..max_loops {
                let block = &self.shared_blocks[step % self.shared_blocks.len()];
                let (candidate, kv) = block.forward_cached(&hidden, cos, sin, mask, past)?;
                last_kv = Some(kv);

                // Freeze halted tokens: hidden = running*candidate + (1-running)*hidden.
                let running_typed = running_f32.to_dtype(self.dtype).map_err(cand)?;
                let halted_typed =
                    (running_typed.ones_like().map_err(cand)? - &running_typed).map_err(cand)?;
                hidden = (candidate.broadcast_mul(&running_typed).map_err(cand)?
                    + hidden.broadcast_mul(&halted_typed).map_err(cand)?)
                .map_err(cand)?;

                // Depth: increment per-token count for tokens that were running this step.
                depth_f32 = (&depth_f32 + &running_f32).map_err(cand)?;

                // Halting decision — fully on device, no weight-vector transfer.
                let p_halt = self.halting_router.p_halt(&hidden)?;
                let p_halt_f32 = p_halt.to_dtype(DType::F32).map_err(cand)?;
                let should_halt = p_halt_f32
                    .ge(self.cfg.halt_threshold as f64)
                    .map_err(cand)?
                    .to_dtype(DType::F32)
                    .map_err(cand)?;
                // Newly halted = was running AND p_halt >= threshold.
                let newly_halted = (&should_halt * &running_f32).map_err(cand)?;
                running_f32 = (&running_f32 - &newly_halted).map_err(cand)?;

                tracing::trace!(step, "rdt loop iteration");

                // Early-exit: one scalar sync (cheap vs. the eliminated weight-vector transfers).
                let any_running = running_f32
                    .sum_all()
                    .map_err(cand)?
                    .to_scalar::<f32>()
                    .map_err(cand)?
                    > 0.5;
                if !any_running {
                    break;
                }
            }

            // Single depth sync at the end (not in the hot path).
            let depth_vec: Vec<f32> = depth_f32
                .reshape((n,))
                .map_err(cand)?
                .to_vec1()
                .map_err(cand)?;
            let depth: Vec<usize> = depth_vec.into_iter().map(|d| d as usize).collect();
            self.telemetry.record(&depth);

            let kv = last_kv.expect("at least one loop iteration runs");
            Ok((hidden, kv))
        }
    }

    /// KV state for one RDT decode session.
    pub enum RdtKvCache {
        /// Grows via Tensor::cat (legacy / first-call path).
        Cat(Tensor, Tensor),
        /// Pre-allocated `[b, num_heads, max_seq, head_dim]` buffers with pre-repeated
        /// KV. Uses scatter_set for O(1) appends and eliminates repeat_kv per step.
        Prealloc {
            k: Tensor,
            v: Tensor,
            seq_len: usize,
            max_seq: usize,
        },
    }

    impl RdtKvCache {
        pub fn seq_len(&self) -> usize {
            match self {
                RdtKvCache::Cat(k, _) => k.dim(2).unwrap_or(0),
                RdtKvCache::Prealloc { seq_len, .. } => *seq_len,
            }
        }
    }

    /// Incremental KV cache for RDT decode (final-iteration K/V of the shared
    /// block, concatenated across decode steps).
    pub struct RdtCache {
        pub(crate) kv: Option<RdtKvCache>,
        seq_len: usize,
    }

    impl Default for RdtCache {
        fn default() -> Self {
            Self {
                kv: None,
                seq_len: 0,
            }
        }
    }

    impl RdtCache {
        pub fn new() -> Self {
            Self::default()
        }

        /// Pre-allocated cache: eliminates Tensor::cat growth and repeat_kv.
        pub fn with_prealloc(
            cfg: &RdtConfig,
            b: usize,
            device: &Device,
            dtype: DType,
        ) -> Result<Self> {
            let max_seq = cfg.max_position_embeddings;
            let n_heads = cfg.num_heads;
            let head_dim = cfg.head_dim();
            let k = Tensor::zeros((b, n_heads, max_seq, head_dim), dtype, device).map_err(cand)?;
            let v = Tensor::zeros((b, n_heads, max_seq, head_dim), dtype, device).map_err(cand)?;
            Ok(Self {
                kv: Some(RdtKvCache::Prealloc {
                    k,
                    v,
                    seq_len: 0,
                    max_seq,
                }),
                seq_len: 0,
            })
        }

        pub fn len(&self) -> usize {
            self.seq_len
        }
        pub fn is_empty(&self) -> bool {
            self.seq_len == 0
        }
        pub fn reset(&mut self) {
            if let Some(RdtKvCache::Prealloc { seq_len, .. }) = &mut self.kv {
                *seq_len = 0;
            } else {
                self.kv = None;
            }
            self.seq_len = 0;
        }
    }

    /// Extract sorted top-k `(values, token_ids)` at the last position using
    /// on-device sort — transfers `2 * k * 4` bytes instead of `vocab * 4`.
    fn last_logits_topk(
        logits: &Tensor,
        k: usize,
        _device: &Device,
    ) -> Result<(Vec<f32>, Vec<u32>)> {
        let (_b, seq, vocab) = logits.dims3().map_err(cand)?;
        let last = logits
            .i((0, seq - 1))
            .map_err(cand)?
            .to_dtype(DType::F32)
            .map_err(cand)?
            .contiguous()
            .map_err(cand)?;
        let k = if k == 0 || k >= vocab { vocab } else { k };
        let (vals, idxs) = last.sort_last_dim(false).map_err(cand)?;
        let vals_k: Vec<f32> = vals
            .narrow(D::Minus1, 0, k)
            .map_err(cand)?
            .to_vec1()
            .map_err(cand)?;
        let idxs_k: Vec<u32> = idxs
            .narrow(D::Minus1, 0, k)
            .map_err(cand)?
            .to_vec1()
            .map_err(cand)?;
        Ok((vals_k, idxs_k))
    }

    /// Argmax over vocab at the last sequence position of `[1, seq, vocab]`.
    ///
    /// `Tensor::argmax` runs on-device; only the winning index (4 bytes) is
    /// transferred vs the full vocab row (~128 KB for vocab=32000).
    fn last_argmax(logits: &Tensor) -> Result<u32> {
        let (_b, seq, _v) = logits.dims3().map_err(cand)?;
        let last = logits.i((0, seq - 1)).map_err(cand)?; // [vocab]
        last.argmax(D::Minus1)
            .map_err(cand)?
            .to_scalar::<u32>()
            .map_err(cand)
    }

    /// Apply rotary position embeddings to `[b, n, seq, head_dim]`.
    fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        // cos/sin: [seq, head_dim] -> broadcast over [b, n, seq, head_dim].
        let (_b, _n, seq, hd) = x.dims4().map_err(cand)?;
        let cos = cos
            .narrow(0, 0, seq)
            .map_err(cand)?
            .reshape((1, 1, seq, hd))
            .map_err(cand)?;
        let sin = sin
            .narrow(0, 0, seq)
            .map_err(cand)?
            .reshape((1, 1, seq, hd))
            .map_err(cand)?;
        let rotated = rotate_half(x)?;
        let out = (x.broadcast_mul(&cos).map_err(cand)?
            + rotated.broadcast_mul(&sin).map_err(cand)?)
        .map_err(cand)?;
        Ok(out)
    }

    /// `rotate_half([x1, x2]) = [-x2, x1]` along the last dimension.
    fn rotate_half(x: &Tensor) -> Result<Tensor> {
        let hd = x.dim(D::Minus1).map_err(cand)?;
        let half = hd / 2;
        let x1 = x.narrow(D::Minus1, 0, half).map_err(cand)?;
        let x2 = x.narrow(D::Minus1, half, hd - half).map_err(cand)?;
        let neg_x2 = x2.neg().map_err(cand)?;
        Tensor::cat(&[&neg_x2, &x1], D::Minus1).map_err(cand)
    }

    /// Repeat KV heads `n_rep` times for grouped-query attention.
    fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            return Ok(x.clone());
        }
        let (b, kv_heads, seq, hd) = x.dims4().map_err(cand)?;
        // [b, kv, seq, hd] -> [b, kv, 1, seq, hd] -> expand -> [b, kv*n_rep, seq, hd]
        x.unsqueeze(2)
            .map_err(cand)?
            .expand((b, kv_heads, n_rep, seq, hd))
            .map_err(cand)?
            .reshape((b, kv_heads * n_rep, seq, hd))
            .map_err(cand)
    }

    /// Map a candle error into the crate error type.
    fn cand(e: candle_core::Error) -> RuvLLMError {
        RuvLLMError::Model(format!("candle (rdt): {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn config_default_is_valid() {
        assert!(RdtConfig::default().validate().is_ok());
    }

    #[test]
    fn config_rejects_bad_shapes() {
        let mut c = RdtConfig::default();
        c.num_heads = 7; // 256 % 7 != 0
        assert!(c.validate().is_err());

        let mut c = RdtConfig::default();
        c.halt_threshold = 1.5;
        assert!(c.validate().is_err());

        let mut c = RdtConfig::default();
        c.max_loops = 0;
        assert!(c.validate().is_err());

        let mut c = RdtConfig::default();
        c.num_heads = 8;
        c.num_kv_heads = 3; // 8 % 3 != 0
        assert!(c.validate().is_err());
    }

    #[test]
    fn honest_boundary_rejects_llama() {
        let m = meta(&[("general.architecture", "llama")]);
        let err = validate_rdt_metadata(&m).unwrap_err();
        assert_eq!(err.detected_architecture, "llama");
        assert!(err.to_string().contains("garbage tokens"));
    }

    #[test]
    fn honest_boundary_rejects_qwen2() {
        let m = meta(&[("general.architecture", "qwen2")]);
        assert!(validate_rdt_metadata(&m).is_err());
    }

    #[test]
    fn honest_boundary_rejects_missing_metadata() {
        let m = meta(&[]);
        assert!(validate_rdt_metadata(&m).is_err());
    }

    #[test]
    fn honest_boundary_accepts_rdt_architecture() {
        for arch in RDT_ARCHITECTURES {
            let m = meta(&[("general.architecture", arch)]);
            assert!(validate_rdt_metadata(&m).is_ok(), "arch {arch} should pass");
        }
    }

    #[test]
    fn honest_boundary_accepts_recurrence_flag() {
        // A llama-derived RDT fine-tune declares recurrence explicitly.
        let m = meta(&[("general.architecture", "llama"), ("rdt.recurrent", "true")]);
        assert!(validate_rdt_metadata(&m).is_ok());

        let m = meta(&[("recurrent_depth.enabled", "1")]);
        assert!(validate_rdt_metadata(&m).is_ok());
    }

    #[test]
    fn honest_boundary_rejects_falsey_flag() {
        let m = meta(&[
            ("general.architecture", "llama"),
            ("rdt.recurrent", "false"),
        ]);
        assert!(validate_rdt_metadata(&m).is_err());
    }

    #[test]
    fn telemetry_aggregates() {
        let t = DepthTelemetry::new();
        t.record(&[2, 4, 6]); // mean 4, max 6, min 2
        t.record(&[1, 1, 1]); // mean 1, max 1, min 1
        let s = t.stats();
        assert_eq!(s.samples, 2);
        assert_eq!(s.max_inference_depth, 6);
        assert_eq!(s.min_inference_depth, 1);
        assert!((s.mean_inference_depth - 2.5).abs() < 1e-6);

        t.reset();
        assert_eq!(t.stats().samples, 0);
    }

    #[test]
    fn telemetry_ignores_empty() {
        let t = DepthTelemetry::new();
        t.record(&[]);
        assert_eq!(t.stats().samples, 0);
    }

    // ---- candle-backed execution tests (synthetic weights) ----

    #[cfg(feature = "candle")]
    mod candle_tests {
        use super::*;
        use candle_core::{DType, Device, IndexOp, Tensor};
        use candle_nn::VarBuilder;

        fn tiny_cfg() -> RdtConfig {
            RdtConfig {
                hidden_size: 32,
                intermediate_size: 64,
                num_heads: 4,
                num_kv_heads: 2,
                vocab_size: 48,
                max_position_embeddings: 64,
                rope_theta: 10_000.0,
                rms_norm_eps: 1e-5,
                num_shared_blocks: 1,
                max_loops: 8,
                halt_threshold: 0.9,
            }
        }

        #[test]
        fn forward_shapes_are_correct() {
            let dev = Device::Cpu;
            let cfg = tiny_cfg();
            // Zeros builder fabricates any requested tensor as zeros — enough to
            // exercise the full execution graph deterministically.
            let vb = VarBuilder::zeros(DType::F32, &dev);
            let model = RdtModel::load(vb, cfg.clone()).expect("load");

            let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4, 5], (1, 5), &dev).unwrap();
            let logits = model.forward(&input_ids).expect("forward");
            assert_eq!(logits.dims(), &[1, 5, cfg.vocab_size]);
        }

        #[test]
        fn zero_router_runs_to_max_loops() {
            // With zero weights, p_halt = sigmoid(0) = 0.5 < 0.9 threshold, so
            // every token runs to the ceiling. This proves the loop honors
            // max_loops and records depth telemetry.
            let dev = Device::Cpu;
            let cfg = tiny_cfg();
            let vb = VarBuilder::zeros(DType::F32, &dev);
            let model = RdtModel::load(vb, cfg.clone()).unwrap();

            let input_ids = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &dev).unwrap();
            let _ = model.forward(&input_ids).unwrap();

            let stats = model.telemetry.stats();
            assert_eq!(stats.samples, 1);
            assert_eq!(stats.max_inference_depth, cfg.max_loops);
            assert_eq!(stats.min_inference_depth, cfg.max_loops);
        }

        #[test]
        fn batched_forward_works() {
            let dev = Device::Cpu;
            let cfg = tiny_cfg();
            let vb = VarBuilder::zeros(DType::F32, &dev);
            let model = RdtModel::load(vb, cfg.clone()).unwrap();

            let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6], (2, 3), &dev).unwrap();
            let logits = model.forward(&input_ids).unwrap();
            assert_eq!(logits.dims(), &[2, 3, cfg.vocab_size]);
            assert_eq!(model.telemetry.stats().samples, 1);
        }

        #[test]
        fn multi_block_sharing_loads_and_runs() {
            let dev = Device::Cpu;
            let mut cfg = tiny_cfg();
            cfg.num_shared_blocks = 2; // ALBERT-style grouped sharing
            let vb = VarBuilder::zeros(DType::F32, &dev);
            let model = RdtModel::load(vb, cfg.clone()).unwrap();
            let input_ids = Tensor::from_vec(vec![7u32, 8, 9, 10], (1, 4), &dev).unwrap();
            let logits = model.forward(&input_ids).unwrap();
            assert_eq!(logits.dims(), &[1, 4, cfg.vocab_size]);
        }

        #[test]
        fn forward_output_is_finite() {
            let dev = Device::Cpu;
            let cfg = tiny_cfg();
            let vb = VarBuilder::zeros(DType::F32, &dev);
            let model = RdtModel::load(vb, cfg).unwrap();
            let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &dev).unwrap();
            let logits = model.forward(&input_ids).unwrap();
            let flat: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
            assert!(flat.iter().all(|x| x.is_finite()), "logits must be finite");
        }

        #[test]
        fn generate_produces_tokens() {
            let dev = Device::Cpu;
            let cfg = tiny_cfg();
            let vb = VarBuilder::zeros(DType::F32, &dev);
            let model = RdtModel::load(vb, cfg.clone()).unwrap();
            let out = model.generate(&[1, 2, 3], 5, None).unwrap();
            assert_eq!(out.len(), 5);
            assert!(out.iter().all(|&t| (t as usize) < cfg.vocab_size));
        }

        #[test]
        fn cached_decode_matches_full_at_single_loop() {
            // At max_loops=1 the recurrent caching is exact (single iteration),
            // so incremental decode must match a full forward. Use random
            // weights so the check is non-degenerate.
            use candle_nn::VarMap;
            let dev = Device::Cpu;
            let mut cfg = tiny_cfg();
            cfg.max_loops = 1;
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
            let model = RdtModel::load(vb, cfg.clone()).unwrap();

            let ids = vec![3u32, 7, 1, 9, 4];
            let full_ids = Tensor::from_vec(ids.clone(), (1, ids.len()), &dev).unwrap();
            let full = model.forward(&full_ids).unwrap();
            let full_last: Vec<f32> = full.i((0, ids.len() - 1)).unwrap().to_vec1().unwrap();

            let mut cache = RdtCache::new();
            let mut last: Vec<f32> = vec![];
            for (k, &tok) in ids.iter().enumerate() {
                let step = Tensor::from_vec(vec![tok], (1, 1), &dev).unwrap();
                let logits = model.forward_cached(&step, &mut cache).unwrap();
                assert_eq!(cache.len(), k + 1);
                last = logits.i((0, 0)).unwrap().to_vec1().unwrap();
            }
            let max_diff = full_last
                .iter()
                .zip(last.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(max_diff < 1e-3, "RDT KV-cache decode diverged: {max_diff}");
        }

        #[test]
        fn telemetry_report_json_roundtrips() {
            let dev = Device::Cpu;
            let cfg = tiny_cfg();
            let vb = VarBuilder::zeros(DType::F32, &dev);
            let model = RdtModel::load(vb, cfg).unwrap();
            let ids = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &dev).unwrap();
            let _ = model.forward(&ids).unwrap();
            let json = model.telemetry.report_json();
            assert!(json.contains("mean_inference_depth"));
            let parsed: DepthStats = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed.samples, 1);
        }
    }
}
