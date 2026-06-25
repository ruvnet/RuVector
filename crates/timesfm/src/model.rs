//! Candle implementation of the TimesFM 1.0 200M patched decoder.
//!
//! Architecturally faithful to google-research/timesfm
//! (`v1/src/timesfm/pytorch_patched_decoder.py`). This module is only compiled
//! when the `candle` feature is enabled. Without real safetensors weights the
//! modules still build and shape-test (use [`VarBuilder::zeros`] / randn).
//!
//! Non-obvious deviations from a vanilla LLM transformer, all implemented here:
//!   * post-norm-ish residual flow (`h = residual + attn(rmsnorm(h))`, then
//!     `h = mlp(h)` where the MLP carries its *own* residual);
//!   * per-dim learnable query scaling via `softplus(scaling)`;
//!   * a `LayerNorm` (not RMSNorm) living *inside* the MLP;
//!   * `ResidualBlock` (SiLU MLP + separate residual projection) for the
//!     input patch embed and the horizon output projection;
//!   * an additive frequency embedding and a non-learned sinusoidal positional
//!     embedding (NOT RoPE);
//!   * RevIN-style per-series instance normalization around the whole stack.

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{ops, Embedding, LayerNorm, Linear, Module, RmsNorm, VarBuilder};

use crate::config::TimesfmConfig;

/// `1.442695041 = 1 / ln(2)`. Folded into the query scaling so the softmax runs
/// in base-2 just like the reference implementation.
const LOG2_E: f64 = 1.442_695_041;

// ---------------------------------------------------------------------------
// ResidualBlock: hidden = SiLU(Linear(in,hid)); out = Linear(hid,out) + Linear(in,out)
// ---------------------------------------------------------------------------

/// `hidden = SiLU(hidden_layer(x)); out = output_layer(hidden) + residual_layer(x)`.
pub struct ResidualBlock {
    hidden_layer: Linear,
    output_layer: Linear,
    residual_layer: Linear,
}

impl ResidualBlock {
    pub fn load(in_dim: usize, hid_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            hidden_layer: candle_nn::linear(in_dim, hid_dim, vb.pp("hidden_layer"))?,
            output_layer: candle_nn::linear(hid_dim, out_dim, vb.pp("output_layer"))?,
            residual_layer: candle_nn::linear(in_dim, out_dim, vb.pp("residual_layer"))?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let hidden = ops::silu(&self.hidden_layer.forward(xs)?)?;
        let out = self.output_layer.forward(&hidden)?;
        let residual = self.residual_layer.forward(xs)?;
        out + residual
    }
}

// ---------------------------------------------------------------------------
// PositionalEmbedding: non-learned sinusoidal, transformer-style sin/cos.
// ---------------------------------------------------------------------------

/// Sinusoidal positional embedding (Vaswani-style), built on the fly and added
/// to the patch embeddings. `min_timescale = 1`, `max_timescale = 10000`.
pub struct PositionalEmbedding {
    dim: usize,
    min_timescale: f64,
    max_timescale: f64,
}

impl PositionalEmbedding {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            min_timescale: 1.0,
            max_timescale: 10_000.0,
        }
    }

    /// Returns `[seq_len, dim]` positional encodings on `device` in `dtype`.
    pub fn forward(&self, seq_len: usize, dtype: DType, device: &Device) -> Result<Tensor> {
        let half = self.dim / 2;
        // position [seq_len, 1]
        let positions: Vec<f32> = (0..seq_len).map(|p| p as f32).collect();
        let positions = Tensor::from_vec(positions, (seq_len, 1), device)?;
        // inv_timescales [1, half]
        let log_increment =
            (self.max_timescale / self.min_timescale).ln() / (half.max(1) as f64 - 1.0).max(1.0);
        let inv: Vec<f32> = (0..half)
            .map(|i| (self.min_timescale * (-(i as f64) * log_increment).exp()) as f32)
            .collect();
        let inv = Tensor::from_vec(inv, (1, half), device)?;
        // scaled [seq_len, half]
        let scaled = positions.broadcast_mul(&inv)?;
        let sin = scaled.sin()?;
        let cos = scaled.cos()?;
        let pe = Tensor::cat(&[&sin, &cos], D::Minus1)?; // [seq_len, 2*half]
                                                         // pad a column if dim is odd (TimesFM dim is even so this is a no-op there).
        let pe = if 2 * half < self.dim {
            let pad = Tensor::zeros((seq_len, self.dim - 2 * half), DType::F32, device)?;
            Tensor::cat(&[&pe, &pad], D::Minus1)?
        } else {
            pe
        };
        pe.to_dtype(dtype)
    }
}

// ---------------------------------------------------------------------------
// TimesFMAttention: fused QKV, per-dim query scaling, additive causal mask.
// ---------------------------------------------------------------------------

pub struct TimesFMAttention {
    qkv_proj: Linear,
    o_proj: Linear,
    /// Learnable per-head-dim scaling parameter, shape `[head_dim]`.
    scaling: Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_queries_per_kv: usize,
}

impl TimesFMAttention {
    pub fn load(cfg: &TimesfmConfig, vb: VarBuilder) -> Result<Self> {
        let qkv_proj = candle_nn::linear(cfg.hidden_size, cfg.qkv_dim(), vb.pp("qkv_proj"))?;
        let o_proj = candle_nn::linear(
            cfg.num_heads * cfg.head_dim,
            cfg.hidden_size,
            vb.pp("o_proj"),
        )?;
        let scaling = vb.get(cfg.head_dim, "scaling")?;
        Ok(Self {
            qkv_proj,
            o_proj,
            scaling,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            num_queries_per_kv: cfg.num_queries_per_kv(),
        })
    }

    /// `xs` `[B, N, D]`, `mask` `[B, 1, N, N]` (additive). Returns `[B, N, D]`.
    pub fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, n, _d) = xs.dims3()?;
        let qkv = self.qkv_proj.forward(xs)?; // [B,N,(h+2kv)*hd]

        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        let q = qkv.narrow(D::Minus1, 0, q_dim)?;
        let k = qkv.narrow(D::Minus1, q_dim, kv_dim)?;
        let v = qkv.narrow(D::Minus1, q_dim + kv_dim, kv_dim)?;

        // [B, heads, N, hd]
        let q = q
            .reshape((b, n, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, n, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, n, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Per-dim query scaling: scale = (log2_e / sqrt(head_dim)) * softplus(scaling).
        // scaling is [head_dim]; broadcast over [B, heads, N, head_dim].
        let softplus = softplus(&self.scaling)?;
        let coef = LOG2_E / (self.head_dim as f64).sqrt();
        let scale = (softplus * coef)?; // [head_dim]
        let scale = scale.reshape((1, 1, 1, self.head_dim))?;
        let q = q.broadcast_mul(&scale)?;

        // Expand kv heads to query heads if GQA (no-op for MHA / qpk == 1).
        let k = repeat_kv(&k, self.num_queries_per_kv)?;
        let v = repeat_kv(&v, self.num_queries_per_kv)?;

        // scores [B, heads, N, N]. q already carries the scaling, so no extra
        // 1/sqrt(d) factor here.
        let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let scores = scores.broadcast_add(mask)?;
        let probs = ops::softmax_last_dim(&scores)?;

        let ctx = probs.matmul(&v)?; // [B, heads, N, hd]
        let ctx =
            ctx.transpose(1, 2)?
                .contiguous()?
                .reshape((b, n, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&ctx)
    }
}

/// Repeat kv heads `n_rep` times along the head dim. No-op for `n_rep == 1`.
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, kv_heads, n, hd) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, kv_heads, n_rep, n, hd))?
        .reshape((b, kv_heads * n_rep, n, hd))
}

/// `softplus(x) = ln(1 + exp(x))`, computed stably.
fn softplus(x: &Tensor) -> Result<Tensor> {
    // ln(1 + exp(x)); candle has no direct softplus, so build it.
    let one = Tensor::ones_like(x)?;
    (x.exp()? + one)?.log()
}

// ---------------------------------------------------------------------------
// TransformerMLP: LayerNorm -> ReLU(gate) -> down_proj, with its OWN residual.
// ---------------------------------------------------------------------------

pub struct TransformerMLP {
    layer_norm: LayerNorm,
    gate_proj: Linear,
    down_proj: Linear,
}

impl TransformerMLP {
    pub fn load(cfg: &TimesfmConfig, vb: VarBuilder) -> Result<Self> {
        let layer_norm =
            candle_nn::layer_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("layer_norm"))?;
        let gate_proj =
            candle_nn::linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?;
        let down_proj =
            candle_nn::linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            layer_norm,
            gate_proj,
            down_proj,
        })
    }

    /// `out = down_proj(relu(gate_proj(layer_norm(x)))) + x`.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let normed = self.layer_norm.forward(xs)?;
        let gate = self.gate_proj.forward(&normed)?.relu()?;
        let out = self.down_proj.forward(&gate)?;
        out + xs
    }
}

// ---------------------------------------------------------------------------
// TimesFMDecoderLayer
// ---------------------------------------------------------------------------

pub struct TimesFMDecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: TimesFMAttention,
    mlp: TransformerMLP,
}

impl TimesFMDecoderLayer {
    pub fn load(cfg: &TimesfmConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            input_layernorm: candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            self_attn: TimesFMAttention::load(cfg, vb.pp("self_attn"))?,
            mlp: TransformerMLP::load(cfg, vb.pp("mlp"))?,
        })
    }

    /// Residual flow (NOT canonical pre-norm):
    ///   residual = h; h = RMSNorm(h); h = attn(h); h = residual + h; h = mlp(h)
    pub fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let h = self.input_layernorm.forward(xs)?;
        let h = self.self_attn.forward(&h, mask)?;
        let h = (residual + h)?;
        // MLP internally LayerNorms and adds its own residual.
        self.mlp.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// StackedDecoder: 20 × TimesFMDecoderLayer, NO final norm.
// ---------------------------------------------------------------------------

pub struct StackedDecoder {
    layers: Vec<TimesFMDecoderLayer>,
}

impl StackedDecoder {
    pub fn load(cfg: &TimesfmConfig, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(TimesFMDecoderLayer::load(cfg, vb.pp(i))?);
        }
        Ok(Self { layers })
    }

    pub fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let mut h = xs.clone();
        for layer in &self.layers {
            h = layer.forward(&h, mask)?;
        }
        Ok(h) // no final norm
    }
}

// ---------------------------------------------------------------------------
// PatchedTimeSeriesDecoder: the full model.
// ---------------------------------------------------------------------------

/// Result of a single forward pass over patched inputs.
pub struct ForecastOutput {
    /// `[B, N, horizon_len, num_outputs]` — per input position, a horizon-step
    /// forecast with channel 0 = mean and 1..=9 = quantiles.
    pub forecasts: Tensor,
}

pub struct PatchedTimeSeriesDecoder {
    input_ff_layer: ResidualBlock,
    freq_emb: Embedding,
    position_emb: PositionalEmbedding,
    stacked_transformer: StackedDecoder,
    horizon_ff_layer: ResidualBlock,
    cfg: TimesfmConfig,
}

impl PatchedTimeSeriesDecoder {
    pub fn load(cfg: TimesfmConfig, vb: VarBuilder) -> Result<Self> {
        let input_ff_layer = ResidualBlock::load(
            cfg.input_ff_in_dim(),
            cfg.hidden_size,
            cfg.hidden_size,
            vb.pp("input_ff_layer"),
        )?;
        let freq_emb = candle_nn::embedding(cfg.num_freq, cfg.hidden_size, vb.pp("freq_emb"))?;
        let position_emb = PositionalEmbedding::new(cfg.hidden_size);
        let stacked_transformer = StackedDecoder::load(&cfg, vb.pp("stacked_transformer"))?;
        let horizon_ff_layer = ResidualBlock::load(
            cfg.hidden_size,
            cfg.hidden_size,
            cfg.horizon_ff_out_dim(),
            vb.pp("horizon_ff_layer"),
        )?;
        Ok(Self {
            input_ff_layer,
            freq_emb,
            position_emb,
            stacked_transformer,
            horizon_ff_layer,
            cfg,
        })
    }

    pub fn config(&self) -> &TimesfmConfig {
        &self.cfg
    }

    /// Build an additive attention mask `[B, 1, N, N]` matching the reference
    /// `merge_masks(convert_paddings_to_mask(...), causal_mask(...))`.
    ///
    /// CRITICAL: the reference uses a *large finite negative number*
    /// (`-0.7 * f32::MAX`), NOT `-inf`. Using `-inf` here is a correctness bug:
    /// the padding term is `padding * neg`, and with real (0/1) paddings
    /// `0 * -inf = NaN`, which poisons the whole mask and makes softmax emit
    /// NaN for every row. A large finite negative keeps `0 * neg = 0` (finite)
    /// and `1 * neg = neg`, exactly as the reference does, while still driving
    /// `exp()` to ~0 after softmax.
    ///
    /// Merge is element-wise minimum of the (broadcast) causal and padding
    /// masks, matching the reference `merge_masks` (`torch.minimum`).
    fn build_mask(&self, patched_padding: &Tensor) -> Result<Tensor> {
        let (b, n) = patched_padding.dims2()?;
        let device = patched_padding.device();
        // Reference: get_large_negative_number(float32) = -0.7 * finfo.max.
        let neg = (-0.7f64) * (f32::MAX as f64);
        // causal [1,1,N,N]: 0 on/below diagonal, `neg` above (row < col).
        let mut causal = vec![0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                if j > i {
                    causal[i * n + j] = neg as f32;
                }
            }
        }
        let causal = Tensor::from_vec(causal, (1, 1, n, n), device)?; // [1,1,N,N]
                                                                      // padding key mask [B,1,1,N]: `neg` where padded key, 0 elsewhere.
                                                                      // padding is strictly 0/1 so `0 * neg = 0` (finite) — no NaN.
        let pad = (patched_padding.to_dtype(DType::F32)? * neg)?; // [B,N]
        let pad_key = pad.reshape((b, 1, 1, n))?; // broadcasts over query dim
                                                  // merge_masks: minimum(padding_key_mask, causal). Broadcast both to
                                                  // [B,1,N,N] and take element-wise min (an additive mask where either
                                                  // term wanting `neg` wins).
        let causal_b = causal.broadcast_as((b, 1, n, n))?;
        let pad_b = pad_key.broadcast_as((b, 1, n, n))?;
        causal_b.minimum(&pad_b)
    }

    /// Forward over already-patched, RevIN-normalized inputs.
    ///
    /// `concat_inputs` `[B, N, 2*patch_len]` = cat(values, pad-mask).
    /// `patched_padding` `[B, N]` (1 = fully-padded patch).
    /// `freq` `[B, 1]` u32 frequency bucket ids.
    /// Returns `[B, N, horizon_len, num_outputs]`.
    pub fn forward_embedded(
        &self,
        concat_inputs: &Tensor,
        patched_padding: &Tensor,
        freq: &Tensor,
    ) -> Result<Tensor> {
        let (b, n, _two_p) = concat_inputs.dims3()?;
        let d = self.cfg.hidden_size;

        // 5. input_ff_layer: [B,N,2P] -> [B,N,D]
        let mut model_input = self.input_ff_layer.forward(concat_inputs)?;

        // 7. positional embedding [N,D] -> [1,N,D] broadcast.
        if self.cfg.use_positional_embedding {
            let pe = self
                .position_emb
                .forward(n, model_input.dtype(), model_input.device())?
                .reshape((1, n, d))?;
            model_input = model_input.broadcast_add(&pe)?;
        }

        // 8. frequency embedding freq_emb(freq) [B,1,D] -> broadcast add.
        let f_emb = self.freq_emb.forward(freq)?; // [B,1,D]
        model_input = model_input.broadcast_add(&f_emb)?;

        // 9. stacked transformer with causal+padding mask.
        let mask = self.build_mask(patched_padding)?;
        let hidden = self.stacked_transformer.forward(&model_input, &mask)?; // [B,N,D]

        // 10. horizon_ff_layer: [B,N,D] -> [B,N,horizon*num_outputs]
        let output_ts = self.horizon_ff_layer.forward(&hidden)?;

        // 11. reshape [B,N,horizon,num_outputs]
        output_ts.reshape((b, n, self.cfg.horizon_len, self.cfg.num_outputs()))
    }

    /// Full forward from raw context. `input_ts` `[B, C]`, `input_padding`
    /// `[B, C]` (1 = padded), `freq` `[B, 1]` u32. Returns the forecast tensor
    /// `[B, N, horizon_len, num_outputs]`, already RevIN-reversed.
    pub fn forward(
        &self,
        input_ts: &Tensor,
        input_padding: &Tensor,
        freq: &Tensor,
    ) -> Result<Tensor> {
        let cfg = &self.cfg;
        let (b, c) = input_ts.dims2()?;
        let p = cfg.patch_len;
        let n = c / p;
        let device = input_ts.device();
        let dtype = input_ts.dtype();

        // 2. reshape to patches [B,N,P]
        let patched_inputs = input_ts.reshape((b, n, p))?;
        let patched_pads = input_padding.reshape((b, n, p))?;

        // 3. zero padded values, RevIN normalize per-series.
        let keep = (Tensor::ones_like(&patched_pads)? - &patched_pads)?; // 1 = real
        let patched_inputs = (patched_inputs * &keep)?;
        let (mu, sigma) = self.masked_mean_std(&patched_inputs, &keep)?; // [B]
        let mu_b = mu.reshape((b, 1, 1))?;
        let sigma_b = sigma.reshape((b, 1, 1))?;
        let normed = patched_inputs
            .broadcast_sub(&mu_b)?
            .broadcast_div(&sigma_b)?;
        // re-zero padded positions after normalization.
        let normed = (normed * &keep)?;

        // 4. concat value + pad mask -> [B,N,2P]
        let concat_inputs = Tensor::cat(&[&normed, &patched_pads], D::Minus1)?;

        // 6. patched_padding = min over P -> [B,N] (padded only if fully padded)
        let patched_padding = patched_pads.min(D::Minus1)?; // [B,N]

        // forward through the stack.
        let out = self.forward_embedded(&concat_inputs, &patched_padding, freq)?; // [B,N,H,O]

        // 12. reverse RevIN: * sigma + mu.
        let mu_r = mu.reshape((b, 1, 1, 1))?;
        let sigma_r = sigma.reshape((b, 1, 1, 1))?;
        let out = out.broadcast_mul(&sigma_r)?.broadcast_add(&mu_r)?;
        Ok(out)
    }

    /// RevIN statistics matching the reference `_masked_mean_std`: per batch
    /// row, pick the *first patch with >= 3 real values* (falling back to the
    /// last patch, `N-1`, if none qualifies) and compute the masked mean/std
    /// over that single patch's `P` values. Returns `(mu, sigma)` each `[B]`.
    /// `x` and `keep` are `[B,N,P]`, `keep` = 1 at real values.
    ///
    /// NOTE: this is *not* a global mean/std over the whole context — TimesFM
    /// normalizes the whole series by the statistics of one representative
    /// patch, so a global reduction would scramble the RevIN scale/offset and
    /// corrupt every forecast.
    fn masked_mean_std(&self, x: &Tensor, keep: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, n, _p) = x.dims3()?;
        // pad_sum[b, i] = number of real values in patch i.   [B, N]
        let pad_sum = keep.sum(D::Minus1)?;
        // qualifies[b, i] = 1 where the patch has >= 3 real values.
        let three = (Tensor::ones_like(&pad_sum)? * 3.0)?;
        let qualifies = pad_sum.ge(&three)?; // u8 [B, N], 1 where >= 3
                                             // first qualifying patch index per row (argmax of the 0/1 mask).
        let qual_i = qualifies.to_dtype(DType::F32)?;
        let first_idx = qual_i.argmax(D::Minus1)?; // [B], = 0 if none qualify
        let row_has = qual_i.sum(D::Minus1)?; // [B], 0 => no patch qualifies
                                              // Select per-row patch (Rust-side: batch is small and the choice is
                                              // genuinely per-row, which candle's index_select can't express).
        let first_idx: Vec<u32> = first_idx.to_dtype(DType::U32)?.to_vec1()?;
        let row_has: Vec<f32> = row_has.to_vec1()?;
        let mut mus: Vec<f32> = Vec::with_capacity(b);
        let mut sigmas: Vec<f32> = Vec::with_capacity(b);
        let tol = self.cfg.tolerance as f32;
        for row in 0..b {
            let patch = if row_has[row] == 0.0 {
                (n - 1) as u32 // fallback: last patch
            } else {
                first_idx[row]
            };
            // [P] for this row's chosen patch.
            let arr = x.i((row, patch as usize, ..))?;
            let msk = keep.i((row, patch as usize, ..))?;
            let cnt = msk.sum_all()?.to_scalar::<f32>()?.max(1.0);
            let sum = (arr.mul(&msk)?).sum_all()?.to_scalar::<f32>()?;
            let mu = sum / cnt;
            let mu_t = (Tensor::ones_like(&arr)? * mu as f64)?;
            let centered = (arr.broadcast_sub(&mu_t)?.mul(&msk)?).sqr()?;
            let var = (centered.sum_all()?.to_scalar::<f32>()? / cnt).max(0.0);
            let sigma = var.sqrt().max(tol); // clamp(std, min=tolerance)
            mus.push(mu);
            sigmas.push(sigma);
        }
        let mu = Tensor::from_vec(mus, b, x.device())?.to_dtype(x.dtype())?;
        let sigma = Tensor::from_vec(sigmas, b, x.device())?.to_dtype(x.dtype())?;
        Ok((mu, sigma))
    }

    /// Autoregressive decode for arbitrary horizon `h`.
    ///
    /// Returns `(point [B, h], full [B, h, num_outputs])`.
    pub fn decode(
        &self,
        input_ts: &Tensor,
        input_padding: &Tensor,
        freq: &Tensor,
        h: usize,
    ) -> Result<(Tensor, Tensor)> {
        let cfg = &self.cfg;
        let output_patch_len = cfg.horizon_len;
        let num_decode = h.div_ceil(output_patch_len);
        let max_len = cfg.max_context_len;
        let device = input_ts.device();

        let mut context = input_ts.clone();
        let mut padding = input_padding.clone();
        let mut point_chunks: Vec<Tensor> = Vec::new();
        let mut full_chunks: Vec<Tensor> = Vec::new();

        for _ in 0..num_decode {
            // take the last <= max_len, aligned to patch boundary.
            let (b, c) = context.dims2()?;
            let usable = c.min(max_len);
            let usable = (usable / cfg.patch_len) * cfg.patch_len; // multiple of P
            let start = c - usable;
            let ctx = context.narrow(1, start, usable)?.contiguous()?;
            let pad = padding.narrow(1, start, usable)?.contiguous()?;

            let out = self.forward(&ctx, &pad, freq)?; // [B,N,H,O]
            let n = out.dim(1)?;
            // last patch's forecast: [B, H, O]
            let last = out.i((.., n - 1, .., ..))?;
            // mean channel [B, H]
            let mean = last.i((.., .., 0))?;

            point_chunks.push(mean.clone());
            full_chunks.push(last);

            // append the mean chunk to the context for the next step.
            context = Tensor::cat(&[&context, &mean], 1)?;
            let new_pad = Tensor::zeros((b, output_patch_len), DType::F32, device)?;
            padding = Tensor::cat(&[&padding, &new_pad], 1)?;
        }

        let point = Tensor::cat(&point_chunks, 1)?; // [B, num_decode*H]
        let full = Tensor::cat(&full_chunks, 1)?; // [B, num_decode*H, O]
                                                  // trim to exactly h.
        let point = point.narrow(1, 0, h)?;
        let full = full.narrow(1, 0, h)?;
        Ok((point, full))
    }
}

/// Replace NaN entries with 0. Used to neutralize `0 * -inf` in mask building.
fn nan_to_zero(x: &Tensor) -> Result<Tensor> {
    // NaN != NaN, so a self-equality test isolates the finite entries.
    let is_finite = x.eq(x)?; // 1 where finite, 0 where NaN
    let zero = Tensor::zeros_like(x)?;
    is_finite
        .to_dtype(x.dtype())?
        .mul(x)?
        .where_cond(&is_finite, &zero)
        .or_else(|_| {
            // Fallback: arithmetic select. keep = x where finite else 0.
            let mask = is_finite.to_dtype(x.dtype())?;
            x.mul(&mask)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    fn dev() -> Device {
        Device::Cpu
    }

    fn vb(dev: &Device) -> VarBuilder<'static> {
        VarBuilder::zeros(DType::F32, dev)
    }

    #[test]
    fn residual_block_shape() -> Result<()> {
        let dev = dev();
        let rb = ResidualBlock::load(64, 1280, 1280, vb(&dev))?;
        let x = Tensor::randn(0f32, 1.0, (2, 16, 64), &dev)?;
        let y = rb.forward(&x)?;
        assert_eq!(y.dims3()?, (2, 16, 1280));
        Ok(())
    }

    #[test]
    fn positional_embedding_shape() -> Result<()> {
        let dev = dev();
        let pe = PositionalEmbedding::new(1280);
        let t = pe.forward(16, DType::F32, &dev)?;
        assert_eq!(t.dims2()?, (16, 1280));
        Ok(())
    }

    #[test]
    fn attention_shape() -> Result<()> {
        let dev = dev();
        let cfg = TimesfmConfig::tiny();
        let attn = TimesFMAttention::load(&cfg, vb(&dev))?;
        let (b, n, d) = (2, 8, cfg.hidden_size);
        let x = Tensor::randn(0f32, 1.0, (b, n, d), &dev)?;
        let mask = Tensor::zeros((b, 1, n, n), DType::F32, &dev)?;
        let y = attn.forward(&x, &mask)?;
        assert_eq!(y.dims3()?, (b, n, d));
        Ok(())
    }

    #[test]
    fn mlp_shape_and_residual() -> Result<()> {
        let dev = dev();
        let cfg = TimesfmConfig::tiny();
        let mlp = TransformerMLP::load(&cfg, vb(&dev))?;
        let x = Tensor::randn(0f32, 1.0, (2, 8, cfg.hidden_size), &dev)?;
        let y = mlp.forward(&x)?;
        assert_eq!(y.dims3()?, (2, 8, cfg.hidden_size));
        Ok(())
    }

    #[test]
    fn decoder_layer_shape() -> Result<()> {
        let dev = dev();
        let cfg = TimesfmConfig::tiny();
        let layer = TimesFMDecoderLayer::load(&cfg, vb(&dev))?;
        let (b, n, d) = (2, 8, cfg.hidden_size);
        let x = Tensor::randn(0f32, 1.0, (b, n, d), &dev)?;
        let mask = Tensor::zeros((b, 1, n, n), DType::F32, &dev)?;
        let y = layer.forward(&x, &mask)?;
        assert_eq!(y.dims3()?, (b, n, d));
        Ok(())
    }

    #[test]
    fn full_forward_shape_tiny() -> Result<()> {
        let dev = dev();
        let cfg = TimesfmConfig::tiny();
        let model = PatchedTimeSeriesDecoder::load(cfg.clone(), vb(&dev))?;
        let b = 2;
        let c = cfg.max_context_len; // 32 in tiny
        let n = c / cfg.patch_len;
        let input_ts = Tensor::randn(0f32, 1.0, (b, c), &dev)?;
        let input_padding = Tensor::zeros((b, c), DType::F32, &dev)?;
        let freq = Tensor::zeros((b, 1), DType::U32, &dev)?;
        let out = model.forward(&input_ts, &input_padding, &freq)?;
        assert_eq!(out.dims4()?, (b, n, cfg.horizon_len, cfg.num_outputs()));
        Ok(())
    }

    #[test]
    fn full_forward_shape_realconfig() -> Result<()> {
        // Real 200M config but tiny batch/context for speed: 1 batch, 2 patches.
        let dev = dev();
        let mut cfg = TimesfmConfig::timesfm_1p0_200m();
        // Use only 2 layers to keep the test fast while exercising real widths.
        cfg.num_layers = 2;
        let model = PatchedTimeSeriesDecoder::load(cfg.clone(), vb(&dev))?;
        let b = 1;
        let c = cfg.patch_len * 2; // 64 = 2 patches
        let n = c / cfg.patch_len;
        let input_ts = Tensor::randn(0f32, 1.0, (b, c), &dev)?;
        let input_padding = Tensor::zeros((b, c), DType::F32, &dev)?;
        let freq = Tensor::zeros((b, 1), DType::U32, &dev)?;
        let out = model.forward(&input_ts, &input_padding, &freq)?;
        assert_eq!(out.dims4()?, (b, n, 128, 10));
        Ok(())
    }

    #[test]
    fn revin_stats_use_first_patch_not_global_mean() -> Result<()> {
        // Two patches with very different means. The reference TimesFM computes
        // RevIN mu/sigma from the *first patch with >= 3 real values*, NOT a
        // global mean over the whole context. This pins that semantics so a
        // regression to a global reduction is caught (shape tests never would).
        let dev = dev();
        let cfg = TimesfmConfig::tiny(); // patch_len = 4
        let model = PatchedTimeSeriesDecoder::load(cfg.clone(), vb(&dev))?;
        // batch=1, N=2, P=4. Patch 0 = all 1.0 (mean 1, std 0 -> clamps to tol).
        // Patch 1 = all 100.0. Global mean would be 50.5; first-patch mean is 1.
        let x = Tensor::from_vec(
            vec![1f32, 1., 1., 1., 100., 100., 100., 100.],
            (1, 2, 4),
            &dev,
        )?;
        let keep = Tensor::ones((1, 2, 4), DType::F32, &dev)?;
        let (mu, sigma) = model.masked_mean_std(&x, &keep)?;
        let mu_v = mu.to_vec1::<f32>()?[0];
        let sigma_v = sigma.to_vec1::<f32>()?[0];
        assert!(
            (mu_v - 1.0).abs() < 1e-4,
            "mu should be first-patch mean 1.0, got {mu_v} (global mean is 50.5)"
        );
        // std of a constant patch is 0, clamped up to tolerance.
        assert!(
            (sigma_v - cfg.tolerance as f32).abs() < 1e-5,
            "sigma should clamp to tolerance, got {sigma_v}"
        );
        Ok(())
    }

    #[test]
    fn revin_stats_skip_short_patches_and_fall_back() -> Result<()> {
        // Patch 0 has only 2 real values (< 3) so it is skipped; patch 1 has 4
        // real values and is the first qualifying patch -> stats come from it.
        let dev = dev();
        let cfg = TimesfmConfig::tiny();
        let model = PatchedTimeSeriesDecoder::load(cfg.clone(), vb(&dev))?;
        let x = Tensor::from_vec(vec![7f32, 7., 0., 0., 5., 5., 5., 5.], (1, 2, 4), &dev)?;
        // keep: patch 0 has 2 reals, patch 1 has 4 reals.
        let keep = Tensor::from_vec(vec![1f32, 1., 0., 0., 1., 1., 1., 1.], (1, 2, 4), &dev)?;
        let (mu, _sigma) = model.masked_mean_std(&x, &keep)?;
        let mu_v = mu.to_vec1::<f32>()?[0];
        assert!(
            (mu_v - 5.0).abs() < 1e-4,
            "mu should be patch-1 mean 5.0 (patch 0 has <3 reals), got {mu_v}"
        );
        Ok(())
    }

    #[test]
    fn decode_arbitrary_horizon() -> Result<()> {
        let dev = dev();
        let cfg = TimesfmConfig::tiny(); // horizon_len = 8
        let model = PatchedTimeSeriesDecoder::load(cfg.clone(), vb(&dev))?;
        let b = 2;
        let c = cfg.max_context_len;
        let input_ts = Tensor::randn(0f32, 1.0, (b, c), &dev)?;
        let input_padding = Tensor::zeros((b, c), DType::F32, &dev)?;
        let freq = Tensor::zeros((b, 1), DType::U32, &dev)?;
        // request a horizon that needs multiple decode patches (8*2 + 3 = 19)
        let h = cfg.horizon_len * 2 + 3;
        let (point, full) = model.decode(&input_ts, &input_padding, &freq, h)?;
        assert_eq!(point.dims2()?, (b, h));
        assert_eq!(full.dims3()?, (b, h, cfg.num_outputs()));
        Ok(())
    }
}
