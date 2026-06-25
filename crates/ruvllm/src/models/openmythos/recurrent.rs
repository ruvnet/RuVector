//! The recurrent block: the deep loop with loop-index embedding, LTI-stable
//! state injection, per-depth LoRA, and Adaptive Computation Time halting.

use candle_core::{DType, Device, Tensor};
use candle_nn::{ops, Linear, Module, RmsNorm, VarBuilder};

use super::attention::KvLayerCache;
use super::block::TransformerBlock;
use super::config::MythosConfig;
use super::rope::cand;
use crate::error::Result;
use crate::models::rdt::DepthTelemetry;

/// LTI-constrained state update guaranteeing a contractive (spectral radius < 1)
/// recurrence: `h_{t+1} = A·h_t + B·e + transformer_out`, with diagonal
/// `A = exp(-exp(log_dt + log_A)) ∈ (0, 1)`.
pub struct LtiInjection {
    /// Cached diagonal `A = exp(-exp(clamp(log_dt+log_A, -20, 20)))` `[dim]`.
    /// Computed once at load time; the weights are frozen after training.
    a_diag_cached: Tensor,
    b_gain: Tensor,
}

impl LtiInjection {
    pub fn load(vb: VarBuilder, dim: usize) -> Result<Self> {
        let log_a = vb.get(dim, "log_a").map_err(cand)?;
        let log_dt = vb.get(dim, "log_dt").map_err(cand)?;
        let b_gain = vb.get(dim, "b_gain").map_err(cand)?;
        // Precompute the contractive diagonal — constant for fixed weights.
        let a_diag_cached = (&log_dt + &log_a)
            .map_err(cand)?
            .clamp(-20.0, 20.0)
            .map_err(cand)?
            .exp()
            .map_err(cand)?
            .neg()
            .map_err(cand)?
            .exp()
            .map_err(cand)?;
        Ok(Self {
            a_diag_cached,
            b_gain,
        })
    }

    /// Cached diagonal `A ∈ (0,1)^dim`.
    pub fn a_diag(&self) -> Result<Tensor> {
        Ok(self.a_diag_cached.clone())
    }

    pub fn forward(&self, h: &Tensor, e: &Tensor, trans_out: &Tensor) -> Result<Tensor> {
        let a_h = h.broadcast_mul(&self.a_diag_cached).map_err(cand)?;
        let b_e = e.broadcast_mul(&self.b_gain).map_err(cand)?;
        ((a_h + b_e).map_err(cand)? + trans_out).map_err(cand)
    }
}

/// Per-depth LoRA: `delta = down(x) @ effective_w[t]` where
/// `effective_w[t] = diag(scale[t]) @ B` is precomputed at load time.
///
/// Original: `delta = (down(x) ⊙ scale[t]) @ B`.
/// Equivalent: `down(x) @ (scale[t].unsqueeze(1) * B)` = `down(x) @ effective_w[t]`.
/// Precomputing saves 3 kernel ops (narrow, reshape, broadcast_mul on scale)
/// per ACT loop iteration.
pub struct DepthLora {
    down: Linear,
    /// `[max_loop_iters]` × `[rank, dim]` — one fused weight per depth index.
    effective_w: Vec<Tensor>,
    rank: usize,
}

impl DepthLora {
    pub fn load(vb: VarBuilder, cfg: &MythosConfig) -> Result<Self> {
        let down =
            candle_nn::linear_no_bias(cfg.dim, cfg.lora_rank, vb.pp("down")).map_err(cand)?;
        let b_mat = vb.get((cfg.lora_rank, cfg.dim), "b_mat").map_err(cand)?;
        let scale = vb
            .get((cfg.max_loop_iters, cfg.lora_rank), "scale")
            .map_err(cand)?;
        // effective_w[t] = scale[t, :, None] * b_mat  →  [rank, dim]
        let effective_w = (0..cfg.max_loop_iters)
            .map(|t| {
                let scale_t = scale
                    .narrow(0, t, 1)
                    .map_err(cand)?
                    .reshape((cfg.lora_rank, 1))
                    .map_err(cand)?; // [rank, 1]
                scale_t.broadcast_mul(&b_mat).map_err(cand) // [rank, dim]
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            down,
            effective_w,
            rank: cfg.lora_rank,
        })
    }

    pub fn forward(&self, x: &Tensor, t: usize) -> Result<Tensor> {
        let (b, seq, _dim) = x.dims3().map_err(cand)?;
        let w = &self.effective_w[t.min(self.effective_w.len() - 1)];
        let dim = w.dim(1).map_err(cand)?;
        let d = self.down.forward(x).map_err(cand)?; // [b, seq, rank]
        d.reshape((b * seq, self.rank))
            .map_err(cand)?
            .matmul(w)
            .map_err(cand)?
            .reshape((b, seq, dim))
            .map_err(cand)
    }
}

/// Sinusoidal loop-index embedding `[1, 1, dim]` (first `loop_dim` channels).
fn compute_loop_embedding(
    t: usize,
    dim: usize,
    loop_dim: usize,
    rope_theta: f32,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let half = loop_dim / 2;
    let mut data = vec![0f32; dim];
    for i in 0..half {
        let freq = 1.0f32 / rope_theta.powf(2.0 * i as f32 / loop_dim as f32);
        let angle = t as f32 * freq;
        data[i] = angle.sin();
        data[half + i] = angle.cos();
    }
    Tensor::from_vec(data, (1, 1, dim), device)
        .map_err(cand)?
        .to_dtype(dtype)
        .map_err(cand)
}

/// The recurrent block executed repeatedly by the model.
pub struct RecurrentBlock {
    inject_norm: RmsNorm,
    block: TransformerBlock,
    lti: LtiInjection,
    lora: DepthLora,
    act_head: Linear,
    loop_dim: usize,
    dim: usize,
    act_threshold: f32,
    rope_theta: f32,
    /// Precomputed loop-index embeddings `[1, 1, dim]` for `t in 0..max_loops`.
    /// These are constant across forward passes, so they are built once.
    loop_embeds: Vec<Tensor>,
}

/// Output of one recurrent forward: ACT-weighted state plus the updated KV cache
/// (final-iteration keys/values for the processed positions).
pub struct RecurrentOut {
    pub hidden: Tensor,
    pub kv: KvLayerCache,
}

impl RecurrentBlock {
    pub fn load(vb: VarBuilder, cfg: &MythosConfig) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        let loop_embeds = (0..cfg.max_loop_iters)
            .map(|t| {
                compute_loop_embedding(t, cfg.dim, cfg.loop_dim, cfg.rope_theta, &device, dtype)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            inject_norm: candle_nn::rms_norm(cfg.dim, cfg.rms_norm_eps, vb.pp("inject_norm"))
                .map_err(cand)?,
            block: TransformerBlock::load(vb.pp("block"), cfg, cfg.use_moe)?,
            lti: LtiInjection::load(vb.pp("lti"), cfg.dim)?,
            lora: DepthLora::load(vb.pp("lora"), cfg)?,
            act_head: candle_nn::linear(cfg.dim, 1, vb.pp("act_head")).map_err(cand)?,
            loop_dim: cfg.loop_dim,
            dim: cfg.dim,
            act_threshold: cfg.act_threshold,
            rope_theta: cfg.rope_theta,
            loop_embeds,
        })
    }

    pub(crate) fn lti(&self) -> &LtiInjection {
        &self.lti
    }

    /// Sinusoidal loop-index embedding for iteration `t`. Returns the precomputed
    /// tensor for `t < max_loops`, else computes it on demand (depth extrapolation).
    fn loop_embedding(&self, t: usize, device: &Device, dtype: DType) -> Result<Tensor> {
        if let Some(emb) = self.loop_embeds.get(t) {
            return Ok(emb.clone());
        }
        compute_loop_embedding(t, self.dim, self.loop_dim, self.rope_theta, device, dtype)
    }

    /// Run the recurrent loop. `past` is the recurrent KV cache for prior
    /// positions; `offset` is its length. `cos`/`sin`/`mask_fn` cover the current
    /// query positions over `offset + seq` keys. Returns the ACT-weighted state
    /// and the updated KV cache, recording per-token depth in `telemetry`.
    ///
    /// ACT state (cumulative probability, halted mask) is maintained as GPU
    /// tensors throughout, eliminating per-iteration GPU→CPU transfers. A
    /// sync only occurs every 4 iterations for the early-exit check.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        h0: &Tensor,
        e: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
        past: Option<&KvLayerCache>,
        n_loops: usize,
        telemetry: &DepthTelemetry,
    ) -> Result<RecurrentOut> {
        let (b, seq, _dim) = h0.dims3().map_err(cand)?;
        let device = h0.device().clone();
        let dtype = h0.dtype();
        let n = b * seq;

        let mut h = h0.clone();
        let mut h_out = h0.zeros_like().map_err(cand)?;

        // ACT state as f32 tensors on device — replaces the per-iteration CPU vecs.
        // `cum_f32`:        running probability mass, [b, seq, 1]
        // `not_halted_f32`: 1.0 for tokens still computing, 0.0 for halted, [b, seq, 1]
        // `depth_f32`:      iteration index when each token halted (0 = not yet), [b, seq, 1]
        let mut cum_f32 = Tensor::zeros((b, seq, 1), DType::F32, &device).map_err(cand)?;
        let mut not_halted_f32 = Tensor::ones((b, seq, 1), DType::F32, &device).map_err(cand)?;
        let mut depth_f32 = Tensor::zeros((b, seq, 1), DType::F32, &device).map_err(cand)?;
        // `ones_f32` removed — replaced by affine(-1, 1) = (1 - x) without a constant tensor.

        // Precompute step tensors (t+1) for depth tracking — avoids Tensor::new per iteration.
        let step_tensors: Vec<Tensor> = (0..n_loops)
            .map(|t| {
                Tensor::new((t + 1) as f32, &device)
                    .map_err(cand)?
                    .broadcast_as((b, seq, 1))
                    .map_err(cand)?
                    .to_dtype(DType::F32)
                    .map_err(cand)
            })
            .collect::<Result<Vec<_>>>()?;

        // KV cache: the final iteration's KV wins (see original design note).
        let mut last_kv: Option<KvLayerCache> = None;
        let mut final_t = 0usize;

        for t in 0..n_loops {
            let loop_emb = self.loop_embedding(t, &device, dtype)?;
            let h_loop = h.broadcast_add(&loop_emb).map_err(cand)?;
            let injected = (h_loop + e).map_err(cand)?;
            let normed = self.inject_norm.forward(&injected).map_err(cand)?;
            let (trans_out, kv) = self.block.forward(&normed, cos, sin, mask, past)?;
            last_kv = Some(kv);
            let trans_out = (trans_out + self.lora.forward(&normed, t)?).map_err(cand)?;

            // Stable state update.
            h = self.lti.forward(&h, e, &trans_out)?;

            // ACT halting — vectorized tensor ops, no per-iteration weight-vector transfer.
            let p_raw = ops::sigmoid(&self.act_head.forward(&h).map_err(cand)?).map_err(cand)?;
            let p_f32 = p_raw.to_dtype(DType::F32).map_err(cand)?;

            // Effective probability for still-running tokens only.
            let p_eff = (&p_f32 * &not_halted_f32).map_err(cand)?;
            // Candidate cumulative after this step (used for weight calc before state update).
            let new_cum = (&cum_f32 + &p_eff).map_err(cand)?;

            // Tokens that newly halt this iteration (cross threshold, not yet halted).
            let exceeds = new_cum
                .ge(self.act_threshold as f64)
                .map_err(cand)?
                .to_dtype(DType::F32)
                .map_err(cand)?;
            let will_halt = (&exceeds * &not_halted_f32).map_err(cand)?;

            // Weight = remainder weight (newly halting) + continuation weight (still running).
            //   newly halting: w = 1 − cumulative_before_this_step
            //   still running: w = p
            //   already halted: w = 0 (not_halted_f32 = 0 for them)
            // affine(-1, 1) = 1 - cum_f32, avoids a constant ones tensor.
            let remainder = cum_f32.affine(-1.0, 1.0).map_err(cand)?;
            let w_halt = (&will_halt * &remainder).map_err(cand)?;
            let still_running = (&not_halted_f32 - &will_halt).map_err(cand)?;
            let w_run = (&still_running * &p_eff).map_err(cand)?;
            let w_f32 = (&w_halt + &w_run).map_err(cand)?;

            // Accumulate weighted output (cast to model dtype once).
            let w = if dtype != DType::F32 {
                w_f32.to_dtype(dtype).map_err(cand)?
            } else {
                w_f32.clone()
            };
            h_out = (&h_out + &h.broadcast_mul(&w).map_err(cand)?).map_err(cand)?;

            // Update ACT state.
            // Cumulative grows only for still-running tokens (frozen on halt).
            cum_f32 = (&cum_f32 + &(&still_running * &p_eff).map_err(cand)?).map_err(cand)?;
            not_halted_f32 = (&not_halted_f32 - &will_halt).map_err(cand)?;

            // Record per-token halt iteration using the precomputed step tensor.
            let step_f32 = &step_tensors[t];
            depth_f32 = (&depth_f32
                + &(&will_halt * &(step_f32 - &depth_f32).map_err(cand)?).map_err(cand)?)
                .map_err(cand)?;

            final_t = t + 1;

            tracing::trace!(loop = t, "openmythos recurrent step");

            // Early-exit: cheap scalar sync (not the hot-path weight-vector transfer).
            let remaining = not_halted_f32
                .sum_all()
                .map_err(cand)?
                .to_scalar::<f32>()
                .map_err(cand)?;
            if remaining < 0.5 {
                break;
            }
        }

        // Tail: tokens still running at max_loops get their probability remainder.
        let remaining_final = not_halted_f32
            .sum_all()
            .map_err(cand)?
            .to_scalar::<f32>()
            .map_err(cand)?;
        if remaining_final > 0.5 {
            let tail_f32 =
                (cum_f32.affine(-1.0, 1.0).map_err(cand)? * &not_halted_f32).map_err(cand)?;
            let w = if dtype != DType::F32 {
                tail_f32.to_dtype(dtype).map_err(cand)?
            } else {
                tail_f32
            };
            h_out = (&h_out + &h.broadcast_mul(&w).map_err(cand)?).map_err(cand)?;
            // Still-running tokens: depth = final_t (ran to ceiling).
            let final_step = Tensor::new(final_t as f64, &device)
                .map_err(cand)?
                .broadcast_as((b, seq, 1))
                .map_err(cand)?
                .to_dtype(DType::F32)
                .map_err(cand)?;
            depth_f32 = (&depth_f32
                + &(&not_halted_f32 * &(&final_step - &depth_f32).map_err(cand)?).map_err(cand)?)
                .map_err(cand)?;
        }

        // ONE GPU→CPU transfer for telemetry (not in the hot path).
        let depth_vec: Vec<f32> = depth_f32
            .reshape((n,))
            .map_err(cand)?
            .to_vec1()
            .map_err(cand)?;
        let depth: Vec<usize> = depth_vec.into_iter().map(|d| d as usize).collect();
        telemetry.record(&depth);

        let kv = last_kv.expect("at least one loop iteration runs");
        Ok(RecurrentOut { hidden: h_out, kv })
    }
}
