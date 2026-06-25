//! Feed-forward networks: dense SwiGLU expert and fine-grained MoE.

use candle_core::Tensor;
use candle_nn::{ops, Linear, Module, VarBuilder};

use super::config::MythosConfig;
use super::rope::cand;
use crate::error::Result;

/// A single SwiGLU expert: `down(silu(gate(x)) * up(x))`.
pub struct Expert {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Expert {
    pub fn load(vb: VarBuilder, dim: usize, inter: usize) -> Result<Self> {
        Ok(Self {
            gate: candle_nn::linear_no_bias(dim, inter, vb.pp("gate")).map_err(cand)?,
            up: candle_nn::linear_no_bias(dim, inter, vb.pp("up")).map_err(cand)?,
            down: candle_nn::linear_no_bias(inter, dim, vb.pp("down")).map_err(cand)?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let g = ops::silu(&self.gate.forward(xs).map_err(cand)?).map_err(cand)?;
        let u = self.up.forward(xs).map_err(cand)?;
        self.down.forward(&(g * u).map_err(cand)?).map_err(cand)
    }
}

/// Either a dense SwiGLU FFN (prelude/coda) or fine-grained MoE (recurrent).
pub enum Ffn {
    Dense(Expert),
    Moe(MoeFfn),
}

impl Ffn {
    pub fn load(vb: VarBuilder, cfg: &MythosConfig, use_moe: bool) -> Result<Self> {
        Ok(if use_moe {
            Ffn::Moe(MoeFfn::load(vb.pp("moe"), cfg)?)
        } else {
            let inter = cfg.expert_dim * cfg.n_shared_experts.max(2);
            Ffn::Dense(Expert::load(vb.pp("ffn"), cfg.dim, inter)?)
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Ffn::Dense(e) => e.forward(xs),
            Ffn::Moe(m) => m.forward(xs),
        }
    }
}

/// Fine-grained Mixture-of-Experts with routed + always-on shared experts.
///
/// Routing computes a softmax over experts and keeps the top-`k` per token
/// (kept weights renormalized). Shared experts always contribute. Each routed
/// expert runs only on the tokens routed to it (sparse dispatch via
/// `index_select` gather + `index_add` scatter), so FFN compute scales with
/// `top_k`, not `n_experts`.
pub struct MoeFfn {
    router: Linear,
    routed: Vec<Expert>,
    shared: Vec<Expert>,
    top_k: usize,
}

impl MoeFfn {
    pub fn load(vb: VarBuilder, cfg: &MythosConfig) -> Result<Self> {
        let router =
            candle_nn::linear_no_bias(cfg.dim, cfg.n_experts, vb.pp("router")).map_err(cand)?;
        let rvb = vb.pp("experts");
        let mut routed = Vec::with_capacity(cfg.n_experts);
        for i in 0..cfg.n_experts {
            routed.push(Expert::load(rvb.pp(i), cfg.dim, cfg.expert_dim)?);
        }
        let svb = vb.pp("shared_experts");
        let mut shared = Vec::with_capacity(cfg.n_shared_experts);
        for i in 0..cfg.n_shared_experts {
            shared.push(Expert::load(svb.pp(i), cfg.dim, cfg.expert_dim)?);
        }
        Ok(Self {
            router,
            routed,
            shared,
            top_k: cfg.n_experts_per_tok,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, seq, dim) = xs.dims3().map_err(cand)?;
        let n_tok = b * seq;
        let device = xs.device().clone();
        let dtype = xs.dtype();
        let flat = xs.reshape((n_tok, dim)).map_err(cand)?;

        let logits = self.router.forward(&flat).map_err(cand)?;
        // Cast to F32 before to_vec2: softmax output may be BF16/F16 when the
        // model runs in reduced precision, but the routing CPU scatter requires f32.
        let probs = ops::softmax_last_dim(&logits)
            .map_err(cand)?
            .to_dtype(candle_core::DType::F32)
            .map_err(cand)?;
        let rows: Vec<Vec<f32>> = probs.to_vec2().map_err(cand)?;

        // Build per-expert token lists and renormalized top-k weights.
        let n_experts = self.routed.len();
        let mut tok_ids: Vec<Vec<u32>> = vec![Vec::new(); n_experts];
        let mut tok_w: Vec<Vec<f32>> = vec![Vec::new(); n_experts];
        let mut order: Vec<usize> = (0..n_experts).collect();
        for (t, row) in rows.iter().enumerate() {
            order.sort_by(|&a, &c| row[c].partial_cmp(&row[a]).unwrap());
            let keep = &order[..self.top_k.min(n_experts)];
            let denom: f32 = keep.iter().map(|&e| row[e]).sum::<f32>().max(1e-9);
            for &e in keep {
                tok_ids[e].push(t as u32);
                tok_w[e].push(row[e] / denom);
            }
        }

        // Sparse dispatch: each expert processes only its routed tokens, so FFN
        // compute scales with top_k rather than n_experts.
        let mut out = flat.zeros_like().map_err(cand)?;
        for (e, expert) in self.routed.iter().enumerate() {
            let n_e = tok_ids[e].len();
            if n_e == 0 {
                continue;
            }
            // from_slice avoids the .clone() heap allocation per expert.
            let idx = Tensor::from_slice(&tok_ids[e], (n_e,), &device).map_err(cand)?;
            let gathered = flat.index_select(&idx, 0).map_err(cand)?; // [n_e, dim]
            let y = expert.forward(&gathered)?;
            let w = Tensor::from_slice(&tok_w[e], (n_e, 1), &device)
                .map_err(cand)?
                .to_dtype(dtype)
                .map_err(cand)?;
            let y = y.broadcast_mul(&w).map_err(cand)?;
            out = out.index_add(&idx, &y, 0).map_err(cand)?;
        }

        // Shared experts always contribute (dense over all tokens).
        for expert in &self.shared {
            out = (out + expert.forward(&flat)?).map_err(cand)?;
        }
        out.reshape((b, seq, dim)).map_err(cand)
    }
}
