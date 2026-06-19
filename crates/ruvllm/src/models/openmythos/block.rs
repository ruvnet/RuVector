//! Standard pre-norm transformer block (used in prelude, coda, and as the
//! shared recurrent block body).

use candle_core::Tensor;
use candle_nn::{Module, RmsNorm, VarBuilder};

use super::attention::{Attention, KvLayerCache};
use super::config::MythosConfig;
use super::ffn::Ffn;
use super::rope::cand;
use crate::error::Result;

/// Pre-norm block: `x += Attn(norm(x)); x += FFN(norm(x))`.
pub struct TransformerBlock {
    attn_norm: RmsNorm,
    attn: Attention,
    ffn_norm: RmsNorm,
    ffn: Ffn,
}

impl TransformerBlock {
    pub fn load(vb: VarBuilder, cfg: &MythosConfig, use_moe: bool) -> Result<Self> {
        Ok(Self {
            attn_norm: candle_nn::rms_norm(cfg.dim, cfg.rms_norm_eps, vb.pp("attn_norm"))
                .map_err(cand)?,
            attn: Attention::load(vb.pp("attn"), cfg)?,
            ffn_norm: candle_nn::rms_norm(cfg.dim, cfg.rms_norm_eps, vb.pp("ffn_norm"))
                .map_err(cand)?,
            ffn: Ffn::load(vb, cfg, use_moe)?,
        })
    }

    /// Forward over `xs` `[b, seq, dim]`. Returns the block output and the
    /// updated (full) KV cache for the attention layer.
    pub fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
        past: Option<&KvLayerCache>,
    ) -> Result<(Tensor, KvLayerCache)> {
        let normed = self.attn_norm.forward(xs).map_err(cand)?;
        let (attn_out, kv) = self.attn.forward(&normed, cos, sin, mask, past)?;
        let xs = (xs + attn_out).map_err(cand)?;
        let normed = self.ffn_norm.forward(&xs).map_err(cand)?;
        let out = (&xs + self.ffn.forward(&normed)?).map_err(cand)?;
        Ok((out, kv))
    }
}
