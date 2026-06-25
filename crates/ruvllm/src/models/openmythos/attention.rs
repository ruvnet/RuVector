//! Attention variants: Grouped-Query Attention and Multi-Latent Attention,
//! both with incremental KV caching for one-token-at-a-time decode.

use candle_core::{Tensor, D};
use candle_nn::{ops, Linear, Module, RmsNorm, VarBuilder};

use super::config::{AttnType, MythosConfig};
use super::rope::{apply_rope, cand, repeat_kv};
use crate::error::Result;

/// Per-layer KV cache holding all *past* key/value state.
#[derive(Clone)]
pub enum KvLayerCache {
    /// GQA: rotated keys and values `[b, kv_heads, len, head_dim]`.
    /// Grows via `Tensor::cat` on each decode step (legacy path).
    Gqa { k: Tensor, v: Tensor },
    /// GQA with pre-allocated `[b, kv_heads, max_seq, head_dim]` buffers.
    /// Uses `scatter_set` for O(1) per-step appends instead of O(N) cat copies.
    GqaPrealloc {
        k: Tensor, // [b, kv_heads, max_seq, head_dim]
        v: Tensor,
        seq_len: usize,
        max_seq: usize,
    },
    /// MLA: compressed latent `[b, len, kv_lora_rank]` and rotated shared
    /// rope keys `[b, len, qk_rope_head_dim]`. Grows via cat (legacy path).
    Mla { c_kv: Tensor, k_rope: Tensor },
    /// MLA pre-allocated: same semantics as GqaPrealloc but for the two MLA
    /// tensors. Shape: `[b, max_seq, kv_lora_rank]` and `[b, max_seq, qk_rope]`.
    MlaPrealloc {
        c_kv: Tensor,
        k_rope: Tensor,
        seq_len: usize,
        max_seq: usize,
    },
}

impl KvLayerCache {
    /// Number of cached positions.
    pub fn len(&self) -> usize {
        match self {
            KvLayerCache::Gqa { k, .. } => k.dim(2).unwrap_or(0),
            KvLayerCache::GqaPrealloc { seq_len, .. } => *seq_len,
            KvLayerCache::Mla { c_kv, .. } => c_kv.dim(1).unwrap_or(0),
            KvLayerCache::MlaPrealloc { seq_len, .. } => *seq_len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Either attention variant.
pub enum Attention {
    Gqa(GqaAttention),
    Mla(MlaAttention),
}

impl Attention {
    pub fn load(vb: VarBuilder, cfg: &MythosConfig) -> Result<Self> {
        Ok(match cfg.attn_type {
            AttnType::Gqa => Attention::Gqa(GqaAttention::load(vb, cfg)?),
            AttnType::Mla => Attention::Mla(MlaAttention::load(vb, cfg)?),
        })
    }

    /// Run attention over `xs` `[b, seq, dim]`.
    ///
    /// `past` is the read-only KV state for positions before this call; `offset`
    /// is its length. `cos`/`sin` are RoPE tables for the current query
    /// positions, and `mask` is `[seq, offset + seq]`. Returns the attention
    /// output and the *full* (past + current) KV cache, which the caller stores.
    pub fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
        past: Option<&KvLayerCache>,
    ) -> Result<(Tensor, KvLayerCache)> {
        match self {
            Attention::Gqa(a) => a.forward(xs, cos, sin, mask, past),
            Attention::Mla(a) => a.forward(xs, cos, sin, mask, past),
        }
    }
}

// ---------------------------------------------------------------------------
// Grouped-Query Attention
// ---------------------------------------------------------------------------

pub struct GqaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl GqaAttention {
    pub fn load(vb: VarBuilder, cfg: &MythosConfig) -> Result<Self> {
        let h = cfg.dim;
        let hd = cfg.head_dim();
        let q_out = cfg.n_heads * hd;
        let kv_out = cfg.n_kv_heads * hd;
        Ok(Self {
            q_proj: candle_nn::linear_no_bias(h, q_out, vb.pp("q_proj")).map_err(cand)?,
            k_proj: candle_nn::linear_no_bias(h, kv_out, vb.pp("k_proj")).map_err(cand)?,
            v_proj: candle_nn::linear_no_bias(h, kv_out, vb.pp("v_proj")).map_err(cand)?,
            o_proj: candle_nn::linear_no_bias(q_out, h, vb.pp("o_proj")).map_err(cand)?,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: hd,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
        past: Option<&KvLayerCache>,
    ) -> Result<(Tensor, KvLayerCache)> {
        let (b, seq, _h) = xs.dims3().map_err(cand)?;
        let q = heads(
            &self.q_proj.forward(xs).map_err(cand)?,
            b,
            seq,
            self.n_heads,
            self.head_dim,
        )?;
        let k = heads(
            &self.k_proj.forward(xs).map_err(cand)?,
            b,
            seq,
            self.n_kv_heads,
            self.head_dim,
        )?;
        let v = heads(
            &self.v_proj.forward(xs).map_err(cand)?,
            b,
            seq,
            self.n_kv_heads,
            self.head_dim,
        )?;

        let q = apply_rope(&q, cos, sin)?;
        let k_cur = apply_rope(&k, cos, sin)?;

        // Accumulate KV: two paths depending on cache variant.
        let (k_full, v_full, new_cache) = match past {
            // Pre-allocated: scatter_set is O(new_data) not O(total); no new tensor.
            Some(KvLayerCache::GqaPrealloc {
                k: buf_k,
                v: buf_v,
                seq_len,
                max_seq,
            }) => {
                let idx =
                    Tensor::full(*seq_len as u32, k_cur.shape(), k_cur.device()).map_err(cand)?;
                buf_k.scatter_set(&idx, &k_cur, 2).map_err(cand)?;
                buf_v.scatter_set(&idx, &v, 2).map_err(cand)?;
                let new_seq = seq_len + seq;
                let k_view = buf_k.narrow(2, 0, new_seq).map_err(cand)?;
                let v_view = buf_v.narrow(2, 0, new_seq).map_err(cand)?;
                let cache = KvLayerCache::GqaPrealloc {
                    k: buf_k.clone(),
                    v: buf_v.clone(),
                    seq_len: new_seq,
                    max_seq: *max_seq,
                };
                (k_view, v_view, cache)
            }
            // Legacy cat path (first call or non-preallocated cache).
            Some(KvLayerCache::Gqa { k: pk, v: pv }) => {
                let k_f = Tensor::cat(&[pk, &k_cur], 2).map_err(cand)?;
                let v_f = Tensor::cat(&[pv, &v], 2).map_err(cand)?;
                let cache = KvLayerCache::Gqa {
                    k: k_f.clone(),
                    v: v_f.clone(),
                };
                (k_f, v_f, cache)
            }
            _ => {
                let cache = KvLayerCache::Gqa {
                    k: k_cur.clone(),
                    v: v.clone(),
                };
                (k_cur, v, cache)
            }
        };

        let n_rep = self.n_heads / self.n_kv_heads;
        let k_rep = repeat_kv(&k_full, n_rep)?;
        let v_rep = repeat_kv(&v_full, n_rep)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q
            .matmul(&k_rep.transpose(2, 3).map_err(cand)?)
            .map_err(cand)?
            * scale)
            .map_err(cand)?;
        let scores = scores.broadcast_add(mask).map_err(cand)?;
        let probs = ops::softmax_last_dim(&scores).map_err(cand)?;
        let ctx = probs.matmul(&v_rep).map_err(cand)?;
        let ctx = ctx
            .transpose(1, 2)
            .map_err(cand)?
            .contiguous()
            .map_err(cand)?
            .reshape((b, seq, self.n_heads * self.head_dim))
            .map_err(cand)?;
        let out = self.o_proj.forward(&ctx).map_err(cand)?;
        Ok((out, new_cache))
    }
}

fn heads(x: &Tensor, b: usize, seq: usize, n: usize, hd: usize) -> Result<Tensor> {
    x.reshape((b, seq, n, hd))
        .map_err(cand)?
        .transpose(1, 2)
        .map_err(cand)?
        .contiguous()
        .map_err(cand)
}

// ---------------------------------------------------------------------------
// Multi-Latent Attention (DeepSeek-V2 style)
// ---------------------------------------------------------------------------

pub struct MlaAttention {
    // Query path (optionally low-rank compressed).
    q_a_proj: Option<Linear>,
    q_a_norm: Option<RmsNorm>,
    q_proj: Linear, // q_b_proj when compressed, else direct dim->n_heads*qk_head_dim
    // KV path (compressed).
    kv_a_proj: Linear, // dim -> kv_lora_rank + qk_rope_head_dim
    kv_a_norm: RmsNorm,
    kv_b_proj: Linear, // kv_lora_rank -> n_heads*(qk_nope_head_dim + v_head_dim)
    o_proj: Linear,
    n_heads: usize,
    kv_lora_rank: usize,
    qk_nope: usize,
    qk_rope: usize,
    v_head_dim: usize,
}

impl MlaAttention {
    pub fn load(vb: VarBuilder, cfg: &MythosConfig) -> Result<Self> {
        let h = cfg.dim;
        let qk_head_dim = cfg.mla_qk_head_dim();
        let q_total = cfg.n_heads * qk_head_dim;

        let (q_a_proj, q_a_norm, q_proj) = if cfg.q_lora_rank > 0 {
            let a =
                candle_nn::linear_no_bias(h, cfg.q_lora_rank, vb.pp("q_a_proj")).map_err(cand)?;
            let n = candle_nn::rms_norm(cfg.q_lora_rank, cfg.rms_norm_eps, vb.pp("q_a_norm"))
                .map_err(cand)?;
            let b = candle_nn::linear_no_bias(cfg.q_lora_rank, q_total, vb.pp("q_b_proj"))
                .map_err(cand)?;
            (Some(a), Some(n), b)
        } else {
            let p = candle_nn::linear_no_bias(h, q_total, vb.pp("q_proj")).map_err(cand)?;
            (None, None, p)
        };

        let kv_a_proj = candle_nn::linear_no_bias(
            h,
            cfg.kv_lora_rank + cfg.qk_rope_head_dim,
            vb.pp("kv_a_proj"),
        )
        .map_err(cand)?;
        let kv_a_norm = candle_nn::rms_norm(cfg.kv_lora_rank, cfg.rms_norm_eps, vb.pp("kv_a_norm"))
            .map_err(cand)?;
        let kv_b_proj = candle_nn::linear_no_bias(
            cfg.kv_lora_rank,
            cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            vb.pp("kv_b_proj"),
        )
        .map_err(cand)?;
        let o_proj = candle_nn::linear_no_bias(cfg.n_heads * cfg.v_head_dim, h, vb.pp("o_proj"))
            .map_err(cand)?;

        Ok(Self {
            q_a_proj,
            q_a_norm,
            q_proj,
            kv_a_proj,
            kv_a_norm,
            kv_b_proj,
            o_proj,
            n_heads: cfg.n_heads,
            kv_lora_rank: cfg.kv_lora_rank,
            qk_nope: cfg.qk_nope_head_dim,
            qk_rope: cfg.qk_rope_head_dim,
            v_head_dim: cfg.v_head_dim,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
        past: Option<&KvLayerCache>,
    ) -> Result<(Tensor, KvLayerCache)> {
        let (b, seq, _h) = xs.dims3().map_err(cand)?;
        let qk_head_dim = self.qk_nope + self.qk_rope;

        // --- Query ---
        let q = match (&self.q_a_proj, &self.q_a_norm) {
            (Some(a), Some(n)) => {
                let c = a.forward(xs).map_err(cand)?;
                let c = n.forward(&c).map_err(cand)?;
                self.q_proj.forward(&c).map_err(cand)?
            }
            _ => self.q_proj.forward(xs).map_err(cand)?,
        };
        // [b, seq, n_heads, qk_head_dim] -> [b, n_heads, seq, qk_head_dim]
        let q = q
            .reshape((b, seq, self.n_heads, qk_head_dim))
            .map_err(cand)?
            .transpose(1, 2)
            .map_err(cand)?
            .contiguous()
            .map_err(cand)?;
        let q_nope = q.narrow(D::Minus1, 0, self.qk_nope).map_err(cand)?;
        let q_rope = q
            .narrow(D::Minus1, self.qk_nope, self.qk_rope)
            .map_err(cand)?;
        let q_rope = apply_rope(&q_rope, cos, sin)?;
        let q = Tensor::cat(&[&q_nope, &q_rope], D::Minus1).map_err(cand)?;

        // --- Compressed KV ---
        let kv_a = self.kv_a_proj.forward(xs).map_err(cand)?; // [b, seq, lora + rope]
        let c_kv_cur = kv_a.narrow(D::Minus1, 0, self.kv_lora_rank).map_err(cand)?;
        let k_rope_cur = kv_a
            .narrow(D::Minus1, self.kv_lora_rank, self.qk_rope)
            .map_err(cand)?; // [b, seq, rope]
                             // Apply rope to the shared k_rope as a single head.
        let k_rope_cur = k_rope_cur
            .reshape((b, seq, 1, self.qk_rope))
            .map_err(cand)?
            .transpose(1, 2)
            .map_err(cand)?
            .contiguous()
            .map_err(cand)?; // [b, 1, seq, rope]
        let k_rope_cur = apply_rope(&k_rope_cur, cos, sin)?;
        // Store k_rope as [b, seq, rope].
        let k_rope_cur_store = k_rope_cur
            .transpose(1, 2)
            .map_err(cand)?
            .reshape((b, seq, self.qk_rope))
            .map_err(cand)?;

        let (c_kv_full, k_rope_store, new_mla_cache) = match past {
            // Pre-allocated MLA: O(1) scatter_set vs O(N) cat.
            Some(KvLayerCache::MlaPrealloc {
                c_kv: buf_ckv,
                k_rope: buf_rope,
                seq_len,
                max_seq,
            }) => {
                let idx_ckv = Tensor::full(*seq_len as u32, c_kv_cur.shape(), c_kv_cur.device())
                    .map_err(cand)?;
                buf_ckv.scatter_set(&idx_ckv, &c_kv_cur, 1).map_err(cand)?;
                let idx_rope = Tensor::full(
                    *seq_len as u32,
                    k_rope_cur_store.shape(),
                    k_rope_cur_store.device(),
                )
                .map_err(cand)?;
                buf_rope
                    .scatter_set(&idx_rope, &k_rope_cur_store, 1)
                    .map_err(cand)?;
                let new_seq = seq_len + seq;
                let ckv_v = buf_ckv.narrow(1, 0, new_seq).map_err(cand)?;
                let rope_v = buf_rope.narrow(1, 0, new_seq).map_err(cand)?;
                let cache = KvLayerCache::MlaPrealloc {
                    c_kv: buf_ckv.clone(),
                    k_rope: buf_rope.clone(),
                    seq_len: new_seq,
                    max_seq: *max_seq,
                };
                (ckv_v, rope_v, cache)
            }
            Some(KvLayerCache::Mla { c_kv, k_rope }) => {
                let c = Tensor::cat(&[c_kv, &c_kv_cur], 1).map_err(cand)?;
                let r = Tensor::cat(&[k_rope, &k_rope_cur_store], 1).map_err(cand)?;
                let cache = KvLayerCache::Mla {
                    c_kv: c.clone(),
                    k_rope: r.clone(),
                };
                (c, r, cache)
            }
            _ => {
                let cache = KvLayerCache::Mla {
                    c_kv: c_kv_cur.clone(),
                    k_rope: k_rope_cur_store.clone(),
                };
                (c_kv_cur, k_rope_cur_store, cache)
            }
        };
        let kv_len = c_kv_full.dim(1).map_err(cand)?;

        // Reconstruct per-head k_nope and v from the full latent.
        let kv = self.kv_a_norm.forward(&c_kv_full).map_err(cand)?;
        let kv = self.kv_b_proj.forward(&kv).map_err(cand)?; // [b, kv_len, n_heads*(nope+v)]
        let kv = kv
            .reshape((b, kv_len, self.n_heads, self.qk_nope + self.v_head_dim))
            .map_err(cand)?
            .transpose(1, 2)
            .map_err(cand)?
            .contiguous()
            .map_err(cand)?; // [b, n_heads, kv_len, nope+v]
        let k_nope = kv.narrow(D::Minus1, 0, self.qk_nope).map_err(cand)?;
        let v = kv
            .narrow(D::Minus1, self.qk_nope, self.v_head_dim)
            .map_err(cand)?;

        // Broadcast shared rope keys across heads: [b, kv_len, rope] -> [b, n_heads, kv_len, rope]
        let k_rope_full = k_rope_store
            .reshape((b, 1, kv_len, self.qk_rope))
            .map_err(cand)?
            .broadcast_as((b, self.n_heads, kv_len, self.qk_rope))
            .map_err(cand)?
            .contiguous()
            .map_err(cand)?;
        let k = Tensor::cat(&[&k_nope, &k_rope_full], D::Minus1).map_err(cand)?;

        let scale = 1.0 / (qk_head_dim as f64).sqrt();
        let scores =
            (q.matmul(&k.transpose(2, 3).map_err(cand)?).map_err(cand)? * scale).map_err(cand)?;
        let scores = scores.broadcast_add(mask).map_err(cand)?;
        let probs = ops::softmax_last_dim(&scores).map_err(cand)?;
        let ctx = probs.matmul(&v.contiguous().map_err(cand)?).map_err(cand)?; // [b, n_heads, seq, v_head_dim]
        let ctx = ctx
            .transpose(1, 2)
            .map_err(cand)?
            .contiguous()
            .map_err(cand)?
            .reshape((b, seq, self.n_heads * self.v_head_dim))
            .map_err(cand)?;
        let out = self.o_proj.forward(&ctx).map_err(cand)?;
        Ok((out, new_mla_cache))
    }
}
