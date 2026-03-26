//! Multi-Head Latent Attention (MLA) from DeepSeek-V2/V3.
//!
//! Achieves ~93% KV-cache reduction by compressing key-value pairs into a
//! low-dimensional latent space. Instead of caching full K,V per head per
//! position (`2 * num_heads * head_dim` floats), MLA caches only the latent
//! vector `c_kv` (`latent_dim` floats) and decompresses K,V on-the-fly:
//!
//! 1. Down-project: `c_kv = x @ W_dkv` (d_model -> latent_dim)
//! 2. Up-project:   `K = c_kv @ W_uk`, `V = c_kv @ W_uv`
//! 3. Query path:   `c_q = x @ W_dq`, `Q = c_q @ W_uq` (same low-rank trick)
//! 4. RoPE bypass:  A `rope_dim`-sized portion of each key skips compression
//!    and receives Rotary Position Embeddings directly.

use crate::error::{AttentionError, AttentionResult};
use crate::traits::Attention;

/// Configuration for Multi-Head Latent Attention.
#[derive(Clone, Debug)]
pub struct MLAConfig {
    pub d_model: usize,
    pub latent_dim: usize,
    pub latent_dim_q: Option<usize>,
    pub num_heads: usize,
    pub head_dim: usize,
    /// Must be even and <= head_dim. Set to 0 to disable RoPE decoupling.
    pub rope_dim: usize,
}

impl MLAConfig {
    pub fn validate(&self) -> AttentionResult<()> {
        let err = |msg: &str| Err(AttentionError::InvalidConfig(msg.into()));
        if self.d_model == 0 { return err("d_model must be > 0"); }
        if self.num_heads == 0 { return err("num_heads must be > 0"); }
        if self.head_dim == 0 { return err("head_dim must be > 0"); }
        if self.latent_dim == 0 { return err("latent_dim must be > 0"); }
        if self.latent_dim >= self.full_kv_dim() {
            return err("latent_dim must be < num_heads * head_dim");
        }
        if self.rope_dim > self.head_dim {
            return err("rope_dim must be <= head_dim");
        }
        if self.rope_dim > 0 && self.rope_dim % 2 != 0 {
            return err("rope_dim must be even (RoPE operates on pairs)");
        }
        Ok(())
    }

    pub fn effective_latent_dim_q(&self) -> usize {
        self.latent_dim_q.unwrap_or(self.latent_dim)
    }

    pub fn full_kv_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }
}

/// KV cache storing only latent vectors instead of full K,V per head.
#[derive(Clone, Debug)]
pub struct MLACache {
    pub latent_vectors: Vec<Vec<f32>>,
    pub rope_keys: Vec<Vec<f32>>,
    latent_dim: usize,
    rope_dim: usize,
    num_heads: usize,
    head_dim: usize,
}

impl MLACache {
    pub fn new(config: &MLAConfig) -> Self {
        Self {
            latent_vectors: Vec::new(), rope_keys: Vec::new(),
            latent_dim: config.latent_dim, rope_dim: config.rope_dim,
            num_heads: config.num_heads, head_dim: config.head_dim,
        }
    }

    pub fn push(&mut self, latent: Vec<f32>, rope_key: Vec<f32>) {
        self.latent_vectors.push(latent);
        self.rope_keys.push(rope_key);
    }

    pub fn len(&self) -> usize { self.latent_vectors.len() }
    pub fn is_empty(&self) -> bool { self.latent_vectors.is_empty() }

    /// Total floats stored in this MLA cache.
    pub fn cache_size(&self) -> usize {
        self.len() * (self.latent_dim + self.rope_dim)
    }

    /// Total floats standard MHA would store for the same positions.
    pub fn mha_equivalent_size(&self) -> usize {
        self.len() * 2 * self.num_heads * self.head_dim
    }

    /// KV-cache reduction ratio (e.g. 0.9375 = 93.75% reduction vs MHA).
    pub fn reduction_ratio(&self) -> f32 {
        if self.len() == 0 { return 0.0; }
        1.0 - (self.cache_size() as f32 / self.mha_equivalent_size() as f32)
    }
}

/// Multi-Head Latent Attention layer with projection weights (row-major).
pub struct MLALayer {
    config: MLAConfig,
    w_dkv: Vec<f32>,  // d_model -> latent_dim
    w_uk: Vec<f32>,   // latent_dim -> full_kv_dim (keys)
    w_uv: Vec<f32>,   // latent_dim -> full_kv_dim (values)
    w_dq: Vec<f32>,   // d_model -> latent_dim_q
    w_uq: Vec<f32>,   // latent_dim_q -> full_kv_dim
    w_rope: Vec<f32>, // d_model -> rope_dim
    w_out: Vec<f32>,  // full_kv_dim -> d_model
}

impl MLALayer {
    /// Creates a new MLA layer with deterministic Xavier-style initialization.
    pub fn new(config: MLAConfig) -> AttentionResult<Self> {
        config.validate()?;
        let fd = config.full_kv_dim();
        let lq = config.effective_latent_dim_q();
        Ok(Self {
            w_dkv: init_weight(config.d_model, config.latent_dim),
            w_uk: init_weight(config.latent_dim, fd),
            w_uv: init_weight(config.latent_dim, fd),
            w_dq: init_weight(config.d_model, lq),
            w_uq: init_weight(lq, fd),
            w_rope: init_weight(config.d_model, config.rope_dim),
            w_out: init_weight(fd, config.d_model),
            config,
        })
    }

    pub fn config(&self) -> &MLAConfig { &self.config }

    /// Compress input to KV latent: `c_kv = x @ W_dkv`.
    pub fn compress_kv(&self, x: &[f32]) -> Vec<f32> {
        matvec(&self.w_dkv, x, self.config.d_model, self.config.latent_dim)
    }

    /// Decompress latent to keys: `K = c_kv @ W_uk`.
    pub fn decompress_keys(&self, c: &[f32]) -> Vec<f32> {
        matvec(&self.w_uk, c, self.config.latent_dim, self.config.full_kv_dim())
    }

    /// Decompress latent to values: `V = c_kv @ W_uv`.
    pub fn decompress_values(&self, c: &[f32]) -> Vec<f32> {
        matvec(&self.w_uv, c, self.config.latent_dim, self.config.full_kv_dim())
    }

    fn compute_rope_keys(&self, x: &[f32]) -> Vec<f32> {
        if self.config.rope_dim == 0 { return Vec::new(); }
        matvec(&self.w_rope, x, self.config.d_model, self.config.rope_dim)
    }

    fn compute_query(&self, x: &[f32]) -> Vec<f32> {
        let lq = self.config.effective_latent_dim_q();
        let c_q = matvec(&self.w_dq, x, self.config.d_model, lq);
        matvec(&self.w_uq, &c_q, lq, self.config.full_kv_dim())
    }

    /// Applies RoPE rotation to pairs of dimensions based on position.
    fn apply_rope(v: &mut [f32], position: usize) {
        let dim = v.len();
        for i in (0..dim).step_by(2) {
            if i + 1 >= dim { break; }
            let freq = 1.0 / (10000.0_f32).powf(i as f32 / dim as f32);
            let theta = position as f32 * freq;
            let (cos_t, sin_t) = (theta.cos(), theta.sin());
            let (x0, x1) = (v[i], v[i + 1]);
            v[i] = x0 * cos_t - x1 * sin_t;
            v[i + 1] = x0 * sin_t + x1 * cos_t;
        }
    }

    /// Core attention computation shared by `forward` and `forward_cached`.
    fn attend(
        &self, q_full: &[f32], all_keys: &[Vec<f32>], all_values: &[Vec<f32>],
    ) -> Vec<f32> {
        let (nh, hd) = (self.config.num_heads, self.config.head_dim);
        let scale = (hd as f32).sqrt();
        let mut out = vec![0.0_f32; nh * hd];
        for h in 0..nh {
            let off = h * hd;
            let qh = &q_full[off..off + hd];
            let mut scores: Vec<f32> = all_keys
                .iter()
                .map(|k| dot(&k[off..off + hd], qh) / scale)
                .collect();
            softmax_inplace(&mut scores);
            for (si, &w) in scores.iter().enumerate() {
                let vh = &all_values[si][off..off + hd];
                for d in 0..hd { out[off + d] += w * vh[d]; }
            }
        }
        matvec(&self.w_out, &out, self.config.full_kv_dim(), self.config.d_model)
    }

    /// Prepares query with RoPE applied to the decoupled portion of each head.
    fn prepare_query(&self, input: &[f32], pos: usize) -> Vec<f32> {
        let mut q = self.compute_query(input);
        let (nh, hd, rd) = (self.config.num_heads, self.config.head_dim, self.config.rope_dim);
        if rd > 0 {
            for h in 0..nh { Self::apply_rope(&mut q[h * hd..h * hd + rd], pos); }
        }
        q
    }

    /// Decompresses a latent+rope pair into full keys/values for one position.
    fn decompress_position(
        &self, latent: &[f32], rope: &[f32], pos: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut keys = self.decompress_keys(latent);
        let values = self.decompress_values(latent);
        let (nh, hd, rd) = (self.config.num_heads, self.config.head_dim, self.config.rope_dim);
        if rd > 0 {
            let mut rp = rope.to_vec();
            Self::apply_rope(&mut rp, pos);
            for h in 0..nh { keys[h * hd..h * hd + rd].copy_from_slice(&rp); }
        }
        (keys, values)
    }

    /// Full MLA forward pass for a single query position.
    pub fn forward(
        &self, query_input: &[f32], kv_inputs: &[&[f32]],
        query_pos: usize, kv_positions: &[usize],
    ) -> AttentionResult<Vec<f32>> {
        if query_input.len() != self.config.d_model {
            return Err(AttentionError::DimensionMismatch {
                expected: self.config.d_model, actual: query_input.len(),
            });
        }
        if kv_inputs.is_empty() {
            return Err(AttentionError::EmptyInput("kv_inputs".into()));
        }
        if kv_inputs.len() != kv_positions.len() {
            return Err(AttentionError::DimensionMismatch {
                expected: kv_inputs.len(), actual: kv_positions.len(),
            });
        }
        let q_full = self.prepare_query(query_input, query_pos);
        let mut all_k = Vec::with_capacity(kv_inputs.len());
        let mut all_v = Vec::with_capacity(kv_inputs.len());
        for (i, &kv) in kv_inputs.iter().enumerate() {
            if kv.len() != self.config.d_model {
                return Err(AttentionError::DimensionMismatch {
                    expected: self.config.d_model, actual: kv.len(),
                });
            }
            let c = self.compress_kv(kv);
            let rope = self.compute_rope_keys(kv);
            let (k, v) = self.decompress_position(&c, &rope, kv_positions[i]);
            all_k.push(k);
            all_v.push(v);
        }
        Ok(self.attend(&q_full, &all_k, &all_v))
    }

    /// Forward pass using incremental MLA cache (for autoregressive decoding).
    pub fn forward_cached(
        &self, query_input: &[f32], new_kv_input: &[f32],
        query_pos: usize, cache: &mut MLACache,
    ) -> AttentionResult<Vec<f32>> {
        if new_kv_input.len() != self.config.d_model {
            return Err(AttentionError::DimensionMismatch {
                expected: self.config.d_model, actual: new_kv_input.len(),
            });
        }
        cache.push(self.compress_kv(new_kv_input), self.compute_rope_keys(new_kv_input));
        let q_full = self.prepare_query(query_input, query_pos);
        let mut all_k = Vec::with_capacity(cache.len());
        let mut all_v = Vec::with_capacity(cache.len());
        for pos in 0..cache.len() {
            let (k, v) = self.decompress_position(
                &cache.latent_vectors[pos], &cache.rope_keys[pos], pos,
            );
            all_k.push(k);
            all_v.push(v);
        }
        Ok(self.attend(&q_full, &all_k, &all_v))
    }

    /// Memory comparison report: MLA vs standard MHA caching.
    pub fn memory_comparison(&self, seq_len: usize) -> MemoryComparison {
        let mha = seq_len * 2 * self.config.num_heads * self.config.head_dim;
        let mla = seq_len * (self.config.latent_dim + self.config.rope_dim);
        MemoryComparison {
            seq_len, mha_cache_floats: mha, mla_cache_floats: mla,
            mha_cache_bytes: mha * 4, mla_cache_bytes: mla * 4,
            reduction_ratio: 1.0 - (mla as f32 / mha as f32),
        }
    }
}

/// Report comparing MLA vs MHA cache memory usage.
#[derive(Clone, Debug)]
pub struct MemoryComparison {
    pub seq_len: usize,
    pub mha_cache_floats: usize,
    pub mla_cache_floats: usize,
    pub mha_cache_bytes: usize,
    pub mla_cache_bytes: usize,
    pub reduction_ratio: f32,
}

impl Attention for MLALayer {
    fn compute(
        &self, query: &[f32], keys: &[&[f32]], values: &[&[f32]],
    ) -> AttentionResult<Vec<f32>> {
        let _ = values; // MLA derives V from the same inputs as K
        let positions: Vec<usize> = (0..keys.len()).collect();
        self.forward(query, keys, 0, &positions)
    }

    fn compute_with_mask(
        &self, query: &[f32], keys: &[&[f32]], values: &[&[f32]],
        _mask: Option<&[bool]>,
    ) -> AttentionResult<Vec<f32>> {
        self.compute(query, keys, values)
    }

    fn dim(&self) -> usize { self.config.d_model }
    fn num_heads(&self) -> usize { self.config.num_heads }
}

// -- Utility functions --------------------------------------------------------

fn matvec(w: &[f32], x: &[f32], in_d: usize, out_d: usize) -> Vec<f32> {
    (0..out_d)
        .map(|r| {
            let off = r * in_d;
            (0..in_d).map(|c| w[off + c] * x[c]).sum()
        })
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn softmax_inplace(s: &mut [f32]) {
    let max = s.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0_f32;
    for v in s.iter_mut() { *v = (*v - max).exp(); sum += *v; }
    for v in s.iter_mut() { *v /= sum; }
}

fn init_weight(in_d: usize, out_d: usize) -> Vec<f32> {
    let scale = (2.0 / (in_d + out_d) as f32).sqrt();
    let period = (in_d + out_d).max(1);
    (0..in_d * out_d)
        .map(|i| scale * ((i % period) as f32 / period as f32 - 0.5))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> MLAConfig {
        MLAConfig {
            d_model: 32, latent_dim: 8, latent_dim_q: None,
            num_heads: 4, head_dim: 8, rope_dim: 4,
        }
    }

    #[test]
    fn test_config_valid() { assert!(cfg().validate().is_ok()); }

    #[test]
    fn test_config_latent_too_large() {
        let mut c = cfg(); c.latent_dim = 999;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_rope_dim_odd() {
        let mut c = cfg(); c.rope_dim = 3;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_zero_heads() {
        let mut c = cfg(); c.num_heads = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_forward_output_shape() {
        let c = cfg();
        let layer = MLALayer::new(c.clone()).unwrap();
        let q = vec![0.1_f32; c.d_model];
        let kv1 = vec![0.2_f32; c.d_model];
        let kv2 = vec![0.3_f32; c.d_model];
        let out = layer.forward(&q, &[&kv1, &kv2], 0, &[0, 1]).unwrap();
        assert_eq!(out.len(), c.d_model);
    }

    #[test]
    fn test_forward_dimension_mismatch() {
        let layer = MLALayer::new(cfg()).unwrap();
        let bad_q = vec![0.1_f32; 5];
        let kv = vec![0.2_f32; 32];
        assert!(layer.forward(&bad_q, &[&kv[..]], 0, &[0]).is_err());
    }

    #[test]
    fn test_cache_size_reduction() {
        let c = cfg();
        let mut cache = MLACache::new(&c);
        for _ in 0..10 { cache.push(vec![0.0; c.latent_dim], vec![0.0; c.rope_dim]); }
        assert_eq!(cache.len(), 10);
        assert_eq!(cache.cache_size(), 120);        // 10 * (8+4)
        assert_eq!(cache.mha_equivalent_size(), 640); // 10 * 2*4*8
        assert!((cache.reduction_ratio() - 0.8125).abs() < 1e-4);
    }

    #[test]
    fn test_memory_comparison_report() {
        let c = MLAConfig {
            d_model: 2048, latent_dim: 256, latent_dim_q: None,
            num_heads: 16, head_dim: 128, rope_dim: 0,
        };
        let layer = MLALayer::new(c).unwrap();
        let r = layer.memory_comparison(1024);
        assert_eq!(r.mha_cache_floats, 4_194_304);
        assert_eq!(r.mla_cache_floats, 262_144);
        assert!((r.reduction_ratio - 0.9375).abs() < 1e-4);
    }

    #[test]
    fn test_cached_forward_multi_position() {
        let c = cfg();
        let layer = MLALayer::new(c.clone()).unwrap();
        let mut cache = MLACache::new(&c);
        let q = vec![0.1_f32; c.d_model];
        for pos in 0..3 {
            let kv = vec![(pos as f32 + 1.0) * 0.1; c.d_model];
            let out = layer.forward_cached(&q, &kv, pos, &mut cache).unwrap();
            assert_eq!(out.len(), c.d_model);
        }
        assert_eq!(cache.len(), 3);
        let kv_last = vec![0.4_f32; c.d_model];
        let out = layer.forward_cached(&q, &kv_last, 3, &mut cache).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
        assert_eq!(cache.len(), 4);
    }

    #[test]
    fn test_rope_identity_at_zero() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        let orig = v.clone();
        MLALayer::apply_rope(&mut v, 0);
        for (a, b) in v.iter().zip(&orig) { assert!((a - b).abs() < 1e-6); }
    }

    #[test]
    fn test_rope_preserves_norm() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        let norm_before: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        MLALayer::apply_rope(&mut v, 42);
        let norm_after: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm_before - norm_after).abs() < 1e-5);
    }

    #[test]
    fn test_compress_decompress_dimensions() {
        let c = cfg();
        let layer = MLALayer::new(c.clone()).unwrap();
        let x = vec![0.5_f32; c.d_model];
        let ckv = layer.compress_kv(&x);
        assert_eq!(ckv.len(), c.latent_dim);
        assert_eq!(layer.decompress_keys(&ckv).len(), c.full_kv_dim());
        assert_eq!(layer.decompress_values(&ckv).len(), c.full_kv_dim());
    }

    #[test]
    fn test_attention_trait() {
        let c = cfg();
        let layer = MLALayer::new(c.clone()).unwrap();
        assert_eq!(layer.dim(), c.d_model);
        assert_eq!(layer.num_heads(), c.num_heads);
        let q = vec![0.1_f32; c.d_model];
        let kv1 = vec![0.2_f32; c.d_model];
        let kv2 = vec![0.3_f32; c.d_model];
        let out = layer.compute(&q, &[&kv1[..], &kv2[..]], &[&kv1[..], &kv2[..]]).unwrap();
        assert_eq!(out.len(), c.d_model);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_empty_cache_ratio() {
        let cache = MLACache::new(&cfg());
        assert_eq!(cache.reduction_ratio(), 0.0);
        assert!(cache.is_empty());
    }
}
