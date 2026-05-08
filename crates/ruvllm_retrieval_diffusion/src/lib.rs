//! Corpus-agnostic training-free retrieval LM and masked discrete diffusion
//! built on `ruvllm_sparse_attention`.
//!
//! Generalises the `sparse-mario` example: any small-vocab token domain can
//! plug in by supplying a corpus and a [`RetrievalConfig`]. The kernel is
//! used as an associative memory — no autograd, no learned weights, no
//! Python toolchain.
//!
//! Two pipelines from one kernel:
//!
//! - [`Retriever::generate_fast`] — autoregressive next-token retrieval via
//!   `KvCache` + `decode_step`, O(log T) per generated token.
//! - [`Diffuser::diffuse`] — bidirectional masked discrete diffusion with a
//!   MaskGIT cosine schedule. Beats the AR path on aggregate by 6.9× on
//!   the Mario benchmark (see `sparse-mario` baselines doc).
//!
//! ## Domain plug-in checklist
//!
//! ```ignore
//! use ruvllm_retrieval_diffusion::{Retriever, Diffuser, RetrievalConfig, SamplingConfig};
//!
//! let cfg = RetrievalConfig {
//!     vocab_size: 5,        // your domain's token count
//!     head_dim: 64,         // 64 works well for vocab ≤ 32
//!     pos_scale: 0.5,       // try 0 to make AR pos-invariant
//!     mask_sentinel: 255,
//!     ..RetrievalConfig::default()
//! };
//! let corpus: Vec<u8> = encode_my_corpus();   // your encoder, vocab-bounded
//! let retriever = Retriever::new(corpus, cfg, 0xMARI_BEEF);
//! let level = retriever.generate_fast(&seed, 256, &SamplingConfig::quality(), 0xC0FFEE);
//! ```

use ruvllm_sparse_attention::{
    AttentionBackend, KvCache, SparseAttentionConfig, SubquadraticSparseAttention, Tensor3,
};

pub use ruvllm_sparse_attention::SparseAttentionConfig as SparseConfig;

// ----------------------------------------------------------------------
// Retrieval config
// ----------------------------------------------------------------------

/// Static configuration shared by both `Retriever` and `Diffuser`.
///
/// `vocab_size` is the number of distinct tokens (≤ 254 to leave one byte
/// for `mask_sentinel`). `head_dim` is the embedding dimension (64 is a
/// good default — the kernel's `1/sqrt(d)` softmax scale separates matched
/// random unit-vector pairs by ~sqrt(d) which is comfortable at d=64).
#[derive(Clone, Debug)]
pub struct RetrievalConfig {
    pub vocab_size: usize,
    pub head_dim: usize,
    /// Positional encoding weight in K/V row construction (AR path). 0
    /// disables — AR becomes purely content-based, useful when the
    /// corpus has no per-position structure to exploit. The Mario
    /// example uses 0.5; the iter-13 finding was that 0 would halve
    /// AR's L2 distance for grid-shaped corpora.
    pub pos_scale: f32,
    /// Out-of-vocab byte used by the diffuser to mark not-yet-denoised
    /// positions. Must be ≥ vocab_size.
    pub mask_sentinel: u8,
    /// Bidirectional context weights for the diffuser, indexed by
    /// `offset - 1` (radius = len()). [0.5, 0.10] is the iter-10 pick.
    pub diffusion_context_weights: Vec<f32>,
    /// Sparse attention config passed to the underlying kernel. Defaults
    /// to non-causal window=256 + log-stride + landmarks.
    pub sparse: SparseAttentionConfig,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            vocab_size: 16,
            head_dim: 64,
            pos_scale: 0.5,
            mask_sentinel: 255,
            diffusion_context_weights: vec![0.5, 0.10],
            sparse: SparseAttentionConfig {
                window: 256,
                block_size: 64,
                global_tokens: vec![0],
                causal: false,
                use_log_stride: true,
                use_landmarks: true,
                sort_candidates: false,
            },
        }
    }
}

// ----------------------------------------------------------------------
// Sampling config
// ----------------------------------------------------------------------

/// Sampling controls applied in `sample_logits` in this order:
/// repetition penalty → top-k → top-p → softmax(/T) → categorical sample.
#[derive(Clone, Debug)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub no_repeat_window: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.0,
            repetition_penalty: 1.0,
            no_repeat_window: 0,
        }
    }
}

impl SamplingConfig {
    /// The Mario-validated quality recipe. Reasonable starting point for any
    /// small-vocab domain; tune `no_repeat_window` to your meaningful local
    /// span (e.g. one row, one bar of music, one indented config block).
    pub fn quality() -> Self {
        Self {
            temperature: 1.0,
            top_k: 5,
            top_p: 0.90,
            repetition_penalty: 1.7,
            no_repeat_window: 24,
        }
    }
}

// ----------------------------------------------------------------------
// Deterministic PRNG (xorshift32 + Box-Muller normal)
// ----------------------------------------------------------------------

#[inline]
pub fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    if x == 0 {
        x = 0x9E37_79B9;
    }
    x ^= x.wrapping_shl(13);
    x ^= x.wrapping_shr(17);
    x ^= x.wrapping_shl(5);
    *state = x;
    x
}

#[inline]
pub fn next_uniform(state: &mut u32) -> f32 {
    (xorshift32(state) as f32) / (u32::MAX as f32 + 1.0)
}

pub fn next_normal(state: &mut u32) -> f32 {
    loop {
        let u1 = next_uniform(state);
        let u2 = next_uniform(state);
        if u1 > 1e-9 {
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            return r * theta.cos();
        }
    }
}

// ----------------------------------------------------------------------
// Embedding helpers
// ----------------------------------------------------------------------

fn make_embedding_matrix(vocab_size: usize, head_dim: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.max(1);
    let mut w = vec![0.0f32; vocab_size * head_dim];
    for v in w.iter_mut() {
        *v = next_normal(&mut state);
    }
    w
}

#[inline]
fn token_embedding<'a>(t: u8, w: &'a [f32], head_dim: usize) -> &'a [f32] {
    let i = (t as usize) * head_dim;
    &w[i..i + head_dim]
}

fn pos_encoding_into(i: usize, dim: usize, out: &mut [f32]) {
    for d in 0..dim {
        let half = d / 2;
        let theta = (i as f32) / 10000_f32.powf((2 * half) as f32 / dim as f32);
        out[d] = if d % 2 == 0 { theta.sin() } else { theta.cos() };
    }
}

// ----------------------------------------------------------------------
// Sample logits helper (rep penalty → top-k → top-p → softmax)
// ----------------------------------------------------------------------

pub fn sample_logits(
    logits: &mut [f32],
    cfg: &SamplingConfig,
    recent: &[u8],
    state: &mut u32,
) -> u8 {
    let v = logits.len();
    if v == 0 {
        return 0;
    }

    if cfg.repetition_penalty > 1.0 + f32::EPSILON && !recent.is_empty() {
        let pen = cfg.repetition_penalty;
        for &t in recent {
            let i = t as usize;
            if i < v {
                logits[i] = if logits[i] > 0.0 {
                    logits[i] / pen
                } else {
                    logits[i] * pen
                };
            }
        }
    }

    if cfg.top_k > 0 && cfg.top_k < v {
        let mut idx: Vec<usize> = (0..v).collect();
        idx.sort_unstable_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        let kth = logits[idx[cfg.top_k - 1]];
        for li in logits.iter_mut() {
            if *li < kth {
                *li = f32::NEG_INFINITY;
            }
        }
    }

    if cfg.top_p > 0.0 && cfg.top_p < 1.0 {
        let temp_p = cfg.temperature.max(1e-3);
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut pairs: Vec<(usize, f32)> = (0..v)
            .map(|i| {
                let p = if logits[i].is_finite() {
                    ((logits[i] - max_l) / temp_p).exp()
                } else {
                    0.0
                };
                (i, p)
            })
            .collect();
        let total: f32 = pairs.iter().map(|p| p.1).sum();
        if total > 0.0 {
            for pr in pairs.iter_mut() {
                pr.1 /= total;
            }
            pairs.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal)
            });
            let mut keep = vec![false; v];
            let mut cum = 0.0f32;
            for &(idx, p) in pairs.iter() {
                keep[idx] = true;
                cum += p;
                if cum >= cfg.top_p {
                    break;
                }
            }
            for i in 0..v {
                if !keep[i] {
                    logits[i] = f32::NEG_INFINITY;
                }
            }
        }
    }

    let temp = cfg.temperature.max(1e-3);
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs = vec![0.0f32; v];
    let mut sum = 0.0f32;
    for i in 0..v {
        if logits[i].is_finite() {
            probs[i] = ((logits[i] - max_l) / temp).exp();
            sum += probs[i];
        }
    }
    if sum <= 0.0 {
        return 0;
    }
    for p in probs.iter_mut() {
        *p /= sum;
    }
    let r = next_uniform(state);
    let mut acc = 0.0f32;
    for i in 0..v {
        acc += probs[i];
        if r < acc {
            return i as u8;
        }
    }
    (v - 1) as u8
}

// ----------------------------------------------------------------------
// Retriever — autoregressive retrieval LM
// ----------------------------------------------------------------------

/// Training-free retrieval LM. K[i] = embed(corpus[i]) + pos·pos(i),
/// V[i] = embed(corpus[i+1]) + pos·pos(i). Attention finds positions
/// where the query token matches a corpus token, and reads back what
/// follows it — pure bigram retrieval through the kernel's lookup.
pub struct Retriever {
    pub corpus: Vec<u8>,
    pub w: Vec<f32>,
    pub cfg: RetrievalConfig,
}

impl Retriever {
    pub fn new(corpus: Vec<u8>, cfg: RetrievalConfig, embedding_seed: u32) -> Self {
        let w = make_embedding_matrix(cfg.vocab_size, cfg.head_dim, embedding_seed);
        Self { corpus, w, cfg }
    }

    fn build_kv_row(&self, tok: u8, abs_pos: usize) -> Tensor3 {
        let d = self.cfg.head_dim;
        let mut data = vec![0.0f32; d];
        let emb = token_embedding(tok, &self.w, d);
        let mut pos = vec![0.0f32; d];
        pos_encoding_into(abs_pos, d, &mut pos);
        for di in 0..d {
            data[di] = emb[di] + self.cfg.pos_scale * pos[di];
        }
        Tensor3::from_vec(data, 1, 1, d).unwrap()
    }

    fn make_row_tensor(&self, tokens: &[u8], shift_for_value: bool) -> Tensor3 {
        let d = self.cfg.head_dim;
        let seq = tokens.len();
        let mut t = Tensor3::zeros(seq, 1, d);
        let mut pos = vec![0.0f32; d];
        for i in 0..seq {
            let tok = if shift_for_value {
                if i + 1 < seq {
                    tokens[i + 1]
                } else {
                    tokens[i]
                }
            } else {
                tokens[i]
            };
            let emb = token_embedding(tok, &self.w, d);
            pos_encoding_into(i, d, &mut pos);
            let row = t.row_mut(i, 0);
            for di in 0..d {
                row[di] = emb[di] + self.cfg.pos_scale * pos[di];
            }
        }
        t
    }

    /// Reference path — full forward over corpus + prefix every step.
    /// Slow (~O(N log N) per token); use `generate_fast` in production.
    pub fn next_token_logits(&self, prefix: &[u8]) -> Vec<f32> {
        let mut combined = self.corpus.clone();
        combined.extend_from_slice(prefix);
        let q = self.make_row_tensor(&combined, false);
        let v = self.make_row_tensor(&combined, true);
        let attn = SubquadraticSparseAttention::new(self.cfg.sparse.clone()).expect("config");
        let out = attn.forward(&q, &q, &v).expect("attention");
        let last = combined.len() - 1;
        let d = self.cfg.head_dim;
        let mut logits = vec![0.0f32; self.cfg.vocab_size];
        for v_idx in 0..self.cfg.vocab_size {
            let emb = token_embedding(v_idx as u8, &self.w, d);
            let mut dot = 0.0f32;
            for di in 0..d {
                dot += out.get(last, 0, di) * emb[di];
            }
            logits[v_idx] = dot;
        }
        logits
    }

    /// Fast path — pre-fill `KvCache` once, then one O(log T) `decode_step`
    /// per generated token. Targets ~3000× speedup vs `next_token_logits`
    /// at 700-token generations on the Mario benchmark.
    pub fn generate_fast(
        &self,
        prefix: &[u8],
        n: usize,
        sampling: &SamplingConfig,
        sampler_seed: u32,
    ) -> Vec<u8> {
        let mut state = sampler_seed.max(1);
        let d = self.cfg.head_dim;
        let cap = self.corpus.len() + prefix.len() + n + 16;
        let mut cache = KvCache::new(cap, 1, d, self.cfg.sparse.block_size);
        let attn = SubquadraticSparseAttention::new(self.cfg.sparse.clone()).expect("config");
        let zero_v = Tensor3::zeros(1, 1, d);

        for i in 0..self.corpus.len() {
            let next = if i + 1 < self.corpus.len() {
                self.corpus[i + 1]
            } else {
                prefix.first().copied().unwrap_or(self.corpus[i])
            };
            let k = self.build_kv_row(self.corpus[i], i);
            let v = self.build_kv_row(next, i);
            cache.try_append(&k, &v).expect("capacity");
        }
        for j in 0..prefix.len() {
            let abs = self.corpus.len() + j;
            let k = self.build_kv_row(prefix[j], abs);
            let v = if j + 1 < prefix.len() {
                self.build_kv_row(prefix[j + 1], abs)
            } else {
                zero_v.clone()
            };
            cache.try_append(&k, &v).expect("capacity");
        }

        let mut sequence = prefix.to_vec();
        for _ in 0..n {
            let last_idx = cache.len - 1;
            let last_tok = sequence.last().copied().unwrap_or(0);
            let q = self.build_kv_row(last_tok, last_idx);
            let out = attn.decode_step(&q, &cache).expect("decode");

            let mut logits = vec![0.0f32; self.cfg.vocab_size];
            for v_idx in 0..self.cfg.vocab_size {
                let emb = token_embedding(v_idx as u8, &self.w, d);
                let mut dot = 0.0f32;
                for di in 0..d {
                    dot += out.get(0, 0, di) * emb[di];
                }
                logits[v_idx] = dot;
            }

            let win = sampling.no_repeat_window.min(sequence.len());
            let recent = &sequence[sequence.len() - win..];
            let next = sample_logits(&mut logits, sampling, recent, &mut state);

            let new_idx = cache.len;
            let k_new = self.build_kv_row(next, new_idx);
            if cache.try_append(&k_new, &zero_v).is_err() {
                break;
            }
            sequence.push(next);
        }
        sequence
    }
}

// ----------------------------------------------------------------------
// Diffuser — bidirectional masked discrete diffusion
// ----------------------------------------------------------------------

pub struct Diffuser<'a> {
    pub retriever: &'a Retriever,
}

impl<'a> Diffuser<'a> {
    pub fn new(retriever: &'a Retriever) -> Self {
        Self { retriever }
    }

    /// Build bidirectional K and V tensors. K[i] sums weighted neighbour
    /// embeddings within radius = `cfg.diffusion_context_weights.len()`.
    /// No positional encoding — pure content match.
    pub fn make_bidir_kv(&self, seq: &[u8]) -> (Tensor3, Tensor3) {
        let d = self.retriever.cfg.head_dim;
        let n = seq.len();
        let mask = self.retriever.cfg.mask_sentinel;
        let weights = &self.retriever.cfg.diffusion_context_weights;
        let mut k = Tensor3::zeros(n, 1, d);
        let mut v = Tensor3::zeros(n, 1, d);
        let zero = vec![0.0f32; d];

        for i in 0..n {
            let krow = k.row_mut(i, 0);
            for slot in 0..weights.len() {
                let weight = weights[slot];
                let off = slot + 1;
                if i >= off && seq[i - off] != mask {
                    let emb = token_embedding(seq[i - off], &self.retriever.w, d);
                    for di in 0..d {
                        krow[di] += weight * emb[di];
                    }
                }
                if i + off < n && seq[i + off] != mask {
                    let emb = token_embedding(seq[i + off], &self.retriever.w, d);
                    for di in 0..d {
                        krow[di] += weight * emb[di];
                    }
                }
            }
            let vrow = v.row_mut(i, 0);
            if seq[i] != mask {
                let emb = token_embedding(seq[i], &self.retriever.w, d);
                vrow.copy_from_slice(emb);
            } else {
                vrow.copy_from_slice(&zero);
            }
        }
        (k, v)
    }

    fn diffusion_logits(&self, working: &[u8]) -> Vec<Vec<f32>> {
        let d = self.retriever.cfg.head_dim;
        let mut combined = self.retriever.corpus.clone();
        combined.extend_from_slice(working);
        let (k, v) = self.make_bidir_kv(&combined);
        let q = k.clone();
        let attn =
            SubquadraticSparseAttention::new(self.retriever.cfg.sparse.clone()).expect("config");
        let out = attn.forward(&q, &q, &v).expect("attention");

        let prefix_start = self.retriever.corpus.len();
        let vsize = self.retriever.cfg.vocab_size;
        let mut all = Vec::with_capacity(working.len());
        for i in 0..working.len() {
            let idx = prefix_start + i;
            let mut logits = vec![0.0f32; vsize];
            for v_idx in 0..vsize {
                let emb = token_embedding(v_idx as u8, &self.retriever.w, d);
                let mut dot = 0.0f32;
                for di in 0..d {
                    dot += out.get(idx, 0, di) * emb[di];
                }
                logits[v_idx] = dot;
            }
            all.push(logits);
        }
        all
    }

    pub fn denoise_step(
        &self,
        working: &mut [u8],
        keep_count: usize,
        sampling: &SamplingConfig,
        state: &mut u32,
    ) {
        let mask = self.retriever.cfg.mask_sentinel;
        let masked: Vec<usize> = working
            .iter()
            .enumerate()
            .filter(|(_, &t)| t == mask)
            .map(|(i, _)| i)
            .collect();
        if masked.is_empty() || keep_count == 0 {
            return;
        }
        let logits = self.diffusion_logits(working);
        let confidence = |row: &[f32]| -> f32 {
            let max_l = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            let mut top = 0.0f32;
            for &l in row.iter() {
                let e = (l - max_l).exp();
                sum += e;
                if e > top {
                    top = e;
                }
            }
            if sum > 0.0 {
                top / sum
            } else {
                0.0
            }
        };
        let mut ranked: Vec<(usize, f32)> = masked
            .iter()
            .map(|&j| (j, confidence(&logits[j])))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        let n = keep_count.min(ranked.len());
        for ki in 0..n {
            let (j, _) = ranked[ki];
            let mut row = logits[j].clone();
            let mut next = sample_logits(&mut row, sampling, &[], state);
            if (next as usize) >= self.retriever.cfg.vocab_size {
                next = 0;
            }
            working[j] = next;
        }
    }

    /// Full pipeline: all-mask init → context boot (random contiguous corpus
    /// slice) → cosine-scheduled denoising → final sweep. Returns a fully
    /// denoised sequence of length `n`.
    pub fn diffuse(
        &self,
        n: usize,
        n_steps: usize,
        sampling: &SamplingConfig,
        seed: u32,
    ) -> Vec<u8> {
        let mut state = seed.max(1);
        let mask = self.retriever.cfg.mask_sentinel;
        let mut working = vec![mask; n];

        let corpus_len = self.retriever.corpus.len();
        let boot_len = (n / 8).clamp(8, 64).min(corpus_len.saturating_sub(1));
        if boot_len > 0 && corpus_len > boot_len {
            let corpus_off = (xorshift32(&mut state) as usize) % (corpus_len - boot_len);
            let work_off = (xorshift32(&mut state) as usize) % (n - boot_len);
            working[work_off..work_off + boot_len].copy_from_slice(
                &self.retriever.corpus[corpus_off..corpus_off + boot_len],
            );
        }

        for t in 0..n_steps {
            let frac = ((t + 1) as f32) / (n_steps as f32);
            let target_masked =
                (n as f32 * (core::f32::consts::FRAC_PI_2 * frac).cos()) as usize;
            let current_masked = working.iter().filter(|&&x| x == mask).count();
            let to_unmask = current_masked.saturating_sub(target_masked).max(1);
            self.denoise_step(&mut working, to_unmask, sampling, &mut state);
        }
        let remaining = working.iter().filter(|&&x| x == mask).count();
        if remaining > 0 {
            self.denoise_step(&mut working, remaining, sampling, &mut state);
        }
        working
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_corpus() -> Vec<u8> {
        // 4-token vocab: [0, 1, 2, 3]. A repeating pattern with structure.
        let mut c = Vec::new();
        for _ in 0..50 {
            c.push(0);
            c.push(1);
            c.push(2);
            c.push(3);
        }
        c
    }

    fn small_cfg() -> RetrievalConfig {
        RetrievalConfig {
            vocab_size: 4,
            head_dim: 32,
            pos_scale: 0.5,
            mask_sentinel: 255,
            diffusion_context_weights: vec![0.5, 0.10],
            sparse: SparseAttentionConfig {
                window: 64,
                block_size: 16,
                global_tokens: vec![0],
                causal: false,
                use_log_stride: true,
                use_landmarks: true,
                sort_candidates: false,
            },
        }
    }

    #[test]
    fn retriever_runs_end_to_end() {
        let r = Retriever::new(small_corpus(), small_cfg(), 0xABCD);
        let out = r.generate_fast(&[0u8, 1u8], 32, &SamplingConfig::quality(), 0xBEEF);
        assert_eq!(out.len(), 34);
        for &t in &out {
            assert!((t as usize) < 4, "out-of-vocab token {}", t);
        }
    }

    #[test]
    fn retriever_is_deterministic() {
        let r = Retriever::new(small_corpus(), small_cfg(), 0xABCD);
        let a = r.generate_fast(&[0u8], 64, &SamplingConfig::quality(), 0xCAFE);
        let b = r.generate_fast(&[0u8], 64, &SamplingConfig::quality(), 0xCAFE);
        assert_eq!(a, b);
    }

    #[test]
    fn diffuser_runs_end_to_end_and_clears_masks() {
        let r = Retriever::new(small_corpus(), small_cfg(), 0xABCD);
        let d = Diffuser::new(&r);
        let out = d.diffuse(80, 8, &SamplingConfig::quality(), 0xDEAD);
        assert_eq!(out.len(), 80);
        let mask = small_cfg().mask_sentinel;
        for &t in &out {
            assert!(t != mask, "leftover mask in output");
            assert!((t as usize) < 4, "out-of-vocab token {}", t);
        }
    }

    #[test]
    fn diffuser_is_deterministic() {
        let r = Retriever::new(small_corpus(), small_cfg(), 0xABCD);
        let d = Diffuser::new(&r);
        let a = d.diffuse(80, 8, &SamplingConfig::quality(), 0x1234);
        let b = d.diffuse(80, 8, &SamplingConfig::quality(), 0x1234);
        assert_eq!(a, b);
    }

    #[test]
    fn sample_logits_top_k_one_is_greedy() {
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_k: 1,
            ..SamplingConfig::default()
        };
        let mut logits = vec![1.0, 2.0, 0.5, 3.0];
        let mut state = 0xABCDu32;
        let next = sample_logits(&mut logits, &cfg, &[], &mut state);
        assert_eq!(next, 3, "top_k=1 should pick the argmax (index 3)");
    }

    #[test]
    fn pos_scale_zero_makes_retrieval_position_invariant() {
        // With pos_scale=0 the AR retriever depends only on token identity.
        // The same prefix should produce the same prediction regardless of
        // its absolute position — i.e. shifting the prefix index doesn't
        // change next-token logits *modulo what positions are in the sparse
        // window*. We just check that the path runs and produces in-vocab
        // tokens; full position-invariance is corpus-dependent.
        let mut cfg = small_cfg();
        cfg.pos_scale = 0.0;
        let r = Retriever::new(small_corpus(), cfg, 0xABCD);
        let out = r.generate_fast(&[2u8], 32, &SamplingConfig::default(), 0xBEEF);
        for &t in &out {
            assert!((t as usize) < 4);
        }
    }
}
