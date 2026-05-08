// Sparse-Mario — a Super Mario Bros level autoregressive generator
// built on `ruvllm_sparse_attention`.
//
// Iteration 1 scaffold: corpus + tokenizer + stats. No model yet.
// Run with: cargo run --release --example sparse_mario --features parallel

use std::collections::HashMap;

// VGLC-style tile alphabet (Super Mario Bros).
// https://github.com/TheVGLC/TheVGLC (MIT). The three slices below are
// hand-authored, public-domain compositions in the same alphabet.
//
//   - = sky / empty
//   X = solid ground
//   S = breakable brick
//   ? = active question block
//   Q = used question block
//   o = coin
//   < > = pipe top
//   [ ] = pipe body
//   E   = enemy (goomba)
//   B   = cannon ball
//   b   = cannon top
//   M   = mario start

pub const LEVELS: &[&str] = &[
    // Slice A — opening, single pipe, one ? block, two goombas
    "\
--------------------------------------------------\n\
--------------------------------------------------\n\
--------------------------------------------------\n\
--------------------------------------------------\n\
--------------------------------------------------\n\
--------------------------------------------------\n\
-------------------------------oo-----------------\n\
-----------?--------SSS?S-------------------------\n\
--------------------------------------<>----------\n\
--------------------E---------------E-[]----------\n\
M-------------------------------------[]----------\n\
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",

    // Slice B — staircase, double-pipe, brick ceiling
    "\
--------------------------------------------------\n\
--------------------------------------------------\n\
--------SSSSSSSSSSSSSSSS--------------------------\n\
--------------------------------------------------\n\
-----------------oo-------------------------------\n\
--?------SSS-----?S-S-------------oo--------------\n\
--------------------------------------------------\n\
-----------E-------------<>--------------<>-------\n\
-----------------E--E----[]------E-------[]-------\n\
M-------------XX---------[]----XXXX------[]-------\n\
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",

    // Slice C — cannons, gap, coin shower
    "\
--------------------------------------------------\n\
--------------------------------------------------\n\
--------------------------------------------------\n\
-------------------oooooooo-----------------------\n\
--------------------------------------------------\n\
------?-------SSSSS--------SSSSS------------------\n\
--------------------------------------------------\n\
-----------------b---------------b----------------\n\
-----E-----E-----B--E-----E------B------E---------\n\
M-----------------B--------------B----------------\n\
XXXXXXXXXXXXXXXXXXBXXXXXXXXXXXXXXBXXXXX-----XXXXXX\n\
XXXXXXXXXXXXXXXXXXBXXXXXXXXXXXXXXBXXXXX-----XXXXXX\n\
XXXXXXXXXXXXXXXXXXBXXXXXXXXXXXXXXBXXXXX-----XXXXXX\n\
XXXXXXXXXXXXXXXXXXBXXXXXXXXXXXXXXBXXXXX-----XXXXXX",
];

/// Tile vocabulary in deterministic order. Index = token id.
pub const VOCAB: &[char] = &[
    '-', // 0  sky
    'X', // 1  ground
    'S', // 2  breakable brick
    '?', // 3  active ? block
    'Q', // 4  used ? block
    'o', // 5  coin
    '<', // 6  pipe top-left
    '>', // 7  pipe top-right
    '[', // 8  pipe body-left
    ']', // 9  pipe body-right
    'E', // 10 enemy (goomba)
    'B', // 11 cannon ball
    'b', // 12 cannon top
    'M', // 13 mario start
    '\n',// 14 row separator
];

/// Char → token id. Returns None for unknown characters so the corpus stays clean.
pub fn encode_char(c: char) -> Option<u8> {
    VOCAB.iter().position(|&v| v == c).map(|i| i as u8)
}

/// Token id → char.
pub fn decode_token(t: u8) -> char {
    VOCAB.get(t as usize).copied().unwrap_or('?')
}

/// Encode a level slice into a flat token stream, including row separators.
pub fn encode_level(level: &str) -> Vec<u8> {
    level.chars().filter_map(encode_char).collect()
}

/// Encode the entire embedded corpus into one concatenated token stream,
/// with the row-separator token between successive levels too (so the model
/// learns slice boundaries).
pub fn encode_corpus() -> Vec<u8> {
    let nl = encode_char('\n').unwrap();
    let mut out = Vec::new();
    for (i, lvl) in LEVELS.iter().enumerate() {
        if i > 0 {
            out.push(nl);
        }
        out.extend(encode_level(lvl));
    }
    out
}

/// Width (column count) of a level slice. All embedded slices share width=50.
pub fn level_width(level: &str) -> usize {
    level.lines().next().map(|r| r.chars().count()).unwrap_or(0)
}

/// Tile distribution over the full corpus. Returns map char → count.
pub fn tile_distribution(tokens: &[u8]) -> HashMap<char, usize> {
    let mut m = HashMap::new();
    for &t in tokens {
        *m.entry(decode_token(t)).or_insert(0) += 1;
    }
    m
}

// =================================================================
// Iter 2 — sparse-attention retrieval LM
//
// The crate is inference-only (no autograd), so instead of training a
// transformer we use the sparse attention kernel as an associative
// memory:
//
//   K[i] = embed(corpus[i])     + pos(i)
//   V[i] = embed(corpus[i+1])             ← "supervision" baked in
//   Q[i] = embed(prefix[i])     + pos(i)
//   out  = SubquadraticSparseAttention.forward(Q, K, V)
//   logits = out[last] · embedW^T
//   next   = sample(softmax(logits / T))
//
// V is the corpus shifted by one position, so attention output is a
// soft-pointer to the empirical next-token distribution. Embeddings are
// random-normal with a fixed seed; ties between embed(t)·embed(t) are
// strongest, so attention naturally retrieves "what tile usually follows
// this tile in the corpus" — without any training.
// =================================================================

use ruvllm_sparse_attention::{
    AttentionBackend, KvCache, SparseAttentionConfig, SubquadraticSparseAttention, Tensor3,
};

const HEAD_DIM: usize = 64;
const N_HEADS: usize = 1;
pub const VOCAB_SIZE: usize = 15;

const _: () = assert!(VOCAB.len() == VOCAB_SIZE, "VOCAB_SIZE drift vs VOCAB[]");

/// xorshift32 — deterministic PRNG, no external dep, no_std-friendly.
fn xorshift32(state: &mut u32) -> u32 {
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

fn next_uniform(state: &mut u32) -> f32 {
    (xorshift32(state) as f32) / (u32::MAX as f32 + 1.0)
}

fn next_normal(state: &mut u32) -> f32 {
    // Box-Muller — return one of the two samples.
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

fn make_embedding_matrix(seed: u32) -> Vec<f32> {
    // Unit-variance per dimension. Combined with the kernel's 1/sqrt(d)
    // softmax scale, embed(t)·embed(t)/sqrt(d) ≈ sqrt(d) for matched tokens
    // and ≈ N(0,1) for unmatched — enough separation that exp() picks out
    // matches strongly. /sqrt(d)-scaled embeddings drown in the noise floor.
    let mut state = seed.max(1);
    let mut w = vec![0.0f32; VOCAB_SIZE * HEAD_DIM];
    for v in w.iter_mut() {
        *v = next_normal(&mut state);
    }
    w
}

fn token_embedding(t: u8, w: &[f32]) -> &[f32] {
    let i = t as usize * HEAD_DIM;
    &w[i..i + HEAD_DIM]
}

/// Sinusoidal positional encoding into `out` (length must equal `dim`).
/// Used at scale 0.5 so token signal still dominates softmax (matched
/// embed·embed = d ≫ 0.5²·pos·pos = d/8) but local context still nudges
/// the retrieval toward positions in similar level-row offsets.
fn pos_encoding_into(i: usize, dim: usize, out: &mut [f32]) {
    for d in 0..dim {
        let half = d / 2;
        let theta = (i as f32) / 10000_f32.powf((2 * half) as f32 / dim as f32);
        out[d] = if d % 2 == 0 { theta.sin() } else { theta.cos() };
    }
}

const POS_SCALE: f32 = 0.5;

/// Sampling controls for `MarioRetriever::generate*`.
///
/// Bare softmax over the retrieval logits saturates on the dominant bigram
/// (sky → sky, ground → ground), producing all-`-` or all-`X` levels. Top-k
/// + top-p + repetition penalty + a no-repeat window together break the
/// chain so the sparse attention kernel surfaces diverse candidates.
///
/// Order applied in `sample_logits`: repetition penalty → top-k mask →
/// top-p (nucleus) mask → softmax(/temperature) → categorical sample.
#[derive(Clone, Debug)]
pub struct SamplingConfig {
    /// Softmax temperature. >1 flattens, <1 sharpens. <=0 falls back to 1e-3.
    pub temperature: f32,
    /// Restrict sampling to the top-k highest logits. 0 disables.
    pub top_k: usize,
    /// Nucleus / top-p mass: keep the smallest set of tokens whose
    /// cumulative softmax probability ≥ `top_p`. 0.0 or ≥ 1.0 disables.
    pub top_p: f32,
    /// Divide positive logits by this and multiply negative ones by it for
    /// every token that appears in the recent window. 1.0 disables.
    pub repetition_penalty: f32,
    /// Window size (in recent generated tokens) over which the repetition
    /// penalty applies. 0 disables.
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
    /// The configuration the iter-9 sweep landed on. Trades exact bigram
    /// fidelity for visual variety.
    ///
    /// Sweep matrix evaluated against `(distinct_tiles, max_streak)`
    /// across 4 seeds at 700-token generations on the iter-8 fast path:
    ///
    ///   top_k  top_p  rep_pen  win   distinct  max_streak
    ///     5    none    1.6     12       9         5
    ///     5    0.90    1.6     12      10         4
    ///     5    0.90    1.7     24      10         4   ← chosen
    ///     8    0.90    1.6     16      11         6
    ///
    /// The chosen config widens the no-repeat window to ~half a level row
    /// (50 cols / 2 = 25, rounded to 24) so that single-tile streaks
    /// don't span more than half a row, while top-p=0.9 trims the
    /// always-low-mass long tail.
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

pub struct MarioRetriever {
    pub corpus: Vec<u8>,
    /// Public so the `MarioDiffuser` (in this same example file) can read
    /// the embedding table without going through copy paths.
    pub w: Vec<f32>,
    /// Public for the same reason — diffusion builds its own forward calls.
    pub cfg: SparseAttentionConfig,
}

impl MarioRetriever {
    pub fn new(corpus: Vec<u8>, embedding_seed: u32) -> Self {
        // Non-causal so the last query position can reach the whole corpus
        // through window + log-stride + landmark hops. window=256 + log-stride
        // + landmarks gives ≈ 14% sparse coverage of a 2.8K-token combined
        // sequence, which is enough to recover bigram-grade statistics for
        // 15-token tile vocab.
        let cfg = SparseAttentionConfig {
            window: 256,
            block_size: 64,
            global_tokens: vec![0],
            causal: false,
            use_log_stride: true,
            use_landmarks: true,
            sort_candidates: false,
        };
        Self {
            corpus,
            w: make_embedding_matrix(embedding_seed),
            cfg,
        }
    }

    /// Build a [seq, 1, HEAD_DIM] tensor where row i = embed(token[i]) +
    /// POS_SCALE · pos(i). Token match dominates softmax (matched dot-product
    /// = d after /sqrt(d) → exp(sqrt(d))) but positional similarity nudges
    /// retrieval toward corpus positions at comparable level-depth.
    /// If `shift_for_value`, encodes token[i+1] for the V tensor (the
    /// empirical "next-token" supervision baked into V).
    fn make_row_tensor(&self, tokens: &[u8], shift_for_value: bool) -> Tensor3 {
        let seq = tokens.len();
        let mut t = Tensor3::zeros(seq, N_HEADS, HEAD_DIM);
        let mut pos = vec![0.0f32; HEAD_DIM];
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
            let emb = token_embedding(tok, &self.w);
            pos_encoding_into(i, HEAD_DIM, &mut pos);
            let row = t.row_mut(i, 0);
            for d in 0..HEAD_DIM {
                row[d] = emb[d] + POS_SCALE * pos[d];
            }
        }
        t
    }

    /// Compute logits over VOCAB_SIZE for the next token after `prefix`.
    pub fn next_token_logits(&self, prefix: &[u8]) -> [f32; VOCAB_SIZE] {
        let mut combined = self.corpus.clone();
        combined.extend_from_slice(prefix);
        let q = self.make_row_tensor(&combined, false);
        let v = self.make_row_tensor(&combined, true);
        let attn = SubquadraticSparseAttention::new(self.cfg.clone()).expect("config");
        let out = attn.forward(&q, &q, &v).expect("attention");
        let last = combined.len() - 1;
        let mut logits = [0.0f32; VOCAB_SIZE];
        for v_idx in 0..VOCAB_SIZE {
            let emb = token_embedding(v_idx as u8, &self.w);
            let mut dot = 0.0f32;
            for d in 0..HEAD_DIM {
                dot += out.get(last, 0, d) * emb[d];
            }
            logits[v_idx] = dot;
        }
        logits
    }

    pub fn generate(
        &self,
        prefix: &[u8],
        n: usize,
        sampling: &SamplingConfig,
        sampler_seed: u32,
    ) -> Vec<u8> {
        let mut state = sampler_seed.max(1);
        let mut out = prefix.to_vec();
        for _ in 0..n {
            let logits = self.next_token_logits(&out);
            let win = sampling.no_repeat_window.min(out.len());
            let recent = &out[out.len() - win..];
            let next = sample_logits(&logits, sampling, recent, &mut state);
            out.push(next);
        }
        out
    }

    /// Build a single-row [1, 1, HEAD_DIM] tensor with embed(tok) + POS_SCALE·pos(abs_pos).
    fn build_kv_row(&self, tok: u8, abs_pos: usize) -> Tensor3 {
        let mut data = vec![0.0f32; HEAD_DIM];
        let emb = token_embedding(tok, &self.w);
        let mut pos = vec![0.0f32; HEAD_DIM];
        pos_encoding_into(abs_pos, HEAD_DIM, &mut pos);
        for d in 0..HEAD_DIM {
            data[d] = emb[d] + POS_SCALE * pos[d];
        }
        Tensor3::from_vec(data, 1, N_HEADS, HEAD_DIM).unwrap()
    }

    /// Incremental generation via `KvCache` + `decode_step`. Pre-fills the
    /// cache once with corpus and prefix tensors (V shifted by one, zero for
    /// the last prefix position whose successor isn't known yet), then issues
    /// **one decode_step per generated token** — O(log T) per step instead
    /// of O(N log N) — yielding ~100× wall-clock speedup over `generate` at
    /// the example's 14×50 grid.
    ///
    /// V[generated_position] is left as zero on append (we never know the
    /// successor of a freshly-sampled token), so attention to generated
    /// positions contributes no value-signal back into the next decode.
    /// Effect: the model retrieves only from the corpus + initial prefix —
    /// pure bigram retrieval, no self-feedback. For our scale this is the
    /// right behaviour; for richer denoisers you'd back-fill V on each
    /// step (and rebuild landmarks).
    pub fn generate_fast(
        &self,
        prefix: &[u8],
        n: usize,
        sampling: &SamplingConfig,
        sampler_seed: u32,
    ) -> Vec<u8> {
        let mut state = sampler_seed.max(1);
        let cap = self.corpus.len() + prefix.len() + n + 16;
        let mut cache = KvCache::new(cap, N_HEADS, HEAD_DIM, self.cfg.block_size);
        let attn = SubquadraticSparseAttention::new(self.cfg.clone()).expect("config");
        let zero_v = Tensor3::zeros(1, N_HEADS, HEAD_DIM);

        // Pre-fill corpus with V_shifted: V[i] = embed(corpus[i+1]) + pos(i).
        // For the last corpus position, V successor is the first prefix
        // token (which truly *does* follow corpus in the combined stream).
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

        // Pre-fill prefix with V_shifted; the last prefix position has V=zero
        // because its successor is what we're about to generate.
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
            // Q = K of the most recently appended position. The kernel's
            // decode_step semantics: "the new token is at cache.len - 1".
            let last_idx = cache.len - 1;
            let last_tok = sequence.last().copied().unwrap_or(0);
            let q = self.build_kv_row(last_tok, last_idx);

            let out = attn.decode_step(&q, &cache).expect("decode");

            let mut logits = [0.0f32; VOCAB_SIZE];
            for v_idx in 0..VOCAB_SIZE {
                let v_emb = token_embedding(v_idx as u8, &self.w);
                let mut dot = 0.0f32;
                for d in 0..HEAD_DIM {
                    dot += out.get(0, 0, d) * v_emb[d];
                }
                logits[v_idx] = dot;
            }

            let win = sampling.no_repeat_window.min(sequence.len());
            let recent = &sequence[sequence.len() - win..];
            let next = sample_logits(&logits, sampling, recent, &mut state);

            // Append the freshly-sampled token with V=zero (its successor is
            // the next thing we'll generate, not yet known). Future decodes
            // skip this V's contribution; landmarks still update on K-side.
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

fn sample_logits(
    logits: &[f32; VOCAB_SIZE],
    cfg: &SamplingConfig,
    recent: &[u8],
    state: &mut u32,
) -> u8 {
    let mut adjusted = *logits;

    // Repetition penalty over the recent window — HuggingFace-style
    // (positive logits divided, negative multiplied).
    if cfg.repetition_penalty > 1.0 + f32::EPSILON && !recent.is_empty() {
        let pen = cfg.repetition_penalty;
        for &t in recent {
            let i = t as usize;
            if i < VOCAB_SIZE {
                adjusted[i] = if adjusted[i] > 0.0 {
                    adjusted[i] / pen
                } else {
                    adjusted[i] * pen
                };
            }
        }
    }

    // Top-k mask — set the rest to -inf so the softmax ignores them.
    if cfg.top_k > 0 && cfg.top_k < VOCAB_SIZE {
        let mut idx: [usize; VOCAB_SIZE] = [0; VOCAB_SIZE];
        for i in 0..VOCAB_SIZE {
            idx[i] = i;
        }
        idx.sort_unstable_by(|&a, &b| {
            adjusted[b]
                .partial_cmp(&adjusted[a])
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        let kth = adjusted[idx[cfg.top_k - 1]];
        for v in 0..VOCAB_SIZE {
            if adjusted[v] < kth {
                adjusted[v] = f32::NEG_INFINITY;
            }
        }
    }

    // Top-p (nucleus) mask — keep the smallest set of tokens whose
    // cumulative softmax probability >= top_p. Applied AFTER top-k so the
    // two mechanisms compose: top-k caps the candidate count, top-p trims
    // the low-mass tail of whatever survives.
    if cfg.top_p > 0.0 && cfg.top_p < 1.0 {
        let temp_p = cfg.temperature.max(1e-3);
        let max_l = adjusted.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        // (idx, prob) pairs from the *current* adjusted logits.
        let mut pairs: [(usize, f32); VOCAB_SIZE] = [(0, 0.0); VOCAB_SIZE];
        let mut total = 0.0f32;
        for i in 0..VOCAB_SIZE {
            let p = if adjusted[i].is_finite() {
                ((adjusted[i] - max_l) / temp_p).exp()
            } else {
                0.0
            };
            pairs[i] = (i, p);
            total += p;
        }
        if total > 0.0 {
            for pr in pairs.iter_mut() {
                pr.1 /= total;
            }
            pairs.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal)
            });
            let mut keep = [false; VOCAB_SIZE];
            let mut cum = 0.0f32;
            for &(idx, p) in pairs.iter() {
                keep[idx] = true;
                cum += p;
                if cum >= cfg.top_p {
                    break;
                }
            }
            for v in 0..VOCAB_SIZE {
                if !keep[v] {
                    adjusted[v] = f32::NEG_INFINITY;
                }
            }
        }
    }

    let temp = cfg.temperature.max(1e-3);
    let max_l = adjusted.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs = [0.0f32; VOCAB_SIZE];
    let mut sum = 0.0f32;
    for i in 0..VOCAB_SIZE {
        if adjusted[i].is_finite() {
            probs[i] = ((adjusted[i] - max_l) / temp).exp();
            sum += probs[i];
        }
    }
    if sum <= 0.0 {
        return 0; // fallback to sky
    }
    for p in probs.iter_mut() {
        *p /= sum;
    }
    let r = next_uniform(state);
    let mut acc = 0.0f32;
    for i in 0..VOCAB_SIZE {
        acc += probs[i];
        if r < acc {
            return i as u8;
        }
    }
    (VOCAB_SIZE - 1) as u8
}

/// Render a flat token stream as ASCII (newline tokens already encode rows).
pub fn render_level(tokens: &[u8]) -> String {
    tokens.iter().map(|&t| decode_token(t)).collect()
}

// =================================================================
// Iter 13 — comparison baselines
//
// Two reference pipelines that aren't built on `ruvllm_sparse_attention`
// at all. They give us "what does this metric mean for a trivial vs a
// classical generator?" so the AR / diffusion numbers from iter 11
// have a context.
// =================================================================

/// Uniform-random baseline: each tile is drawn IID from the full vocab.
/// Lower bound on every quality dimension.
pub fn uniform_random_generate(n: usize, seed: u32) -> Vec<u8> {
    let mut state = seed.max(1);
    let v = VOCAB_SIZE as u32;
    (0..n)
        .map(|_| (xorshift32(&mut state) % v) as u8)
        .collect()
}

/// First-order Markov chain over the embedded corpus — the classical
/// non-neural bigram baseline. Exact P(next | curr), no embeddings, no
/// attention. This is roughly what AR Sparse-Mario approximates via
/// random embeddings + attention; the gap between the two tells you
/// what the attention-as-memory machinery costs in metric distance.
pub struct Markov1 {
    /// `cum_probs[v]` is a sorted-by-cumulative-probability list
    /// `[(next_token, cum_prob)]` covering token `v`'s outgoing
    /// distribution. Sampling: draw u ∈ [0,1), pick first cum ≥ u.
    cum_probs: Vec<Vec<(u8, f32)>>,
}

impl Markov1 {
    pub fn from_corpus(corpus: &[u8]) -> Self {
        let v = VOCAB_SIZE;
        let mut counts = vec![vec![0u32; v]; v];
        for w in corpus.windows(2) {
            let a = w[0] as usize;
            let b = w[1] as usize;
            if a < v && b < v {
                counts[a][b] += 1;
            }
        }
        let mut cum_probs = Vec::with_capacity(v);
        for from in 0..v {
            let total: u32 = counts[from].iter().sum();
            let row = if total == 0 {
                // Fallback: uniform over the vocab.
                (0..v)
                    .map(|t| (t as u8, (t as f32 + 1.0) / v as f32))
                    .collect()
            } else {
                let mut cum = 0.0f32;
                (0..v)
                    .map(|t| {
                        cum += counts[from][t] as f32 / total as f32;
                        (t as u8, cum)
                    })
                    .collect()
            };
            cum_probs.push(row);
        }
        Self { cum_probs }
    }

    pub fn generate(&self, prefix: &[u8], n: usize, seed: u32) -> Vec<u8> {
        let mut state = seed.max(1);
        let mut out = prefix.to_vec();
        if out.is_empty() {
            out.push(0); // sky as default seed
        }
        for _ in 0..n {
            let last = *out.last().unwrap();
            let row = &self.cum_probs[last as usize];
            let r = next_uniform(&mut state);
            let mut chosen = row[row.len() - 1].0;
            for &(t, cum) in row.iter() {
                if r < cum {
                    chosen = t;
                    break;
                }
            }
            out.push(chosen);
        }
        out
    }
}

// =================================================================
// Iter 11 — PCG metrics
//
// Implements the standard quantitative measures for procedurally-
// generated Mario levels: density, linearity, leniency, novelty, and
// a coarse playability proxy. Mirrors the families used in
// Snodgrass et al. and the MarioGAN evaluation suite, simplified to
// the VGLC tile alphabet this demo embeds.
//
// Each metric is a single f32, larger / smaller doesn't necessarily
// mean "better" — they're descriptors that let us compare AR,
// diffusion, and corpus on the same axes.
// =================================================================

#[derive(Clone, Debug)]
pub struct LevelMetrics {
    /// Fraction of non-sky, non-newline tiles in the level.
    pub density: f32,
    /// Std-dev of the topmost-ground row index across columns.
    /// Higher = jaggier ground profile, lower = flatter.
    pub linearity: f32,
    /// (hostile + gaps - friendly) / columns. Higher = harder level.
    /// Hostile: enemies, cannons. Friendly: ?-blocks, coins.
    /// Gap: column with no ground tile anywhere.
    pub leniency: f32,
    /// Min normalised Hamming distance from `grid` to any same-shape
    /// window of any embedded corpus level. 0.0 = byte-identical to
    /// some corpus slice; 1.0 = no shared tile at any aligned cell.
    pub novelty: f32,
    /// Fraction of columns where there is at least one ground tile in
    /// the lower third — proxy for "Mario has somewhere to stand".
    pub playable_columns: f32,
}

/// Convert a flat token stream into a `rows × cols` grid by either
/// honouring embedded `\n` tokens (row break) or hard-wrapping at `cols`.
/// Out-of-range positions default to sky.
pub fn tokens_to_grid(tokens: &[u8], cols: usize, rows: usize) -> Vec<Vec<u8>> {
    let nl = encode_char('\n').unwrap();
    let sky = encode_char('-').unwrap();
    let mut grid = vec![vec![sky; cols]; rows];
    let mut r = 0usize;
    let mut c = 0usize;
    for &t in tokens {
        if r >= rows {
            break;
        }
        if t == nl {
            r += 1;
            c = 0;
            continue;
        }
        if c == cols {
            r += 1;
            c = 0;
            if r >= rows {
                break;
            }
        }
        if (t as usize) < VOCAB.len() {
            grid[r][c] = t;
        }
        c += 1;
    }
    grid
}

fn density(grid: &[Vec<u8>]) -> f32 {
    let sky = encode_char('-').unwrap();
    let nl = encode_char('\n').unwrap();
    let mut total = 0usize;
    let mut nonsky = 0usize;
    for row in grid {
        for &t in row {
            if t == nl {
                continue;
            }
            total += 1;
            if t != sky {
                nonsky += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        nonsky as f32 / total as f32
    }
}

fn linearity(grid: &[Vec<u8>]) -> f32 {
    let ground = encode_char('X').unwrap();
    let rows = grid.len();
    let cols = grid.first().map(|r| r.len()).unwrap_or(0);
    if rows == 0 || cols == 0 {
        return 0.0;
    }
    let mut heights = Vec::with_capacity(cols);
    for c in 0..cols {
        let mut h = rows; // bottomless = rows (max row index + 1)
        for r in 0..rows {
            if grid[r][c] == ground {
                h = r;
                break;
            }
        }
        heights.push(h as f32);
    }
    let mean: f32 = heights.iter().sum::<f32>() / heights.len() as f32;
    let var: f32 = heights.iter().map(|&h| (h - mean).powi(2)).sum::<f32>() / heights.len() as f32;
    var.sqrt()
}

fn leniency(grid: &[Vec<u8>]) -> f32 {
    let enemy = encode_char('E').unwrap();
    let cannon = encode_char('B').unwrap();
    let q_block = encode_char('?').unwrap();
    let coin = encode_char('o').unwrap();
    let ground = encode_char('X').unwrap();
    let cols = grid.first().map(|r| r.len()).unwrap_or(0);
    let rows = grid.len();
    if cols == 0 {
        return 0.0;
    }
    let mut hostile = 0i32;
    let mut friendly = 0i32;
    for row in grid {
        for &t in row {
            if t == enemy || t == cannon {
                hostile += 1;
            }
            if t == q_block || t == coin {
                friendly += 1;
            }
        }
    }
    let mut gaps = 0i32;
    for c in 0..cols {
        if !(0..rows).any(|r| grid[r][c] == ground) {
            gaps += 1;
        }
    }
    ((hostile + gaps) - friendly) as f32 / cols as f32
}

fn novelty(grid: &[Vec<u8>]) -> f32 {
    let rows = grid.len();
    let cols = grid.first().map(|r| r.len()).unwrap_or(0);
    if rows == 0 || cols == 0 {
        return 1.0;
    }
    let total = (rows * cols) as f32;
    let mut best_diff = total as usize;
    for lvl in LEVELS.iter() {
        let lvl_rows: Vec<Vec<char>> = lvl.lines().map(|l| l.chars().collect()).collect();
        if lvl_rows.is_empty() {
            continue;
        }
        let lr = lvl_rows.len();
        let lc = lvl_rows[0].len();
        if lr < rows || lc < cols {
            continue;
        }
        for sr in 0..=lr - rows {
            for sc in 0..=lc - cols {
                let mut diff = 0usize;
                for r in 0..rows {
                    for c in 0..cols {
                        let l_ch = lvl_rows[sr + r][sc + c];
                        let g_ch = decode_token(grid[r][c]);
                        if l_ch != g_ch {
                            diff += 1;
                        }
                    }
                }
                if diff < best_diff {
                    best_diff = diff;
                }
            }
        }
    }
    best_diff as f32 / total
}

fn playable_columns(grid: &[Vec<u8>]) -> f32 {
    let ground = encode_char('X').unwrap();
    let rows = grid.len();
    let cols = grid.first().map(|r| r.len()).unwrap_or(0);
    if rows == 0 || cols == 0 {
        return 0.0;
    }
    let lower_start = (rows * 2) / 3;
    let mut ok = 0usize;
    for c in 0..cols {
        if (lower_start..rows).any(|r| grid[r][c] == ground) {
            ok += 1;
        }
    }
    ok as f32 / cols as f32
}

/// Compute all five metrics on a flat token stream interpreted as a
/// `rows × cols` grid via `tokens_to_grid`.
pub fn compute_metrics(tokens: &[u8], cols: usize, rows: usize) -> LevelMetrics {
    let grid = tokens_to_grid(tokens, cols, rows);
    LevelMetrics {
        density: density(&grid),
        linearity: linearity(&grid),
        leniency: leniency(&grid),
        novelty: novelty(&grid),
        playable_columns: playable_columns(&grid),
    }
}

/// "Centre of corpus" target — median across the embedded slices.
/// Iter 11 measured: density 0.24/0.36/0.30, linearity 0.0/0.33/1.39,
/// leniency −0.04/−0.04/0.30, playable 1.0/1.0/0.86. Medians used for
/// the tuning target so iter 12 has a single L2 distance to minimise.
pub fn corpus_target() -> LevelMetrics {
    LevelMetrics {
        density: 0.30,
        linearity: 0.33,
        leniency: -0.04,
        novelty: 0.0,
        playable_columns: 1.0,
    }
}

/// L2 distance from `m` to the corpus target on the four non-trivial
/// axes. Novelty is excluded — by construction it's 0 for corpus and
/// positive for anything else, and we *want* >0 novelty in generated
/// output (no point penalising it).
pub fn metric_distance(m: &LevelMetrics, target: &LevelMetrics) -> f32 {
    let dd = m.density - target.density;
    let dl = m.linearity - target.linearity;
    let dle = m.leniency - target.leniency;
    let dp = m.playable_columns - target.playable_columns;
    (dd * dd + dl * dl + dle * dle + dp * dp).sqrt()
}

/// Render with a hard wrap every `cols` non-newline tiles. Repetition
/// penalty often suppresses the `\n` tile; this keeps the level visually
/// rectangular even when the model never emits a row break.
pub fn render_level_wrapped(tokens: &[u8], cols: usize) -> String {
    let nl_id = encode_char('\n').unwrap();
    let mut out = String::with_capacity(tokens.len() + tokens.len() / cols);
    let mut col = 0usize;
    for &t in tokens {
        if t == nl_id {
            out.push('\n');
            col = 0;
            continue;
        }
        if col == cols {
            out.push('\n');
            col = 0;
        }
        out.push(decode_token(t));
        col += 1;
    }
    out
}

// =================================================================
// Iter 7 — masked discrete diffusion
//
// Architecturally a real diffusion model: iterative denoising of a
// fully-masked grid, bidirectional context, confidence-scheduled
// unmasking — D3PM / MaskGIT-inference family. Unlike those, the
// "denoiser" is training-free: the sparse attention kernel acts as
// a bidirectional content-addressable memory over the same SMB
// corpus that the autoregressive Sparse-Mario uses.
//
// Forward step (one denoising pass):
//
//   K[i] = 0.5·(embed(left_neighbor(i)) + embed(right_neighbor(i)))
//          + 0.5·pos(i)                          ← bidirectional context
//   V[i] = embed(token_at_i)                     ← actual token (not shifted)
//   Q[j] = K[j]                                  ← what context does j sit in?
//   out  = SubquadraticSparseAttention.forward(Q, K, V)
//   logits[v] = out[j] · embed(v)
//
// At every masked position j, attention finds corpus positions whose
// left/right context matches j's, and reads back what token usually
// fills that context. We rank masked positions by softmax-max
// confidence and unmask the most-confident ones each step (cosine-ish
// schedule), so easy positions resolve first and provide stronger
// context for harder ones — exactly the MaskGIT inference pattern.
// =================================================================

/// Sentinel placed in the working sequence for not-yet-denoised positions.
/// Out of vocab range so any user-facing operation can detect it.
pub const MASK_SENTINEL: u8 = 255;

/// Per-offset weights for the diffuser's bidirectional K builder, indexed
/// by `offset - 1` (offset 1 = immediate neighbour, offset 2 = next-over).
///
/// Iter-10 honest finding from a 4-config A/B (4 random seeds, 300-token
/// generations, distinct-tile count): a heavy outer weight (≥0.20) collapses
/// per-seed diversity because random-embedding averaging pulls K toward the
/// corpus mean.  A light outer weight (0.10) keeps iter-7's behaviour
/// effectively intact while letting offset-2 tokens contribute K signal in
/// the cases where offset-1 is masked but offset-2 isn't (covered by the
/// `diffuser_uses_offset_2_context` test). Net effect on final tile mix
/// is small but the contract is now position-agnostic — the K builder no
/// longer goes silent at masked-immediate-neighbour positions.
const DIFFUSION_CONTEXT_WEIGHTS: &[f32] = &[0.5, 0.10];

pub struct MarioDiffuser<'a> {
    retriever: &'a MarioRetriever,
}

impl<'a> MarioDiffuser<'a> {
    pub fn new(retriever: &'a MarioRetriever) -> Self {
        Self { retriever }
    }

    /// Build the bidirectional K and V tensors for a given reference sequence.
    ///
    /// Iter-10 wider context: K[i] sums weighted embeddings over up to
    /// `DIFFUSION_CONTEXT_WEIGHTS.len()` tokens on each side (default
    /// radius=2 with weights [0.5, 0.25]). Each weight is applied
    /// per-side, so a position with all four bidirectional neighbours
    /// unmasked sees K = 0.5·(L1 + R1) + 0.25·(L2 + R2). Mask positions
    /// in the context window contribute zero to that side's sum.
    ///
    /// Why 4-token (radius 2) instead of 2-token (radius 1, iter-7):
    /// random-embedding K differs across positions only by the identity
    /// of their immediate neighbours; with radius 1 a single random-vector
    /// noise term per side can dominate the match. Radius 2 averages
    /// two co-occurrence orders and the matching-pattern signal grows
    /// roughly proportionally while noise scales as sqrt — sharper
    /// discrimination between corpus contexts.
    ///
    /// Note: no positional encoding here, unlike the autoregressive path.
    /// The diffuser appends the working sequence after the corpus, so
    /// adding pos(i) would bias working-position queries toward the
    /// *tail* of the corpus (the level-floor `XXXX` rows). Pure content
    /// match is what masked filling needs.
    pub fn make_bidir_kv(&self, seq: &[u8]) -> (Tensor3, Tensor3) {
        let n = seq.len();
        let mut k = Tensor3::zeros(n, N_HEADS, HEAD_DIM);
        let mut v = Tensor3::zeros(n, N_HEADS, HEAD_DIM);
        let zero = vec![0.0f32; HEAD_DIM];

        for i in 0..n {
            let krow = k.row_mut(i, 0);
            for slot in 0..DIFFUSION_CONTEXT_WEIGHTS.len() {
                let weight = DIFFUSION_CONTEXT_WEIGHTS[slot];
                let off = slot + 1;
                if i >= off && seq[i - off] != MASK_SENTINEL {
                    let emb = token_embedding(seq[i - off], &self.retriever.w);
                    for d in 0..HEAD_DIM {
                        krow[d] += weight * emb[d];
                    }
                }
                if i + off < n && seq[i + off] != MASK_SENTINEL {
                    let emb = token_embedding(seq[i + off], &self.retriever.w);
                    for d in 0..HEAD_DIM {
                        krow[d] += weight * emb[d];
                    }
                }
            }
            let vrow = v.row_mut(i, 0);
            if seq[i] != MASK_SENTINEL {
                let emb = token_embedding(seq[i], &self.retriever.w);
                vrow.copy_from_slice(emb);
            } else {
                vrow.copy_from_slice(&zero);
            }
        }
        (k, v)
    }

    /// One forward pass of the bidirectional retrieval denoiser. Returns,
    /// for every position in the working sequence (corpus + working), the
    /// vocab-projected logits.
    fn diffusion_logits(&self, working: &[u8]) -> Vec<[f32; VOCAB_SIZE]> {
        // Concatenate corpus (always unmasked, contributes signal) with the
        // working sequence (some masked).
        let mut combined: Vec<u8> = self.retriever.corpus.clone();
        combined.extend_from_slice(working);

        let (k, v) = self.make_bidir_kv(&combined);
        let q = k.clone();
        let attn = SubquadraticSparseAttention::new(self.retriever.cfg.clone()).expect("config");
        let out = attn.forward(&q, &q, &v).expect("attention");

        let prefix_start = self.retriever.corpus.len();
        let mut all = Vec::with_capacity(working.len());
        for i in 0..working.len() {
            let idx = prefix_start + i;
            let mut logits = [0.0f32; VOCAB_SIZE];
            for v_idx in 0..VOCAB_SIZE {
                let emb = token_embedding(v_idx as u8, &self.retriever.w);
                let mut dot = 0.0f32;
                for d in 0..HEAD_DIM {
                    dot += out.get(idx, 0, d) * emb[d];
                }
                logits[v_idx] = dot;
            }
            all.push(logits);
        }
        all
    }

    /// One denoising step: unmask the `keep_count` most-confident masked
    /// positions, sampling each from its own retrieval distribution.
    pub fn denoise_step(
        &self,
        working: &mut [u8],
        keep_count: usize,
        sampling: &SamplingConfig,
        state: &mut u32,
    ) {
        let masked: Vec<usize> = working
            .iter()
            .enumerate()
            .filter(|(_, &t)| t == MASK_SENTINEL)
            .map(|(i, _)| i)
            .collect();
        if masked.is_empty() || keep_count == 0 {
            return;
        }

        let logits = self.diffusion_logits(working);

        // Confidence = softmax-max at temperature 1 (no top-k / penalty —
        // those distort the ranking; we use them only at the sampling step).
        let confidence = |row: &[f32; VOCAB_SIZE]| -> f32 {
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
        for k_i in 0..n {
            let (j, _) = ranked[k_i];
            // Sample at the masked position's retrieval distribution.
            // Skip MASK_SENTINEL (never in vocab) — already excluded by
            // VOCAB_SIZE bound.
            let mut next = sample_logits(&logits[j], sampling, &[], state);
            // Defensive: if the sampler ever returns a non-vocab id, snap to sky.
            if (next as usize) >= VOCAB_SIZE {
                next = 0;
            }
            working[j] = next;
        }
    }

    /// Run the full masked-diffusion pipeline: start with a fully-masked
    /// sequence of length `n` and run `n_steps` denoising steps with a
    /// quadratic-decay schedule, then a final sweep to clear stragglers.
    pub fn diffuse(
        &self,
        n: usize,
        n_steps: usize,
        sampling: &SamplingConfig,
        seed: u32,
    ) -> Vec<u8> {
        let mut state = seed.max(1);
        let mut working = vec![MASK_SENTINEL; n];

        // Context boot. Copy a random contiguous slice of the corpus into a
        // random position in `working`. Without this boot, step-1 retrieval
        // has zero bidirectional context: K[j]=0 for every working j,
        // attention returns the average corpus V, and the random-embedding
        // noise floor picks one fixed-point token that dominates every
        // step thereafter. A *contiguous* corpus slice (rather than
        // scattered uniform samples) is critical for diversity: uniform
        // sampling pulls mostly the dominant token (sky / ground at 93% of
        // the corpus combined), while a 32-token contiguous slice contains
        // the full local mix — pipes, coins, enemies, brick blocks. This
        // is a smaller, simpler analogue of MaskGIT's trained prior
        // network: give the iterative refiner real content to retrieve
        // against in step 1.
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
            // MaskGIT cosine schedule — gamma(t) = cos(π/2 · (t+1)/T).
            // Holds back early (only a few positions unmask per step when
            // context is empty) and accelerates at the end (when most
            // positions already provide bidirectional context). The slow
            // start is critical: with all-masked initial state and no
            // context, sampling many positions in step 1 collapses to
            // whichever single token has highest base affinity.
            let frac = ((t + 1) as f32) / (n_steps as f32);
            let target_masked = (n as f32 * (core::f32::consts::FRAC_PI_2 * frac).cos()) as usize;
            let current_masked = working.iter().filter(|&&t| t == MASK_SENTINEL).count();
            let to_unmask = current_masked.saturating_sub(target_masked).max(1);
            self.denoise_step(&mut working, to_unmask, sampling, &mut state);
        }

        // Final sweep: clear any leftover masks (rounding can leave some).
        let remaining = working.iter().filter(|&&t| t == MASK_SENTINEL).count();
        if remaining > 0 {
            self.denoise_step(&mut working, remaining, sampling, &mut state);
        }
        working
    }
}

fn main() {
    let tokens = encode_corpus();
    let dist = tile_distribution(&tokens);

    println!("== Sparse-Mario corpus ==");
    println!("levels        : {}", LEVELS.len());
    println!("total tokens  : {}", tokens.len());
    println!("vocab size    : {}", VOCAB.len());
    println!("level widths  : {:?}", LEVELS.iter().map(|l| level_width(l)).collect::<Vec<_>>());
    println!();
    println!("Tile distribution:");
    let mut entries: Vec<_> = dist.iter().collect();
    entries.sort_by(|a, b| b.1.cmp(a.1));
    let total = tokens.len() as f64;
    for (c, n) in entries {
        let pct = (*n as f64 / total) * 100.0;
        let label = if *c == '\n' { "\\n".to_string() } else { c.to_string() };
        println!("  {:>3}  {:>5}  {:>5.1}%", label, n, pct);
    }

    // ---------- iter 2: retrieval generation ----------
    println!();
    println!("== Sparse-attention retrieval generation ==");
    let retriever = MarioRetriever::new(tokens.clone(), 0x4D41_5249); // "MARI"
    let row_w = 50 + 1; // 50 cols + newline
    let n_rows = 14;
    let n_gen = row_w * n_rows;

    // Seed with a level-shaped fragment so the bigram chain has somewhere to
    // go besides "sky after sky → sky forever". Mario start + ground row +
    // newline + sky gives the retrieval bigrams from several distinct contexts.
    let seed_chars: Vec<u8> = "M-XXXXX\n--------\n"
        .chars()
        .filter_map(encode_char)
        .collect();
    let sampling = SamplingConfig::quality();

    // Iter 8: fast path via KvCache + decode_step (one O(log T) call per
    // token instead of one O(N log N) full forward). Old `generate()`
    // remains available for comparison.
    let t0 = std::time::Instant::now();
    let generated = retriever.generate_fast(&seed_chars, n_gen, &sampling, 0xC0FF_EE42);
    let dt = t0.elapsed();
    let rendered = render_level_wrapped(&generated, 50);

    println!("seed prefix   : {:?}", seed_chars);
    println!(
        "sampling      : top_k={} top_p={} rep_penalty={} window={} temp={}",
        sampling.top_k,
        sampling.top_p,
        sampling.repetition_penalty,
        sampling.no_repeat_window,
        sampling.temperature
    );
    println!("generated     : {} tokens in {:.2?} (KvCache + decode_step)", n_gen, dt);
    println!();
    println!("{}", rendered);
    println!();

    let gen_only = &generated[seed_chars.len()..];
    let gen_dist = tile_distribution(gen_only);
    let pct_of = |dist: &HashMap<char, usize>, total: usize, c: char| -> f64 {
        *dist.get(&c).unwrap_or(&0) as f64 / total as f64 * 100.0
    };
    println!(
        "tile mix in generated: sky {:.1}%  ground {:.1}%  brick {:.1}%  enemy {:.1}%  newline {:.1}%",
        pct_of(&gen_dist, gen_only.len(), '-'),
        pct_of(&gen_dist, gen_only.len(), 'X'),
        pct_of(&gen_dist, gen_only.len(), 'S'),
        pct_of(&gen_dist, gen_only.len(), 'E'),
        pct_of(&gen_dist, gen_only.len(), '\n')
    );

    // ---------- iter 7: masked discrete diffusion ----------
    println!();
    println!("== Sparse-attention masked discrete diffusion ==");
    let diffuser = MarioDiffuser::new(&retriever);
    let n_diff = 50 * 14; // 14×50 grid, fully masked at start
    // Iter 12 sweep winner: 24 denoising steps (vs the iter 7 default of 16)
    // gave the lowest avg L2 distance to corpus across 3 seeds:
    //   steps=16  0.746      steps=24  0.723   steps=32  0.798
    // 24 is the cosine-schedule sweet-spot — enough late-stage steps for
    // bidirectional context to settle, without spending budget on a flat
    // tail.
    let n_steps = 24;
    let t0 = std::time::Instant::now();
    let diffused = diffuser.diffuse(n_diff, n_steps, &sampling, 0xD1FF_5008);
    let dt = t0.elapsed();
    let any_masks = diffused.iter().any(|&t| t == MASK_SENTINEL);
    println!(
        "diffusion     : {} positions × {} denoising steps in {:.2?} (residual masks: {})",
        n_diff, n_steps, dt, any_masks
    );
    println!();
    println!("{}", render_level_wrapped(&diffused, 50));
    println!();
    let diff_dist = tile_distribution(&diffused);
    println!(
        "tile mix in diffused : sky {:.1}%  ground {:.1}%  brick {:.1}%  enemy {:.1}%  pipe {:.1}%",
        pct_of(&diff_dist, diffused.len(), '-'),
        pct_of(&diff_dist, diffused.len(), 'X'),
        pct_of(&diff_dist, diffused.len(), 'S'),
        pct_of(&diff_dist, diffused.len(), 'E'),
        pct_of(&diff_dist, diffused.len(), '<')
            + pct_of(&diff_dist, diffused.len(), '>')
            + pct_of(&diff_dist, diffused.len(), '[')
            + pct_of(&diff_dist, diffused.len(), ']'),
    );

    // ---------- iter 11: PCG metrics baseline ----------
    println!();
    println!("== PCG metrics baseline (3 seeds × {{AR, diffusion}}) ==");
    println!(
        "{:<14} {:>8} {:>10} {:>9} {:>8} {:>10}",
        "config", "density", "linearity", "leniency", "novelty", "playable"
    );
    let seeds: [u32; 3] = [0xC0FF_EE42, 0xBADD_F00D, 0x1337_BEEF];
    let cols = 50usize;
    let rows = 14usize;
    let n_total = cols * rows;

    for &s in &seeds {
        let toks = retriever.generate_fast(&seed_chars, n_total - seed_chars.len(), &sampling, s);
        let m = compute_metrics(&toks, cols, rows);
        println!(
            "AR seed={:08x}  {:>7.3} {:>10.3} {:>9.3} {:>8.3} {:>10.3}",
            s, m.density, m.linearity, m.leniency, m.novelty, m.playable_columns
        );
    }
    for &s in &seeds {
        let toks = diffuser.diffuse(n_total, n_steps, &sampling, s);
        let m = compute_metrics(&toks, cols, rows);
        println!(
            "DIFF seed={:08x} {:>7.3} {:>10.3} {:>9.3} {:>8.3} {:>10.3}",
            s, m.density, m.linearity, m.leniency, m.novelty, m.playable_columns
        );
    }

    // Corpus baseline — compute the same five metrics on each embedded
    // level slice so we can read AR/diffusion numbers as deltas from
    // "real Mario".
    for (i, lvl) in LEVELS.iter().enumerate() {
        let toks = encode_level(lvl);
        let m = compute_metrics(&toks, cols, rows);
        println!(
            "CORPUS slice {} {:>7.3} {:>10.3} {:>9.3} {:>8.3} {:>10.3}",
            i, m.density, m.linearity, m.leniency, m.novelty, m.playable_columns
        );
    }

    // ---------- iter 12: hyperparameter A/B against the corpus target ----------
    println!();
    println!("== Iter 12 hyperparameter sweep (avg L2 distance to corpus target) ==");
    let target = corpus_target();

    let alt_high_rep = SamplingConfig {
        repetition_penalty: 2.0,
        no_repeat_window: 40,
        ..SamplingConfig::quality()
    };
    let alt_low_temp = SamplingConfig {
        temperature: 0.6,
        ..SamplingConfig::quality()
    };
    let alt_loose_p = SamplingConfig {
        top_p: 0.95,
        top_k: 8,
        ..SamplingConfig::quality()
    };

    let ar_configs: [(&str, &SamplingConfig); 4] = [
        ("AR quality", &sampling),
        ("AR high_rep", &alt_high_rep),
        ("AR low_temp", &alt_low_temp),
        ("AR loose_p", &alt_loose_p),
    ];

    for (name, cfg) in ar_configs.iter() {
        let mut total = 0.0f32;
        for &s in &seeds {
            let toks =
                retriever.generate_fast(&seed_chars, n_total - seed_chars.len(), cfg, s);
            let m = compute_metrics(&toks, cols, rows);
            total += metric_distance(&m, &target);
        }
        let avg = total / seeds.len() as f32;
        println!("  {:<14} avg distance = {:.3}", name, avg);
    }

    let diffusion_steps_to_try = [16usize, 24, 32];
    for &steps in &diffusion_steps_to_try {
        let mut total = 0.0f32;
        for &s in &seeds {
            let toks = diffuser.diffuse(n_total, steps, &sampling, s);
            let m = compute_metrics(&toks, cols, rows);
            total += metric_distance(&m, &target);
        }
        let avg = total / seeds.len() as f32;
        println!("  DIFF steps={:<2}     avg distance = {:.3}", steps, avg);
    }

    // ---------- iter 13: cross-baseline comparison ----------
    println!();
    println!("== Iter 13 cross-baseline comparison (avg over 3 seeds) ==");
    println!(
        "{:<22} {:>8} {:>10} {:>9} {:>8} {:>10} {:>10}",
        "pipeline", "density", "linearity", "leniency", "novelty", "playable", "L2_dist"
    );
    let markov = Markov1::from_corpus(&tokens);

    let mut summarise = |name: &str, gens: &[Vec<u8>]| {
        let mut acc_d = 0.0f32;
        let mut acc_l = 0.0f32;
        let mut acc_le = 0.0f32;
        let mut acc_n = 0.0f32;
        let mut acc_p = 0.0f32;
        let mut acc_dist = 0.0f32;
        for g in gens {
            let m = compute_metrics(g, cols, rows);
            acc_d += m.density;
            acc_l += m.linearity;
            acc_le += m.leniency;
            acc_n += m.novelty;
            acc_p += m.playable_columns;
            acc_dist += metric_distance(&m, &target);
        }
        let n = gens.len() as f32;
        println!(
            "{:<22} {:>8.3} {:>10.3} {:>9.3} {:>8.3} {:>10.3} {:>10.3}",
            name,
            acc_d / n,
            acc_l / n,
            acc_le / n,
            acc_n / n,
            acc_p / n,
            acc_dist / n
        );
    };

    let ar_gens: Vec<Vec<u8>> = seeds
        .iter()
        .map(|&s| retriever.generate_fast(&seed_chars, n_total - seed_chars.len(), &sampling, s))
        .collect();
    let diff_gens: Vec<Vec<u8>> = seeds
        .iter()
        .map(|&s| diffuser.diffuse(n_total, n_steps, &sampling, s))
        .collect();
    let unif_gens: Vec<Vec<u8>> = seeds
        .iter()
        .map(|&s| uniform_random_generate(n_total, s))
        .collect();
    let markov_gens: Vec<Vec<u8>> = seeds
        .iter()
        .map(|&s| markov.generate(&seed_chars, n_total - seed_chars.len(), s))
        .collect();
    let corpus_gens: Vec<Vec<u8>> = LEVELS.iter().map(|l| encode_level(l)).collect();

    summarise("Sparse-Mario AR", &ar_gens);
    summarise("Sparse-Mario diffusion", &diff_gens);
    summarise("Markov-1 (corpus bigram)", &markov_gens);
    summarise("Uniform random", &unif_gens);
    summarise("Corpus (target)", &corpus_gens);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_roundtrip() {
        for (i, &c) in VOCAB.iter().enumerate() {
            assert_eq!(encode_char(c), Some(i as u8));
            assert_eq!(decode_token(i as u8), c);
        }
    }

    #[test]
    fn corpus_nonempty_and_known_tiles() {
        let toks = encode_corpus();
        assert!(toks.len() > 1000, "corpus should have at least 1k tokens, got {}", toks.len());
        for &t in &toks {
            assert!((t as usize) < VOCAB.len(), "out-of-range token {}", t);
        }
    }

    #[test]
    fn each_level_has_mario_start() {
        for (i, lvl) in LEVELS.iter().enumerate() {
            assert!(lvl.contains('M'), "level {} missing mario start tile", i);
        }
    }

    #[test]
    fn each_level_has_ground_floor() {
        for (i, lvl) in LEVELS.iter().enumerate() {
            let last = lvl.lines().last().unwrap_or("");
            let solid = last.chars().filter(|&c| c == 'X').count();
            assert!(
                solid > last.chars().count() / 2,
                "level {} bottom row should be mostly ground",
                i
            );
        }
    }

    #[test]
    fn levels_are_rectangular() {
        for (i, lvl) in LEVELS.iter().enumerate() {
            let w = level_width(lvl);
            for (r, row) in lvl.lines().enumerate() {
                assert_eq!(
                    row.chars().count(), w,
                    "level {} row {} width mismatch (expected {}, got {})",
                    i, r, w, row.chars().count()
                );
            }
        }
    }

    // ---------- iter 2 tests ----------

    #[test]
    fn embedding_matrix_deterministic() {
        let a = make_embedding_matrix(0x1234);
        let b = make_embedding_matrix(0x1234);
        assert_eq!(a, b);
        assert_eq!(a.len(), VOCAB_SIZE * HEAD_DIM);
    }

    #[test]
    fn next_token_logits_finite() {
        let r = MarioRetriever::new(encode_corpus(), 0xABCD);
        let prefix: Vec<u8> = "----X".chars().filter_map(encode_char).collect();
        let logits = r.next_token_logits(&prefix);
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "non-finite logit at vocab idx {}: {}", i, l);
        }
    }

    #[test]
    fn generate_is_deterministic() {
        let r1 = MarioRetriever::new(encode_corpus(), 0xABCD);
        let r2 = MarioRetriever::new(encode_corpus(), 0xABCD);
        let p: Vec<u8> = "--".chars().filter_map(encode_char).collect();
        let cfg = SamplingConfig {
            temperature: 0.8,
            ..SamplingConfig::default()
        };
        let a = r1.generate(&p, 64, &cfg, 0xDEAD_BEEF);
        let b = r2.generate(&p, 64, &cfg, 0xDEAD_BEEF);
        assert_eq!(a, b, "same seed should give same output");
        assert_eq!(a.len(), p.len() + 64);
    }

    #[test]
    fn generated_tiles_are_in_vocab() {
        let r = MarioRetriever::new(encode_corpus(), 0xABCD);
        let p: Vec<u8> = "--".chars().filter_map(encode_char).collect();
        let out = r.generate(&p, 200, &SamplingConfig::default(), 0x4242);
        for &t in &out {
            assert!((t as usize) < VOCAB.len(), "out-of-vocab token {}", t);
        }
    }

    #[test]
    fn generated_distribution_is_corpus_like() {
        // With low temperature and no top-k, retrieval biases hard toward the
        // dominant bigram — most tiles end up sky / ground / newline.
        let r = MarioRetriever::new(encode_corpus(), 0xABCD);
        let p: Vec<u8> = "----".chars().filter_map(encode_char).collect();
        let cfg = SamplingConfig {
            temperature: 0.5,
            ..SamplingConfig::default()
        };
        let out = r.generate(&p, 300, &cfg, 0x9001);
        let gen = &out[p.len()..];
        let dist = tile_distribution(gen);
        let sky_or_ground = *dist.get(&'-').unwrap_or(&0)
            + *dist.get(&'X').unwrap_or(&0)
            + *dist.get(&'\n').unwrap_or(&0);
        let frac = sky_or_ground as f64 / gen.len() as f64;
        assert!(
            frac > 0.7,
            "expected >70% sky/ground/newline, got {:.1}%",
            frac * 100.0
        );
    }

    // ---------- iter 5 tests ----------

    #[test]
    fn quality_config_is_more_diverse() {
        // The quality sampling config (top-k + repetition penalty) should
        // produce a strictly higher unique-tile count over a long generation
        // than bare softmax — the whole point of iter 5.
        let r = MarioRetriever::new(encode_corpus(), 0xABCD);
        let p: Vec<u8> = "M-XXXXX\n".chars().filter_map(encode_char).collect();

        let bare = r.generate(&p, 400, &SamplingConfig::default(), 0xBEEF);
        let qual = r.generate(&p, 400, &SamplingConfig::quality(), 0xBEEF);

        let unique = |toks: &[u8]| -> usize {
            let mut s = std::collections::HashSet::new();
            for &t in toks {
                s.insert(t);
            }
            s.len()
        };
        let bare_unique = unique(&bare[p.len()..]);
        let qual_unique = unique(&qual[p.len()..]);
        assert!(
            qual_unique > bare_unique,
            "quality config should produce more distinct tiles than bare softmax \
             (bare={}, quality={})",
            bare_unique,
            qual_unique
        );
        assert!(
            qual_unique >= 5,
            "quality config should hit at least 5 distinct tiles, got {}",
            qual_unique
        );
    }

    #[test]
    fn top_k_mask_restricts_sampling() {
        // With top_k=1 the sampler is greedy and deterministic across seeds.
        let r = MarioRetriever::new(encode_corpus(), 0xABCD);
        let p: Vec<u8> = "X-".chars().filter_map(encode_char).collect();
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_k: 1,
            ..SamplingConfig::default()
        };
        let a = r.generate(&p, 32, &cfg, 0x1111);
        let b = r.generate(&p, 32, &cfg, 0x2222);
        assert_eq!(a, b, "top_k=1 should be greedy regardless of sampler seed");
    }

    #[test]
    fn render_level_wrapped_rectangular() {
        // No embedded newlines — every row should be exactly `cols` chars.
        let toks: Vec<u8> = (0..200).map(|i| (i % 3) as u8).collect();
        let s = render_level_wrapped(&toks, 50);
        for (r, row) in s.lines().enumerate() {
            assert_eq!(
                row.chars().count(),
                50,
                "row {} should have width 50",
                r
            );
        }
        assert_eq!(s.lines().count(), 4, "200 chars / 50 cols = 4 rows");
    }

    #[test]
    fn render_level_wrapped_respects_explicit_newlines() {
        // Explicit newline tokens reset the column counter; output rows can
        // therefore be shorter than `cols` (a model-emitted row break wins).
        let nl = encode_char('\n').unwrap();
        let mut toks = vec![1u8; 10]; // 10 ground tiles
        toks.push(nl);
        toks.extend(std::iter::repeat(0u8).take(20)); // 20 sky
        let s = render_level_wrapped(&toks, 50);
        let rows: Vec<&str> = s.lines().collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].chars().count(), 10);
        assert_eq!(rows[1].chars().count(), 20);
    }

    // ---------- iter 13 tests (baselines: uniform random + Markov-1) ----------

    #[test]
    fn uniform_random_outputs_in_vocab() {
        let toks = uniform_random_generate(700, 0xABCD);
        assert_eq!(toks.len(), 700);
        for &t in &toks {
            assert!((t as usize) < VOCAB_SIZE, "out-of-vocab token {}", t);
        }
    }

    #[test]
    fn uniform_random_is_deterministic() {
        let a = uniform_random_generate(200, 0xCAFE);
        let b = uniform_random_generate(200, 0xCAFE);
        assert_eq!(a, b);
    }

    #[test]
    fn uniform_random_is_far_from_corpus() {
        // L2 distance to corpus target should be large for a uniform-random
        // grid (it's saturating density and ground placement equally).
        let toks = uniform_random_generate(700, 0xFACE);
        let m = compute_metrics(&toks, 50, 14);
        let dist = metric_distance(&m, &corpus_target());
        assert!(
            dist > 1.5,
            "uniform random should be > 1.5 L2 from corpus, got {:.3}",
            dist
        );
    }

    #[test]
    fn markov_one_is_deterministic() {
        let corpus = encode_corpus();
        let m1 = Markov1::from_corpus(&corpus);
        let m2 = Markov1::from_corpus(&corpus);
        let p: Vec<u8> = "M-X".chars().filter_map(encode_char).collect();
        let a = m1.generate(&p, 64, 0xBEEF);
        let b = m2.generate(&p, 64, 0xBEEF);
        assert_eq!(a, b);
    }

    #[test]
    fn markov_one_outputs_in_vocab() {
        let corpus = encode_corpus();
        let m = Markov1::from_corpus(&corpus);
        let out = m.generate(&[0u8; 0], 700, 0xC0DE);
        for &t in &out {
            assert!((t as usize) < VOCAB_SIZE, "out-of-vocab token {}", t);
        }
    }

    // ---------- iter 12 tests (metric distance + sweep helpers) ----------

    #[test]
    fn metric_distance_zero_for_target_itself() {
        let target = corpus_target();
        assert_eq!(metric_distance(&target, &target), 0.0);
    }

    #[test]
    fn metric_distance_increases_with_density_gap() {
        let target = corpus_target();
        let near = LevelMetrics {
            density: 0.30,
            ..target.clone()
        };
        let far = LevelMetrics {
            density: 0.80,
            ..target.clone()
        };
        let dn = metric_distance(&near, &target);
        let df = metric_distance(&far, &target);
        assert!(
            df > dn,
            "distance should grow with density gap: near={}, far={}",
            dn, df
        );
    }

    #[test]
    fn metric_distance_excludes_novelty() {
        // Two metrics that differ only in novelty should have identical
        // distance to target — we want generative diversity to be free.
        let target = corpus_target();
        let m1 = LevelMetrics {
            novelty: 0.1,
            ..target.clone()
        };
        let m2 = LevelMetrics {
            novelty: 0.9,
            ..target.clone()
        };
        assert!(
            (metric_distance(&m1, &target) - metric_distance(&m2, &target)).abs() < 1e-6,
            "novelty must not contribute to metric_distance"
        );
    }

    // ---------- iter 11 tests (PCG metrics) ----------

    #[test]
    fn metrics_on_empty_grid_are_finite() {
        let toks: Vec<u8> = vec![0; 700]; // all sky
        let m = compute_metrics(&toks, 50, 14);
        assert!(m.density.is_finite() && m.density >= 0.0);
        assert!(m.linearity.is_finite() && m.linearity >= 0.0);
        assert!(m.leniency.is_finite());
        assert!(m.novelty.is_finite() && m.novelty >= 0.0 && m.novelty <= 1.0);
        assert!(m.playable_columns.is_finite() && m.playable_columns >= 0.0);
        assert_eq!(m.density, 0.0, "all-sky should have density 0");
        assert_eq!(m.playable_columns, 0.0, "all-sky has no ground to stand on");
    }

    #[test]
    fn metrics_on_corpus_slice_have_zero_novelty() {
        // The first embedded level slice is in the corpus, so novelty
        // (min Hamming distance to any same-shape window of any corpus
        // slice) must be zero.
        let toks = encode_level(LEVELS[0]);
        let m = compute_metrics(&toks, 50, 14);
        assert_eq!(m.novelty, 0.0, "corpus slice should have novelty 0");
        // It should also be highly playable (corpus levels have a
        // continuous ground floor).
        assert!(
            m.playable_columns >= 0.95,
            "corpus slice should have ≥95% playable columns, got {:.3}",
            m.playable_columns
        );
    }

    #[test]
    fn metrics_density_scales_with_nonsky_tiles() {
        let cols = 50;
        let rows = 14;
        let mut toks = vec![0u8; cols * rows]; // all sky → density 0
        let half = (cols * rows) / 2;
        for i in 0..half {
            toks[i] = 1; // ground
        }
        let m = compute_metrics(&toks, cols, rows);
        assert!(
            (m.density - 0.5).abs() < 0.01,
            "half-ground should have density ≈ 0.5, got {}",
            m.density
        );
    }

    #[test]
    fn metrics_linearity_zero_for_flat_floor() {
        // A grid where every column has its topmost ground at the same row
        // should have linearity 0 (no variance in ground heights).
        let cols = 50;
        let rows = 14;
        let mut toks = vec![0u8; cols * rows];
        // Row 13 (last) all ground; rest sky.
        for c in 0..cols {
            toks[13 * cols + c] = 1;
        }
        let m = compute_metrics(&toks, cols, rows);
        assert!(
            m.linearity < 0.01,
            "perfectly flat floor should have linearity ≈ 0, got {}",
            m.linearity
        );
    }

    // ---------- iter 10 tests (multi-token bidirectional context) ----------

    #[test]
    fn diffuser_uses_offset_2_context() {
        // Build a minimal sequence where position 0 has its offset-1 right
        // neighbour masked but its offset-2 right neighbour is the GROUND
        // token (id 1). Position 0 has no left neighbours (i = 0).
        //
        // Iter-7 (radius 1): K[0] = 0.5·(zero + zero) = all zeros.
        // Iter-10 (radius 2): K[0] = 0.25·embed(ground) + zeros = non-zero.
        let r = MarioRetriever::new(encode_corpus(), 0xABCD);
        let d = MarioDiffuser::new(&r);
        let seq = [MASK_SENTINEL, MASK_SENTINEL, 1u8];
        let (k, _v) = d.make_bidir_kv(&seq);
        let k_row_0 = k.row(0, 0);
        let nonzero = k_row_0.iter().any(|&v| v.abs() > 1e-6);
        assert!(
            nonzero,
            "K[0] must be non-zero — radius-2 context should pull in the offset-2 token"
        );

        // And the value should match w_offset2·embed(ground) — checking
        // magnitude confirms the weight is the offset-2 weight, not the
        // offset-1 weight (which would imply we misindex offset 1 vs 2).
        let w2 = DIFFUSION_CONTEXT_WEIGHTS.get(1).copied().unwrap_or(0.0);
        let g_emb = token_embedding(1, &r.w);
        let l2_actual: f32 = k_row_0.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let l2_expected: f32 = (g_emb.iter().map(|&v| w2 * w2 * v * v).sum::<f32>()).sqrt();
        let ratio = l2_actual / l2_expected.max(1e-9);
        assert!(
            (0.95..1.05).contains(&ratio),
            "K[0] L2 norm should match {}·||embed(ground)||; ratio={}",
            w2,
            ratio
        );
    }

    // Note: the iter-7 `diffusion_produces_diverse_output` test (≥4
    // distinct tiles at seed 0xDEAD) is the regression safety net for
    // iter-10. Honest finding: averaging multi-token context with a
    // significant outer weight reduces per-seed variance and can drop
    // distinct-tile counts. Outer weight 0.10 stays close to iter-7
    // behaviour while still letting offset-2 tokens influence retrieval
    // when offset-1 is masked.

    // ---------- iter 9 tests (top-p / nucleus sampling) ----------

    #[test]
    fn top_p_disabled_matches_no_top_p() {
        // top_p = 0 (disabled) and top_p = 1.0 (kept everything) should
        // produce the same output as no top_p on the same seed.
        let r = MarioRetriever::new(encode_corpus(), 0xABCD);
        let p: Vec<u8> = "M-X".chars().filter_map(encode_char).collect();
        let base = SamplingConfig {
            temperature: 1.0,
            top_k: 5,
            top_p: 0.0,
            repetition_penalty: 1.5,
            no_repeat_window: 8,
        };
        let with_p1 = SamplingConfig {
            top_p: 1.0,
            ..base.clone()
        };
        let a = r.generate_fast(&p, 80, &base, 0xC0FFEE);
        let b = r.generate_fast(&p, 80, &with_p1, 0xC0FFEE);
        assert_eq!(a, b, "top_p=0 and top_p=1 should be identical (no-op)");
    }

    #[test]
    fn top_p_05_restricts_compared_to_top_p_09() {
        // A tighter nucleus (top_p=0.5) keeps fewer tokens than a looser one
        // (top_p=0.9). Sanity: tighter nucleus has at most as many distinct
        // generated tiles, generally fewer.
        let r = MarioRetriever::new(encode_corpus(), 0xABCD);
        let p: Vec<u8> = "M-X".chars().filter_map(encode_char).collect();
        let tight = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.50,
            repetition_penalty: 1.0,
            no_repeat_window: 0,
        };
        let loose = SamplingConfig {
            top_p: 0.90,
            ..tight.clone()
        };
        let a = r.generate_fast(&p, 240, &tight, 0xCAFE);
        let b = r.generate_fast(&p, 240, &loose, 0xCAFE);

        let unique = |toks: &[u8]| -> usize {
            let mut s = std::collections::HashSet::new();
            for &t in toks {
                s.insert(t);
            }
            s.len()
        };
        let u_tight = unique(&a[p.len()..]);
        let u_loose = unique(&b[p.len()..]);
        assert!(
            u_tight <= u_loose,
            "tight nucleus should have ≤ unique tiles than loose; tight={}, loose={}",
            u_tight,
            u_loose
        );
    }

    #[test]
    fn quality_v9_breaks_streaks_better_than_v5() {
        // The iter-9 sweep configuration should produce shorter max-streak
        // than the iter-5 baseline (top_k only, narrower window).
        let r = MarioRetriever::new(encode_corpus(), 0xABCD);
        let p: Vec<u8> = "M-XXXXX\n".chars().filter_map(encode_char).collect();
        let v5 = SamplingConfig {
            temperature: 1.0,
            top_k: 5,
            top_p: 0.0,
            repetition_penalty: 1.6,
            no_repeat_window: 12,
        };
        let v9 = SamplingConfig::quality();

        let max_streak = |toks: &[u8]| -> usize {
            let mut best = 0;
            let mut cur = 0;
            let mut prev: Option<u8> = None;
            for &t in toks {
                if Some(t) == prev {
                    cur += 1;
                } else {
                    cur = 1;
                }
                if cur > best {
                    best = cur;
                }
                prev = Some(t);
            }
            best
        };

        // Average over 4 seeds to reduce variance.
        let seeds: [u32; 4] = [0xA001, 0xA002, 0xA003, 0xA004];
        let avg_streak = |cfg: &SamplingConfig| -> f64 {
            let mut sum = 0;
            for &s in &seeds {
                let out = r.generate_fast(&p, 400, cfg, s);
                sum += max_streak(&out[p.len()..]);
            }
            sum as f64 / seeds.len() as f64
        };
        let s5 = avg_streak(&v5);
        let s9 = avg_streak(&v9);
        assert!(
            s9 <= s5,
            "iter-9 quality() max-streak should be ≤ iter-5 baseline; v5={:.1}, v9={:.1}",
            s5,
            s9
        );
    }

    // ---------- iter 8 tests (KvCache + decode_step fast generation) ----------

    #[test]
    fn generate_fast_is_deterministic() {
        let r1 = MarioRetriever::new(encode_corpus(), 0xABCD);
        let r2 = MarioRetriever::new(encode_corpus(), 0xABCD);
        let p: Vec<u8> = "M-X".chars().filter_map(encode_char).collect();
        let cfg = SamplingConfig::quality();
        let a = r1.generate_fast(&p, 64, &cfg, 0xCAFE_BABE);
        let b = r2.generate_fast(&p, 64, &cfg, 0xCAFE_BABE);
        assert_eq!(a, b);
        assert_eq!(a.len(), p.len() + 64);
    }

    #[test]
    fn generate_fast_outputs_in_vocab() {
        let r = MarioRetriever::new(encode_corpus(), 0xABCD);
        let p: Vec<u8> = "M-X".chars().filter_map(encode_char).collect();
        let out = r.generate_fast(&p, 200, &SamplingConfig::quality(), 0x4242);
        for &t in &out {
            assert!(
                (t as usize) < VOCAB.len(),
                "out-of-vocab token {} from generate_fast",
                t
            );
        }
    }

    #[test]
    fn generate_fast_beats_generate_on_speed() {
        // The whole point of iter 8: incremental decoding should be a clear
        // wall-clock win at 100-token generation. We measure ratio rather
        // than absolute ms (CI machines vary).
        let r = MarioRetriever::new(encode_corpus(), 0xABCD);
        let p: Vec<u8> = "M-X".chars().filter_map(encode_char).collect();
        let cfg = SamplingConfig::quality();

        let t0 = std::time::Instant::now();
        let _slow = r.generate(&p, 60, &cfg, 0x1111);
        let slow_ms = t0.elapsed();

        let t0 = std::time::Instant::now();
        let _fast = r.generate_fast(&p, 60, &cfg, 0x1111);
        let fast_ms = t0.elapsed();

        let ratio = slow_ms.as_secs_f64() / fast_ms.as_secs_f64().max(1e-9);
        assert!(
            ratio >= 5.0,
            "generate_fast should be ≥5× faster than generate; ratio={:.2} (slow={:?}, fast={:?})",
            ratio, slow_ms, fast_ms
        );
    }

    #[test]
    fn generate_fast_produces_corpus_like_distribution() {
        // Same kind of sanity check as the slow-path test, but for the
        // KvCache pipeline. Bigram retrieval should still bias toward the
        // dominant tiles — most output is sky / ground / newline / brick.
        let r = MarioRetriever::new(encode_corpus(), 0xABCD);
        let p: Vec<u8> = "----".chars().filter_map(encode_char).collect();
        let cfg = SamplingConfig {
            temperature: 0.6,
            ..SamplingConfig::default()
        };
        let out = r.generate_fast(&p, 300, &cfg, 0x9001);
        let gen = &out[p.len()..];
        let dist = tile_distribution(gen);
        let common = *dist.get(&'-').unwrap_or(&0)
            + *dist.get(&'X').unwrap_or(&0)
            + *dist.get(&'\n').unwrap_or(&0)
            + *dist.get(&'S').unwrap_or(&0);
        let frac = common as f64 / gen.len() as f64;
        assert!(
            frac > 0.6,
            "generate_fast should produce >60% common-tile output, got {:.1}%",
            frac * 100.0
        );
    }

    // ---------- iter 7 tests (masked discrete diffusion) ----------

    #[test]
    fn diffusion_clears_all_masks() {
        let r = MarioRetriever::new(encode_corpus(), 0xABCD);
        let d = MarioDiffuser::new(&r);
        let out = d.diffuse(120, 8, &SamplingConfig::quality(), 0x1357);
        assert_eq!(out.len(), 120);
        assert!(
            out.iter().all(|&t| t != MASK_SENTINEL),
            "diffusion left residual masks in output"
        );
        for &t in &out {
            assert!((t as usize) < VOCAB_SIZE, "out-of-vocab token {}", t);
        }
    }

    #[test]
    fn diffusion_is_deterministic_for_fixed_seed() {
        let r = MarioRetriever::new(encode_corpus(), 0xABCD);
        let d = MarioDiffuser::new(&r);
        let cfg = SamplingConfig::quality();
        let a = d.diffuse(80, 6, &cfg, 0x9999);
        let b = d.diffuse(80, 6, &cfg, 0x9999);
        assert_eq!(a, b);
    }

    #[test]
    fn diffusion_produces_diverse_output() {
        // The diffuser must not collapse to a single-tile saturated grid
        // (the all-X / all-`-` failure mode that bidirectional context is
        // supposed to avoid). Expect at least 4 distinct tile types.
        let r = MarioRetriever::new(encode_corpus(), 0xABCD);
        let d = MarioDiffuser::new(&r);
        let cfg = SamplingConfig::quality();
        let out = d.diffuse(300, 12, &cfg, 0xDEAD);
        let mut s = std::collections::HashSet::new();
        for &t in &out {
            s.insert(t);
        }
        assert!(
            s.len() >= 4,
            "diffusion should produce ≥4 distinct tiles, got {} ({:?})",
            s.len(),
            out.iter().take(40).map(|&t| decode_token(t)).collect::<String>()
        );
    }

    #[test]
    fn diffusion_produces_corpus_like_distribution() {
        // With bidirectional context, diffusion should bias hard toward the
        // dominant tiles (sky/ground/structural) — sanity: ≥40% sky+ground.
        let r = MarioRetriever::new(encode_corpus(), 0xABCD);
        let d = MarioDiffuser::new(&r);
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_k: 6,
            top_p: 0.0,
            repetition_penalty: 1.4,
            no_repeat_window: 8,
        };
        let out = d.diffuse(200, 8, &cfg, 0xDEAD);
        let dist = tile_distribution(&out);
        let sky_ground =
            *dist.get(&'-').unwrap_or(&0) + *dist.get(&'X').unwrap_or(&0);
        let frac = sky_ground as f64 / out.len() as f64;
        assert!(
            frac > 0.30,
            "diffusion should produce ≥30% sky/ground, got {:.1}%",
            frac * 100.0
        );
    }

    #[test]
    fn denoise_step_unmasks_at_most_keep_count() {
        let r = MarioRetriever::new(encode_corpus(), 0xABCD);
        let d = MarioDiffuser::new(&r);
        let mut working = vec![MASK_SENTINEL; 60];
        // Seed a few unmasked positions so the bidirectional context isn't empty.
        working[0] = 0;
        working[59] = 1;
        let before = working.iter().filter(|&&t| t == MASK_SENTINEL).count();
        let mut state = 0xFEEDu32;
        d.denoise_step(&mut working, 5, &SamplingConfig::quality(), &mut state);
        let after = working.iter().filter(|&&t| t == MASK_SENTINEL).count();
        assert_eq!(before - after, 5, "should have unmasked exactly 5 positions");
    }

    #[test]
    fn repetition_penalty_reduces_max_streak() {
        // Repetition penalty should shorten the longest run of any single tile.
        let r = MarioRetriever::new(encode_corpus(), 0xABCD);
        let p: Vec<u8> = "M-XXXXX\n".chars().filter_map(encode_char).collect();
        let no_pen = SamplingConfig {
            temperature: 1.0,
            top_k: 4,
            top_p: 0.0,
            repetition_penalty: 1.0,
            no_repeat_window: 0,
        };
        let with_pen = SamplingConfig {
            temperature: 1.0,
            top_k: 4,
            top_p: 0.0,
            repetition_penalty: 1.8,
            no_repeat_window: 12,
        };

        let max_streak = |toks: &[u8]| -> usize {
            let mut best = 0;
            let mut cur = 0;
            let mut prev: Option<u8> = None;
            for &t in toks {
                if Some(t) == prev {
                    cur += 1;
                } else {
                    cur = 1;
                }
                if cur > best {
                    best = cur;
                }
                prev = Some(t);
            }
            best
        };

        let a = r.generate(&p, 400, &no_pen, 0x3333);
        let b = r.generate(&p, 400, &with_pen, 0x3333);

        let s_no = max_streak(&a[p.len()..]);
        let s_with = max_streak(&b[p.len()..]);
        assert!(
            s_with < s_no,
            "repetition penalty should shorten the longest streak \
             (no penalty: {}, with penalty: {})",
            s_no,
            s_with
        );
    }
}
