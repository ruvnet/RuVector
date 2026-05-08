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
    AttentionBackend, SparseAttentionConfig, SubquadraticSparseAttention, Tensor3,
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

pub struct MarioRetriever {
    pub corpus: Vec<u8>,
    w: Vec<f32>,
    cfg: SparseAttentionConfig,
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

    pub fn generate(&self, prefix: &[u8], n: usize, temperature: f32, sampler_seed: u32) -> Vec<u8> {
        let mut state = sampler_seed.max(1);
        let mut out = prefix.to_vec();
        for _ in 0..n {
            let logits = self.next_token_logits(&out);
            let next = sample_logits(&logits, temperature, &mut state);
            out.push(next);
        }
        out
    }
}

fn sample_logits(logits: &[f32; VOCAB_SIZE], temperature: f32, state: &mut u32) -> u8 {
    let temp = temperature.max(1e-3);
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs = [0.0f32; VOCAB_SIZE];
    let mut sum = 0.0f32;
    for i in 0..VOCAB_SIZE {
        probs[i] = ((logits[i] - max_l) / temp).exp();
        sum += probs[i];
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
    let t0 = std::time::Instant::now();
    let generated = retriever.generate(&seed_chars, n_gen, 1.2, 0xC0FF_EE42);
    let dt = t0.elapsed();
    let rendered = render_level(&generated);

    println!("seed prefix   : {:?}", seed_chars);
    println!("generated     : {} tokens in {:.2?}", n_gen, dt);
    println!();
    println!("{}", rendered);
    println!();

    let gen_only = &generated[seed_chars.len()..];
    let gen_dist = tile_distribution(gen_only);
    let pct = |c: char| -> f64 {
        *gen_dist.get(&c).unwrap_or(&0) as f64 / gen_only.len() as f64 * 100.0
    };
    println!(
        "tile mix in generated: sky {:.1}%  ground {:.1}%  brick {:.1}%  enemy {:.1}%  newline {:.1}%",
        pct('-'),
        pct('X'),
        pct('S'),
        pct('E'),
        pct('\n')
    );
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
        let a = r1.generate(&p, 64, 0.8, 0xDEAD_BEEF);
        let b = r2.generate(&p, 64, 0.8, 0xDEAD_BEEF);
        assert_eq!(a, b, "same seed should give same output");
        assert_eq!(a.len(), p.len() + 64);
    }

    #[test]
    fn generated_tiles_are_in_vocab() {
        let r = MarioRetriever::new(encode_corpus(), 0xABCD);
        let p: Vec<u8> = "--".chars().filter_map(encode_char).collect();
        let out = r.generate(&p, 200, 1.0, 0x4242);
        for &t in &out {
            assert!((t as usize) < VOCAB.len(), "out-of-vocab token {}", t);
        }
    }

    #[test]
    fn generated_distribution_is_corpus_like() {
        // With temperature=0.5 the retrieval should bias toward the corpus
        // distribution: most tiles should be sky or ground.
        let r = MarioRetriever::new(encode_corpus(), 0xABCD);
        let p: Vec<u8> = "----".chars().filter_map(encode_char).collect();
        let out = r.generate(&p, 300, 0.5, 0x9001);
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
}
