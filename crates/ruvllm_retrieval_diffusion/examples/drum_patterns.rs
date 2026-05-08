//! Drum patterns — second-domain demo of `ruvllm_retrieval_diffusion`.
//!
//! Same pattern as `sparse-mario`, different corpus: instead of Super
//! Mario level tiles we use 5-token drum-machine notation:
//!
//!   K = kick, S = snare, h = closed hi-hat, H = open hi-hat, . = silence
//!
//! Four classic 16-step patterns are embedded as the corpus (rock, funk,
//! reggae, boom-bap). The retriever learns the bigram statistics; the
//! diffuser fills bidirectional context. Output is a 64-step (4-bar) loop.
//!
//! Run with: cargo run --release --features parallel --example drum_patterns

use ruvllm_retrieval_diffusion::{Diffuser, RetrievalConfig, Retriever, SamplingConfig};

const VOCAB: &[char] = &['.', 'K', 'S', 'h', 'H']; // index = token id

fn encode_char(c: char) -> Option<u8> {
    VOCAB.iter().position(|&v| v == c).map(|i| i as u8)
}

fn decode_token(t: u8) -> char {
    VOCAB.get(t as usize).copied().unwrap_or('?')
}

fn encode(s: &str) -> Vec<u8> {
    s.chars().filter_map(encode_char).collect()
}

/// Embedded corpus — four 16-step drum loops, hand-authored.
/// Total = 64 tokens (the full corpus is short by design — the demo
/// shows the same training-free retrieval picking up *any* small-vocab
/// rhythmic prior, not specific drum knowledge).
const PATTERNS: &[&str] = &[
    // Basic rock beat — 4 on the floor
    "K.h.S.h.K.h.S.h.",
    // Funk — sixteenth-note hi-hats with snare on 5/13
    "KhhhShhhKhhhShhh",
    // Reggae one-drop — snare on 3, kick on 4-and
    "..S...S...S...S.",
    // Boom-bap — sparse kick + ghost snares, open hi-hat lift on 7
    "K..K.S..K..K.S..",
];

fn render_bars(tokens: &[u8], steps_per_bar: usize) -> String {
    let mut out = String::new();
    let mut col = 0;
    for &t in tokens {
        out.push(decode_token(t));
        col += 1;
        if col == steps_per_bar {
            out.push('\n');
            col = 0;
        }
    }
    out
}

fn drum_config() -> RetrievalConfig {
    RetrievalConfig {
        vocab_size: VOCAB.len(),
        head_dim: 64,
        // Drum patterns repeat every 16 steps; positional bias would push
        // queries late in the prefix toward late corpus positions, which is
        // the wrong inductive bias for a strictly-cyclic domain.
        pos_scale: 0.0,
        mask_sentinel: 255,
        diffusion_context_weights: vec![0.5, 0.10],
        sparse: ruvllm_retrieval_diffusion::SparseConfig {
            window: 32,
            block_size: 16,
            global_tokens: vec![0],
            causal: false,
            use_log_stride: true,
            use_landmarks: true,
            sort_candidates: false,
        },
    }
}

fn build_corpus() -> Vec<u8> {
    // Concatenate all patterns, no separator (the shape is fixed at 16
    // steps per bar, so absolute index modulo 16 is the bar position).
    let mut c = Vec::new();
    for p in PATTERNS {
        c.extend(encode(p));
    }
    c
}

fn main() {
    let corpus = build_corpus();
    println!("== Drum-pattern retrieval-diffusion demo ==");
    println!(
        "corpus       : {} tokens ({} patterns × 16 steps)",
        corpus.len(),
        PATTERNS.len()
    );
    println!("vocab        : {:?}", VOCAB);

    // Tile distribution
    let mut dist = std::collections::HashMap::new();
    for &t in &corpus {
        *dist.entry(decode_token(t)).or_insert(0usize) += 1;
    }
    print!("tile mix     : ");
    for &c in VOCAB {
        let n = dist.get(&c).copied().unwrap_or(0);
        print!("{}={:.1}%  ", c, n as f32 / corpus.len() as f32 * 100.0);
    }
    println!();

    let cfg = drum_config();
    let retriever = Retriever::new(corpus.clone(), cfg, 0xD7_5_BABE);

    // Seed with the first half of a familiar pattern, ask the model to
    // continue. AR walks bigram statistics; should mostly stay in groove.
    let seed = encode("K.h.S.h.");
    let sampling = SamplingConfig::quality();

    println!();
    println!("--- AR (KvCache + decode_step) ---");
    let t0 = std::time::Instant::now();
    let ar = retriever.generate_fast(&seed, 64, &sampling, 0xC0_FFEE_42);
    let dt_ar = t0.elapsed();
    println!("seed         : \"{}\"", String::from_utf8_lossy(&seed.iter().map(|&t| decode_token(t) as u8).collect::<Vec<_>>()));
    println!("generated    : {} tokens in {:.2?}", 64, dt_ar);
    println!();
    println!("{}", render_bars(&ar, 16));

    // Diffusion — start fully masked, denoise to 4 bars (64 steps) with
    // bidirectional context. Boot slice is taken from the corpus.
    println!("--- Diffusion (D3PM-style, cosine schedule) ---");
    let diffuser = Diffuser::new(&retriever);
    let t0 = std::time::Instant::now();
    let diff = diffuser.diffuse(64, 24, &sampling, 0xD1_FF_BEEF);
    let dt_diff = t0.elapsed();
    println!(
        "diffused     : {} tokens × 24 denoising steps in {:.2?}",
        64, dt_diff
    );
    println!();
    println!("{}", render_bars(&diff, 16));

    // Compute simple "groove sanity" stats: density (non-silence rate)
    // and longest-streak. A real corpus has density ≈ 0.5–0.75.
    let density = |toks: &[u8]| -> f32 {
        let nonsilence = toks.iter().filter(|&&t| t != 0).count();
        nonsilence as f32 / toks.len().max(1) as f32
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

    println!("--- groove sanity ---");
    println!(
        "corpus    density={:.2}  max_streak={}",
        density(&corpus),
        max_streak(&corpus)
    );
    println!(
        "AR        density={:.2}  max_streak={}",
        density(&ar),
        max_streak(&ar)
    );
    println!(
        "diffusion density={:.2}  max_streak={}",
        density(&diff),
        max_streak(&diff)
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
    fn corpus_is_64_tokens() {
        let c = build_corpus();
        assert_eq!(c.len(), 64, "4 patterns × 16 steps = 64");
        for &t in &c {
            assert!((t as usize) < VOCAB.len(), "out-of-vocab token {}", t);
        }
    }

    #[test]
    fn ar_generation_in_vocab() {
        let r = Retriever::new(build_corpus(), drum_config(), 0x1111);
        let out = r.generate_fast(&[1u8], 64, &SamplingConfig::quality(), 0x2222);
        for &t in &out {
            assert!((t as usize) < VOCAB.len());
        }
    }

    #[test]
    fn diffusion_clears_all_masks() {
        let r = Retriever::new(build_corpus(), drum_config(), 0x1111);
        let d = Diffuser::new(&r);
        let out = d.diffuse(64, 16, &SamplingConfig::quality(), 0x3333);
        let mask = drum_config().mask_sentinel;
        for &t in &out {
            assert_ne!(t, mask);
            assert!((t as usize) < VOCAB.len());
        }
    }
}
