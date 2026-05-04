//! `WordPieceTokenizer::encode` throughput benchmark.
//!
//! Motivation (May 2026 design check): the worker hot path is
//! `tokenize -> NPU forward pass -> reply`. NPU forward on Hailo-8 is
//! ~1-3 ms for a single 128-token sequence. If tokenization on Cortex-A76
//! costs more than ~500 µs, the NPU is starved.
//!
//! This bench builds a realistic-size synthetic vocabulary (~30k entries
//! to mirror BERT-base) and runs `encode` against representative English
//! text at four sequence-length targets: 16, 64, 128, 256 tokens.
//!
//! It is hardware-agnostic — the same harness runs on x86 dev hosts and
//! on the Pi 5 over SSH; the absolute numbers from each give a
//! before/after comparison for any optimisation work (SIMD basic-tokenize,
//! interned vocab, etc.).
//!
//! Run with:
//!   cargo bench --bench wordpiece_throughput
//!
//! On the Pi 5 (cross-compiled or native) this is the canonical signal
//! for whether tokenization is the bottleneck before / after the HEF
//! lands. Measurements logged to PR #413 review thread.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ruvector_hailo::tokenizer::WordPieceTokenizer;

/// Build a synthetic ~30k-entry vocab that mirrors BERT-base's structure:
/// 4 specials, then a mix of 2-6 char base tokens and `##xxx` continuations.
/// Deterministic so bench numbers are comparable across runs.
fn synthetic_vocab() -> String {
    let mut v: Vec<String> = Vec::with_capacity(30_522);
    v.push("[PAD]".into());
    v.push("[UNK]".into());
    v.push("[CLS]".into());
    v.push("[SEP]".into());
    // Pad to 100 with [unusedN] so common BERT IDs land where users expect.
    for i in 4..100 {
        v.push(format!("[unused{}]", i));
    }
    // Common single chars first (high hit rate on real text).
    for c in 'a'..='z' {
        v.push(c.to_string());
        v.push(format!("##{}", c));
    }
    for c in '0'..='9' {
        v.push(c.to_string());
    }
    for p in [
        ",", ".", "!", "?", "-", ":", ";", "(", ")", "'", "\"", "/", "&",
    ] {
        v.push(p.to_string());
    }
    // Bigrams (2-char). 26*26 = 676.
    for a in 'a'..='z' {
        for b in 'a'..='z' {
            v.push(format!("{}{}", a, b));
        }
    }
    // 3-grams sampled procedurally to fill out the vocab. Use a tiny LCG
    // so we don't need a rand crate dep. ~28k entries land here.
    let mut state: u32 = 0xc0ffee;
    let next = |s: &mut u32| -> u32 {
        *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *s
    };
    while v.len() < 30_500 {
        let l = 3 + (next(&mut state) % 4) as usize; // 3..=6 char tokens
        let prefix = if next(&mut state) % 3 == 0 { "##" } else { "" };
        let mut s = String::from(prefix);
        for _ in 0..l {
            let c = b'a' + (next(&mut state) % 26) as u8;
            s.push(c as char);
        }
        v.push(s);
    }
    v.join("\n")
}

/// Realistic English-ish text generator. Avoids a network or embedded
/// fixture — produces deterministic prose-shaped strings of approximately
/// the requested character length.
fn sample_text(target_chars: usize) -> String {
    const STOCK: &[&str] = &[
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "the",
        "lazy",
        "dog",
        "ruvector",
        "embeddings",
        "search",
        "system",
        "produces",
        "high-quality",
        "dense",
        "vectors",
        "from",
        "natural",
        "language",
        "queries",
        "for",
        "use",
        "in",
        "downstream",
        "retrieval",
        "pipelines",
        "and",
        "neural",
        "ranking",
        "models",
        "trained",
        "on",
        "scientific",
        "literature",
        "and",
        "general",
        "domain",
        "corpora",
    ];
    let mut out = String::with_capacity(target_chars);
    let mut i = 0usize;
    while out.len() < target_chars {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(STOCK[i % STOCK.len()]);
        i += 1;
        // Sprinkle some punctuation every ~12 words.
        if i % 12 == 11 {
            out.push(',');
        }
    }
    out
}

fn bench_encode(c: &mut Criterion) {
    let vocab = synthetic_vocab();
    let tok = WordPieceTokenizer::from_vocab_str(&vocab).expect("build tokenizer");

    let mut group = c.benchmark_group("wordpiece_encode");
    // Target sequence lengths cover the realistic span: short queries
    // (16), single-sentence chunks (64), paragraph chunks (128 — the
    // all-MiniLM-L6-v2 default), and long passages (256).
    for max_seq in &[16usize, 64, 128, 256] {
        // Aim for ~max_seq*4 chars so post-tokenization length lands
        // close to the target before truncation kicks in.
        let text = sample_text(max_seq * 4);
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(max_seq),
            max_seq,
            |b, &max_seq| {
                b.iter(|| {
                    let enc = tok.encode(black_box(&text), black_box(max_seq), true);
                    black_box(enc);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_encode);
criterion_main!(benches);
