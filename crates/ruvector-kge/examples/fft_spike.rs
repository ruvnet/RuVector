//! ADR-002 §3 spike: FFT vs direct circular correlation, and batch matmul vs
//! triple-by-triple scoring, for d ∈ {128, 256, 512}. Native timing only —
//! examples may use `std::time`; the core crate may not (ADR-005). Emits JSON
//! on stdout; redirect into `npm/packages/kge/bench/fft-spike-2026-09-21.json`.
//!
//!   cargo run -p ruvector-kge --example fft_spike --release > <that file>
//!
//! The wasm32 half of the gate is a compile check, not a timing:
//!   cargo check -p ruvector-kge --target wasm32-unknown-unknown

use ruvector_kge::scorer::fft::{corr_direct, FftPlan};
use ruvector_kge::{BatchScorer, HolE, Scorer, Side, Tables};
use std::hint::black_box;
use std::time::Instant;

fn rand_vec(d: usize, mut state: u64) -> Vec<f32> {
    (0..d)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state >> 11) as f32 / (1u64 << 53) as f32;
            2.0 * u - 1.0
        })
        .collect()
}

/// Median ns/op of `f` over `reps` timed runs (each run does one op).
fn bench<F: FnMut()>(reps: usize, warmup: usize, mut f: F) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_nanos() as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn main() {
    let dims = [128usize, 256, 512];
    let mut per_dim = Vec::new();

    for &d in &dims {
        let plan = FftPlan::new(d).unwrap();
        let a = rand_vec(d, 0x11 ^ d as u64);
        let b = rand_vec(d, 0x22 ^ d as u64);

        let fft_ns = bench(2000, 200, || {
            black_box(plan.circular_correlation(black_box(&a), black_box(&b)));
        });
        let direct_ns = bench(2000, 200, || {
            black_box(corr_direct(black_box(&a), black_box(&b)));
        });

        // Batch matmul vs triple-by-triple scoring over a fixed relation.
        let n = 2000usize;
        let scorer = HolE::new(d).unwrap();
        let tables = Tables::new(n, 8, d, 0x1357 ^ d as u64);
        let s = tables.entity(3).unwrap().to_vec();
        let r = tables.relation(2).unwrap().to_vec();

        let build_t = Instant::now();
        let batch = BatchScorer::build(&tables, &scorer);
        let batch_build_ns = build_t.elapsed().as_nanos() as f64;

        let q = scorer.query_vector(&r, &s, Side::Tail);
        let batch_ns = bench(200, 20, || {
            black_box(batch.scores(black_box(&q)));
        });
        let triple_ns = bench(200, 20, || {
            let mut acc = 0.0f32;
            for o in 0..n {
                acc += scorer.score(&s, &r, tables.entity(o as u32).unwrap());
            }
            black_box(acc);
        });

        per_dim.push(serde_json::json!({
            "d": d,
            "fft_corr_ns": fft_ns,
            "direct_corr_ns": direct_ns,
            "fft_corr_speedup": direct_ns / fft_ns,
            "batch_scores_ns": batch_ns,
            "batch_build_ns": batch_build_ns,
            "batch_scores_plus_build_ns": batch_ns + batch_build_ns,
            "triple_scores_ns": triple_ns,
            "batch_vs_triple_speedup": triple_ns / batch_ns,
            "batch_plus_build_vs_triple_speedup": triple_ns / (batch_ns + batch_build_ns),
            "entities": n
        }));
    }

    let fft_beats_direct_512 = {
        let last = per_dim.last().unwrap();
        last["fft_corr_speedup"].as_f64().unwrap() > 1.0
    };
    let out = serde_json::json!({
        "spike": "ADR-002 §3 — FFT vs direct correlation, batch vs per-triple",
        "date": "2026-09-21",
        "fft_crate": "rustfft 6.4.1 (MIT OR Apache-2.0)",
        "target_native": std::env::consts::ARCH,
        "wasm32_gate": "compile-only: cargo check --target wasm32-unknown-unknown passes",
        "method": "median ns/op, black_box, 2000 reps (corr) / 200 reps (scoring); Fft::process allocates scratch per call",
        "pass_criteria": "FFT beats direct at d=512 native AND rustfft compiles+runs on wasm32",
        "fft_beats_direct_at_d512": fft_beats_direct_512,
        "results": per_dim
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
