//! Latency/throughput benchmark for the GNN-diffusion reranker (productionizes #479).
//!
//! Times only the rerank hot path (candidate sets are pre-built) and reports
//! µs/query + QPS for the GnnDiffusion variant vs the NoisyScore baseline. Runs
//! under `cargo test --release` so CI can track it without the AV-blocked
//! standalone benchmark bin. The assertion is a loose upper bound (catches a
//! large regression) — the printed numbers are the real signal.
//!
//! Run: `cargo test -p ruvector-gnn-rerank --release --test perf_benchmark -- --nocapture`

use std::time::Instant;

use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use ruvector_gnn_rerank::{Candidate, CandidateReranker, GnnDiffusionReranker, NoisyScoreReranker};

const DIM: usize = 128;
const CORPUS: usize = 2_000;
const RETRIEVAL_K: usize = 80;
const K: usize = 10;
const N_SETS: usize = 500; // pre-built candidate sets (≈ queries)
const REPS: usize = 5; // timing repetitions
const SEED: u64 = 42;
// Generous per-query budget (µs) for ~80-candidate, 1-hop diffusion in release.
const BUDGET_US: f64 = 700.0; // ~1.8x the measured ~393us/q — guards a real regression, not the true cost

fn build_candidate_sets() -> (Vec<Vec<f32>>, Vec<Vec<Candidate>>) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let corpus: Vec<Vec<f32>> = (0..CORPUS)
        .map(|_| (0..DIM).map(|_| rng.gen_range(-2.0_f32..2.0)).collect())
        .collect();
    let queries: Vec<Vec<f32>> = (0..N_SETS)
        .map(|_| {
            let base = &corpus[rng.gen_range(0..CORPUS)];
            base.iter().map(|&x| x + rng.gen_range(-0.1_f32..0.1)).collect()
        })
        .collect();
    let noise = Normal::new(0.0_f32, 0.40).unwrap();
    let sets = queries
        .iter()
        .map(|q| {
            let mut scored: Vec<(usize, f32)> = corpus
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let d2: f32 = q.iter().zip(v).map(|(a, b)| (a - b).powi(2)).sum();
                    (i, -d2.sqrt() + noise.sample(&mut rng))
                })
                .collect();
            scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            scored
                .into_iter()
                .take(RETRIEVAL_K)
                .map(|(id, s)| Candidate { id: id as u32, vector: corpus[id].clone(), noisy_score: s })
                .collect::<Vec<_>>()
        })
        .collect();
    (queries, sets)
}

fn time_reranker<R: CandidateReranker>(r: &R, queries: &[Vec<f32>], sets: &[Vec<Candidate>]) -> f64 {
    // warm up
    for (q, c) in queries.iter().zip(sets).take(16) {
        let _ = r.rerank(q, c, K).unwrap();
    }
    let mut best = f64::MAX;
    for _ in 0..REPS {
        let t = Instant::now();
        for (q, c) in queries.iter().zip(sets) {
            let out = r.rerank(q, c, K).unwrap();
            std::hint::black_box(&out);
        }
        best = best.min(t.elapsed().as_secs_f64());
    }
    best / queries.len() as f64 * 1e6 // µs per query (best-of-REPS)
}

#[test]
#[ignore = "perf benchmark; run with: cargo test --release -- --ignored"]
fn rerank_latency_throughput() {
    let (queries, sets) = build_candidate_sets();
    let noisy_us = time_reranker(&NoisyScoreReranker, &queries, &sets);
    let gnn_us = time_reranker(&GnnDiffusionReranker::default(), &queries, &sets);

    eprintln!("rerank latency (DIM={DIM}, candidates={RETRIEVAL_K}, k={K}, n={N_SETS}):");
    eprintln!("  NoisyScore   {noisy_us:8.2} µs/q   {:.2} M QPS", 1.0 / noisy_us);
    eprintln!("  GnnDiffusion {gnn_us:8.2} µs/q   {:.2} M QPS", 1.0 / gnn_us);
    eprintln!("  diffusion overhead: {:.1}× baseline", gnn_us / noisy_us.max(1e-6));

    assert!(
        gnn_us < BUDGET_US,
        "GnnDiffusion rerank latency {gnn_us:.1} µs/q exceeds {BUDGET_US} µs budget (perf regression)"
    );
}
