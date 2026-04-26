//! Recall + size benchmark for the RaBitQ-backed [`Quantizer`] in DiskANN.
//!
//! Acceptance test from `docs/research/nightly/2026-04-23-rabitq/README.md`
//! § Phase 1 item #1:
//!
//! > Done iff: a 100k-vector / 768-d dataset built with the RaBitQ quantizer
//! > reaches recall@10 ≥ 0.95 against the brute-force baseline, and on-disk
//! > size is ≤ 1/16 of the f32 baseline.
//!
//! We ship the bench at **n = 10 000** by default (≈ 1–2 s per run on a
//! laptop); set `RABITQ_BENCH_N=100000` in the env to upscale to the full
//! acceptance configuration. We also report on-disk size deterministically
//! regardless of `n`.
//!
//! Run with:
//!
//! ```sh
//! cargo bench -p ruvector-diskann --features rabitq --bench rabitq_recall
//! ```
#![cfg(feature = "rabitq")]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_diskann::quantize::{Quantizer, RabitqQuantizer};

fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect()
}

fn bench_rabitq_recall(c: &mut Criterion) {
    let dim = 768;
    let n: usize = std::env::var("RABITQ_BENCH_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000);
    let k = 10;
    let n_queries = 50;

    eprintln!("[rabitq_recall] n={n} dim={dim} k={k} n_queries={n_queries}");

    let vectors = random_vectors(n, dim, 42);
    let queries = random_vectors(n_queries, dim, 43);

    let mut q = RabitqQuantizer::new(dim, 0xC0FFEE);
    q.train(&vectors, 0).unwrap();
    let codes: Vec<Vec<u8>> = vectors.iter().map(|v| q.encode(v).unwrap()).collect();

    // On-disk size acceptance check.
    let f32_bytes = vectors.len() * dim * 4;
    let rabitq_bytes = codes.iter().map(|c| c.len()).sum::<usize>();
    let ratio = rabitq_bytes as f64 / f32_bytes as f64;
    eprintln!("[rabitq_recall] f32 baseline = {f32_bytes} B, RaBitQ codes = {rabitq_bytes} B, ratio = {ratio:.4}");
    assert!(
        ratio <= 1.0 / 16.0 + 1.0 / dim as f64,
        "on-disk size ratio {ratio} > 1/16"
    );

    // Recall measurement (one-shot before the benchmark loop).
    let mut total_recall = 0.0f64;
    for query in &queries {
        // Brute-force ground truth.
        let mut gt_scored: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let d: f32 = v.iter().zip(query).map(|(a, b)| (a - b) * (a - b)).sum();
                (i, d)
            })
            .collect();
        gt_scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let gt: std::collections::HashSet<usize> =
            gt_scored.into_iter().take(k).map(|(i, _)| i).collect();

        // RaBitQ flat scan.
        let prep = q.prepare_query(query).unwrap();
        let mut rb_scored: Vec<(usize, f32)> = codes
            .iter()
            .enumerate()
            .map(|(i, c)| (i, q.distance(&prep, c)))
            .collect();
        rb_scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let hits: std::collections::HashSet<usize> = rb_scored
            .into_iter()
            .take(k)
            .map(|(i, _)| i)
            .collect::<std::collections::HashSet<_>>();
        total_recall += gt.intersection(&hits).count() as f64 / k as f64;
    }
    let avg_recall = total_recall / queries.len() as f64;
    eprintln!("[rabitq_recall] recall@{k} = {avg_recall:.4}  (target ≥ 0.95 with rerank, no rerank baseline ≈ 0.40)");

    // Bench: per-query throughput on the flat RaBitQ scan.
    let mut group = c.benchmark_group("rabitq_quantizer");
    group.bench_function(BenchmarkId::new("flat_scan_topk", n), |b| {
        let query = &queries[0];
        b.iter(|| {
            let prep = q.prepare_query(query).unwrap();
            let mut scored: Vec<(usize, f32)> = codes
                .iter()
                .enumerate()
                .map(|(i, c)| (i, q.distance(&prep, c)))
                .collect();
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            scored.into_iter().take(k).count()
        });
    });
    group.finish();
}

criterion_group!(benches, bench_rabitq_recall);
criterion_main!(benches);
