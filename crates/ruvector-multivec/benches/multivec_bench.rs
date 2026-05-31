//! Criterion benchmarks for ruvector-multivec.
//!
//! Two groups:
//!
//!   `scoring_kernels` — per-query cost of centroid dot, MaxSim, Chamfer,
//!                       and FDE encode+dot at dim ∈ {64, 128, 256} with
//!                       T ∈ {8, 32} tokens per document.
//!
//!   `index_search`    — end-to-end search at n ∈ {1K, 5K, 10K} for all
//!                       three index variants.
//!
//! Run: cargo bench -p ruvector-multivec --bench multivec_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use ruvector_multivec::{
    index::{CentroidIndex, MaxSimIndex, MultiVecIndex, MuveraFdeIndex},
    scoring::{centroid_dot, chamfer_score, dot, l2_normalize, maxsim_exact, FdeEncoder},
};

fn make_tokens(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f64, 1.0).unwrap();
    (0..count)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rng) as f32).collect();
            l2_normalize(&mut v);
            v
        })
        .collect()
}

fn make_corpus(n_docs: usize, t: usize, dim: usize, seed: u64) -> Vec<(usize, Vec<Vec<f32>>)> {
    (0..n_docs)
        .map(|id| {
            let tokens = make_tokens(t, dim, seed.wrapping_add(id as u64));
            (id, tokens)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Scoring kernel benchmarks
// ---------------------------------------------------------------------------

fn bench_scoring_kernels(c: &mut Criterion) {
    let mut g = c.benchmark_group("scoring_kernels");

    for (dim, t) in [(64usize, 8usize), (128, 8), (128, 32), (256, 32)] {
        let qt = make_tokens(8, dim, 1);
        let dt = make_tokens(t, dim, 2);
        let label = format!("D{dim}_T{t}");

        g.bench_with_input(BenchmarkId::new("centroid_dot", &label), &(), |b, _| {
            b.iter(|| black_box(centroid_dot(black_box(&qt), black_box(&dt))))
        });

        g.bench_with_input(BenchmarkId::new("maxsim_exact", &label), &(), |b, _| {
            b.iter(|| black_box(maxsim_exact(black_box(&qt), black_box(&dt))))
        });

        g.bench_with_input(BenchmarkId::new("chamfer_score", &label), &(), |b, _| {
            b.iter(|| black_box(chamfer_score(black_box(&qt), black_box(&dt))))
        });

        // FDE encode + dot.
        let m = if dim >= 128 { 8 } else { 4 };
        let enc = FdeEncoder::new(dim, m, 2, 42);
        g.bench_with_input(BenchmarkId::new("fde_encode_dot", &label), &(), |b, _| {
            b.iter(|| {
                let qfde = enc.encode(black_box(&qt));
                let dfde = enc.encode(black_box(&dt));
                black_box(dot(&qfde, &dfde))
            })
        });
    }
    g.finish();
}

// ---------------------------------------------------------------------------
// End-to-end index search benchmarks
// ---------------------------------------------------------------------------

fn bench_index_search(c: &mut Criterion) {
    let mut g = c.benchmark_group("index_search");
    let dim = 128;
    let t = 16;
    let k = 10;

    for n in [1_000usize, 5_000, 10_000] {
        let corpus = make_corpus(n, t, dim, 77);
        let query = make_tokens(8, dim, 999);
        let label = format!("n{n}_D{dim}_T{t}");

        // CentroidIndex
        let mut cidx = CentroidIndex::new(dim);
        for (id, toks) in &corpus {
            cidx.add(*id, toks.clone()).unwrap();
        }
        g.bench_with_input(BenchmarkId::new("centroid", &label), &(), |b, _| {
            b.iter(|| black_box(cidx.search(black_box(&query), k).unwrap()))
        });

        // MaxSimIndex
        let mut midx = MaxSimIndex::new(dim);
        for (id, toks) in &corpus {
            midx.add(*id, toks.clone()).unwrap();
        }
        g.bench_with_input(BenchmarkId::new("maxsim", &label), &(), |b, _| {
            b.iter(|| black_box(midx.search(black_box(&query), k).unwrap()))
        });

        // MuveraFdeIndex
        let mut fidx = MuveraFdeIndex::new(dim, 8, 4, 42).unwrap();
        for (id, toks) in &corpus {
            fidx.add(*id, toks.clone()).unwrap();
        }
        g.bench_with_input(BenchmarkId::new("muvera_fde", &label), &(), |b, _| {
            b.iter(|| black_box(fidx.search(black_box(&query), k).unwrap()))
        });
    }
    g.finish();
}

criterion_group!(benches, bench_scoring_kernels, bench_index_search);
criterion_main!(benches);
