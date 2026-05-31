use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_muvera::{FdeConfig, FdeEncoder, MuveraIndex};

const DIM: usize = 128;
const N_TOKENS: usize = 32;
const N_DOCS_BENCH: usize = 1_000;

fn random_unit_vec(rng: &mut impl Rng, dim: usize) -> Vec<f32> {
    let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(f32::EPSILON);
    v.into_iter().map(|x| x / norm).collect()
}

fn maxsim(doc: &[Vec<f32>], query: &[Vec<f32>]) -> f32 {
    query
        .iter()
        .map(|q| {
            doc.iter()
                .map(|d| q.iter().zip(d.iter()).map(|(a, b)| a * b).sum::<f32>())
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

// ── Encode benchmark (single document) ────────────────────────────────────────

fn bench_encode(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let tokens: Vec<Vec<f32>> =
        (0..N_TOKENS).map(|_| random_unit_vec(&mut rng, DIM)).collect();

    let mut g = c.benchmark_group("fde_encode_single_doc");
    g.throughput(Throughput::Elements(N_TOKENS as u64));

    for (label, cfg) in [
        ("B=8,dp=8,R=4", FdeConfig { dim: DIM, buckets: 8, d_proj: 8, reps: 4 }),
        ("B=16,dp=16,R=4", FdeConfig { dim: DIM, buckets: 16, d_proj: 16, reps: 4 }),
        ("B=32,dp=16,R=4", FdeConfig { dim: DIM, buckets: 32, d_proj: 16, reps: 4 }),
    ] {
        let mut enc_rng = StdRng::seed_from_u64(7);
        let encoder = FdeEncoder::new(cfg, &mut enc_rng).unwrap();
        g.bench_with_input(BenchmarkId::new("encode", label), &encoder, |b, enc| {
            b.iter(|| enc.encode(black_box(&tokens)).unwrap())
        });
    }
    g.finish();
}

// ── Search benchmark (1 K docs, flat scan) ────────────────────────────────────

fn bench_search(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let docs: Vec<Vec<Vec<f32>>> = (0..N_DOCS_BENCH)
        .map(|_| (0..N_TOKENS).map(|_| random_unit_vec(&mut rng, DIM)).collect())
        .collect();
    let query: Vec<Vec<f32>> =
        (0..N_TOKENS).map(|_| random_unit_vec(&mut rng, DIM)).collect();

    let mut g = c.benchmark_group("search_1k_docs");
    g.throughput(Throughput::Elements(N_DOCS_BENCH as u64));

    // Baseline: brute-force MaxSim.
    g.bench_function("brute_force_maxsim", |b| {
        b.iter(|| {
            let mut best = f32::NEG_INFINITY;
            let mut best_idx = 0usize;
            for (i, doc) in black_box(&docs).iter().enumerate() {
                let s = maxsim(doc, black_box(&query));
                if s > best {
                    best = s;
                    best_idx = i;
                }
            }
            black_box(best_idx)
        })
    });

    for (label, cfg) in [
        ("fde_B8_dp8_R4", FdeConfig { dim: DIM, buckets: 8, d_proj: 8, reps: 4 }),
        ("fde_B16_dp16_R4", FdeConfig { dim: DIM, buckets: 16, d_proj: 16, reps: 4 }),
        ("fde_B32_dp16_R4", FdeConfig { dim: DIM, buckets: 32, d_proj: 16, reps: 4 }),
    ] {
        let mut enc_rng = StdRng::seed_from_u64(7);
        let encoder = FdeEncoder::new(cfg, &mut enc_rng).unwrap();
        let mut index = MuveraIndex::new(encoder);
        for (i, doc) in docs.iter().enumerate() {
            index.insert(i.to_string(), doc).unwrap();
        }
        g.bench_with_input(BenchmarkId::new("muvera_flat", label), &index, |b, idx| {
            b.iter(|| idx.search(black_box(&query), 10).unwrap())
        });
    }
    g.finish();
}

criterion_group!(benches, bench_encode, bench_search);
criterion_main!(benches);
