use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Uniform};
use ruvector_soar::{IndexKind, SoarConfig, SoarIndex};

fn gen_data(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let range = Uniform::new(-1.0f32, 1.0);
    let noise = Normal::new(0.0f64, 0.3).unwrap();
    let centroids: Vec<Vec<f32>> = (0..20)
        .map(|_| (0..d).map(|_| range.sample(&mut rng)).collect())
        .collect();
    (0..n)
        .map(|i| {
            let c = &centroids[i % 20];
            c.iter()
                .map(|&x| x + noise.sample(&mut rng) as f32)
                .collect()
        })
        .collect()
}

fn bench_search(c: &mut Criterion) {
    let n = 5_000;
    let d = 128;
    let nq = 50;
    let k = 10;
    let corpus = gen_data(n, d, 1);
    let queries = gen_data(nq, d, 2);

    let mut group = c.benchmark_group("soar_search");

    for &nprobe in &[4usize, 8, 16] {
        // IVF-PQ
        let ivf = SoarIndex::build(
            corpus.clone(),
            SoarConfig {
                kind: IndexKind::IvfPq,
                nlist: 64,
                nprobe,
                m_pq: 8,
                ..Default::default()
            },
        )
        .unwrap();
        group.bench_with_input(BenchmarkId::new("IVF-PQ", nprobe), &nprobe, |b, _| {
            b.iter(|| {
                for q in &queries {
                    let _ = ivf.search(q, k).unwrap();
                }
            })
        });

        // SOAR-IVF-PQ
        let soar = SoarIndex::build(
            corpus.clone(),
            SoarConfig {
                kind: IndexKind::SoarIvfPq,
                nlist: 64,
                nprobe,
                m_pq: 8,
                lambda: 1.0,
                n_secondary_candidates: 10,
                ..Default::default()
            },
        )
        .unwrap();
        group.bench_with_input(BenchmarkId::new("SOAR-IVF-PQ", nprobe), &nprobe, |b, _| {
            b.iter(|| {
                for q in &queries {
                    let _ = soar.search(q, k).unwrap();
                }
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
