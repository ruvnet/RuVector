use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::SmallRng, SeedableRng};
use rand_distr::{Distribution, Normal};
use ruvector_drift::{
    centroid::CentroidDriftDetector, graph::GraphDriftDetector, mmd::MmdDriftDetector,
    DriftDetector,
};

fn make_vecs(n: usize, dims: usize, mean: f32, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let dist = Normal::new(mean, 1.0f32).unwrap();
    (0..n)
        .map(|_| (0..dims).map(|_| dist.sample(&mut rng)).collect())
        .collect()
}

fn bench_observe(c: &mut Criterion) {
    let mut group = c.benchmark_group("observe_latency");

    for dims in [64usize, 128, 256] {
        let ref_data = make_vecs(500, dims, 0.0, 1);
        let query = make_vecs(1, dims, 0.0, 99);
        let v = &query[0];

        let sigma = (dims as f32).sqrt();

        group.bench_with_input(BenchmarkId::new("centroid", dims), &dims, |b, _| {
            let mut det = CentroidDriftDetector::new(&ref_data, 200, 0.3);
            b.iter(|| det.observe(criterion::black_box(v)));
        });

        group.bench_with_input(BenchmarkId::new("mmd-rff/D=128", dims), &dims, |b, _| {
            let mut det = MmdDriftDetector::new(&ref_data, 128, sigma, 200, 0.05);
            b.iter(|| det.observe(criterion::black_box(v)));
        });

        group.bench_with_input(BenchmarkId::new("graph-knn/k=7", dims), &dims, |b, _| {
            let ref_small = make_vecs(100, dims, 0.0, 1);
            let mut det = GraphDriftDetector::new(&ref_small, 7, 50, 0.25);
            b.iter(|| det.observe(criterion::black_box(v)));
        });
    }

    group.finish();
}

fn bench_report(c: &mut Criterion) {
    let mut group = c.benchmark_group("report_latency");
    let dims = 128;

    let ref_data = make_vecs(500, dims, 0.0, 1);
    let query = make_vecs(100, dims, 1.0, 77);

    group.bench_function("centroid/report", |b| {
        let mut det = CentroidDriftDetector::new(&ref_data, 200, 0.3);
        for v in &query {
            det.observe(v);
        }
        b.iter(|| det.report());
    });

    group.bench_function("mmd-rff/report", |b| {
        let sigma = (dims as f32).sqrt();
        let mut det = MmdDriftDetector::new(&ref_data, 128, sigma, 200, 0.05);
        for v in &query {
            det.observe(v);
        }
        b.iter(|| det.report());
    });

    group.bench_function("graph-knn/report", |b| {
        let ref_small = make_vecs(100, dims, 0.0, 1);
        let cur = make_vecs(50, dims, 1.0, 77);
        let mut det = GraphDriftDetector::new(&ref_small, 7, 50, 0.25);
        for v in &cur {
            det.observe(v);
        }
        b.iter(|| det.report());
    });

    group.finish();
}

criterion_group!(benches, bench_observe, bench_report);
criterion_main!(benches);
