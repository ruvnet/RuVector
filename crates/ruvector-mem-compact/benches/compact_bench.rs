use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ruvector_mem_compact::dataset::{generate_store, DatasetParams};
use ruvector_mem_compact::{
    AgeTtlCompactor, GraphCutCompactor, MemoryCompactor, ThresholdCompactor,
};

fn bench_compaction(c: &mut Criterion) {
    let params = DatasetParams {
        n: 2000,
        dims: 64,
        n_clusters: 10,
        cluster_std: 0.25,
        seed: 42,
    };
    let store = generate_store(&params);

    let mut group = c.benchmark_group("compaction");

    for target in [0.5f32] {
        group.bench_with_input(
            BenchmarkId::new("age_ttl", format!("r{target:.0}")),
            &target,
            |b, &ratio| {
                b.iter(|| AgeTtlCompactor.compact(&store, ratio));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("threshold", format!("r{target:.0}")),
            &target,
            |b, &ratio| {
                b.iter(|| ThresholdCompactor::new(0.90).compact(&store, ratio));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("graph_cut", format!("r{target:.0}")),
            &target,
            |b, &ratio| {
                b.iter(|| GraphCutCompactor::new(8, 0.85).compact(&store, ratio));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_compaction);
criterion_main!(benches);
