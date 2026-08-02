//! Row-major batch distance vs PDX vertical layout (ADR-279 §5.1).
//!
//! Both compute the same thing over the same data. The only difference is
//! memory layout and whether a horizontal reduction happens per vector.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ruvector_core::pdx::PdxIndex;
use ruvector_core::simd_intrinsics::batch_euclidean;

fn corpus(n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            (0..dim)
                .map(|d| ((i * 31 + d * 17) % 101) as f32 / 101.0)
                .collect()
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    // 1536 is the OpenAI embedding width; 768 covers most sentence encoders.
    // (n, dim) pairs: the first three are cache-resident, the rest stream.
    for (n, dim) in [(256usize, 768usize), (512, 768), (1024, 768), (4096, 768), (4096, 1536)] {
        let vecs = corpus(n, dim);
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let query: Vec<f32> = (0..dim).map(|d| (d % 7) as f32 / 7.0).collect();

        let index = PdxIndex::from_rows(&refs);
        let mut out = vec![0.0f32; n];

        let mut group = c.benchmark_group(format!("batch_euclidean_{n}x{dim}"));
        // Bytes of vector data scanned per query — makes the two comparable
        // in bandwidth terms rather than just wall time.
        group.throughput(Throughput::Bytes((n * dim * 4) as u64));

        group.bench_function(BenchmarkId::new("row_major", dim), |b| {
            b.iter(|| {
                batch_euclidean(black_box(&query), black_box(&refs), black_box(&mut out));
            })
        });

        group.bench_function(BenchmarkId::new("pdx_vertical", dim), |b| {
            b.iter(|| {
                index.euclidean_sq_into(black_box(&query), black_box(&mut out));
            })
        });

        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
