//! Quantization Benchmarks
//!
//! Benchmarks quantization throughput, distance computation speedups, and
//! reconstruction accuracy for genomic embedding vectors.
//!
//! Run: cargo bench -p ruvector-dna-bench --bench bench_quantization

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};

use ruvector_core::quantization::{
    BinaryQuantized, Int4Quantized, QuantizedVector, ScalarQuantized,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random vector generator.
fn gen_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..dim)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = ((state >> 33) ^ state) as u32;
            // Map to [-1.0, 1.0] -- typical range for normalized embeddings
            (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn gen_vectors(count: usize, dim: usize, base_seed: u64) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| gen_vector(dim, base_seed.wrapping_add(i as u64)))
        .collect()
}

/// Naive f32 Euclidean squared distance for baseline comparison.
fn f32_distance_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

// ===========================================================================
// Benchmark: f32 -> Binary Quantization Throughput
// ===========================================================================

fn bench_binary_quantize_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization/binary_quantize");

    let dim = 384;

    for &batch_size in &[1usize, 100, 1_000, 10_000] {
        let vectors = gen_vectors(batch_size, dim, 42);

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("vectors", batch_size),
            &vectors,
            |b, vectors| {
                b.iter(|| {
                    let quantized: Vec<BinaryQuantized> = vectors
                        .iter()
                        .map(|v| BinaryQuantized::quantize(v))
                        .collect();
                    quantized
                });
            },
        );
    }

    group.finish();
}

// ===========================================================================
// Benchmark: f32 -> INT4 Quantization Throughput
// ===========================================================================

fn bench_int4_quantize_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization/int4_quantize");

    let dim = 384;

    for &batch_size in &[1usize, 100, 1_000, 10_000] {
        let vectors = gen_vectors(batch_size, dim, 42);

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("vectors", batch_size),
            &vectors,
            |b, vectors| {
                b.iter(|| {
                    let quantized: Vec<Int4Quantized> = vectors
                        .iter()
                        .map(|v| Int4Quantized::quantize(v))
                        .collect();
                    quantized
                });
            },
        );
    }

    group.finish();
}

// ===========================================================================
// Benchmark: f32 -> Scalar (u8) Quantization Throughput
// ===========================================================================

fn bench_scalar_quantize_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization/scalar_quantize");

    let dim = 384;

    for &batch_size in &[1usize, 100, 1_000, 10_000] {
        let vectors = gen_vectors(batch_size, dim, 42);

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("vectors", batch_size),
            &vectors,
            |b, vectors| {
                b.iter(|| {
                    let quantized: Vec<ScalarQuantized> = vectors
                        .iter()
                        .map(|v| ScalarQuantized::quantize(v))
                        .collect();
                    quantized
                });
            },
        );
    }

    group.finish();
}

// ===========================================================================
// Benchmark: Distance Computation with Quantized Vectors
// ===========================================================================

fn bench_binary_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization/binary_distance");

    for &dim in &[128usize, 384, 768, 1536] {
        let v1 = gen_vector(dim, 42);
        let v2 = gen_vector(dim, 123);
        let q1 = BinaryQuantized::quantize(&v1);
        let q2 = BinaryQuantized::quantize(&v2);

        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &(q1.clone(), q2.clone()),
            |b, (q1, q2)| {
                b.iter(|| q1.distance(q2));
            },
        );
    }

    group.finish();
}

fn bench_int4_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization/int4_distance");

    for &dim in &[128usize, 384, 768, 1536] {
        let v1 = gen_vector(dim, 42);
        let v2 = gen_vector(dim, 123);
        let q1 = Int4Quantized::quantize(&v1);
        let q2 = Int4Quantized::quantize(&v2);

        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &(q1.clone(), q2.clone()),
            |b, (q1, q2)| {
                b.iter(|| q1.distance(&q2));
            },
        );
    }

    group.finish();
}

fn bench_scalar_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization/scalar_distance");

    for &dim in &[128usize, 384, 768, 1536] {
        let v1 = gen_vector(dim, 42);
        let v2 = gen_vector(dim, 123);
        let q1 = ScalarQuantized::quantize(&v1);
        let q2 = ScalarQuantized::quantize(&v2);

        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &(q1.clone(), q2.clone()),
            |b, (q1, q2)| {
                b.iter(|| q1.distance(q2));
            },
        );
    }

    group.finish();
}

// ===========================================================================
// Benchmark: Distance Speedup (quantized vs f32 baseline)
// ===========================================================================

fn bench_distance_speedup_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization/distance_speedup");

    let dim = 384;
    let v1 = gen_vector(dim, 42);
    let v2 = gen_vector(dim, 123);

    // Pre-quantize
    let scalar_q1 = ScalarQuantized::quantize(&v1);
    let scalar_q2 = ScalarQuantized::quantize(&v2);
    let int4_q1 = Int4Quantized::quantize(&v1);
    let int4_q2 = Int4Quantized::quantize(&v2);
    let binary_q1 = BinaryQuantized::quantize(&v1);
    let binary_q2 = BinaryQuantized::quantize(&v2);

    group.throughput(Throughput::Elements(dim as u64));

    group.bench_function("f32_euclidean_sq", |b| {
        b.iter(|| f32_distance_sq(&v1, &v2));
    });

    group.bench_function("scalar_u8_distance", |b| {
        b.iter(|| scalar_q1.distance(&scalar_q2));
    });

    group.bench_function("int4_distance", |b| {
        b.iter(|| int4_q1.distance(&int4_q2));
    });

    group.bench_function("binary_hamming", |b| {
        b.iter(|| binary_q1.distance(&binary_q2));
    });

    group.finish();
}

// ===========================================================================
// Benchmark: Batch Distance Computation
// ===========================================================================

fn bench_batch_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization/batch_distance");
    group.sample_size(10);

    let dim = 384;
    let num_vectors = 10_000;

    let query = gen_vector(dim, 42);
    let vectors = gen_vectors(num_vectors, dim, 100);

    // Pre-quantize all vectors
    let binary_query = BinaryQuantized::quantize(&query);
    let binary_db: Vec<BinaryQuantized> = vectors
        .iter()
        .map(|v| BinaryQuantized::quantize(v))
        .collect();

    let scalar_query = ScalarQuantized::quantize(&query);
    let scalar_db: Vec<ScalarQuantized> = vectors
        .iter()
        .map(|v| ScalarQuantized::quantize(v))
        .collect();

    group.throughput(Throughput::Elements(num_vectors as u64));

    group.bench_function("f32_batch_10k", |b| {
        b.iter(|| {
            let dists: Vec<f32> = vectors.iter().map(|v| f32_distance_sq(&query, v)).collect();
            dists
        });
    });

    group.bench_function("binary_batch_10k", |b| {
        b.iter(|| {
            let dists: Vec<f32> = binary_db
                .iter()
                .map(|v| binary_query.distance(v))
                .collect();
            dists
        });
    });

    group.bench_function("scalar_batch_10k", |b| {
        b.iter(|| {
            let dists: Vec<f32> = scalar_db
                .iter()
                .map(|v| scalar_query.distance(v))
                .collect();
            dists
        });
    });

    group.finish();
}

// ===========================================================================
// Benchmark: Reconstruction Accuracy (quality check alongside speed)
// ===========================================================================

fn bench_reconstruction(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization/reconstruction");

    let dim = 384;

    for &batch_size in &[100usize, 1_000] {
        let vectors = gen_vectors(batch_size, dim, 42);

        // Binary reconstruction
        group.bench_with_input(
            BenchmarkId::new("binary_reconstruct", batch_size),
            &vectors,
            |b, vectors| {
                let quantized: Vec<BinaryQuantized> = vectors
                    .iter()
                    .map(|v| BinaryQuantized::quantize(v))
                    .collect();
                b.iter(|| {
                    let reconstructed: Vec<Vec<f32>> =
                        quantized.iter().map(|q| q.reconstruct()).collect();
                    reconstructed
                });
            },
        );

        // Scalar reconstruction
        group.bench_with_input(
            BenchmarkId::new("scalar_reconstruct", batch_size),
            &vectors,
            |b, vectors| {
                let quantized: Vec<ScalarQuantized> = vectors
                    .iter()
                    .map(|v| ScalarQuantized::quantize(v))
                    .collect();
                b.iter(|| {
                    let reconstructed: Vec<Vec<f32>> =
                        quantized.iter().map(|q| q.reconstruct()).collect();
                    reconstructed
                });
            },
        );

        // Int4 reconstruction
        group.bench_with_input(
            BenchmarkId::new("int4_reconstruct", batch_size),
            &vectors,
            |b, vectors| {
                let quantized: Vec<Int4Quantized> = vectors
                    .iter()
                    .map(|v| Int4Quantized::quantize(v))
                    .collect();
                b.iter(|| {
                    let reconstructed: Vec<Vec<f32>> =
                        quantized.iter().map(|q| q.reconstruct()).collect();
                    reconstructed
                });
            },
        );
    }

    group.finish();
}

// ===========================================================================
// Benchmark: Dimension Scaling (quantization cost vs vector size)
// ===========================================================================

fn bench_dimension_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization/dimension_scaling");

    for &dim in &[64usize, 128, 256, 384, 768, 1536, 3072] {
        let v = gen_vector(dim, 42);

        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(
            BenchmarkId::new("binary_quantize", dim),
            &v,
            |b, v| {
                b.iter(|| BinaryQuantized::quantize(v));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("int4_quantize", dim),
            &v,
            |b, v| {
                b.iter(|| Int4Quantized::quantize(v));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("scalar_quantize", dim),
            &v,
            |b, v| {
                b.iter(|| ScalarQuantized::quantize(v));
            },
        );
    }

    group.finish();
}

// ===========================================================================
// Benchmark: Memory Footprint (bytes per quantized vector)
// ===========================================================================

fn bench_memory_footprint(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization/memory_footprint");

    let dim = 384;
    let v = gen_vector(dim, 42);

    group.bench_function("measure_sizes", |b| {
        b.iter(|| {
            let f32_size = dim * std::mem::size_of::<f32>();
            let scalar = ScalarQuantized::quantize(&v);
            let scalar_size = scalar.data.len() + std::mem::size_of::<ScalarQuantized>();
            let int4 = Int4Quantized::quantize(&v);
            let int4_size = int4.data.len() + std::mem::size_of::<Int4Quantized>();
            let binary = BinaryQuantized::quantize(&v);
            let binary_size = binary.bits.len() + std::mem::size_of::<BinaryQuantized>();
            (f32_size, scalar_size, int4_size, binary_size)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_binary_quantize_throughput,
    bench_int4_quantize_throughput,
    bench_scalar_quantize_throughput,
    bench_binary_distance,
    bench_int4_distance,
    bench_scalar_distance,
    bench_distance_speedup_comparison,
    bench_batch_distance,
    bench_reconstruction,
    bench_dimension_scaling,
    bench_memory_footprint,
);
criterion_main!(benches);
