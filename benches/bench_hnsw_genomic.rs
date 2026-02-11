//! HNSW Index Benchmarks for Genomic Workloads
//!
//! Benchmarks HNSW operations with parameters realistic for k-mer embedding
//! vectors (384-dimensional, normalized, cosine metric).
//!
//! Run: cargo bench -p ruvector-dna-bench --bench bench_hnsw_genomic

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use ruvector_core::index::hnsw::HnswIndex;
use ruvector_core::index::VectorIndex;
use ruvector_core::types::{DistanceMetric, HnswConfig};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random vector generator (LCG-based, no external deps).
fn gen_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let raw: Vec<f32> = (0..dim)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = ((state >> 33) ^ state) as u32;
            (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    // Normalize for cosine metric
    let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        raw.iter().map(|x| x / norm).collect()
    } else {
        raw
    }
}

fn gen_vectors(count: usize, dim: usize, base_seed: u64) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| gen_vector(dim, base_seed.wrapping_add(i as u64)))
        .collect()
}

/// Build an HNSW index with `n` vectors of dimension `dim`.
fn build_index(n: usize, dim: usize) -> HnswIndex {
    let config = HnswConfig {
        m: 16,
        ef_construction: 128,
        ef_search: 64,
        max_elements: n + 1024,
    };
    let mut index = HnswIndex::new(dim, DistanceMetric::Cosine, config)
        .expect("Failed to create HNSW index");

    let vectors = gen_vectors(n, dim, 42);
    let entries: Vec<(String, Vec<f32>)> = vectors
        .into_iter()
        .enumerate()
        .map(|(i, v)| (format!("kmer_{}", i), v))
        .collect();

    index.add_batch(entries).expect("Failed to insert batch");
    index
}

// ---------------------------------------------------------------------------
// Benchmark: Index Construction
// ---------------------------------------------------------------------------

fn bench_index_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_genomic/index_construction");
    group.sample_size(10);

    let dim = 384; // Typical k-mer embedding dimension

    for &n in &[10_000u64, 100_000] {
        group.throughput(Throughput::Elements(n));
        group.bench_with_input(BenchmarkId::new("vectors", n), &n, |b, &n| {
            b.iter(|| {
                let config = HnswConfig {
                    m: 16,
                    ef_construction: 128,
                    ef_search: 64,
                    max_elements: n as usize + 1024,
                };
                let mut index = HnswIndex::new(dim, DistanceMetric::Cosine, config)
                    .expect("create index");

                let vectors = gen_vectors(n as usize, dim, 42);
                let entries: Vec<(String, Vec<f32>)> = vectors
                    .into_iter()
                    .enumerate()
                    .map(|(i, v)| (format!("kmer_{}", i), v))
                    .collect();

                index.add_batch(entries).expect("batch insert");
                index
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Single-Query Search Latency
// ---------------------------------------------------------------------------

fn bench_single_query_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_genomic/single_query_search");

    let dim = 384;
    let n = 10_000;
    let index = build_index(n, dim);
    let query = gen_vector(dim, 9999);

    for &k in &[1, 10, 50, 100] {
        group.bench_with_input(BenchmarkId::new("top_k", k), &k, |b, &k| {
            b.iter(|| index.search(&query, k).expect("search"));
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Search with varying ef_search (quality vs speed)
// ---------------------------------------------------------------------------

fn bench_ef_search_tradeoff(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_genomic/ef_search_tradeoff");

    let dim = 384;
    let n = 10_000;
    let index = build_index(n, dim);
    let query = gen_vector(dim, 9999);
    let k = 10;

    for &ef in &[16, 32, 64, 128, 256, 512] {
        group.bench_with_input(BenchmarkId::new("ef_search", ef), &ef, |b, &ef| {
            b.iter(|| index.search_with_ef(&query, k, ef).expect("search"));
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Batch Search (100 queries)
// ---------------------------------------------------------------------------

fn bench_batch_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_genomic/batch_search");
    group.sample_size(10);

    let dim = 384;
    let n = 10_000;
    let index = build_index(n, dim);
    let queries = gen_vectors(100, dim, 80_000);
    let k = 10;

    group.throughput(Throughput::Elements(100));
    group.bench_function("100_queries_top10", |b| {
        b.iter(|| {
            let results: Vec<_> = queries
                .iter()
                .map(|q| index.search(q, k).expect("search"))
                .collect();
            results
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Recall@10 Measurement
//
// We insert known vectors, then search for them to verify recall.
// This is a correctness-aware benchmark: it measures search quality
// alongside latency.
// ---------------------------------------------------------------------------

fn bench_recall_at_10(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_genomic/recall_at_10");
    group.sample_size(10);

    let dim = 384;
    let n = 10_000;
    let index = build_index(n, dim);
    let queries = gen_vectors(100, dim, 42); // Same seed as index vectors

    group.bench_function("recall_measurement", |b| {
        b.iter(|| {
            let mut total_recall = 0.0f64;
            for (i, query) in queries.iter().enumerate() {
                let results = index.search(query, 10).expect("search");
                let expected_id = format!("kmer_{}", i);
                let found = results.iter().any(|r| r.id == expected_id);
                if found {
                    total_recall += 1.0;
                }
            }
            total_recall / queries.len() as f64
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Memory Usage Estimation
//
// Measures serialized index size as a proxy for memory footprint.
// ---------------------------------------------------------------------------

fn bench_memory_estimation(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_genomic/memory_estimation");
    group.sample_size(10);

    let dim = 384;

    for &n in &[1_000u64, 10_000] {
        group.bench_with_input(
            BenchmarkId::new("serialize_index", n),
            &n,
            |b, &n| {
                let index = build_index(n as usize, dim);
                b.iter(|| {
                    let bytes = index.serialize().expect("serialize");
                    bytes.len()
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Incremental Insertion
// ---------------------------------------------------------------------------

fn bench_incremental_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_genomic/incremental_insert");

    let dim = 384;
    let n = 10_000;

    group.bench_function("single_insert_into_10k", |b| {
        let mut index = build_index(n, dim);
        let mut counter = n;
        b.iter(|| {
            let v = gen_vector(dim, counter as u64 + 100_000);
            index
                .add(format!("new_{}", counter), v)
                .expect("insert");
            counter += 1;
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_index_construction,
    bench_single_query_search,
    bench_ef_search_tradeoff,
    bench_batch_search,
    bench_recall_at_10,
    bench_memory_estimation,
    bench_incremental_insert,
);
criterion_main!(benches);
