//! Attention Mechanism Benchmarks for Genomic Sequences
//!
//! Benchmarks attention operations with parameters realistic for genomic
//! sequence analysis (long sequences, varying model dimensions).
//!
//! Run: cargo bench -p ruvector-dna-bench --bench bench_attention_genomic

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};

use ruvector_attention::attention::ScaledDotProductAttention;
use ruvector_attention::sparse::FlashAttention;
use ruvector_attention::traits::Attention;

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
            (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn gen_vectors(count: usize, dim: usize, base_seed: u64) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| gen_vector(dim, base_seed.wrapping_add(i as u64)))
        .collect()
}

// ---------------------------------------------------------------------------
// Benchmark: Flash Attention Forward Pass (varying sequence lengths)
// ---------------------------------------------------------------------------

fn bench_flash_attention_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_genomic/flash_forward");
    group.sample_size(10);

    let dim = 64; // Head dimension
    let block_size = 64;

    // Genomic sequence lengths: 1K, 4K, 16K tokens
    // (64K is very large for single-query attention, included only as a
    //  stress test if the machine has enough memory.)
    for &seq_len in &[1_024usize, 4_096, 16_384] {
        let flash = FlashAttention::new(dim, block_size);

        let query = gen_vector(dim, 42);
        let keys = gen_vectors(seq_len, dim, 100);
        let values = gen_vectors(seq_len, dim, 200);

        let keys_refs: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
        let values_refs: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();

        group.throughput(Throughput::Elements(seq_len as u64));
        group.bench_with_input(
            BenchmarkId::new("seq_len", seq_len),
            &(query.as_slice(), keys_refs.as_slice(), values_refs.as_slice()),
            |b, (q, k, v)| {
                b.iter(|| flash.compute(q, k, v).expect("flash forward"));
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Standard Attention Forward Pass (for comparison)
// ---------------------------------------------------------------------------

fn bench_standard_attention_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_genomic/standard_forward");
    group.sample_size(10);

    let dim = 64;

    // Standard attention is O(n^2) so keep lengths modest
    for &seq_len in &[256usize, 1_024, 4_096] {
        let attention = ScaledDotProductAttention::new(dim);

        let query = gen_vector(dim, 42);
        let keys = gen_vectors(seq_len, dim, 100);
        let values = gen_vectors(seq_len, dim, 200);

        let keys_refs: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
        let values_refs: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();

        group.throughput(Throughput::Elements(seq_len as u64));
        group.bench_with_input(
            BenchmarkId::new("seq_len", seq_len),
            &(query.as_slice(), keys_refs.as_slice(), values_refs.as_slice()),
            |b, (q, k, v)| {
                b.iter(|| attention.compute(q, k, v).expect("standard forward"));
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Flash vs Standard Throughput Comparison
//
// Fixed sequence length, measure tokens/second for both.
// ---------------------------------------------------------------------------

fn bench_throughput_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_genomic/throughput_comparison");

    let dim = 64;
    let seq_len = 4_096;

    let query = gen_vector(dim, 42);
    let keys = gen_vectors(seq_len, dim, 100);
    let values = gen_vectors(seq_len, dim, 200);
    let keys_refs: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
    let values_refs: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();

    group.throughput(Throughput::Elements(seq_len as u64));

    group.bench_function("flash_attention", |b| {
        let flash = FlashAttention::new(dim, 64);
        b.iter(|| flash.compute(&query, &keys_refs, &values_refs).expect("flash"));
    });

    group.bench_function("standard_attention", |b| {
        let standard = ScaledDotProductAttention::new(dim);
        b.iter(|| standard.compute(&query, &keys_refs, &values_refs).expect("standard"));
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Memory Usage Comparison (Flash vs Naive)
//
// We approximate memory usage by measuring the peak allocation size
// difference. In practice, flash attention uses O(block_size) intermediate
// memory vs O(n^2) for standard attention. We benchmark the wall-clock
// cost of the memory-efficient path.
// ---------------------------------------------------------------------------

fn bench_memory_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_genomic/memory_comparison");
    group.sample_size(10);

    let dim = 64;

    for &seq_len in &[1_024usize, 4_096, 16_384] {
        let query = gen_vector(dim, 42);
        let keys = gen_vectors(seq_len, dim, 100);
        let values = gen_vectors(seq_len, dim, 200);
        let keys_refs: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
        let values_refs: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();

        // Flash attention: O(block_size) intermediate memory
        group.bench_with_input(
            BenchmarkId::new("flash_block64", seq_len),
            &(query.as_slice(), keys_refs.as_slice(), values_refs.as_slice()),
            |b, (q, k, v)| {
                let flash = FlashAttention::new(dim, 64);
                b.iter(|| flash.compute(q, k, v).expect("flash"));
            },
        );

        // Flash attention with smaller block size: even less memory
        group.bench_with_input(
            BenchmarkId::new("flash_block16", seq_len),
            &(query.as_slice(), keys_refs.as_slice(), values_refs.as_slice()),
            |b, (q, k, v)| {
                let flash = FlashAttention::new(dim, 16);
                b.iter(|| flash.compute(q, k, v).expect("flash"));
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Varying Block Size Impact
// ---------------------------------------------------------------------------

fn bench_block_size_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_genomic/block_size_impact");

    let dim = 64;
    let seq_len = 4_096;

    let query = gen_vector(dim, 42);
    let keys = gen_vectors(seq_len, dim, 100);
    let values = gen_vectors(seq_len, dim, 200);
    let keys_refs: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
    let values_refs: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();

    for &block_size in &[8, 16, 32, 64, 128, 256] {
        group.bench_with_input(
            BenchmarkId::new("block_size", block_size),
            &(query.as_slice(), keys_refs.as_slice(), values_refs.as_slice()),
            |b, (q, k, v)| {
                let flash = FlashAttention::new(dim, block_size);
                b.iter(|| flash.compute(q, k, v).expect("flash"));
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Causal vs Non-Causal Flash Attention
// ---------------------------------------------------------------------------

fn bench_causal_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_genomic/causal_vs_noncausal");

    let dim = 64;
    let seq_len = 4_096;

    let query = gen_vector(dim, 42);
    let keys = gen_vectors(seq_len, dim, 100);
    let values = gen_vectors(seq_len, dim, 200);
    let keys_refs: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
    let values_refs: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();

    group.throughput(Throughput::Elements(seq_len as u64));

    group.bench_function("non_causal", |b| {
        let flash = FlashAttention::new(dim, 64);
        b.iter(|| flash.compute(&query, &keys_refs, &values_refs).expect("non-causal"));
    });

    group.bench_function("causal", |b| {
        let flash = FlashAttention::causal(dim, 64);
        b.iter(|| flash.compute(&query, &keys_refs, &values_refs).expect("causal"));
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Dimension Scaling
// ---------------------------------------------------------------------------

fn bench_dimension_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_genomic/dimension_scaling");

    let seq_len = 2_048;

    for &dim in &[32, 64, 128, 256, 512] {
        let flash = FlashAttention::new(dim, 64);

        let query = gen_vector(dim, 42);
        let keys = gen_vectors(seq_len, dim, 100);
        let values = gen_vectors(seq_len, dim, 200);
        let keys_refs: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
        let values_refs: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();

        group.throughput(Throughput::Elements(seq_len as u64));
        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &(query, keys_refs, values_refs),
            |b, (q, k, v)| {
                b.iter(|| flash.compute(q, k, v).expect("flash"));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_flash_attention_forward,
    bench_standard_attention_forward,
    bench_throughput_comparison,
    bench_memory_comparison,
    bench_block_size_impact,
    bench_causal_attention,
    bench_dimension_scaling,
);
criterion_main!(benches);
