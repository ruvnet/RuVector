//! TurboQuant KV Cache Compression Benchmarks
//!
//! Comprehensive benchmarks covering all TurboQuant capabilities:
//! - Compression/decompression throughput at all bit widths
//! - Batch compression scaling
//! - Inner product (asymmetric + batch) latency
//! - KV cache tier operations (push, get, get_all_kv)
//! - Three-tier TurboQuantKvCache (append, migration, retrieval)
//! - Embedding store (build_from_batch, search)
//! - Memory efficiency / compression ratios
//! - Dimension scaling (64..1024)
//!
//! Run with: cargo bench -p ruvllm --features quantize --bench turbo_quant_bench

#![allow(unused_imports, dead_code, unused_variables)]
#![cfg(feature = "quantize")]

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use rand::prelude::*;

use ruvllm::kv_cache::{TurboQuantKvCache, TurboQuantKvCacheConfig};
use ruvllm::quantize::turbo_quant::{
    TurboQuantBits, TurboQuantCacheTier, TurboQuantCompressor, TurboQuantConfig,
    TurboQuantEmbeddingStore,
};

// ============================================================================
// Helpers
// ============================================================================

fn random_vec(dim: usize, rng: &mut StdRng) -> Vec<f32> {
    (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
}

fn make_config(bits: TurboQuantBits, block_size: usize) -> TurboQuantConfig {
    TurboQuantConfig {
        bits,
        rotation_seed: 42,
        enable_qjl_residual: true,
        block_size,
    }
}

const ALL_BITS: &[(TurboQuantBits, &str)] = &[
    (TurboQuantBits::Bits2_5, "2.5bit"),
    (TurboQuantBits::Bits3_0, "3.0bit"),
    (TurboQuantBits::Bits3_5, "3.5bit"),
    (TurboQuantBits::Bits4_0, "4.0bit"),
];

const DEFAULT_DIM: usize = 128;

// ============================================================================
// 1. Compression throughput at all 4 bit widths
// ============================================================================

fn bench_compress_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("turbo_quant/compress");
    let mut rng = StdRng::seed_from_u64(0xBEEF);
    let data = random_vec(DEFAULT_DIM, &mut rng);

    for &(bits, label) in ALL_BITS {
        let config = make_config(bits, DEFAULT_DIM);
        let compressor = TurboQuantCompressor::new(config).unwrap();

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("single", label), &data, |b, data| {
            b.iter(|| {
                black_box(compressor.compress(black_box(data)).unwrap());
            });
        });
    }

    group.finish();
}

// ============================================================================
// 2. Decompression throughput
// ============================================================================

fn bench_decompress_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("turbo_quant/decompress");
    let mut rng = StdRng::seed_from_u64(0xCAFE);
    let data = random_vec(DEFAULT_DIM, &mut rng);

    for &(bits, label) in ALL_BITS {
        let config = make_config(bits, DEFAULT_DIM);
        let compressor = TurboQuantCompressor::new(config).unwrap();
        let compressed = compressor.compress(&data).unwrap();

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("single", label),
            &compressed,
            |b, compressed| {
                b.iter(|| {
                    black_box(compressor.decompress(black_box(compressed)).unwrap());
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// 3. Batch compression scaling
// ============================================================================

fn bench_batch_compress(c: &mut Criterion) {
    let mut group = c.benchmark_group("turbo_quant/compress_batch");
    group.sampling_mode(SamplingMode::Flat);
    let mut rng = StdRng::seed_from_u64(0xD00D);

    let batch_sizes: &[usize] = &[1, 10, 100, 1000];

    for &batch_size in batch_sizes {
        let vecs: Vec<Vec<f32>> = (0..batch_size)
            .map(|_| random_vec(DEFAULT_DIM, &mut rng))
            .collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

        let config = make_config(TurboQuantBits::Bits3_5, DEFAULT_DIM);
        let compressor = TurboQuantCompressor::new(config).unwrap();

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(BenchmarkId::new("3.5bit", batch_size), &refs, |b, refs| {
            b.iter(|| {
                black_box(compressor.compress_batch(black_box(refs)).unwrap());
            });
        });
    }

    group.finish();
}

// ============================================================================
// 4. Inner product latency
// ============================================================================

fn bench_inner_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("turbo_quant/inner_product");
    let mut rng = StdRng::seed_from_u64(0xFACE);

    let config = make_config(TurboQuantBits::Bits3_5, DEFAULT_DIM);
    let compressor = TurboQuantCompressor::new(config).unwrap();

    let query = random_vec(DEFAULT_DIM, &mut rng);

    // Single asymmetric inner product
    let target = random_vec(DEFAULT_DIM, &mut rng);
    let compressed_single = compressor.compress(&target).unwrap();

    group.bench_function("asymmetric_single", |b| {
        b.iter(|| {
            black_box(
                compressor
                    .inner_product_asymmetric(black_box(&query), black_box(&compressed_single), 0)
                    .unwrap(),
            );
        });
    });

    // Batch inner product with varying sizes
    for &n in &[10u64, 100, 1000] {
        let vecs: Vec<Vec<f32>> = (0..n).map(|_| random_vec(DEFAULT_DIM, &mut rng)).collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let compressed_batch = compressor.compress_batch(&refs).unwrap();

        group.throughput(Throughput::Elements(n));
        group.bench_with_input(
            BenchmarkId::new("batch", n),
            &compressed_batch,
            |b, compressed| {
                b.iter(|| {
                    black_box(
                        compressor
                            .inner_product_batch(black_box(&query), black_box(compressed))
                            .unwrap(),
                    );
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// 5. KV cache tier operations (TurboQuantCacheTier)
// ============================================================================

fn bench_cache_tier(c: &mut Criterion) {
    let mut group = c.benchmark_group("turbo_quant/cache_tier");
    group.sampling_mode(SamplingMode::Flat);
    let mut rng = StdRng::seed_from_u64(0xABCD);

    let config = make_config(TurboQuantBits::Bits3_5, DEFAULT_DIM);

    // Push
    group.bench_function("push", |b| {
        let keys = random_vec(DEFAULT_DIM, &mut rng);
        let values = random_vec(DEFAULT_DIM, &mut rng);
        b.iter_batched(
            || TurboQuantCacheTier::new(config.clone()).unwrap(),
            |mut tier| {
                tier.push(black_box(&keys), black_box(&values), 0).unwrap();
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // Get from a tier with varying sizes
    for &size in &[10usize, 100, 500] {
        let mut tier = TurboQuantCacheTier::new(config.clone()).unwrap();
        for i in 0..size {
            let k = random_vec(DEFAULT_DIM, &mut rng);
            let v = random_vec(DEFAULT_DIM, &mut rng);
            tier.push(&k, &v, i).unwrap();
        }

        group.bench_with_input(BenchmarkId::new("get", size), &tier, |b, tier| {
            b.iter(|| {
                black_box(tier.get(black_box(0)).unwrap());
            });
        });
    }

    // get_all_kv with varying sizes
    for &size in &[10usize, 50, 200] {
        let mut tier = TurboQuantCacheTier::new(config.clone()).unwrap();
        for i in 0..size {
            let k = random_vec(DEFAULT_DIM, &mut rng);
            let v = random_vec(DEFAULT_DIM, &mut rng);
            tier.push(&k, &v, i).unwrap();
        }

        group.bench_with_input(BenchmarkId::new("get_all_kv", size), &tier, |b, tier| {
            b.iter(|| {
                black_box(tier.get_all_kv().unwrap());
            });
        });
    }

    group.finish();
}

// ============================================================================
// 6. TurboQuantKvCache (three-tier: hot tail + TurboQuant cold)
// ============================================================================

fn bench_kv_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("turbo_quant/kv_cache");
    group.sampling_mode(SamplingMode::Flat);
    let mut rng = StdRng::seed_from_u64(0x1234);

    let num_kv_heads = 8;
    let head_dim = 128; // must be power of 2 for Hadamard
    let stride = num_kv_heads * head_dim;

    let kv_config = TurboQuantKvCacheConfig {
        tail_length: 64,
        max_tokens: 4096,
        num_kv_heads,
        head_dim,
        migration_batch: 32,
        turbo_config: make_config(TurboQuantBits::Bits3_5, head_dim),
    };

    // Append single token
    group.bench_function("append_1_token", |b| {
        let keys = random_vec(stride, &mut rng);
        let values = random_vec(stride, &mut rng);
        b.iter_batched(
            || TurboQuantKvCache::new(kv_config.clone()).unwrap(),
            |cache| {
                cache.append(black_box(&keys), black_box(&values)).unwrap();
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // Append triggering migration (fill past tail_length)
    group.bench_function("append_with_migration", |b| {
        b.iter_batched(
            || {
                let mut setup_rng = StdRng::seed_from_u64(0x9999);
                let cache = TurboQuantKvCache::new(kv_config.clone()).unwrap();
                // Pre-fill to just under tail_length
                for _ in 0..kv_config.tail_length - 1 {
                    let k = random_vec(stride, &mut setup_rng);
                    let v = random_vec(stride, &mut setup_rng);
                    cache.append(&k, &v).unwrap();
                }
                // Pre-generate the trigger token
                let k = random_vec(stride, &mut setup_rng);
                let v = random_vec(stride, &mut setup_rng);
                (cache, k, v)
            },
            |(cache, k, v)| {
                // This append should trigger migration
                cache.append(black_box(&k), black_box(&v)).unwrap();
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // get_all_kv with mixed tiers
    for &total_tokens in &[128usize, 512] {
        group.bench_with_input(
            BenchmarkId::new("get_all_kv", total_tokens),
            &total_tokens,
            |b, &total_tokens| {
                b.iter_batched(
                    || {
                        let cache = TurboQuantKvCache::new(kv_config.clone()).unwrap();
                        let mut rng2 = StdRng::seed_from_u64(0x5678);
                        for _ in 0..total_tokens {
                            let k = random_vec(stride, &mut rng2);
                            let v = random_vec(stride, &mut rng2);
                            cache.append(&k, &v).unwrap();
                        }
                        cache
                    },
                    |cache| {
                        black_box(cache.get_all_kv().unwrap());
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ============================================================================
// 7. Embedding store
// ============================================================================

fn bench_embedding_store(c: &mut Criterion) {
    let mut group = c.benchmark_group("turbo_quant/embedding_store");
    group.sampling_mode(SamplingMode::Flat);
    let mut rng = StdRng::seed_from_u64(0xEEEE);

    let config = make_config(TurboQuantBits::Bits3_5, DEFAULT_DIM);

    // build_from_batch with varying dataset sizes
    for &n in &[100usize, 1000, 5000] {
        let embeddings: Vec<Vec<f32>> = (0..n).map(|_| random_vec(DEFAULT_DIM, &mut rng)).collect();
        let ids: Vec<u64> = (0..n as u64).collect();

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("build_from_batch", n),
            &(embeddings.clone(), ids.clone()),
            |b, (embeddings, ids)| {
                b.iter(|| {
                    let mut store =
                        TurboQuantEmbeddingStore::new(DEFAULT_DIM, config.clone()).unwrap();
                    store
                        .build_from_batch(black_box(embeddings), black_box(ids))
                        .unwrap();
                    black_box(&store);
                });
            },
        );
    }

    // Search over pre-built stores
    for &n in &[100usize, 1000] {
        let embeddings: Vec<Vec<f32>> = (0..n).map(|_| random_vec(DEFAULT_DIM, &mut rng)).collect();
        let ids: Vec<u64> = (0..n as u64).collect();
        let mut store = TurboQuantEmbeddingStore::new(DEFAULT_DIM, config.clone()).unwrap();
        store.build_from_batch(&embeddings, &ids).unwrap();

        let query = random_vec(DEFAULT_DIM, &mut rng);

        group.bench_with_input(
            BenchmarkId::new("search_top10", n),
            &(store, query.clone()),
            |b, (store, query)| {
                b.iter(|| {
                    black_box(store.search(black_box(query), 10).unwrap());
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// 8. Memory efficiency / compression ratios
// ============================================================================

fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("turbo_quant/memory_efficiency");
    let mut rng = StdRng::seed_from_u64(0xAAAA);

    let n = 100;
    let vecs: Vec<Vec<f32>> = (0..n).map(|_| random_vec(DEFAULT_DIM, &mut rng)).collect();
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    for &(bits, label) in ALL_BITS {
        let config = make_config(bits, DEFAULT_DIM);
        let compressor = TurboQuantCompressor::new(config).unwrap();

        // Bench the compress and report compression ratio in the name
        group.throughput(Throughput::Bytes((n * DEFAULT_DIM * 4) as u64));
        group.bench_with_input(BenchmarkId::new("compress_100", label), &refs, |b, refs| {
            b.iter(|| {
                let compressed = compressor.compress_batch(black_box(refs)).unwrap();
                black_box(&compressed);
            });
        });
    }

    group.finish();

    // Print summary stats outside of criterion timing
    println!("\n=== TurboQuant Compression Ratio Summary ===");
    println!(
        "{:<10} {:>12} {:>12} {:>16}",
        "Bits", "Original", "Compressed", "Ratio"
    );
    println!("{}", "-".repeat(54));
    for &(bits, label) in ALL_BITS {
        let config = make_config(bits, DEFAULT_DIM);
        let compressor = TurboQuantCompressor::new(config).unwrap();
        let compressed = compressor.compress_batch(&refs).unwrap();
        let original = n * DEFAULT_DIM * 4;
        let used = compressed.memory_bytes();
        let ratio = original as f64 / used as f64;
        println!(
            "{:<10} {:>10} B {:>10} B {:>14.2}x",
            label, original, used, ratio
        );
    }
    println!();
}

// ============================================================================
// 9. Dimension scaling
// ============================================================================

fn bench_dimension_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("turbo_quant/dim_scaling");
    let mut rng = StdRng::seed_from_u64(0xDDDD);

    let dims: &[usize] = &[64, 128, 256, 512, 1024];

    for &dim in dims {
        let data = random_vec(dim, &mut rng);
        // block_size must be power-of-2 and <= dim; use min(dim, 128)
        let block_size = dim.min(128);
        let config = make_config(TurboQuantBits::Bits3_5, block_size);
        let compressor = TurboQuantCompressor::new(config).unwrap();

        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(BenchmarkId::new("compress", dim), &data, |b, data| {
            b.iter(|| {
                black_box(compressor.compress(black_box(data)).unwrap());
            });
        });

        let compressed = compressor.compress(&data).unwrap();
        group.bench_with_input(
            BenchmarkId::new("decompress", dim),
            &compressed,
            |b, compressed| {
                b.iter(|| {
                    black_box(compressor.decompress(black_box(compressed)).unwrap());
                });
            },
        );

        // Inner product at this dimension
        let query = random_vec(dim, &mut rng);
        group.bench_with_input(
            BenchmarkId::new("inner_product", dim),
            &compressed,
            |b, compressed| {
                b.iter(|| {
                    black_box(
                        compressor
                            .inner_product_asymmetric(black_box(&query), black_box(compressed), 0)
                            .unwrap(),
                    );
                });
            },
        );
    }

    group.finish();

    // Print dimension scaling summary
    println!("\n=== TurboQuant Dimension Scaling Summary (3.5-bit) ===");
    println!(
        "{:<8} {:>12} {:>12} {:>12}",
        "Dim", "Original", "Compressed", "Ratio"
    );
    println!("{}", "-".repeat(48));
    for &dim in dims {
        let block_size = dim.min(128);
        let config = make_config(TurboQuantBits::Bits3_5, block_size);
        let compressor = TurboQuantCompressor::new(config).unwrap();
        let data = random_vec(dim, &mut rng);
        let compressed = compressor.compress(&data).unwrap();
        let original = dim * 4;
        let used = compressed.memory_bytes();
        let ratio = original as f64 / used as f64;
        println!(
            "{:<8} {:>10} B {:>10} B {:>10.2}x",
            dim, original, used, ratio
        );
    }
    println!();
}

// ============================================================================
// Criterion groups and main
// ============================================================================

criterion_group!(
    benches,
    bench_compress_throughput,
    bench_decompress_throughput,
    bench_batch_compress,
    bench_inner_product,
    bench_cache_tier,
    bench_kv_cache,
    bench_embedding_store,
    bench_memory_efficiency,
    bench_dimension_scaling,
);
criterion_main!(benches);
