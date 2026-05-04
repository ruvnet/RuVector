//! Microbenchmarks for the cluster coordinator hot path.
//!
//! Exercises three layers in isolation:
//!   1. P2cPool::choose_two_random — RNG + lock + comparison
//!   2. HashShardRouter::pick      — content-derived hashing
//!   3. HailoClusterEmbedder::embed_one_blocking against an in-memory
//!      transport that returns instantly — measures the dispatch loop's
//!      overhead with the network factored out.
//!
//! Run with `cargo bench --bench dispatch`. Numbers serve as a regression
//! guard — a 2× regression on (3) would mean someone introduced an
//! allocation or contention in the hot path.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ruvector_hailo_cluster::cache::EmbeddingCache;
use ruvector_hailo_cluster::error::ClusterError;
use ruvector_hailo_cluster::pool::P2cPool;
use ruvector_hailo_cluster::shard::{HashShardRouter, ShardRouter};
use ruvector_hailo_cluster::transport::{EmbeddingTransport, HealthReport, WorkerEndpoint};
use ruvector_hailo_cluster::HailoClusterEmbedder;
use std::sync::Arc;

/// Trivial transport that returns a pre-built vector instantly. Used to
/// remove tonic/IO from the bench so we measure pure coordinator cost.
struct InstantTransport {
    fixed: Vec<f32>,
}
impl EmbeddingTransport for InstantTransport {
    fn embed(
        &self,
        _w: &WorkerEndpoint,
        _t: &str,
        _m: u32,
    ) -> Result<(Vec<f32>, u64), ClusterError> {
        Ok((self.fixed.clone(), 0))
    }
    fn health(&self, _w: &WorkerEndpoint) -> Result<HealthReport, ClusterError> {
        Ok(HealthReport {
            version: "instant".into(),
            device_id: "instant:0".into(),
            model_fingerprint: "fp:instant".into(),
            ready: true,
            npu_temp_ts0_celsius: None,
            npu_temp_ts1_celsius: None,
        })
    }
}

fn workers(n: usize) -> Vec<WorkerEndpoint> {
    (0..n)
        .map(|i| WorkerEndpoint::new(format!("w{}", i), format!("10.0.0.{}:50051", i)))
        .collect()
}

fn bench_pool_choose(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool/choose_two_random");
    for n in [2usize, 4, 8, 16, 64] {
        group.bench_function(format!("n={}", n), |b| {
            let pool = P2cPool::new(workers(n));
            b.iter(|| {
                black_box(pool.choose_two_random());
            });
        });
    }
    group.finish();
}

fn bench_shard_router(c: &mut Criterion) {
    let mut group = c.benchmark_group("shard/hash_router_pick");
    let ws = workers(8);
    let texts: Vec<String> = (0..256)
        .map(|i| format!("input text number {}", i))
        .collect();
    let router = HashShardRouter;
    group.bench_function("8_workers/256_inputs", |b| {
        b.iter(|| {
            for t in &texts {
                black_box(router.pick(t, &ws));
            }
        });
    });
    group.finish();
}

fn bench_dispatch_loop(c: &mut Criterion) {
    let mut group = c.benchmark_group("dispatch/embed_one_blocking");
    let dim = 384;
    let fixed: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32).collect();
    for n in [1usize, 2, 8] {
        let cluster = HailoClusterEmbedder::new(
            workers(n),
            Arc::new(InstantTransport {
                fixed: fixed.clone(),
            }),
            dim,
            "fp:bench",
        )
        .unwrap();
        group.bench_function(format!("workers={}", n), |b| {
            b.iter(|| {
                black_box(cluster.embed_one_blocking(black_box("benchmark text"))).unwrap();
            });
        });
    }
    group.finish();
}

/// Cache hot path. Pre-iter-80 baseline (single-Mutex, VecDeque LRU,
/// Vec<f32> clone): ~3.4 µs/get on contended access at dim=384.
/// Post-iter-81 (16-shard Mutex, counter LRU, Arc<Vec<f32>> clone):
/// ~250 ns/get single-threaded; the empirical 30M req/s in
/// `cluster-bench --concurrency 8 --cache 4096 --cache-keyspace 200`
/// derives from the per-shard contention reduction. Single-threaded
/// criterion can't exercise the multi-shard win on its own, so this
/// bench focuses on per-call overhead.
fn bench_cache_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache/get");
    let dim = 384;
    let v: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32).collect();

    // Pre-warm with N keys so we measure hit cost (the common case).
    for n_keys in [10usize, 100, 1000] {
        let cache = EmbeddingCache::new(n_keys * 2);
        for i in 0..n_keys {
            cache.insert("fp:bench", &format!("key-{}", i), v.clone());
        }
        group.bench_function(format!("hit/keyspace={}", n_keys), |b| {
            let mut counter = 0u64;
            b.iter(|| {
                let key = format!("key-{}", counter as usize % n_keys);
                counter = counter.wrapping_add(1);
                black_box(cache.get(black_box("fp:bench"), black_box(&key)))
            });
        });
    }

    // Miss path — empty cache, every get returns None.
    let cache_empty = EmbeddingCache::new(64);
    group.bench_function("miss/empty", |b| {
        b.iter(|| black_box(cache_empty.get(black_box("fp:bench"), black_box("absent"))));
    });

    // Disabled cache — fast-path branch should be ~ns.
    let cache_off = EmbeddingCache::new(0);
    group.bench_function("disabled", |b| {
        b.iter(|| black_box(cache_off.get(black_box("fp:bench"), black_box("any"))));
    });

    group.finish();
}

/// Insert path including occasional eviction. With cap=N and inserting
/// N+1 distinct keys per cycle, every iteration triggers an O(N/16) shard
/// scan to evict the LRU.
fn bench_cache_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache/insert");
    let dim = 384;
    let v: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32).collect();

    for cap in [16usize, 256, 4096] {
        let cache = EmbeddingCache::new(cap);
        // Pre-fill so subsequent inserts evict.
        for i in 0..cap {
            cache.insert("fp:bench", &format!("warmup-{}", i), v.clone());
        }
        group.bench_function(format!("with_eviction/cap={}", cap), |b| {
            let mut counter = 0u64;
            b.iter(|| {
                let key = format!("new-{}", counter);
                counter = counter.wrapping_add(1);
                cache.insert(black_box("fp:bench"), black_box(&key), v.clone());
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_pool_choose,
    bench_shard_router,
    bench_dispatch_loop,
    bench_cache_get,
    bench_cache_insert
);
criterion_main!(benches);
