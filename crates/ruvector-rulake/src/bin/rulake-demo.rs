#![allow(clippy::manual_div_ceil)]
#![allow(clippy::too_many_arguments)]

//! ruLake benchmark harness — measures:
//!
//!   - direct RabitqPlusIndex throughput (baseline from BENCHMARK.md)
//!   - ruLake intermediary throughput (cache-hit path)
//!   - the *intermediary tax* (ratio)
//!   - cache prime time vs direct index build time
//!   - federation across 2 / 4 backends
//!
//! Same dataset, same seed, same queries for every row so the numbers
//! are directly comparable — matches the pattern in rabitq-demo.

use std::sync::Arc;
use std::time::Instant;

use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Uniform};

use ruvector_rabitq::{AnnIndex, RabitqPlusIndex, RandomRotationKind};
use ruvector_rulake::{cache::Consistency, LocalBackend, RuLake, SearchResult};

fn clustered(n: usize, d: usize, n_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    use rand::Rng as _;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let centroid = Uniform::new(-2.0f32, 2.0);
    let centroids: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..d).map(|_| centroid.sample(&mut rng)).collect())
        .collect();
    let noise = Normal::new(0.0f64, 0.6).unwrap();
    (0..n)
        .map(|_| {
            let c = &centroids[rng.gen_range(0..n_clusters)];
            c.iter()
                .map(|&x| x + noise.sample(&mut rng) as f32)
                .collect()
        })
        .collect()
}

fn measure_direct(
    d: usize,
    rerank: usize,
    seed: u64,
    data: &[Vec<f32>],
    queries: &[Vec<f32>],
) -> (f64, f64) {
    // returns (build_ms, qps)
    let t = Instant::now();
    let mut idx = RabitqPlusIndex::new(d, seed, rerank);
    for (i, v) in data.iter().enumerate() {
        idx.add(i, v.clone()).unwrap();
    }
    let build_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    for q in queries {
        let _ = idx.search(q, 10).unwrap();
    }
    let qps = queries.len() as f64 / t.elapsed().as_secs_f64();
    (build_ms, qps)
}

/// Same shape as [`measure_direct`] but uses a randomised-Hadamard
/// rotation instead of the default Haar matrix (ADR-158 feature).
fn measure_direct_hadamard(
    d: usize,
    rerank: usize,
    seed: u64,
    data: &[Vec<f32>],
    queries: &[Vec<f32>],
) -> (f64, f64) {
    let t = Instant::now();
    let mut idx =
        RabitqPlusIndex::new_with_rotation(d, seed, rerank, RandomRotationKind::HadamardSigned);
    for (i, v) in data.iter().enumerate() {
        idx.add(i, v.clone()).unwrap();
    }
    let build_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    for q in queries {
        let _ = idx.search(q, 10).unwrap();
    }
    let qps = queries.len() as f64 / t.elapsed().as_secs_f64();
    (build_ms, qps)
}

fn measure_rulake_single(
    d: usize,
    rerank: usize,
    seed: u64,
    data: &[Vec<f32>],
    queries: &[Vec<f32>],
    consistency: Consistency,
) -> (f64, f64) {
    // returns (prime_ms, qps). Prime = first-miss backend pull + RaBitQ build.
    let backend = Arc::new(LocalBackend::new("b"));
    backend
        .put_collection("c", d, (0..data.len() as u64).collect(), data.to_vec())
        .unwrap();
    let lake = RuLake::new(rerank, seed).with_consistency(consistency);
    lake.register_backend(backend).unwrap();

    let t = Instant::now();
    let _ = lake.search_one("b", "c", &queries[0], 10).unwrap();
    let prime_ms = t.elapsed().as_secs_f64() * 1000.0;

    // Remaining queries are cache hits.
    let t = Instant::now();
    for q in &queries[1..] {
        let _ = lake.search_one("b", "c", q, 10).unwrap();
    }
    let qps = (queries.len() - 1) as f64 / t.elapsed().as_secs_f64();
    (prime_ms, qps)
}

fn measure_rulake_fed(
    d: usize,
    rerank: usize,
    seed: u64,
    total_n: usize,
    n_shards: usize,
    queries: &[Vec<f32>],
    consistency: Consistency,
) -> (f64, f64) {
    // Partition `total_n` vectors across `n_shards` backends.
    let data = clustered(total_n, d, 100, seed);
    let lake = RuLake::new(rerank, seed).with_consistency(consistency);
    let mut targets: Vec<(String, String)> = Vec::new();
    let chunk = (total_n + n_shards - 1) / n_shards;
    for s in 0..n_shards {
        let lo = s * chunk;
        let hi = ((s + 1) * chunk).min(total_n);
        let backend_id = format!("shard-{s}");
        let b = Arc::new(LocalBackend::new(&backend_id));
        b.put_collection(
            "c",
            d,
            (lo as u64..hi as u64).collect(),
            data[lo..hi].to_vec(),
        )
        .unwrap();
        lake.register_backend(b).unwrap();
        targets.push((backend_id, "c".to_string()));
    }

    // Prime all backends with the first query.
    let target_refs: Vec<(&str, &str)> = targets
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let t = Instant::now();
    let _: Vec<SearchResult> = lake
        .search_federated(&target_refs, &queries[0], 10)
        .unwrap();
    let prime_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    for q in &queries[1..] {
        let _ = lake.search_federated(&target_refs, q, 10).unwrap();
    }
    let qps = (queries.len() - 1) as f64 / t.elapsed().as_secs_f64();
    (prime_ms, qps)
}

/// Measures concurrent-client QPS under C parallel clients, each running
/// `queries_per_client` queries over an already-primed `n_shards`-shard
/// federated collection. This is the workload where rayon fan-out
/// actually matters: per-query wall clock drops when a shard's scan
/// can run in parallel with others while clients keep arriving.
fn measure_concurrent_fed(
    d: usize,
    rerank: usize,
    seed: u64,
    total_n: usize,
    n_shards: usize,
    n_clients: usize,
    queries_per_client: usize,
    queries: &[Vec<f32>],
) -> (f64, f64) {
    // (wall_ms, qps)
    use std::thread;

    let data = clustered(total_n, d, 100, seed);
    let lake = Arc::new(
        RuLake::new(rerank, seed).with_consistency(Consistency::Eventual { ttl_ms: 60_000 }),
    );
    let mut targets: Vec<(String, String)> = Vec::new();
    let chunk = (total_n + n_shards - 1) / n_shards;
    for s in 0..n_shards {
        let lo = s * chunk;
        let hi = ((s + 1) * chunk).min(total_n);
        let backend_id = format!("shard-{s}");
        let b = Arc::new(LocalBackend::new(&backend_id));
        b.put_collection(
            "c",
            d,
            (lo as u64..hi as u64).collect(),
            data[lo..hi].to_vec(),
        )
        .unwrap();
        lake.register_backend(b).unwrap();
        targets.push((backend_id, "c".to_string()));
    }

    // Prime all shards in a single pass so client threads only do hits.
    let target_refs_init: Vec<(&str, &str)> = targets
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let _ = lake
        .search_federated(&target_refs_init, &queries[0], 10)
        .unwrap();

    // Each client picks queries deterministically from the shared slice.
    let t = Instant::now();
    let mut handles = Vec::with_capacity(n_clients);
    for c in 0..n_clients {
        let lake = Arc::clone(&lake);
        let targets = targets.clone();
        let queries = queries.to_vec();
        let h = thread::spawn(move || {
            let lo = c * queries_per_client;
            let refs: Vec<(&str, &str)> = targets
                .iter()
                .map(|(a, b)| (a.as_str(), b.as_str()))
                .collect();
            for i in 0..queries_per_client {
                let q = &queries[(lo + i) % queries.len()];
                let _ = lake.search_federated(&refs, q, 10).unwrap();
            }
        });
        handles.push(h);
    }
    for h in handles {
        h.join().unwrap();
    }
    let wall_ms = t.elapsed().as_secs_f64() * 1000.0;
    let total_queries = (n_clients * queries_per_client) as f64;
    let qps = total_queries / (wall_ms / 1000.0);
    (wall_ms, qps)
}

fn main() {
    let fast = std::env::args().any(|a| a == "--fast");
    println!("=== ruLake benchmark harness ===");
    println!("clustered Gaussian, D=128, 100 clusters, rerank×20, Fresh consistency unless noted");
    println!("queries are warm (not timing first miss unless labeled 'prime')");
    println!();

    let d = 128;
    let rerank = 20;
    let seed = 91;
    let nq = if fast { 100 } else { 300 };

    for &n in if fast {
        &[5_000usize][..]
    } else {
        &[5_000, 50_000, 100_000][..]
    } {
        println!("── n = {n} ──────────────────────────────────────────");
        let data = clustered(n, d, 100, seed);
        let queries = clustered(nq, d, 100, seed ^ 0xdead_beef);

        let (direct_build, direct_qps) = measure_direct(d, rerank, seed, &data, &queries);
        println!(
            "  direct RaBitQ+ (Haar)    build={:>8.1} ms   qps={:>8.0}",
            direct_build, direct_qps
        );

        let (hada_build, hada_qps) = measure_direct_hadamard(d, rerank, seed, &data, &queries);
        println!(
            "  direct RaBitQ+ (Hadamard) build={:>8.1} ms   qps={:>8.0}   build_speedup={:.2}×",
            hada_build,
            hada_qps,
            direct_build / hada_build.max(0.001)
        );

        let (lake_prime, lake_qps) =
            measure_rulake_single(d, rerank, seed, &data, &queries, Consistency::Fresh);
        println!(
            "  ruLake (Fresh)           prime={:>8.1} ms   qps={:>8.0}   tax={:.2}×",
            lake_prime,
            lake_qps,
            direct_qps / lake_qps.max(1.0)
        );

        let (lake_prime_e, lake_qps_e) = measure_rulake_single(
            d,
            rerank,
            seed,
            &data,
            &queries,
            Consistency::Eventual { ttl_ms: 60_000 },
        );
        println!(
            "  ruLake (Eventual 60s)    prime={:>8.1} ms   qps={:>8.0}   tax={:.2}×",
            lake_prime_e,
            lake_qps_e,
            direct_qps / lake_qps_e.max(1.0)
        );

        // Federation: split n across 2 and 4 backends.
        let (fed2_prime, fed2_qps) = measure_rulake_fed(
            d,
            rerank,
            seed,
            n,
            2,
            &queries,
            Consistency::Eventual { ttl_ms: 60_000 },
        );
        println!(
            "  ruLake federated (2 shards, Eventual)  prime={:>6.1} ms   qps={:>8.0}",
            fed2_prime, fed2_qps
        );

        if !fast {
            let (fed4_prime, fed4_qps) = measure_rulake_fed(
                d,
                rerank,
                seed,
                n,
                4,
                &queries,
                Consistency::Eventual { ttl_ms: 60_000 },
            );
            println!(
                "  ruLake federated (4 shards, Eventual)  prime={:>6.1} ms   qps={:>8.0}",
                fed4_prime, fed4_qps
            );
        }
        println!();
    }
    if !fast {
        println!("── search_batch vs per-query loop (n=100k) ──");
        let n = 100_000;
        let data = clustered(n, d, 100, seed);
        let queries = clustered(300, d, 100, seed ^ 0xdead_beef);

        let backend = Arc::new(LocalBackend::new("bench"));
        backend
            .put_collection("c", d, (0..n as u64).collect(), data.clone())
            .unwrap();
        let lake =
            RuLake::new(rerank, seed).with_consistency(Consistency::Eventual { ttl_ms: 60_000 });
        lake.register_backend(backend).unwrap();
        // Prime.
        lake.search_one("bench", "c", &queries[0], 10).unwrap();

        // Per-query loop over the full 300-query set.
        let t = Instant::now();
        for q in &queries {
            let _ = lake.search_one("bench", "c", q, 10).unwrap();
        }
        let loop_qps = queries.len() as f64 / t.elapsed().as_secs_f64();

        // Batch the same 300 queries in chunks of 32.
        for &batch_size in &[8usize, 32, 128, 300] {
            let t = Instant::now();
            for chunk in queries.chunks(batch_size) {
                let _ = lake.search_batch("bench", "c", chunk, 10).unwrap();
            }
            let batch_qps = queries.len() as f64 / t.elapsed().as_secs_f64();
            println!(
                "  batch={:>3}   qps={:>8.0}   speedup vs per-query {:.2}×",
                batch_size,
                batch_qps,
                batch_qps / loop_qps
            );
        }
        println!("  per-query loop   qps={:>8.0}   (baseline)", loop_qps);
        println!();

        println!("── concurrent clients × federation (n=100k, 8 clients × 300 queries) ──");
        let n = 100_000;
        let queries = clustered(300, d, 100, seed ^ 0xdead_beef);
        for &shards in &[1, 2, 4] {
            let (wall_ms, qps) =
                measure_concurrent_fed(d, rerank, seed, n, shards, 8, 300, &queries);
            println!(
                "  {} shards × 8 clients × 300 queries   wall={:>7.1} ms   qps={:>8.0}",
                shards, wall_ms, qps
            );
        }
        println!();
    }

    println!("Notes:");
    println!("  - 'tax' is the direct-QPS / lake-QPS ratio (1.0 = free, 2.0 = lake is 2× slower).");
    println!("  - Fresh calls LocalBackend::generation() on every query (one hash-map read —");
    println!("    on a real backend this is a network round-trip, expect materially higher tax).");
    println!("  - Eventual skips the generation check within TTL; this is the production path.");
    println!("  - Federated fan-out runs on rayon (added 2026-04-22).");
    println!("  - Single-threaded QPS undersells rayon on federated reads; see the");
    println!("    concurrent-clients block where inter-shard parallelism overlaps with");
    println!("    inter-client parallelism.");
}
