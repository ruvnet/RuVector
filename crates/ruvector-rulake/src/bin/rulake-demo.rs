#![allow(clippy::manual_div_ceil)]

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

use ruvector_rabitq::{AnnIndex, RabitqPlusIndex};
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
            "  direct RaBitQ+           build={:>8.1} ms   qps={:>8.0}",
            direct_build, direct_qps
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
    println!("Notes:");
    println!("  - 'tax' is the direct-QPS / lake-QPS ratio (1.0 = free, 2.0 = lake is 2× slower).");
    println!("  - Fresh calls LocalBackend::generation() on every query (one hash-map read —");
    println!("    on a real backend this is a network round-trip, expect materially higher tax).");
    println!("  - Eventual skips the generation check within TTL; this is the production path.");
    println!("  - Federated fan-out runs on rayon (added 2026-04-22); shard count");
    println!("    should improve tail latency, but per-shard work still adds to total");
    println!("    wall clock when shards are balanced and CPU-bound.");
}
