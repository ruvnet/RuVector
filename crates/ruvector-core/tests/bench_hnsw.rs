/// Real ruvector-core HNSW benchmark
/// Run: cargo test -p ruvector-core --test bench_hnsw --release -- --nocapture

use ruvector_core::types::{DbOptions, DistanceMetric, HnswConfig, SearchQuery, VectorEntry};
use ruvector_core::VectorDB;
use std::collections::HashSet;
use std::time::Instant;

fn generate_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((state >> 33) as f32 / (u32::MAX as f32)) * 2.0 - 1.0
                })
                .collect()
        })
        .collect()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn brute_force_topk(data: &[Vec<f32>], query: &[f32], k: usize) -> Vec<String> {
    let mut sims: Vec<(usize, f32)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| (i, cosine_similarity(v, query)))
        .collect();
    sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    sims.iter().take(k).map(|(i, _)| format!("v{}", i)).collect()
}

fn run_benchmark(num_vectors: usize, dimensions: usize, num_queries: usize, k: usize) {
    eprintln!(
        "\n=== ruvector-core HNSW: {}K vectors, {}d, {} queries, k={} ===",
        num_vectors / 1000, dimensions, num_queries, k
    );

    let data = generate_vectors(num_vectors, dimensions, 42);
    let queries = generate_vectors(num_queries, dimensions, 123);

    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("bench.db");
    let opts = DbOptions {
        dimensions,
        distance_metric: DistanceMetric::Cosine,
        storage_path: db_path.to_string_lossy().to_string(),
        hnsw_config: Some(HnswConfig {
            m: 32,
            ef_construction: 200,
            ef_search: 200,
            max_elements: num_vectors + 1000,
        }),
        quantization: None,
    };

    let db = VectorDB::new(opts).expect("Failed to create VectorDB");

    // Use insert_batch() for a single REDB transaction instead of individual
    // db.insert() calls. The old code opened a separate write transaction per
    // vector (10K vectors = 10K transaction commits with fsync). This measured
    // REDB transaction overhead, not HNSW build performance.
    //
    // Production code (ruvector-server, ruvector-cli, ruvector-node) already
    // uses insert_batch(). This change aligns the benchmark with real usage.
    let entries: Vec<VectorEntry> = data
        .iter()
        .enumerate()
        .map(|(i, vec)| VectorEntry {
            id: Some(format!("v{}", i)),
            vector: vec.clone(),
            metadata: None,
        })
        .collect();

    let build_start = Instant::now();
    db.insert_batch(entries).expect("Insert batch failed");
    let build_time = build_start.elapsed();
    eprintln!("  Build time: {:.3}s", build_time.as_secs_f64());

    let mut latencies = Vec::new();
    let mut recall_values = Vec::new();

    for query in &queries {
        let gt: HashSet<String> = brute_force_topk(&data, query, k).into_iter().collect();

        let search = SearchQuery {
            vector: query.clone(),
            k,
            filter: None,
            ef_search: Some(200),
        };

        let t0 = Instant::now();
        let results = db.search(search).expect("Search failed");
        let latency = t0.elapsed();

        latencies.push(latency.as_secs_f64() * 1000.0);

        let retrieved: HashSet<String> = results.iter().map(|r| r.id.clone()).collect();
        let recall = retrieved.intersection(&gt).count() as f64 / k as f64;
        recall_values.push(recall);
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let total_sec: f64 = latencies.iter().sum::<f64>() / 1000.0;
    let qps = num_queries as f64 / total_sec;
    let p50 = latencies[latencies.len() / 2];
    let p95 = latencies[(latencies.len() as f64 * 0.95) as usize];
    let avg_recall = recall_values.iter().sum::<f64>() / recall_values.len() as f64;

    eprintln!("  QPS: {:.1}", qps);
    eprintln!("  Recall@{}: {:.4}", k, avg_recall);
    eprintln!("  Latency p50: {:.3}ms, p95: {:.3}ms", p50, p95);
    eprintln!("  Build: {:.3}s", build_time.as_secs_f64());

    assert!(avg_recall > 0.0, "Recall should be > 0");
    assert!(qps > 1.0, "QPS should be > 1");
}

#[test]
fn bench_hnsw_10k() {
    run_benchmark(10_000, 128, 200, 10);
}

#[test]
fn bench_hnsw_100k() {
    run_benchmark(100_000, 128, 200, 10);
}
