//! # EML End-to-End Proof Benchmark
//!
//! Apples-to-apples, production-grade ANN benchmark proving (or disproving)
//! that EML-inspired optimizations deliver real improvements.
//!
//! ## What this measures
//!
//! Builds two HNSW indexes on identical data with identical parameters:
//! - **Baseline**: Standard match-dispatched distance (SimSIMD) + ScalarQuantized
//! - **LogQuantized**: Unified branch-free distance + LogQuantized
//!
//! Then runs identical query workloads measuring:
//! - Recall@1, Recall@10, Recall@100
//! - Search latency percentiles (p50, p95, p99, p99.9)
//! - Throughput (QPS) at ef_search=64 and ef_search=256
//! - Index build time
//! - Reconstruction MSE on actual embeddings
//!
//! ## Running
//!
//! ```bash
//! # Quick Criterion comparison (search QPS + reconstruction MSE)
//! cargo bench -p ruvector-core --bench eml_end_to_end
//!
//! # Full proof report (100K vectors, 3 seeds, Markdown table)
//! cargo bench -p ruvector-core --bench eml_end_to_end -- full_proof --ignored
//! # Or run directly:
//! cargo test -p ruvector-core --release -- eml_proof --ignored --nocapture
//! ```
//!
//! ## Profiling
//!
//! ```bash
//! cargo flamegraph --bench eml_end_to_end -p ruvector-core -- --bench
//! perf record cargo bench -p ruvector-core --bench eml_end_to_end
//! perf report
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_core::distance::euclidean_distance;
use ruvector_core::index::hnsw::HnswIndex;
use ruvector_core::index::VectorIndex;
use ruvector_core::quantization::{LogQuantized, QuantizedVector, ScalarQuantized};
use ruvector_core::types::{DistanceMetric, HnswConfig};
use std::time::Instant;

// =============================================================================
// Configuration
// =============================================================================

const DIMS: usize = 128;
const HNSW_M: usize = 16;
const EF_CONSTRUCTION: usize = 200;

// =============================================================================
// Dataset Generation
// =============================================================================

/// Generate SIFT-like vectors: 128-dim with values in [0, 218].
/// Real SIFT descriptors are histograms of oriented gradients with
/// a roughly log-normal distribution concentrated near small values.
fn generate_sift_like(count: usize, dims: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            (0..dims)
                .map(|_| {
                    let u: f32 = rng.gen::<f32>().max(1e-7);
                    let log_val = (-2.0 * u.ln()).sqrt() * 30.0;
                    log_val.clamp(0.0, 218.0)
                })
                .collect()
        })
        .collect()
}

/// Generate normally-distributed embeddings (typical transformer output).
fn generate_normal_embeddings(count: usize, dims: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            (0..dims)
                .map(|_| {
                    let u1: f32 = rng.gen::<f32>().max(1e-7);
                    let u2: f32 = rng.gen();
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * 0.3
                })
                .collect()
        })
        .collect()
}

// =============================================================================
// Brute-Force Ground Truth
// =============================================================================

fn compute_ground_truth(base: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<Vec<usize>> {
    queries
        .iter()
        .map(|query| {
            let mut dists: Vec<(usize, f32)> = base
                .iter()
                .enumerate()
                .map(|(i, v)| (i, euclidean_distance(query, v)))
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            dists.iter().take(k).map(|(i, _)| *i).collect()
        })
        .collect()
}

// =============================================================================
// Recall Computation
// =============================================================================

fn compute_recall(ground_truth: &[Vec<usize>], ann_results: &[Vec<String>], k: usize) -> f64 {
    let mut total_hits = 0usize;
    let mut total_possible = 0usize;

    for (gt, ann) in ground_truth.iter().zip(ann_results.iter()) {
        let gt_set: std::collections::HashSet<usize> = gt.iter().take(k).copied().collect();
        let ann_set: std::collections::HashSet<usize> = ann
            .iter()
            .take(k)
            .filter_map(|id| id.strip_prefix("v").and_then(|n| n.parse::<usize>().ok()))
            .collect();

        total_hits += gt_set.intersection(&ann_set).count();
        total_possible += gt_set.len();
    }

    if total_possible == 0 { 0.0 } else { total_hits as f64 / total_possible as f64 }
}

// =============================================================================
// Latency Stats
// =============================================================================

struct LatencyStats {
    p50: f64,
    p95: f64,
    p99: f64,
    p999: f64,
}

fn compute_latency_stats(latencies_us: &mut [f64]) -> LatencyStats {
    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = latencies_us.len();
    LatencyStats {
        p50: latencies_us[n / 2],
        p95: latencies_us[(n as f64 * 0.95) as usize],
        p99: latencies_us[(n as f64 * 0.99) as usize],
        p999: latencies_us[((n as f64 * 0.999) as usize).min(n - 1)],
    }
}

// =============================================================================
// Reconstruction MSE
// =============================================================================

fn compute_reconstruction_mse_scalar(vectors: &[Vec<f32>]) -> f64 {
    let mut total_mse = 0.0f64;
    for v in vectors {
        let q = ScalarQuantized::quantize(v);
        let r = q.reconstruct();
        let mse: f64 = v.iter().zip(r.iter())
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum::<f64>() / v.len() as f64;
        total_mse += mse;
    }
    total_mse / vectors.len() as f64
}

fn compute_reconstruction_mse_log(vectors: &[Vec<f32>]) -> f64 {
    let mut total_mse = 0.0f64;
    for v in vectors {
        let q = LogQuantized::quantize(v);
        let r = q.reconstruct();
        let mse: f64 = v.iter().zip(r.iter())
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum::<f64>() / v.len() as f64;
        total_mse += mse;
    }
    total_mse / vectors.len() as f64
}

// =============================================================================
// End-to-End Run
// =============================================================================

struct RunResult {
    build_time_ms: f64,
    recall_at_1: f64,
    recall_at_10: f64,
    recall_at_100: f64,
    latency_ef64: LatencyStats,
    latency_ef256: LatencyStats,
    qps_ef64: f64,
    qps_ef256: f64,
    recon_mse: f64,
}

fn run_end_to_end(
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    ground_truth_100: &[Vec<usize>],
    use_unified: bool,
) -> RunResult {
    let config = HnswConfig {
        m: HNSW_M,
        ef_construction: EF_CONSTRUCTION,
        ef_search: 64,
        max_elements: base.len() + 1000,
    };

    // --- Build Index ---
    let build_start = Instant::now();
    let mut index = if use_unified {
        HnswIndex::new_unified(DIMS, DistanceMetric::Euclidean, config.clone()).unwrap()
    } else {
        HnswIndex::new(DIMS, DistanceMetric::Euclidean, config.clone()).unwrap()
    };

    let entries: Vec<_> = base.iter().enumerate()
        .map(|(i, v)| (format!("v{}", i), v.clone()))
        .collect();
    index.add_batch(entries).unwrap();
    let build_time_ms = build_start.elapsed().as_secs_f64() * 1000.0;

    // --- Reconstruction MSE ---
    let recon_mse = if use_unified {
        compute_reconstruction_mse_log(base)
    } else {
        compute_reconstruction_mse_scalar(base)
    };

    // --- Search at ef_search=64 ---
    let mut latencies_64 = Vec::with_capacity(queries.len());
    let mut results_64 = Vec::with_capacity(queries.len());
    for query in queries {
        let start = Instant::now();
        let res = index.search_with_ef(black_box(query), 100, 64).unwrap();
        latencies_64.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        results_64.push(res.iter().map(|r| r.id.clone()).collect::<Vec<_>>());
    }

    // --- Search at ef_search=256 ---
    let mut latencies_256 = Vec::with_capacity(queries.len());
    let mut results_256 = Vec::with_capacity(queries.len());
    for query in queries {
        let start = Instant::now();
        let res = index.search_with_ef(black_box(query), 100, 256).unwrap();
        latencies_256.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        results_256.push(res.iter().map(|r| r.id.clone()).collect::<Vec<_>>());
    }

    // Use ef=256 results for recall (higher quality)
    let recall_at_1 = compute_recall(ground_truth_100, &results_256, 1);
    let recall_at_10 = compute_recall(ground_truth_100, &results_256, 10);
    let recall_at_100 = compute_recall(ground_truth_100, &results_256, 100);

    let total_time_64 = latencies_64.iter().sum::<f64>();
    let total_time_256 = latencies_256.iter().sum::<f64>();
    let qps_ef64 = queries.len() as f64 / (total_time_64 / 1_000_000.0);
    let qps_ef256 = queries.len() as f64 / (total_time_256 / 1_000_000.0);

    RunResult {
        build_time_ms,
        recall_at_1,
        recall_at_10,
        recall_at_100,
        latency_ef64: compute_latency_stats(&mut latencies_64),
        latency_ef256: compute_latency_stats(&mut latencies_256),
        qps_ef64,
        qps_ef256,
        recon_mse,
    }
}

// =============================================================================
// Aggregation
// =============================================================================

struct AggregatedResult {
    build_time_ms: (f64, f64),
    recall_at_1: (f64, f64),
    recall_at_10: (f64, f64),
    recall_at_100: (f64, f64),
    p50_ef64: (f64, f64),
    p95_ef64: (f64, f64),
    p99_ef64: (f64, f64),
    p999_ef64: (f64, f64),
    p50_ef256: (f64, f64),
    p95_ef256: (f64, f64),
    p99_ef256: (f64, f64),
    p999_ef256: (f64, f64),
    qps_ef64: (f64, f64),
    qps_ef256: (f64, f64),
    recon_mse: (f64, f64),
}

fn mean_stddev(vals: &[f64]) -> (f64, f64) {
    let n = vals.len() as f64;
    let mean = vals.iter().sum::<f64>() / n;
    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

fn aggregate_runs(runs: &[RunResult]) -> AggregatedResult {
    AggregatedResult {
        build_time_ms: mean_stddev(&runs.iter().map(|r| r.build_time_ms).collect::<Vec<_>>()),
        recall_at_1: mean_stddev(&runs.iter().map(|r| r.recall_at_1).collect::<Vec<_>>()),
        recall_at_10: mean_stddev(&runs.iter().map(|r| r.recall_at_10).collect::<Vec<_>>()),
        recall_at_100: mean_stddev(&runs.iter().map(|r| r.recall_at_100).collect::<Vec<_>>()),
        p50_ef64: mean_stddev(&runs.iter().map(|r| r.latency_ef64.p50).collect::<Vec<_>>()),
        p95_ef64: mean_stddev(&runs.iter().map(|r| r.latency_ef64.p95).collect::<Vec<_>>()),
        p99_ef64: mean_stddev(&runs.iter().map(|r| r.latency_ef64.p99).collect::<Vec<_>>()),
        p999_ef64: mean_stddev(&runs.iter().map(|r| r.latency_ef64.p999).collect::<Vec<_>>()),
        p50_ef256: mean_stddev(&runs.iter().map(|r| r.latency_ef256.p50).collect::<Vec<_>>()),
        p95_ef256: mean_stddev(&runs.iter().map(|r| r.latency_ef256.p95).collect::<Vec<_>>()),
        p99_ef256: mean_stddev(&runs.iter().map(|r| r.latency_ef256.p99).collect::<Vec<_>>()),
        p999_ef256: mean_stddev(&runs.iter().map(|r| r.latency_ef256.p999).collect::<Vec<_>>()),
        qps_ef64: mean_stddev(&runs.iter().map(|r| r.qps_ef64).collect::<Vec<_>>()),
        qps_ef256: mean_stddev(&runs.iter().map(|r| r.qps_ef256).collect::<Vec<_>>()),
        recon_mse: mean_stddev(&runs.iter().map(|r| r.recon_mse).collect::<Vec<_>>()),
    }
}

// =============================================================================
// Report Printing
// =============================================================================

fn print_markdown_report(
    dataset_name: &str,
    num_base: usize,
    num_queries: usize,
    seeds: &[u64],
    baseline: &AggregatedResult,
    log_quantized: &AggregatedResult,
) {
    eprintln!("\n### EML Optimizations — End-to-End Proof\n");
    eprintln!(
        "**Dataset**: {} ({} base, {} queries, {}D)",
        dataset_name, num_base, num_queries, DIMS
    );
    eprintln!("**HNSW config**: M={}, ef_construction={}", HNSW_M, EF_CONSTRUCTION);
    eprintln!("**Runs per config**: {} (seeds: {:?})\n", seeds.len(), seeds);

    eprintln!("| Metric | Baseline (Scalar+Dispatch) | LogQuantized (Log+Unified) | Delta |");
    eprintln!("|--------|---------------------------|---------------------------|-------|");

    let rows: Vec<(&str, (f64, f64), (f64, f64), bool)> = vec![
        ("Build Time (ms)", baseline.build_time_ms, log_quantized.build_time_ms, false),
        ("Recon MSE", baseline.recon_mse, log_quantized.recon_mse, false),
        ("Recall@1 (ef=256)", baseline.recall_at_1, log_quantized.recall_at_1, true),
        ("Recall@10 (ef=256)", baseline.recall_at_10, log_quantized.recall_at_10, true),
        ("Recall@100 (ef=256)", baseline.recall_at_100, log_quantized.recall_at_100, true),
        ("QPS ef=64", baseline.qps_ef64, log_quantized.qps_ef64, true),
        ("QPS ef=256", baseline.qps_ef256, log_quantized.qps_ef256, true),
        ("p50 ef=64 (us)", baseline.p50_ef64, log_quantized.p50_ef64, false),
        ("p95 ef=64 (us)", baseline.p95_ef64, log_quantized.p95_ef64, false),
        ("p99 ef=64 (us)", baseline.p99_ef64, log_quantized.p99_ef64, false),
        ("p99.9 ef=64 (us)", baseline.p999_ef64, log_quantized.p999_ef64, false),
        ("p50 ef=256 (us)", baseline.p50_ef256, log_quantized.p50_ef256, false),
        ("p95 ef=256 (us)", baseline.p95_ef256, log_quantized.p95_ef256, false),
        ("p99 ef=256 (us)", baseline.p99_ef256, log_quantized.p99_ef256, false),
        ("p99.9 ef=256 (us)", baseline.p999_ef256, log_quantized.p999_ef256, false),
    ];

    for (name, (bm, bs), (lm, ls), higher_is_better) in &rows {
        let delta_pct = if bm.abs() > 1e-12 { (lm - bm) / bm * 100.0 } else { 0.0 };
        let direction = if *higher_is_better {
            if delta_pct > 0.5 { "better" } else if delta_pct < -0.5 { "WORSE" } else { "~same" }
        } else {
            if delta_pct < -0.5 { "better" } else if delta_pct > 0.5 { "WORSE" } else { "~same" }
        };
        eprintln!(
            "| {} | {:.4} +/- {:.4} | {:.4} +/- {:.4} | {:+.1}% ({}) |",
            name, bm, bs, lm, ls, delta_pct, direction
        );
    }

    eprintln!();

    // CSV for programmatic consumption
    eprintln!("<!-- CSV_START");
    eprintln!("metric,baseline_mean,baseline_stddev,log_mean,log_stddev,delta_pct");
    for (name, (bm, bs), (lm, ls), _) in &rows {
        let delta_pct = if bm.abs() > 1e-12 { (lm - bm) / bm * 100.0 } else { 0.0 };
        eprintln!("{},{:.6},{:.6},{:.6},{:.6},{:.2}", name, bm, bs, lm, ls, delta_pct);
    }
    eprintln!("CSV_END -->");
}

// =============================================================================
// Criterion Benchmarks (fast — pre-built indexes, search-only)
// =============================================================================

fn bench_search_qps(c: &mut Criterion) {
    let mut group = c.benchmark_group("eml_e2e_search");
    group.sample_size(20);

    // Use 10K vectors for Criterion (fast enough for statistical sampling)
    let num_base = 10_000;
    let num_queries = 200;

    let base = generate_sift_like(num_base, DIMS, 42);
    let queries = generate_sift_like(num_queries, DIMS, 99);

    let config = HnswConfig {
        m: HNSW_M,
        ef_construction: EF_CONSTRUCTION,
        ef_search: 64,
        max_elements: num_base + 1000,
    };

    let entries: Vec<_> = base.iter().enumerate()
        .map(|(i, v)| (format!("v{}", i), v.clone()))
        .collect();

    // Build both indexes once (outside the benchmark loop)
    let mut baseline_idx = HnswIndex::new(DIMS, DistanceMetric::Euclidean, config.clone()).unwrap();
    baseline_idx.add_batch(entries.clone()).unwrap();

    let mut log_idx = HnswIndex::new_unified(DIMS, DistanceMetric::Euclidean, config.clone()).unwrap();
    log_idx.add_batch(entries).unwrap();

    for ef in [64, 256] {
        group.bench_with_input(
            BenchmarkId::new("baseline_search", ef),
            &ef,
            |b, &ef| {
                b.iter(|| {
                    let mut total = 0.0f32;
                    for q in &queries {
                        let res = baseline_idx.search_with_ef(black_box(q), 10, ef).unwrap();
                        total += res.first().map(|r| r.score).unwrap_or(0.0);
                    }
                    total
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("log_unified_search", ef),
            &ef,
            |b, &ef| {
                b.iter(|| {
                    let mut total = 0.0f32;
                    for q in &queries {
                        let res = log_idx.search_with_ef(black_box(q), 10, ef).unwrap();
                        total += res.first().map(|r| r.score).unwrap_or(0.0);
                    }
                    total
                });
            },
        );
    }

    group.finish();
}

fn bench_reconstruction_mse(c: &mut Criterion) {
    let mut group = c.benchmark_group("eml_e2e_recon_mse");

    let sift_data = generate_sift_like(10_000, DIMS, 42);
    let normal_data = generate_normal_embeddings(10_000, DIMS, 42);

    for (name, data) in [("sift_like", &sift_data), ("normal_embed", &normal_data)] {
        group.bench_function(format!("scalar_{}", name), |b| {
            b.iter(|| compute_reconstruction_mse_scalar(black_box(data)));
        });

        group.bench_function(format!("log_{}", name), |b| {
            b.iter(|| compute_reconstruction_mse_log(black_box(data)));
        });
    }

    group.finish();
}

/// Full proof report — runs end-to-end on 100K vectors with 3 seeds on 2 distributions.
///
/// **Gated behind `EML_FULL_PROOF=1`** because it takes ~10 minutes. When enabled,
/// prints a comprehensive Markdown comparison table to stderr.
///
/// ```bash
/// EML_FULL_PROOF=1 cargo bench -p ruvector-core --bench eml_end_to_end -- full_proof 2>&1 | tee bench_results/eml_proof.md
/// ```
fn bench_full_proof(c: &mut Criterion) {
    if std::env::var("EML_FULL_PROOF").ok().as_deref() != Some("1") {
        // Skip by default — only runs when explicitly requested
        return;
    }

    // Allow overriding dataset size for faster iteration
    let num_base: usize = std::env::var("EML_PROOF_N")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let num_queries: usize = std::env::var("EML_PROOF_Q")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(1_000);
    let seeds: [u64; 3] = [42, 1337, 2024];

    eprintln!("\n=== EML_FULL_PROOF enabled: N={} Q={} ===\n", num_base, num_queries);

    // --- SIFT-like dataset ---
    let mut baseline_runs = Vec::new();
    let mut log_runs = Vec::new();
    for &seed in &seeds {
        eprintln!("  [proof] SIFT-like seed={}: building baseline...", seed);
        let base = generate_sift_like(num_base, DIMS, seed);
        let queries = generate_sift_like(num_queries, DIMS, seed + 1000);
        let gt = compute_ground_truth(&base, &queries, 100);

        baseline_runs.push(run_end_to_end(&base, &queries, &gt, false));
        eprintln!("  [proof] SIFT-like seed={}: building log+unified...", seed);
        log_runs.push(run_end_to_end(&base, &queries, &gt, true));
    }
    print_markdown_report("SIFT-like (synthetic)", num_base, num_queries, &seeds,
        &aggregate_runs(&baseline_runs), &aggregate_runs(&log_runs));

    // --- Normal embeddings dataset ---
    let mut baseline_runs_n = Vec::new();
    let mut log_runs_n = Vec::new();
    for &seed in &seeds {
        eprintln!("  [proof] Normal seed={}: building baseline...", seed);
        let base = generate_normal_embeddings(num_base, DIMS, seed);
        let queries = generate_normal_embeddings(num_queries, DIMS, seed + 1000);
        let gt = compute_ground_truth(&base, &queries, 100);

        baseline_runs_n.push(run_end_to_end(&base, &queries, &gt, false));
        eprintln!("  [proof] Normal seed={}: building log+unified...", seed);
        log_runs_n.push(run_end_to_end(&base, &queries, &gt, true));
    }
    print_markdown_report("Normal embeddings (synthetic)", num_base, num_queries, &seeds,
        &aggregate_runs(&baseline_runs_n), &aggregate_runs(&log_runs_n));

    // Sentinel Criterion benchmark so the report shows up under a named group
    let mut group = c.benchmark_group("eml_e2e_full_proof");
    group.sample_size(10);
    group.bench_function("_report_generated", |b| {
        b.iter(|| black_box(42));
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_search_qps,
    bench_reconstruction_mse,
    bench_full_proof,
);
criterion_main!(benches);
