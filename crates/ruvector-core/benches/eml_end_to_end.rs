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
//! # Full synthetic proof (100K vectors, 3 seeds, Markdown table)
//! EML_FULL_PROOF=1 cargo bench -p ruvector-core --bench eml_end_to_end -- eml_e2e_full_proof
//!
//! # Full real-dataset proof (SIFT1M + GloVe subsets)
//! # Requires bench_data/sift/ and bench_data/glove.6B.100d.txt (see DATASETS.md)
//! EML_FULL_PROOF=1 EML_REAL_DATASETS=1 EML_PROOF_N=100000 EML_PROOF_Q=500 \
//!   cargo bench -p ruvector-core --bench eml_end_to_end -- eml_e2e_full_proof
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
use ruvector_core::advanced::eml::EmlScoreFusion;
use ruvector_core::distance::{cosine_distance, euclidean_distance};
use ruvector_core::index::hnsw::HnswIndex;
use ruvector_core::index::VectorIndex;
use ruvector_core::quantization::{LogQuantized, QuantizedVector, ScalarQuantized};
use ruvector_core::types::{DistanceMetric, HnswConfig};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

// =============================================================================
// Configuration
// =============================================================================

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
// Real Dataset Loaders (SIFT1M, GloVe)
// =============================================================================

/// Holds a loaded ANN dataset ready for benchmarking.
struct Dataset {
    /// Human-readable name for reporting
    name: String,
    /// Base vectors to index
    base: Vec<Vec<f32>>,
    /// Query vectors
    queries: Vec<Vec<f32>>,
    /// Vector dimensionality
    dims: usize,
    /// Distance metric this dataset is designed for
    metric: DistanceMetric,
}

/// Root of the dataset cache. Override via `EML_BENCH_DATA_DIR`.
fn bench_data_dir() -> PathBuf {
    std::env::var("EML_BENCH_DATA_DIR")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let here = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            // Walk up to workspace root then into bench_data
            here.ancestors()
                .find(|p| p.join("bench_data").is_dir())
                .map(|p| p.join("bench_data"))
                .unwrap_or_else(|| here.join("bench_data"))
        })
}

/// Read a .fvecs file (Texmex format): [d:i32 LE][f32 * d] repeated per vector.
fn read_fvecs(path: &Path, max_vectors: Option<usize>) -> std::io::Result<Vec<Vec<f32>>> {
    let mut file = BufReader::new(File::open(path)?);
    let mut vectors = Vec::new();
    let mut dim_buf = [0u8; 4];
    loop {
        match file.read_exact(&mut dim_buf) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        let d = i32::from_le_bytes(dim_buf) as usize;
        let mut vec = vec![0f32; d];
        // Safety: reading into the Vec<f32>'s byte buffer. f32 has no invalid bit patterns
        // for the arithmetic we do (NaN/Inf would be accepted but SIFT data is clean).
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(vec.as_mut_ptr() as *mut u8, d * 4)
        };
        file.read_exact(bytes)?;
        vectors.push(vec);
        if let Some(max) = max_vectors {
            if vectors.len() >= max {
                break;
            }
        }
    }
    Ok(vectors)
}

/// Load SIFT1M from `bench_data/sift/` (downloaded separately).
/// Returns Err if dataset is not cached — caller should fall back to synthetic.
fn load_sift1m(n_base: usize, n_queries: usize) -> Result<Dataset, String> {
    let root = bench_data_dir().join("sift");
    let base_path = root.join("sift_base.fvecs");
    let query_path = root.join("sift_query.fvecs");

    if !base_path.exists() || !query_path.exists() {
        return Err(format!(
            "SIFT1M not found at {:?}. Download with:\n  \
             cd bench_data && curl -fLO ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz && tar xzf sift.tar.gz",
            root
        ));
    }

    let base = read_fvecs(&base_path, Some(n_base))
        .map_err(|e| format!("Failed to read {:?}: {}", base_path, e))?;
    let queries = read_fvecs(&query_path, Some(n_queries))
        .map_err(|e| format!("Failed to read {:?}: {}", query_path, e))?;

    let dims = base.first().map(|v| v.len()).unwrap_or(128);
    eprintln!(
        "  [dataset] SIFT1M loaded: {} base x {} queries x {}D",
        base.len(),
        queries.len(),
        dims
    );

    Ok(Dataset {
        name: format!("SIFT1M-subset-{}", base.len()),
        base,
        queries,
        dims,
        metric: DistanceMetric::Euclidean,
    })
}

/// Load GloVe word embeddings from `bench_data/glove.6B.100d.txt`.
/// Format: `word v1 v2 ... v100\n` per line. We split off the last
/// `n_queries` vectors as the query set so they don't appear in the base.
fn load_glove(n_base: usize, n_queries: usize) -> Result<Dataset, String> {
    let path = bench_data_dir().join("glove.6B.100d.txt");

    if !path.exists() {
        return Err(format!(
            "GloVe not found at {:?}. Download with:\n  \
             cd bench_data && curl -fLO https://nlp.stanford.edu/data/glove.6B.zip && unzip -o glove.6B.zip glove.6B.100d.txt",
            path
        ));
    }

    let file = File::open(&path).map_err(|e| format!("open glove: {}", e))?;
    let reader = BufReader::new(file);
    let total_needed = n_base + n_queries;
    let mut all: Vec<Vec<f32>> = Vec::with_capacity(total_needed);

    for line in reader.lines().take(total_needed) {
        let line = line.map_err(|e| format!("read glove line: {}", e))?;
        let mut parts = line.split_whitespace();
        parts.next(); // skip the word token
        let vec: Vec<f32> = parts
            .filter_map(|t| t.parse::<f32>().ok())
            .collect();
        if vec.len() == 100 {
            all.push(vec);
        }
    }

    if all.len() < total_needed {
        return Err(format!(
            "GloVe has only {} valid vectors, need {}",
            all.len(),
            total_needed
        ));
    }

    // Split: first n_base = base, next n_queries = queries (disjoint sets)
    let queries: Vec<Vec<f32>> = all.drain(n_base..n_base + n_queries).collect();
    let base = all;

    let dims = base.first().map(|v| v.len()).unwrap_or(100);
    eprintln!(
        "  [dataset] GloVe loaded: {} base x {} queries x {}D",
        base.len(),
        queries.len(),
        dims
    );

    Ok(Dataset {
        name: format!("GloVe-100d-subset-{}", base.len()),
        base,
        queries,
        dims,
        metric: DistanceMetric::Cosine,
    })
}

/// Build a synthetic Dataset for one of the two built-in generators.
fn synthetic_dataset(
    generator_name: &str,
    n_base: usize,
    n_queries: usize,
    seed: u64,
    dims: usize,
) -> Dataset {
    let (base, queries, metric) = match generator_name {
        "sift_like" => (
            generate_sift_like(n_base, dims, seed),
            generate_sift_like(n_queries, dims, seed + 1000),
            DistanceMetric::Euclidean,
        ),
        "normal" => (
            generate_normal_embeddings(n_base, dims, seed),
            generate_normal_embeddings(n_queries, dims, seed + 1000),
            DistanceMetric::Euclidean,
        ),
        _ => panic!("unknown synthetic generator: {}", generator_name),
    };
    Dataset {
        name: format!("{}-seed{}", generator_name, seed),
        base,
        queries,
        dims,
        metric,
    }
}

// =============================================================================
// Brute-Force Ground Truth
// =============================================================================

/// Compute exact ground-truth top-k neighbors via brute force under the
/// given metric. Parallelized with Rayon.
fn compute_ground_truth(
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
    metric: DistanceMetric,
) -> Vec<Vec<usize>> {
    use rayon::prelude::*;
    let dist_fn = |a: &[f32], b: &[f32]| -> f32 {
        match metric {
            DistanceMetric::Euclidean => euclidean_distance(a, b),
            DistanceMetric::Cosine => cosine_distance(a, b),
            // Not used by our datasets; keep scalar for completeness
            DistanceMetric::DotProduct => -a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>(),
            DistanceMetric::Manhattan => {
                a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>()
            }
        }
    };

    queries
        .par_iter()
        .map(|query| {
            let mut dists: Vec<(usize, f32)> = base
                .iter()
                .enumerate()
                .map(|(i, v)| (i, dist_fn(query, v)))
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
    dataset: &Dataset,
    ground_truth_100: &[Vec<usize>],
    use_unified: bool,
) -> RunResult {
    // EML_PAD_UNIFIED=1 opts into `HnswIndex::new_unified_padded()` for the
    // unified-distance variant. This is a v4 test to measure whether zero-
    // padding non-power-of-two dimensions (GloVe 100D) recovers the synthetic
    // win that v3 showed regressing.
    let pad_unified = std::env::var("EML_PAD_UNIFIED").ok().as_deref() == Some("1");

    let base = &dataset.base;
    let queries = &dataset.queries;

    let config = HnswConfig {
        m: HNSW_M,
        ef_construction: EF_CONSTRUCTION,
        ef_search: 64,
        max_elements: base.len() + 1000,
    };

    // --- Build Index ---
    let build_start = Instant::now();
    let mut index = if use_unified {
        if pad_unified {
            HnswIndex::new_unified_padded(dataset.dims, dataset.metric, config.clone()).unwrap()
        } else {
            HnswIndex::new_unified(dataset.dims, dataset.metric, config.clone()).unwrap()
        }
    } else {
        HnswIndex::new(dataset.dims, dataset.metric, config.clone()).unwrap()
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
    dims: usize,
    seeds: &[u64],
    baseline: &AggregatedResult,
    log_quantized: &AggregatedResult,
) {
    eprintln!("\n### EML Optimizations — End-to-End Proof\n");
    eprintln!(
        "**Dataset**: {} ({} base, {} queries, {}D)",
        dataset_name, num_base, num_queries, dims
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
    let dims = 128;

    let base = generate_sift_like(num_base, dims, 42);
    let queries = generate_sift_like(num_queries, dims, 99);

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
    let mut baseline_idx = HnswIndex::new(dims, DistanceMetric::Euclidean, config.clone()).unwrap();
    baseline_idx.add_batch(entries.clone()).unwrap();

    let mut log_idx = HnswIndex::new_unified(dims, DistanceMetric::Euclidean, config.clone()).unwrap();
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

    let sift_data = generate_sift_like(10_000, 128, 42);
    let normal_data = generate_normal_embeddings(10_000, 128, 42);

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

/// Run the proof across three seeds for a dataset-producer function.
/// For synthetic datasets `producer` returns a fresh synthetic Dataset per seed.
/// For real datasets `producer` returns the same fixed Dataset every seed (but
/// seed still varies HNSW construction order indirectly via query shuffling —
/// here we keep queries fixed so results are deterministic across seeds,
/// which means stddev primarily reflects timing noise for real data).
fn run_seeded_proof<F: Fn(u64) -> Dataset>(
    dataset_label: &str,
    producer: F,
    seeds: &[u64],
) -> (AggregatedResult, AggregatedResult, String, usize, usize, usize) {
    let mut baseline_runs = Vec::new();
    let mut log_runs = Vec::new();
    let mut base_n = 0usize;
    let mut q_n = 0usize;
    let mut dims = 0usize;

    for &seed in seeds {
        let dataset = producer(seed);
        eprintln!(
            "  [proof] {} seed={}: computing ground truth ({} base x {} queries, metric={:?})...",
            dataset_label, seed, dataset.base.len(), dataset.queries.len(), dataset.metric
        );
        let gt_start = Instant::now();
        let gt = compute_ground_truth(&dataset.base, &dataset.queries, 100, dataset.metric);
        eprintln!("  [proof] {} seed={}: ground truth done in {:.1}s",
            dataset_label, seed, gt_start.elapsed().as_secs_f64());

        eprintln!("  [proof] {} seed={}: building baseline (Scalar+Dispatched)...", dataset_label, seed);
        baseline_runs.push(run_end_to_end(&dataset, &gt, false));
        eprintln!("  [proof] {} seed={}: building log+unified (Log+SIMD-Unified)...", dataset_label, seed);
        log_runs.push(run_end_to_end(&dataset, &gt, true));

        base_n = dataset.base.len();
        q_n = dataset.queries.len();
        dims = dataset.dims;
    }

    (
        aggregate_runs(&baseline_runs),
        aggregate_runs(&log_runs),
        dataset_label.to_string(),
        base_n,
        q_n,
        dims,
    )
}

/// Full proof report — gated behind `EML_FULL_PROOF=1` because each dataset run
/// takes several minutes (HNSW build is O(n·log(n)·M) and is single-threaded for
/// insertion in hnsw_rs).
///
/// Controls:
/// - `EML_FULL_PROOF=1`          — enable the full proof
/// - `EML_PROOF_N=100000`        — base vectors per run (default 100K)
/// - `EML_PROOF_Q=500`           — query count per run (default 500)
/// - `EML_REAL_DATASETS=1`       — also run on SIFT1M + GloVe (requires bench_data/)
/// - `EML_SYNTHETIC_DATASETS=0`  — skip synthetic (if you only want real data)
///
/// ```bash
/// # Synthetic only (fast)
/// EML_FULL_PROOF=1 cargo bench -p ruvector-core --bench eml_end_to_end -- eml_e2e_full_proof
///
/// # Real datasets too (slower — requires bench_data/sift/ and bench_data/glove.6B.100d.txt)
/// EML_FULL_PROOF=1 EML_REAL_DATASETS=1 \
///   cargo bench -p ruvector-core --bench eml_end_to_end -- eml_e2e_full_proof
/// ```
fn bench_full_proof(c: &mut Criterion) {
    if std::env::var("EML_FULL_PROOF").ok().as_deref() != Some("1") {
        return;
    }

    let num_base: usize = std::env::var("EML_PROOF_N")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let num_queries: usize = std::env::var("EML_PROOF_Q")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(1_000);
    let seeds: [u64; 3] = [42, 1337, 2024];
    let run_synthetic = std::env::var("EML_SYNTHETIC_DATASETS").ok().as_deref() != Some("0");
    let run_real = std::env::var("EML_REAL_DATASETS").ok().as_deref() == Some("1");

    eprintln!("\n=== EML_FULL_PROOF enabled: N={} Q={} (synthetic={}, real={}) ===\n",
        num_base, num_queries, run_synthetic, run_real);

    // ------------- Synthetic datasets -------------
    if run_synthetic {
        let (bl, lg, name, n, q, d) = run_seeded_proof(
            "SIFT-like (synthetic)",
            |seed| synthetic_dataset("sift_like", num_base, num_queries, seed, 128),
            &seeds,
        );
        print_markdown_report(&name, n, q, d, &seeds, &bl, &lg);

        let (bl, lg, name, n, q, d) = run_seeded_proof(
            "Normal embeddings (synthetic)",
            |seed| synthetic_dataset("normal", num_base, num_queries, seed, 128),
            &seeds,
        );
        print_markdown_report(&name, n, q, d, &seeds, &bl, &lg);
    }

    // ------------- Real datasets -------------
    if run_real {
        // Per-dataset skip flags (handy for focused v4 padding test on GloVe only):
        //   EML_SKIP_SIFT1M=1 — skip SIFT1M
        //   EML_SKIP_GLOVE=1  — skip GloVe
        let skip_sift = std::env::var("EML_SKIP_SIFT1M").ok().as_deref() == Some("1");
        let skip_glove = std::env::var("EML_SKIP_GLOVE").ok().as_deref() == Some("1");

        // SIFT1M: Fixed across seeds (deterministic data). Stddev will reflect only
        // build/search timing variance, not data variance. That's fine — it's more
        // realistic than three different synthetic draws.
        if !skip_sift {
            match load_sift1m(num_base, num_queries) {
                Ok(ds) => {
                    let label = format!("SIFT1M (real, {} base)", ds.base.len());
                    eprintln!("\n  [proof] Starting {} with {} seeds", label, seeds.len());
                    let (bl, lg, _, n, q, d) = run_seeded_proof(
                        &label,
                        |_seed| {
                            load_sift1m(num_base, num_queries).expect("SIFT1M already validated")
                        },
                        &seeds,
                    );
                    print_markdown_report(&label, n, q, d, &seeds, &bl, &lg);
                }
                Err(e) => {
                    eprintln!("  [proof] Skipping SIFT1M — {}", e);
                }
            }
        }

        if !skip_glove {
            match load_glove(num_base, num_queries) {
                Ok(ds) => {
                    let label = format!("GloVe-100d (real, {} base, cosine)", ds.base.len());
                    eprintln!("\n  [proof] Starting {} with {} seeds", label, seeds.len());
                    let (bl, lg, _, n, q, d) = run_seeded_proof(
                        &label,
                        |_seed| load_glove(num_base, num_queries).expect("GloVe already validated"),
                        &seeds,
                    );
                    print_markdown_report(&label, n, q, d, &seeds, &bl, &lg);
                }
                Err(e) => {
                    eprintln!("  [proof] Skipping GloVe — {}", e);
                }
            }
        }
    }

    // Sentinel so Criterion has something to run under this group name
    let mut group = c.benchmark_group("eml_e2e_full_proof");
    group.sample_size(10);
    group.bench_function("_report_generated", |b| {
        b.iter(|| black_box(42));
    });
    group.finish();
}

// =============================================================================
// Score Fusion Micro-Benchmark (EmlScoreFusion vs Linear)
// =============================================================================

/// Compares the wall-time + output characteristics of linear hybrid scoring
/// (`alpha*vec + (1-alpha)*bm25`) vs EML non-linear score fusion on synthetic
/// (vector_score, bm25_score) pairs. The benchmark measures overhead; the
/// *quality* claim (3.16x asymmetry) is covered by `tests/eml_proof.rs`.
fn bench_score_fusion_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("eml_e2e_score_fusion");

    let mut rng = StdRng::seed_from_u64(42);
    let pairs: Vec<(f32, f32)> = (0..10_000)
        .map(|_| (rng.gen::<f32>(), rng.gen::<f32>() * 10.0))
        .collect();

    group.bench_function("linear_fusion_10k", |b| {
        let alpha = 0.7f32;
        b.iter(|| {
            let mut sum = 0.0f32;
            for &(v, bm) in &pairs {
                sum += black_box(alpha) * black_box(v) + (1.0 - alpha) * black_box(bm);
            }
            sum
        });
    });

    let fusion = EmlScoreFusion::default();
    group.bench_function("eml_fusion_10k", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for &(v, bm) in &pairs {
                sum += fusion.fuse(black_box(v), black_box(bm));
            }
            sum
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_search_qps,
    bench_reconstruction_mse,
    bench_score_fusion_overhead,
    bench_full_proof,
);
criterion_main!(benches);
