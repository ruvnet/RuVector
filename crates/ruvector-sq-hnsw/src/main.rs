//! Benchmark binary: three HNSW variants compared on recall, latency, and memory.
//!
//! Usage:
//!   cargo run --release -p ruvector-sq-hnsw
//!   N=50000 DIMS=128 cargo run --release -p ruvector-sq-hnsw
//!
//! Environment variables:
//!   N        corpus size (default 10_000)
//!   DIMS     vector dimensions (default 128)
//!   QUERIES  query count (default 500)
//!   K        top-k (default 10)
//!   M        HNSW M parameter (default 16)
//!   EF_C     ef_construction (default 200)
//!   EF_S     ef_search (default 64)

use ruvector_sq_hnsw::{
    exact_knn, AnnIndex, F32Index, HnswConfig, SearchResult, Sq8Index, Sq8RerankIndex,
};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Deterministic pseudo-random number generator
// ---------------------------------------------------------------------------

fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn rand_f32(state: &mut u64) -> f32 {
    (xorshift64(state) as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
}

fn gen_vectors(n: usize, dims: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = seed;
    (0..n)
        .map(|_| (0..dims).map(|_| rand_f32(&mut rng)).collect())
        .collect()
}

// ---------------------------------------------------------------------------
// Latency helper: run `f` `reps` times, return sorted microsecond latencies
// ---------------------------------------------------------------------------

fn measure_latencies<F: FnMut() -> Vec<SearchResult>>(mut f: F, reps: usize) -> Vec<f64> {
    let mut us: Vec<f64> = (0..reps)
        .map(|_| {
            let t = Instant::now();
            std::hint::black_box(f());
            t.elapsed().as_nanos() as f64 / 1_000.0
        })
        .collect();
    us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    us
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = ((sorted.len() - 1) as f64 * p / 100.0) as usize;
    sorted[idx]
}

// ---------------------------------------------------------------------------
// Recall computation
// ---------------------------------------------------------------------------

fn compute_recall(
    index: &dyn AnnIndex,
    corpus: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
) -> f64 {
    let mut hits = 0usize;
    for q in queries {
        let approx: Vec<usize> = index.search(q, k).into_iter().map(|r| r.id).collect();
        let ground = exact_knn(corpus, q, k);
        hits += approx.iter().filter(|id| ground.contains(id)).count();
    }
    hits as f64 / (queries.len() * k) as f64
}

// ---------------------------------------------------------------------------
// System info
// ---------------------------------------------------------------------------

fn print_system_info() {
    println!("=== SQ-HNSW Calibrated Search Benchmark ===");
    println!("OS: {}", std::env::consts::OS);
    println!("Arch: {}", std::env::consts::ARCH);
    println!("Rust: {}", rustc_version_or_unknown());
    println!();
}

fn rustc_version_or_unknown() -> String {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_else(|| "unknown".to_string())
        .trim()
        .to_string()
}

fn parse_env<T: std::str::FromStr>(var: &str, default: T) -> T
where
    T::Err: std::fmt::Debug,
{
    std::env::var(var)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let n: usize = parse_env("N", 10_000);
    let dims: usize = parse_env("DIMS", 128);
    let n_queries: usize = parse_env("QUERIES", 500);
    let k: usize = parse_env("K", 10);
    let m: usize = parse_env("M", 16);
    let ef_c: usize = parse_env("EF_C", 200);
    let ef_s: usize = parse_env("EF_S", 64);

    print_system_info();

    println!("Dataset: n={n}, dims={dims}, queries={n_queries}, k={k}");
    println!("HNSW: M={m}, ef_construction={ef_c}, ef_search={ef_s}");
    println!();

    let config = HnswConfig {
        m,
        ef_construction: ef_c,
        ef_search: ef_s,
    };

    // Generate data
    print!("Generating corpus ({n} × {dims})... ");
    let corpus = gen_vectors(n, dims, 12345);
    let queries = gen_vectors(n_queries, dims, 99999);
    println!("done.");
    println!();

    // -----------------------------------------------------------------------
    // Variant 1: F32 HNSW
    // -----------------------------------------------------------------------
    print!("Building F32Index...");
    let t_start = Instant::now();
    let mut f32_idx = F32Index::new(config, dims);
    for (i, v) in corpus.iter().enumerate() {
        f32_idx.add(i, v.clone());
    }
    let f32_build_ms = t_start.elapsed().as_millis();
    println!(" {f32_build_ms} ms");

    // -----------------------------------------------------------------------
    // Variant 2: SQ8 HNSW
    // -----------------------------------------------------------------------
    print!("Building Sq8Index...");
    let t_start = Instant::now();
    let mut sq8_idx = Sq8Index::new(config, dims, n); // calibrate on full corpus
    for (i, v) in corpus.iter().enumerate() {
        sq8_idx.add(i, v.clone());
    }
    let sq8_build_ms = t_start.elapsed().as_millis();
    println!(" {sq8_build_ms} ms");

    // -----------------------------------------------------------------------
    // Variant 3: SQ8 + rerank
    // -----------------------------------------------------------------------
    print!("Building Sq8RerankIndex...");
    let t_start = Instant::now();
    let mut rerank_idx = Sq8RerankIndex::new(config, dims, n);
    for (i, v) in corpus.iter().enumerate() {
        rerank_idx.add(i, v.clone());
    }
    let rerank_build_ms = t_start.elapsed().as_millis();
    println!(" {rerank_build_ms} ms\n");

    // -----------------------------------------------------------------------
    // Recall
    // -----------------------------------------------------------------------
    print!("Computing recall@{k} (F32)... ");
    let r_f32 = compute_recall(&f32_idx, &corpus, &queries, k);
    println!("{:.4}", r_f32);

    print!("Computing recall@{k} (SQ8)... ");
    let r_sq8 = compute_recall(&sq8_idx, &corpus, &queries, k);
    println!("{:.4}", r_sq8);

    print!("Computing recall@{k} (SQ8+Rerank)... ");
    let r_rerank = compute_recall(&rerank_idx, &corpus, &queries, k);
    println!("{:.4}", r_rerank);
    println!();

    // -----------------------------------------------------------------------
    // Latency
    // -----------------------------------------------------------------------
    let lat_reps = n_queries.min(200);
    let q_slice = &queries[..lat_reps];

    let lats_f32: Vec<f64> = q_slice
        .iter()
        .flat_map(|q| measure_latencies(|| f32_idx.search(q, k), 1))
        .collect();

    let lats_sq8: Vec<f64> = q_slice
        .iter()
        .flat_map(|q| measure_latencies(|| sq8_idx.search(q, k), 1))
        .collect();

    let lats_rerank: Vec<f64> = q_slice
        .iter()
        .flat_map(|q| measure_latencies(|| rerank_idx.search(q, k), 1))
        .collect();

    let mut lats_f32_s = lats_f32.clone();
    let mut lats_sq8_s = lats_sq8.clone();
    let mut lats_rerank_s = lats_rerank.clone();
    lats_f32_s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    lats_sq8_s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    lats_rerank_s.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Throughput
    let qps_f32 = lat_reps as f64 / (lats_f32.iter().sum::<f64>() / 1_000_000.0);
    let qps_sq8 = lat_reps as f64 / (lats_sq8.iter().sum::<f64>() / 1_000_000.0);
    let qps_rerank = lat_reps as f64 / (lats_rerank.iter().sum::<f64>() / 1_000_000.0);

    // Memory
    let mem_f32_mb = (f32_idx.len() * f32_idx.bytes_per_vector()) as f64 / 1_048_576.0;
    let mem_sq8_mb = (sq8_idx.len() * sq8_idx.bytes_per_vector()) as f64 / 1_048_576.0;
    let mem_rerank_mb = (rerank_idx.len() * rerank_idx.bytes_per_vector()) as f64 / 1_048_576.0;

    // -----------------------------------------------------------------------
    // Print results table
    // -----------------------------------------------------------------------
    println!(
        "┌─────────────────┬──────────┬────────────┬────────────┬────────────┬─────────────┬──────────────┬──────────────┐"
    );
    println!(
        "│ Variant         │ Recall@{k:<2}│ Mean(μs)   │ p50(μs)    │ p95(μs)    │ QPS         │ Mem/vec(B)   │ Build(ms)    │"
    );
    println!(
        "├─────────────────┼──────────┼────────────┼────────────┼────────────┼─────────────┼──────────────┼──────────────┤"
    );

    let mean_f32 = lats_f32.iter().sum::<f64>() / lats_f32.len() as f64;
    let mean_sq8 = lats_sq8.iter().sum::<f64>() / lats_sq8.len() as f64;
    let mean_rerank = lats_rerank.iter().sum::<f64>() / lats_rerank.len() as f64;

    println!(
        "│ F32 (baseline)  │ {:.4}   │ {:>10.1} │ {:>10.1} │ {:>10.1} │ {:>11.0} │ {:>12.0} │ {:>12} │",
        r_f32, mean_f32, percentile(&lats_f32_s, 50.0), percentile(&lats_f32_s, 95.0),
        qps_f32, mem_f32_mb * 1_048_576.0 / n as f64, f32_build_ms
    );
    println!(
        "│ SQ8 (no-rerank) │ {:.4}   │ {:>10.1} │ {:>10.1} │ {:>10.1} │ {:>11.0} │ {:>12.0} │ {:>12} │",
        r_sq8, mean_sq8, percentile(&lats_sq8_s, 50.0), percentile(&lats_sq8_s, 95.0),
        qps_sq8, mem_sq8_mb * 1_048_576.0 / n as f64, sq8_build_ms
    );
    println!(
        "│ SQ8 + Rerank    │ {:.4}   │ {:>10.1} │ {:>10.1} │ {:>10.1} │ {:>11.0} │ {:>12.0} │ {:>12} │",
        r_rerank, mean_rerank, percentile(&lats_rerank_s, 50.0), percentile(&lats_rerank_s, 95.0),
        qps_rerank, mem_rerank_mb * 1_048_576.0 / n as f64, rerank_build_ms
    );
    println!(
        "└─────────────────┴──────────┴────────────┴────────────┴────────────┴─────────────┴──────────────┴──────────────┘"
    );

    println!();
    println!("Notes:");
    println!(
        "  - Mem/vec excludes graph topology (~{} B/node at M={m})",
        m * 2 * 8
    );
    println!("  - Calibration: SQ8 variants calibrate on the full corpus before insertion");
    println!(
        "  - Rerank overquery factor: {}×k candidates fetched before exact rerank",
        3
    );
    println!("  - Corpus: deterministic pseudo-random uniform[-1,1]^{dims}");
    println!();

    // -----------------------------------------------------------------------
    // Acceptance test
    // -----------------------------------------------------------------------
    println!("=== Acceptance Test ===");
    let f32_ok = r_f32 >= 0.70;
    let sq8_ok = r_sq8 >= 0.55;
    let rerank_ok = r_rerank >= 0.70;
    let rerank_faster_than_f32 = mean_sq8 <= mean_f32 * 1.50; // SQ8 no more than 50% slower than f32

    let all_pass = f32_ok && sq8_ok && rerank_ok;

    println!(
        "F32 recall@{k} >= 0.70:    {} (got {:.4})",
        if f32_ok { "PASS" } else { "FAIL" },
        r_f32
    );
    println!(
        "SQ8 recall@{k} >= 0.55:    {} (got {:.4})",
        if sq8_ok { "PASS" } else { "FAIL" },
        r_sq8
    );
    println!(
        "Rerank recall@{k} >= 0.70: {} (got {:.4})",
        if rerank_ok { "PASS" } else { "FAIL" },
        r_rerank
    );
    println!(
        "SQ8 mean latency <= 1.5×F32: {} ({:.1}μs vs {:.1}μs × 1.5 = {:.1}μs)",
        if rerank_faster_than_f32 {
            "PASS"
        } else {
            "NOTE"
        },
        mean_sq8,
        mean_f32,
        mean_f32 * 1.5
    );
    println!();

    // Also validate memory ratios
    let sq8_mem_ratio = mem_sq8_mb / mem_f32_mb;
    let expected_ratio_lo = 0.20;
    let expected_ratio_hi = 0.30;
    let mem_ok = sq8_mem_ratio >= expected_ratio_lo && sq8_mem_ratio <= expected_ratio_hi;
    println!(
        "SQ8 mem ratio ∈ [{expected_ratio_lo:.2}, {expected_ratio_hi:.2}]: {} (got {sq8_mem_ratio:.3})",
        if mem_ok { "PASS" } else { "NOTE" }
    );

    println!();
    if all_pass {
        println!("RESULT: PASS — all acceptance criteria met.");
    } else {
        println!("RESULT: FAIL — one or more acceptance criteria not met.");
        std::process::exit(1);
    }
}
