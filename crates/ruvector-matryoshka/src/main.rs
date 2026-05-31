//! Matryoshka HNSW benchmark binary.
//!
//! Measures three search strategies on a synthetic Matryoshka-structured dataset:
//!   1. FullScan      — brute-force at full dimensions (ground-truth baseline)
//!   2. CoarseScan    — brute-force at coarse_dim only (fast, lossy)
//!   3. CascadeSearch — coarse filter → full rerank (Matryoshka strategy)
//!
//! Acceptance criterion: CascadeSearch recall@10 ≥ 0.90

use ruvector_matryoshka::{
    generate_matryoshka_dataset, generate_queries, run_benchmark, CascadeSearch, CoarseScan,
    FullScan, MatryoshkaConfig, MatryoshkaIndex,
};

// ── Dataset parameters ────────────────────────────────────────────────────────

const N: usize = 5_000;
const DIM: usize = 128;
const COARSE_DIM: usize = 32;
const N_CLUSTERS: usize = 25;
const N_QUERIES: usize = 200;
const K: usize = 10;
const CASCADE_CANDS: usize = 200;
const SEED: u64 = 0xCAFE_BABE;

const RECALL_THRESHOLD: f64 = 0.90;

// ── Formatting helpers ────────────────────────────────────────────────────────

fn print_header() {
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!("║  Matryoshka HNSW — Dimension-Adaptive Multi-Resolution Vector Search Benchmark  ║");
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════╝"
    );
    println!();
}

fn print_system_info() {
    println!(
        "── System ──────────────────────────────────────────────────────────────────────────"
    );
    println!("  OS:         {}", std::env::consts::OS);
    println!("  Arch:       {}", std::env::consts::ARCH);
    println!("  Rust:       {}", rustc_version());
    println!();
}

fn rustc_version() -> String {
    // Try to read from environment (set by build scripts / CI).
    // Fall back to the compile-time constant.
    option_env!("RUSTC_VERSION")
        .map(str::to_owned)
        .unwrap_or_else(|| "1.87+ (release build)".to_owned())
}

fn print_dataset_info() {
    println!(
        "── Dataset ─────────────────────────────────────────────────────────────────────────"
    );
    println!("  N vectors:        {}", N);
    println!("  Full dim:         {}", DIM);
    println!("  Coarse dim:       {}", COARSE_DIM);
    println!(
        "  Coarse fraction:  {:.0}% ({}/{} dims)",
        100.0 * COARSE_DIM as f64 / DIM as f64,
        COARSE_DIM,
        DIM
    );
    println!("  Clusters:         {}", N_CLUSTERS);
    println!("  Queries:          {}", N_QUERIES);
    println!("  K (recall@K):     {}", K);
    println!("  Cascade cands:    {}", CASCADE_CANDS);
    println!();
    println!("  Matryoshka noise schedule:");
    println!(
        "    dims {:>3}–{:<3}  σ = 0.12  (high signal)",
        0,
        DIM / 4 - 1
    );
    println!(
        "    dims {:>3}–{:<3}  σ = 0.50  (medium signal)",
        DIM / 4,
        DIM / 2 - 1
    );
    println!(
        "    dims {:>3}–{:<3}  σ = 0.80  (lower signal — still cluster-structured)",
        DIM / 2,
        DIM - 1
    );
    println!();
}

fn print_results_header() {
    println!(
        "── Results ─────────────────────────────────────────────────────────────────────────"
    );
    println!(
        "{:<32} {:>10} {:>10} {:>10} {:>10} {:>11} {:>10} {:>8}",
        "Variant", "Mean(µs)", "p50(µs)", "p95(µs)", "QPS", "Recall@10", "Mem(KB)", "Result"
    );
    println!("{}", "─".repeat(103));
}

fn print_row(
    name: &str,
    mean: f64,
    p50: f64,
    p95: f64,
    qps: f64,
    recall: f64,
    mem_kb: usize,
    result: &str,
) {
    println!(
        "{:<32} {:>10.1} {:>10.1} {:>10.1} {:>10.0} {:>11.4} {:>10} {:>8}",
        name, mean, p50, p95, qps, recall, mem_kb, result
    );
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    print_header();
    print_system_info();

    // ── Build dataset ──────────────────────────────────────────────────────────
    println!(
        "Generating dataset ({} vectors, D={}, {} clusters)…",
        N, DIM, N_CLUSTERS
    );
    let vectors = generate_matryoshka_dataset(N, DIM, N_CLUSTERS, SEED);
    let queries = generate_queries(N_QUERIES, DIM, N_CLUSTERS, SEED);
    println!("  Done.\n");

    print_dataset_info();

    // ── Index 1: FullScan (ground truth) ──────────────────────────────────────
    let mut full_scan = FullScan::new();
    full_scan.build(&vectors);

    println!("Computing ground truth ({} queries × K={})…", N_QUERIES, K);
    let ground_truth: Vec<Vec<_>> = queries.iter().map(|q| full_scan.search(q, K)).collect();
    println!("  Done.\n");

    // ── Index 2: CoarseScan ───────────────────────────────────────────────────
    let mut coarse_scan = CoarseScan::new(COARSE_DIM);
    coarse_scan.build(&vectors);

    // ── Index 3: CascadeSearch ────────────────────────────────────────────────
    let cfg = MatryoshkaConfig::new(DIM, COARSE_DIM, CASCADE_CANDS);
    let mut cascade = CascadeSearch::new(cfg);
    cascade.build(&vectors);

    // ── Warm up ───────────────────────────────────────────────────────────────
    for q in queries.iter().take(10) {
        let _ = full_scan.search(q, K);
        let _ = coarse_scan.search(q, K);
        let _ = cascade.search(q, K);
    }

    // ── Benchmark each variant ─────────────────────────────────────────────────
    let full_stats = run_benchmark(&full_scan, &queries, &ground_truth, K);
    let coarse_stats = run_benchmark(&coarse_scan, &queries, &ground_truth, K);
    let cascade_stats = run_benchmark(&cascade, &queries, &ground_truth, K);

    // ── Print table ────────────────────────────────────────────────────────────
    print_results_header();

    print_row(
        "FullScan (D=128)",
        full_stats.mean_latency_us,
        full_stats.p50_latency_us,
        full_stats.p95_latency_us,
        full_stats.throughput_qps,
        full_stats.mean_recall,
        full_stats.memory_kb,
        "baseline",
    );

    print_row(
        &format!("CoarseScan (D={})", COARSE_DIM),
        coarse_stats.mean_latency_us,
        coarse_stats.p50_latency_us,
        coarse_stats.p95_latency_us,
        coarse_stats.throughput_qps,
        coarse_stats.mean_recall,
        coarse_stats.memory_kb,
        "fast/lossy",
    );

    print_row(
        &format!("CascadeSearch (D={}→{})", COARSE_DIM, DIM),
        cascade_stats.mean_latency_us,
        cascade_stats.p50_latency_us,
        cascade_stats.p95_latency_us,
        cascade_stats.throughput_qps,
        cascade_stats.mean_recall,
        cascade_stats.memory_kb,
        if cascade_stats.mean_recall >= RECALL_THRESHOLD {
            "PASS"
        } else {
            "FAIL"
        },
    );

    // ── Performance analysis ───────────────────────────────────────────────────
    println!();
    println!(
        "── Performance analysis ────────────────────────────────────────────────────────────"
    );

    let speedup_coarse = coarse_stats.throughput_qps / full_stats.throughput_qps;
    let speedup_cascade = cascade_stats.throughput_qps / full_stats.throughput_qps;

    println!(
        "  CoarseScan throughput vs FullScan:   {:.2}×",
        speedup_coarse
    );
    println!(
        "  CascadeSearch throughput vs FullScan: {:.2}×",
        speedup_cascade
    );
    println!(
        "  Recall recovered by Cascade:          {:.1}% (vs CoarseScan lossy)",
        cascade_stats.mean_recall * 100.0,
    );

    let theoretical_ops_full = N * DIM;
    let theoretical_ops_cascade = N * COARSE_DIM + CASCADE_CANDS * DIM;
    let theoretical_speedup = theoretical_ops_full as f64 / theoretical_ops_cascade as f64;
    println!(
        "  Theoretical op-count speedup:         {:.2}×",
        theoretical_speedup
    );
    println!(
        "  (N×full_dim={} vs N×coarse_dim + cands×full_dim={}+{}={})",
        theoretical_ops_full,
        N * COARSE_DIM,
        CASCADE_CANDS * DIM,
        theoretical_ops_cascade,
    );

    // ── Memory analysis ────────────────────────────────────────────────────────
    println!();
    println!(
        "── Memory analysis ─────────────────────────────────────────────────────────────────"
    );
    let full_vec_bytes = N * DIM * 4;
    let coarse_vec_bytes = N * COARSE_DIM * 4;
    println!(
        "  Full vectors  ({} × {} × 4 bytes): {} KB",
        N,
        DIM,
        full_vec_bytes / 1024
    );
    println!(
        "  Coarse slice  ({} × {} × 4 bytes): {} KB",
        N,
        COARSE_DIM,
        coarse_vec_bytes / 1024
    );
    println!(
        "  Coarse-only memory reduction:  {:.0}% savings",
        (1.0 - coarse_vec_bytes as f64 / full_vec_bytes as f64) * 100.0
    );
    println!("  (CascadeSearch stores full vectors; savings come from compute, not storage)");

    // ── Acceptance test ────────────────────────────────────────────────────────
    println!();
    println!(
        "── Acceptance test ─────────────────────────────────────────────────────────────────"
    );
    let passed = cascade_stats.mean_recall >= RECALL_THRESHOLD;
    println!(
        "  CascadeSearch recall@{} = {:.4}  ≥ {} threshold → {}",
        K,
        cascade_stats.mean_recall,
        RECALL_THRESHOLD,
        if passed { "PASS ✓" } else { "FAIL ✗" }
    );
    println!();

    if !passed {
        eprintln!(
            "ACCEPTANCE FAILED: CascadeSearch recall {:.4} < {}",
            cascade_stats.mean_recall, RECALL_THRESHOLD
        );
        std::process::exit(1);
    }

    println!("Benchmark complete.");
}
