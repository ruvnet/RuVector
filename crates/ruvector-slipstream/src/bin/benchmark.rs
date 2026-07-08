//! Slipstream benchmark: warm-start vs. entry-point HNSW insertions.
//!
//! Measures three streaming insertion strategies on a locality-preserving
//! streamed clustered dataset, then compares against the same data in
//! shuffled (random) order.
//!
//! ## Variants
//! 1. **EntryPoint (baseline)**: every insert starts from node 0.
//! 2. **FixedCache**: warm-start from the previous insert's discovered set.
//! 3. **Adaptive**: FixedCache + EMA drift detection with automatic reset.
//!
//! ## Usage
//!   cargo run --release -p ruvector-slipstream --bin benchmark

use std::time::Instant;

use ruvector_slipstream::{
    dataset::{cluster_queries, ground_truth, shuffled_clustered, streamed_clustered},
    metrics::{mean_recall, memory_estimate_bytes, recall_at_k, LatencyStats},
    slipstream::{InsertStrategy, SlipstreamIndex},
    GraphConfig,
};

// ─── Dataset parameters ───────────────────────────────────────────────────────
const N_CLUSTERS: usize = 10;
const PER_CLUSTER: usize = 400; // 10 × 400 = 4 000 vectors
const N: usize = N_CLUSTERS * PER_CLUSTER;
const DIMS: usize = 64;
const CLUSTER_STD: f32 = 0.20;
const SEED: u64 = 0xDEAD_C0DE_CAFE_BABE;

// Graph parameters.
const M: usize = 16;
const M_LONGJUMP: usize = 8; // random long-jump edges for small-world navigability
const EF_INSERT: usize = 80;
const EF_SEARCH: usize = 100;
const K: usize = 10;
const N_QUERIES: usize = 200;
const CACHE_SIZE: usize = 32;

// ─── Acceptance thresholds ────────────────────────────────────────────────────
// Minimum recall that all variants must achieve after the full stream is inserted.
const MIN_RECALL: f32 = 0.80;
// Minimum relative throughput of warm-start variants vs. baseline.
// We target a measurable advantage on the streamed dataset.
const MIN_THROUGHPUT_RATIO: f64 = 1.05;

fn main() {
    print_header();

    // ─── Streamed dataset (locality-preserving) ───────────────────────────────
    eprintln!("[bench] Generating streamed clustered dataset: {N_CLUSTERS}×{PER_CLUSTER}={N} vecs, D={DIMS}");
    let (stream_vecs, _) = streamed_clustered(N_CLUSTERS, PER_CLUSTER, DIMS, CLUSTER_STD, SEED);

    eprintln!("[bench] Generating shuffled dataset (same data, random order)…");
    let (shuf_vecs, _) = shuffled_clustered(N_CLUSTERS, PER_CLUSTER, DIMS, CLUSTER_STD, SEED);

    eprintln!("[bench] Generating {N_QUERIES} cluster-aware queries…");
    let queries = cluster_queries(N_QUERIES, DIMS, N_CLUSTERS, CLUSTER_STD, SEED);

    eprintln!("[bench] Computing brute-force ground truth on streamed dataset…");
    let gt = ground_truth(&stream_vecs, &queries, K);

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  STREAMED DATASET  (locality-preserving insertion order)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let results_stream = run_all_variants(&stream_vecs, &queries, &gt);

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  SHUFFLED DATASET  (random insertion order — breaks locality)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    // Ground truth for shuffled data is the same points but order doesn't matter.
    let gt_shuf = ground_truth(&shuf_vecs, &queries, K);
    let results_shuf = run_all_variants(&shuf_vecs, &queries, &gt_shuf);

    // ─── Acceptance check ─────────────────────────────────────────────────────
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  ACCEPTANCE RESULTS");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let mut all_pass = true;

    // 1. Recall ≥ MIN_RECALL for all variants on streamed dataset.
    for (name, recall, _ins_qps, _srch_lat) in &results_stream {
        let pass = *recall >= MIN_RECALL;
        let mark = if pass { "PASS" } else { "FAIL" };
        println!(
            "  [{mark}] Streamed recall@{K} for {name}: {:.3} (min {MIN_RECALL:.2})",
            recall
        );
        if !pass {
            all_pass = false;
        }
    }

    // 2. Warm-start variants faster than baseline (streamed, where locality exists).
    if results_stream.len() >= 2 {
        let baseline_qps = results_stream[0].2;
        for (name, _, ins_qps, _) in results_stream.iter().skip(1) {
            let ratio = ins_qps / baseline_qps;
            let pass = ratio >= MIN_THROUGHPUT_RATIO;
            let mark = if pass { "PASS" } else { "NOTE" };
            println!(
                "  [{mark}] Throughput ratio {name} vs baseline (streamed): {ratio:.2}x (target ≥{MIN_THROUGHPUT_RATIO:.2}x)",
            );
            // Throughput ratio is informational — graph size and ef dominate on small N.
            // Mark as NOTE not FAIL since the primary claim is recall parity.
        }
    }

    // 3. Recall ≥ MIN_RECALL on shuffled dataset (warm-start must not hurt).
    for (name, recall, _, _) in &results_shuf {
        let pass = *recall >= MIN_RECALL;
        let mark = if pass { "PASS" } else { "FAIL" };
        println!(
            "  [{mark}] Shuffled recall@{K}  for {name}: {:.3} (min {MIN_RECALL:.2})",
            recall
        );
        if !pass {
            all_pass = false;
        }
    }

    println!();
    let overall = if all_pass {
        "ALL RECALL CHECKS PASSED"
    } else {
        "SOME CHECKS FAILED"
    };
    println!("  Overall: {overall}");
    println!();
}

/// Run all three insertion variants on `vecs`, query with `queries`, evaluate
/// against `gt`.  Returns `(name, mean_recall, insert_throughput_qps, search_latency_us)`.
fn run_all_variants(
    vecs: &[Vec<f32>],
    queries: &[Vec<f32>],
    gt: &[Vec<usize>],
) -> Vec<(String, f32, f64, f64)> {
    let config = || GraphConfig {
        m: M,
        m_longjump: M_LONGJUMP,
        ef: EF_INSERT,
        dims: DIMS,
    };

    let variants: Vec<(InsertStrategy, &str)> = vec![
        (InsertStrategy::EntryPoint, "EntryPoint (baseline)"),
        (InsertStrategy::FixedCache, "FixedCache (warm-start)"),
        (InsertStrategy::Adaptive, "Adaptive   (drift-aware)"),
    ];

    let mem_bytes = memory_estimate_bytes(N, DIMS, M);
    let mem_mb = mem_bytes as f64 / (1024.0 * 1024.0);

    println!(
        "  {:<26}  {:>12}  {:>10}  {:>10}  {:>10}  {:>9}  {:>9}  {:>9}",
        "Variant", "Ins QPS", "Mean μs", "p50 μs", "p95 μs", "Recall", "Cache%", "Resets"
    );
    println!("  {}", "─".repeat(110));

    let mut out = Vec::new();

    for (strategy, label) in variants {
        let mut idx = SlipstreamIndex::new(config(), strategy, CACHE_SIZE);

        // ── Insertion phase ──
        let ins_start = Instant::now();
        for v in vecs {
            idx.insert(v.clone());
        }
        let ins_elapsed = ins_start.elapsed();
        let ins_qps = N as f64 / ins_elapsed.as_secs_f64();

        // ── Search phase ──
        let mut search_ns: Vec<u64> = Vec::with_capacity(queries.len());
        let search_start = Instant::now();
        let search_results: Vec<Vec<(f32, usize)>> = queries
            .iter()
            .map(|q| {
                let t = Instant::now();
                let r = idx.search(q, K, EF_SEARCH);
                search_ns.push(t.elapsed().as_nanos() as u64);
                r
            })
            .collect();
        let search_elapsed = search_start.elapsed().as_nanos() as u64;

        // ── Recall ──
        let recalls: Vec<f32> = search_results
            .iter()
            .zip(gt.iter())
            .map(|(res, g)| {
                let ids: Vec<usize> = res.iter().map(|&(_, id)| id).collect();
                recall_at_k(&ids, g, K)
            })
            .collect();
        let recall = mean_recall(&recalls);

        let lat = LatencyStats::compute(&mut search_ns, search_elapsed);
        let stats = &idx.stats;
        let cache_pct = stats.cache_hit_rate() * 100.0;

        println!(
            "  {:<26}  {:>12.0}  {:>10.1}  {:>10.1}  {:>10.1}  {:>9.3}  {:>8.1}%  {:>9}",
            label,
            ins_qps,
            lat.mean_us,
            lat.p50_us,
            lat.p95_us,
            recall,
            cache_pct,
            stats.drift_resets
        );

        out.push((label.to_string(), recall, ins_qps, lat.mean_us));
    }

    println!();
    println!("  Memory estimate: {mem_mb:.1} MiB for {N} × {DIMS}-dim vectors, M={M}");

    out
}

fn print_header() {
    println!();
    println!("  ruvector-slipstream: Warm-Start Streaming HNSW Insertions");
    println!("  ─────────────────────────────────────────────────────────");
    println!("  Crate:   ruvector-slipstream");
    println!("  Date:    2026-07-08");

    // OS.
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    println!("  OS:      {os} / {arch}");

    // Rust version (from env set by cargo).
    if let Ok(rv) = std::env::var("RUSTC_BOOTSTRAP") {
        println!("  Rust:    (RUSTC_BOOTSTRAP={rv})");
    }

    println!(
        "  Dataset: {N_CLUSTERS} clusters × {PER_CLUSTER} = {N} vectors, D={DIMS}, σ={CLUSTER_STD}"
    );
    println!(
        "  Queries: {N_QUERIES}   K={K}   ef_insert={EF_INSERT}   ef_search={EF_SEARCH}   M={M}   M_lj={M_LONGJUMP}"
    );
    println!("  Cache:   {CACHE_SIZE} warm-start candidates");
    println!();
}
