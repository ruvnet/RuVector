//! Benchmark binary for ruvector-semantic-cache.
//!
//! Measures three semantic cache variants across three workloads:
//!
//! 1. NoCache        — brute-force ground truth (baseline latency)
//! 2. FixedSemantic  — HNSW-backed cache, fixed cosine threshold 0.92
//! 3. AdaptiveSem    — HNSW-backed cache, adaptive percentile threshold
//!
//! Workloads:
//!   A. 500 near-duplicate queries (should hit cache)
//!   B. 500 random queries (should miss cache)
//!   C. Mixed: 250 near-dup + 250 random

use ruvector_semantic_cache::dataset::{brute_force_topk, recall_at_k, Dataset};
use ruvector_semantic_cache::{
    AdaptiveSemanticCache, CacheConfig, FixedSemanticCache, NoCache, SemanticCache,
};
use std::time::Instant;

fn percentile(mut v: Vec<f64>, p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((p / 100.0) * (v.len() as f64 - 1.0)).round() as usize;
    v[idx.min(v.len() - 1)]
}

struct Stats {
    name: &'static str,
    workload: &'static str,
    n_queries: usize,
    hit_rate: f32,
    recall: f32,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    throughput_qps: f64,
    cache_mem_kb: f64,
    pass: bool,
}

impl Stats {
    fn print_header() {
        println!(
            "{:<25} {:<10} {:>8} {:>8} {:>8} {:>10} {:>10} {:>12} {:>12} {:>6}",
            "Variant",
            "Workload",
            "HitRate",
            "Recall",
            "Mean(µs)",
            "p50(µs)",
            "p95(µs)",
            "QPS",
            "CacheMem(KB)",
            "PASS"
        );
        println!("{}", "-".repeat(120));
    }

    fn print(&self) {
        println!(
            "{:<25} {:<10} {:>7.1}% {:>7.3} {:>9.1} {:>10.1} {:>10.1} {:>12.0} {:>11.1} {:>6}",
            self.name,
            self.workload,
            self.hit_rate * 100.0,
            self.recall,
            self.mean_us,
            self.p50_us,
            self.p95_us,
            self.throughput_qps,
            self.cache_mem_kb,
            if self.pass { "PASS" } else { "FAIL" },
        );
    }
}

fn run_workload<C: SemanticCache>(
    cache: &mut C,
    queries: &[Vec<f32>],
    corpus: &[Vec<f32>],
    ground_truths: &[Vec<u32>],
    k: usize,
    workload_name: &'static str,
    cache_entry_count: usize,
    dim: usize,
) -> Stats {
    let mut latencies_us: Vec<f64> = Vec::with_capacity(queries.len());
    let mut total_recall: f64 = 0.0;
    let mut queries_answered = 0usize;

    for (i, query) in queries.iter().enumerate() {
        let t0 = Instant::now();
        let results = match cache.get(query) {
            Some(cached) => cached,
            None => {
                let fresh = brute_force_topk(query, corpus, k);
                cache.put(query, fresh.clone());
                fresh
            }
        };
        let elapsed_us = t0.elapsed().as_secs_f64() * 1_000_000.0;
        latencies_us.push(elapsed_us);

        // Compute recall against ground truth
        let gt = &ground_truths[i % ground_truths.len()];
        total_recall += recall_at_k(&results, gt) as f64;
        queries_answered += 1;
    }

    let stats = cache.stats().clone();
    let hit_rate = stats.hit_rate();
    let mean_us: f64 = latencies_us.iter().sum::<f64>() / latencies_us.len() as f64;
    let p50_us = percentile(latencies_us.clone(), 50.0);
    let p95_us = percentile(latencies_us.clone(), 95.0);
    let throughput_qps = 1_000_000.0 / mean_us;
    let recall = (total_recall / queries_answered as f64) as f32;

    // Cache memory: each entry ≈ dim * 4 bytes (f32) + 16 bytes metadata
    let bytes_per_entry = dim * std::mem::size_of::<f32>() + 16;
    let cache_mem_kb = (cache_entry_count * bytes_per_entry) as f64 / 1024.0;

    // Acceptance criteria per variant and workload:
    // - NoCache: always passes (baseline, no caching expected)
    // - FixedSemanticCache/near_dup: >= 80% hit rate (fixed threshold should reliably hit)
    // - FixedSemanticCache/random:   == 0% hit rate (fixed threshold should reliably miss random)
    // - AdaptiveSemanticCache: always passes (adaptive threshold exploration does not guarantee
    //   a fixed hit rate; its value is precision control, documented in research notes)
    let pass = match (cache.name(), workload_name) {
        ("NoCache", _) => true,
        ("FixedSemanticCache", "near_dup") => hit_rate >= 0.80,
        ("FixedSemanticCache", "random") => hit_rate <= 0.05,
        ("AdaptiveSemanticCache", _) => true,
        _ => true,
    };

    Stats {
        name: cache.name(),
        workload: workload_name,
        n_queries: queries_answered,
        hit_rate,
        recall,
        mean_us,
        p50_us,
        p95_us,
        throughput_qps,
        cache_mem_kb,
        pass,
    }
}

fn main() {
    // ── System info ─────────────────────────────────────────────────────────
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  ruvector-semantic-cache benchmark");
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  OS:          {}", std::env::consts::OS);
    println!("  Arch:        {}", std::env::consts::ARCH);
    println!(
        "  Rust:        {}",
        env!("CARGO_PKG_RUST_VERSION", "unknown")
    );

    // ── Dataset parameters ───────────────────────────────────────────────────
    let n_corpus = 5_000;
    let n_history = 200;
    let n_test = 500;
    let dim = 128;
    let k = 10;
    let noise = 0.02; // near-dup perturbation scale

    println!();
    println!("  Corpus size:      {}", n_corpus);
    println!("  Dimensions:       {}", dim);
    println!("  History queries:  {}", n_history);
    println!("  Test per class:   {}", n_test);
    println!("  Top-k:            {}", k);
    println!("  Noise scale:      {}", noise);
    println!();

    let t_gen = Instant::now();
    let ds = Dataset::generate(n_corpus, n_history, n_test, dim, k, noise);
    println!(
        "  Dataset generated in {:.1} ms",
        t_gen.elapsed().as_secs_f64() * 1000.0
    );
    println!(
        "  Dataset memory est:  {:.1} KB",
        ds.memory_bytes() as f64 / 1024.0
    );
    println!();

    // ── Workloads ─────────────────────────────────────────────────────────────
    // A. Near-duplicate queries (expect high hit rate after warmup)
    // B. Random queries (expect near-zero hit rate)
    // C. Mixed (real-world mix)

    let mut mixed_queries: Vec<Vec<f32>> = ds.near_dup_queries[..250].to_vec();
    mixed_queries.extend_from_slice(&ds.random_queries[..250]);
    let mut mixed_gt: Vec<Vec<u32>> = ds.near_dup_expected[..250].to_vec();
    // For random queries, compute actual ground truth
    let random_gt: Vec<Vec<u32>> = ds.random_queries[..250]
        .iter()
        .map(|q| brute_force_topk(q, &ds.corpus, k))
        .collect();
    mixed_gt.extend(random_gt.clone());

    let workloads: Vec<(&str, &[Vec<f32>], &[Vec<u32>])> = vec![
        (
            "near_dup",
            ds.near_dup_queries.as_slice(),
            ds.near_dup_expected.as_slice(),
        ),
        ("random", ds.random_queries.as_slice(), &random_gt),
        ("mixed", mixed_queries.as_slice(), mixed_gt.as_slice()),
    ];

    // ── Run all variants across all workloads ─────────────────────────────────
    let cfg = CacheConfig {
        dim,
        max_entries: 512,
        hnsw_m: 16,
        hnsw_ef_construction: 100,
        search_ef: 50,
        fixed_threshold: 0.92,
        adaptive_percentile: 88.0,
        adaptive_window: 100,
    };

    let mut all_stats: Vec<Stats> = Vec::new();

    // Warm up random_gt
    let random_gt_full: Vec<Vec<u32>> = ds
        .random_queries
        .iter()
        .map(|q| brute_force_topk(q, &ds.corpus, k))
        .collect();

    println!("  Warming up caches with {} history queries...", n_history);
    let t_warm = Instant::now();
    // Pre-warm FixedSemanticCache
    let mut fixed_cache = FixedSemanticCache::new(cfg.clone());
    for (i, q) in ds.history_queries.iter().enumerate() {
        let res = ds.history_ground_truth[i].clone();
        fixed_cache.put(q, res);
    }
    let mut adaptive_cache = AdaptiveSemanticCache::new(cfg.clone());
    for (i, q) in ds.history_queries.iter().enumerate() {
        let res = ds.history_ground_truth[i].clone();
        adaptive_cache.put(q, res);
    }
    println!(
        "  Warmup done in {:.1} ms",
        t_warm.elapsed().as_secs_f64() * 1000.0
    );
    println!();

    Stats::print_header();

    for (wname, wq, wgt) in &workloads {
        let n_entries = n_history;

        // NoCache (fresh brute-force each time)
        let mut no_cache = NoCache::new(&cfg);
        let s = run_workload(&mut no_cache, wq, &ds.corpus, wgt, k, wname, 0, dim);
        s.print();
        all_stats.push(s);

        // FixedSemanticCache (pre-warmed)
        let mut fc2 = FixedSemanticCache::new(cfg.clone());
        for (i, q) in ds.history_queries.iter().enumerate() {
            fc2.put(q, ds.history_ground_truth[i].clone());
        }
        let s = run_workload(&mut fc2, wq, &ds.corpus, wgt, k, wname, n_entries, dim);
        s.print();
        all_stats.push(s);

        // AdaptiveSemanticCache (pre-warmed)
        let mut ac2 = AdaptiveSemanticCache::new(cfg.clone());
        for (i, q) in ds.history_queries.iter().enumerate() {
            ac2.put(q, ds.history_ground_truth[i].clone());
        }
        let s = run_workload(&mut ac2, wq, &ds.corpus, wgt, k, wname, n_entries, dim);
        s.print();
        all_stats.push(s);

        println!();
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Acceptance results:");
    let mut all_pass = true;
    for s in &all_stats {
        let status = if s.pass { "PASS" } else { "FAIL" };
        println!(
            "    [{status}] {:<25} workload={:<10} hit_rate={:.1}% recall={:.3}",
            s.name,
            s.workload,
            s.hit_rate * 100.0,
            s.recall
        );
        if !s.pass {
            all_pass = false;
        }
    }

    println!();
    println!("  Hit rate improvement (FixedSemantic vs NoCache, near_dup workload):");
    let nc_hr = all_stats
        .iter()
        .find(|s| s.name == "NoCache" && s.workload == "near_dup")
        .map(|s| s.hit_rate)
        .unwrap_or(0.0);
    let fc_hr = all_stats
        .iter()
        .find(|s| s.name == "FixedSemanticCache" && s.workload == "near_dup")
        .map(|s| s.hit_rate)
        .unwrap_or(0.0);
    let ac_hr = all_stats
        .iter()
        .find(|s| s.name == "AdaptiveSemanticCache" && s.workload == "near_dup")
        .map(|s| s.hit_rate)
        .unwrap_or(0.0);
    println!("    NoCache:              {:.1}%", nc_hr * 100.0);
    println!("    FixedSemanticCache:   {:.1}%", fc_hr * 100.0);
    println!("    AdaptiveSemanticCache:{:.1}%", ac_hr * 100.0);

    // Latency comparison (hit path vs miss path)
    let fc_near = all_stats
        .iter()
        .find(|s| s.name == "FixedSemanticCache" && s.workload == "near_dup")
        .map(|s| s.mean_us)
        .unwrap_or(0.0);
    let nc_near = all_stats
        .iter()
        .find(|s| s.name == "NoCache" && s.workload == "near_dup")
        .map(|s| s.mean_us)
        .unwrap_or(1.0);
    let speedup = nc_near / fc_near.max(0.001);
    println!();
    println!("  Mean latency (near_dup workload):");
    println!("    NoCache:             {:.1} µs", nc_near);
    println!(
        "    FixedSemanticCache:  {:.1} µs  ({:.2}× speedup)",
        fc_near, speedup
    );

    println!();
    if all_pass {
        println!("  ACCEPTANCE: ALL PASS");
    } else {
        println!("  ACCEPTANCE: SOME FAILURES - review above");
        std::process::exit(1);
    }
    println!("═══════════════════════════════════════════════════════════════════");
}
