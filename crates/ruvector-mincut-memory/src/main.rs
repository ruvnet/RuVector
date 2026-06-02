//! MinCut-guided agent memory compaction — benchmark binary.
//!
//! Generates a synthetic multi-cluster memory store, compacts it to half
//! capacity using each of the three strategies, and reports recall@10
//! before/after plus compaction latency.
//!
//! Usage:
//!   cargo run --release -p ruvector-mincut-memory
//!   cargo run --release -p ruvector-mincut-memory -- --n 2000 --dims 64 --clusters 8

use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use ruvector_mincut_memory::{AgeEvict, CoherenceEvict, Compactor, MemoryStore, MinCutEvict};

// ─── Dataset generation ───────────────────────────────────────────────────────

struct Config {
    n: usize,
    dims: usize,
    clusters: usize,
    queries: usize,
    k: usize,
    target_ratio: f32,
    sim_threshold: f32,
    seed: u64,
}

impl Config {
    fn from_env_args() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let mut cfg = Config {
            n: 500,
            dims: 32,
            clusters: 6,
            queries: 50,
            k: 10,
            target_ratio: 0.5,
            sim_threshold: 0.4,
            seed: 42,
        };
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--n" => {
                    cfg.n = args[i + 1].parse().unwrap_or(cfg.n);
                    i += 2;
                }
                "--dims" => {
                    cfg.dims = args[i + 1].parse().unwrap_or(cfg.dims);
                    i += 2;
                }
                "--clusters" => {
                    cfg.clusters = args[i + 1].parse().unwrap_or(cfg.clusters);
                    i += 2;
                }
                "--queries" => {
                    cfg.queries = args[i + 1].parse().unwrap_or(cfg.queries);
                    i += 2;
                }
                _ => {
                    i += 1;
                }
            }
        }
        cfg
    }
}

fn generate_clustered_dataset(n: usize, dims: usize, clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f32, 0.25).unwrap();

    // One centroid per cluster: random unit vector in `dims` space.
    let centroids: Vec<Vec<f32>> = (0..clusters)
        .map(|_| {
            let raw: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
            let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            raw.iter().map(|x| x / norm).collect()
        })
        .collect();

    (0..n)
        .map(|_| {
            let c = rng.gen_range(0..clusters);
            let mut v: Vec<f32> = centroids[c]
                .iter()
                .map(|x| x + normal.sample(&mut rng))
                .collect();
            // Normalize to unit sphere so cosine similarity is meaningful.
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            v.iter_mut().for_each(|x| *x /= norm);
            v
        })
        .collect()
}

// ─── Recall measurement ───────────────────────────────────────────────────────

/// Compute ground-truth top-k ids from a *clean* reference store.
fn ground_truth_ids(reference: &mut MemoryStore, query: &[f32], k: usize) -> Vec<u64> {
    reference
        .search_k(query, k)
        .iter()
        .map(|(i, _)| reference.entries[*i].id)
        .collect()
}

fn mean_recall(
    reference: &mut MemoryStore,
    compacted: &mut MemoryStore,
    queries: &[Vec<f32>],
    k: usize,
) -> f32 {
    let mut total = 0.0f32;
    for q in queries {
        let gt = ground_truth_ids(reference, q, k);
        // Only ask for ids that still exist in the compacted store.
        let id_set: std::collections::HashSet<u64> =
            compacted.entries.iter().map(|e| e.id).collect();
        let surviving_gt: Vec<u64> = gt
            .iter()
            .filter(|id| id_set.contains(id))
            .cloned()
            .collect();
        if surviving_gt.is_empty() {
            total += 0.0;
            continue;
        }
        total += compacted.recall_at_k(q, &surviving_gt, k);
    }
    total / queries.len() as f32
}

// ─── Latency percentiles ──────────────────────────────────────────────────────

fn percentile(mut samples: Vec<u128>, p: f64) -> u128 {
    samples.sort_unstable();
    let idx = ((p / 100.0) * (samples.len() as f64 - 1.0)).round() as usize;
    samples[idx]
}

// ─── One strategy run ─────────────────────────────────────────────────────────

struct RunResult {
    strategy: &'static str,
    n: usize,
    dims: usize,
    queries: usize,
    target_size: usize,
    entries_before: usize,
    entries_after: usize,
    recall_before: f32,
    recall_after: f32,
    mean_latency_us: f64,
    p50_us: u128,
    p95_us: u128,
    throughput_ops_s: f64,
    memory_before_kb: f64,
    memory_after_kb: f64,
    edges_before: usize,
    edges_after: usize,
}

fn run_strategy(
    strategy: &dyn Compactor,
    name: &'static str,
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    cfg: &Config,
) -> RunResult {
    let target_size = ((vectors.len() as f32) * (1.0 - cfg.target_ratio)) as usize;

    // Build reference store for ground-truth recall computation.
    let mut reference = MemoryStore::new(cfg.dims, cfg.sim_threshold);
    for (ts, v) in vectors.iter().enumerate() {
        reference.insert(v.clone(), ts as u64);
    }

    // Recall BEFORE compaction (full store).
    let recall_before = {
        let mut tmp = MemoryStore::new(cfg.dims, cfg.sim_threshold);
        for (ts, v) in vectors.iter().enumerate() {
            tmp.insert(v.clone(), ts as u64);
        }
        let mut ref2 = MemoryStore::new(cfg.dims, cfg.sim_threshold);
        for (ts, v) in vectors.iter().enumerate() {
            ref2.insert(v.clone(), ts as u64);
        }
        mean_recall(&mut ref2, &mut tmp, queries, cfg.k)
    };

    let memory_before_kb = {
        let tmp = MemoryStore::new(cfg.dims, cfg.sim_threshold);
        // Estimate: entries * (dims * 4 bytes + 24 bytes overhead)
        (vectors.len() * (cfg.dims * 4 + 24)) as f64 / 1024.0
    };

    // Warm up and measure compaction latency over multiple runs.
    const REPS: usize = 5;
    let mut latencies: Vec<u128> = Vec::with_capacity(REPS);
    let mut last_result = None;
    let mut last_store = None;

    for _ in 0..REPS {
        let mut store = MemoryStore::new(cfg.dims, cfg.sim_threshold);
        for (ts, v) in vectors.iter().enumerate() {
            store.insert(v.clone(), ts as u64);
        }
        let t0 = Instant::now();
        let res = strategy.compact(&mut store, target_size);
        latencies.push(t0.elapsed().as_micros());
        last_result = Some(res);
        last_store = Some(store);
    }

    let last_result = last_result.unwrap();
    let mut compacted = last_store.unwrap();

    // Recall AFTER compaction.
    let recall_after = mean_recall(&mut reference, &mut compacted, queries, cfg.k);

    let memory_after_kb = (compacted.len() * (cfg.dims * 4 + 24)) as f64 / 1024.0;

    let mean_latency_us = latencies.iter().sum::<u128>() as f64 / REPS as f64;
    let p50 = percentile(latencies.clone(), 50.0);
    let p95 = percentile(latencies.clone(), 95.0);
    let throughput = 1_000_000.0 / mean_latency_us;

    RunResult {
        strategy: name,
        n: cfg.n,
        dims: cfg.dims,
        queries: queries.len(),
        target_size,
        entries_before: last_result.entries_before,
        entries_after: last_result.entries_after,
        recall_before,
        recall_after,
        mean_latency_us,
        p50_us: p50,
        p95_us: p95,
        throughput_ops_s: throughput,
        memory_before_kb,
        memory_after_kb,
        edges_before: last_result.edges_before,
        edges_after: last_result.edges_after,
    }
}

// ─── Acceptance test ──────────────────────────────────────────────────────────

/// PASS if recall after compaction is ≥ acceptance_floor * recall before.
const RECALL_RETENTION_FLOOR: f32 = 0.60;

fn acceptance(r: &RunResult) -> (&'static str, bool) {
    let ratio = if r.recall_before < 1e-5 {
        1.0
    } else {
        r.recall_after / r.recall_before
    };
    let pass = ratio >= RECALL_RETENTION_FLOOR;
    let label = if pass { "PASS" } else { "FAIL" };
    (label, pass)
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let cfg = Config::from_env_args();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  ruvector-mincut-memory  –  Agent Memory Compaction Benchmark");
    println!("═══════════════════════════════════════════════════════════════");
    println!("OS      : {}", std::env::consts::OS);
    println!("Arch    : {}", std::env::consts::ARCH);
    println!(
        "Dataset : N={} D={} clusters={}",
        cfg.n, cfg.dims, cfg.clusters
    );
    println!("Queries : {}", cfg.queries);
    println!(
        "Target  : {:.0}% reduction (keep {:.0}%)",
        cfg.target_ratio * 100.0,
        (1.0 - cfg.target_ratio) * 100.0
    );
    println!("K       : {}", cfg.k);
    println!("SimThresh: {:.2}", cfg.sim_threshold);
    println!();

    let vectors = generate_clustered_dataset(cfg.n, cfg.dims, cfg.clusters, cfg.seed);
    let queries = generate_clustered_dataset(cfg.queries, cfg.dims, cfg.clusters, cfg.seed + 1);

    let strategies: Vec<(&'static str, Box<dyn Compactor>)> = vec![
        ("AgeEvict", Box::new(AgeEvict)),
        ("CoherenceEvict", Box::new(CoherenceEvict)),
        ("MinCutEvict", Box::new(MinCutEvict)),
    ];

    let mut results: Vec<RunResult> = Vec::new();
    for (name, strat) in &strategies {
        print!("Running {} ... ", name);
        let r = run_strategy(strat.as_ref(), name, &vectors, &queries, &cfg);
        println!("done ({:.0} µs mean)", r.mean_latency_us);
        results.push(r);
    }

    // ─── Print results table ─────────────────────────────────────────────────
    println!();
    println!("┌──────────────────┬───────┬───────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬────────┐");
    println!("│ Strategy         │ N_in  │ N_out │ Recall_b │ Recall_a │ MeanLatµs│   p50µs  │   p95µs  │ Thr(ops) │ Accept │");
    println!("├──────────────────┼───────┼───────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼────────┤");

    let mut all_pass = true;
    for r in &results {
        let (lbl, pass) = acceptance(r);
        if !pass {
            all_pass = false;
        }
        println!(
            "│ {:16} │ {:5} │ {:5} │  {:6.3}  │  {:6.3}  │ {:8.1}  │ {:8} │ {:8} │ {:8.1} │ {:6} │",
            r.strategy,
            r.entries_before,
            r.entries_after,
            r.recall_before,
            r.recall_after,
            r.mean_latency_us,
            r.p50_us,
            r.p95_us,
            r.throughput_ops_s,
            lbl,
        );
    }
    println!("└──────────────────┴───────┴───────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴────────┘");
    println!();

    // Memory and edges detail
    println!("── Memory & Graph Detail ───────────────────────────────────────");
    println!(
        "{:18} {:>12} {:>12} {:>12} {:>12}",
        "Strategy", "Mem_before", "Mem_after", "Edges_bef", "Edges_aft"
    );
    for r in &results {
        println!(
            "{:18} {:>10.1}KB {:>10.1}KB {:>12} {:>12}",
            r.strategy, r.memory_before_kb, r.memory_after_kb, r.edges_before, r.edges_after
        );
    }

    println!();
    println!(
        "Acceptance floor: recall_after / recall_before >= {:.2}",
        RECALL_RETENTION_FLOOR
    );
    println!(
        "Overall: {}",
        if all_pass {
            "ALL PASS ✓"
        } else {
            "SOME FAIL ✗"
        }
    );

    if !all_pass {
        std::process::exit(1);
    }
}
