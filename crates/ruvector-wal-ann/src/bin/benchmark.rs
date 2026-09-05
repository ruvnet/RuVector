//! Benchmark: Streaming WAL-ANN with three merge gate strategies.
//!
//! Measures insert throughput, merge count, query latency, and Recall@10
//! for EagerMerge, LazyMerge, and CoherenceGatedMerge on a synthetic dataset.
//!
//! Usage:
//!   cargo run --release -p ruvector-wal-ann --bin benchmark
//!   cargo run --release -p ruvector-wal-ann --bin benchmark -- --n 5000 --dims 64 --queries 200

use ruvector_wal_ann::{
    brute_force_search, recall_at_k, CoherenceGate, EagerGate, LazyGate, MergeGate, WalAnnIndex,
};

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::time::{Duration, Instant};

// ── CLI args ────────────────────────────────────────────────────────────────

struct Args {
    n: usize,
    dims: usize,
    queries: usize,
    k: usize,
}

impl Args {
    fn from_env() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let mut n = 3_000usize;
        let mut dims = 64usize;
        let mut queries = 100usize;
        let mut k = 10usize;
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--n" => {
                    i += 1;
                    n = args[i].parse().unwrap();
                }
                "--dims" => {
                    i += 1;
                    dims = args[i].parse().unwrap();
                }
                "--queries" => {
                    i += 1;
                    queries = args[i].parse().unwrap();
                }
                "--k" => {
                    i += 1;
                    k = args[i].parse().unwrap();
                }
                _ => {}
            }
            i += 1;
        }
        Args {
            n,
            dims,
            queries,
            k,
        }
    }
}

// ── Dataset ─────────────────────────────────────────────────────────────────

fn make_dataset(n: usize, dims: usize, seed: u64) -> Vec<(u64, Vec<f32>)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let dist = Normal::new(0.0f32, 1.0).unwrap();
    (0..n as u64)
        .map(|id| {
            let v: Vec<f32> = (0..dims).map(|_| dist.sample(&mut rng)).collect();
            (id, v)
        })
        .collect()
}

fn make_queries(n: usize, dims: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed + 1_000_007);
    let dist = Normal::new(0.0f32, 1.0).unwrap();
    (0..n)
        .map(|_| (0..dims).map(|_| dist.sample(&mut rng)).collect())
        .collect()
}

// ── Latency stats ────────────────────────────────────────────────────────────

struct LatencyStats {
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    throughput_qps: f64,
}

fn latency_stats(samples: &mut Vec<Duration>) -> LatencyStats {
    samples.sort_unstable();
    let n = samples.len();
    let total: Duration = samples.iter().sum();
    let mean_us = total.as_secs_f64() * 1e6 / n as f64;
    let p50_us = samples[n / 2].as_secs_f64() * 1e6;
    let p95_us = samples[(n * 95) / 100].as_secs_f64() * 1e6;
    let throughput_qps = n as f64 / total.as_secs_f64();
    LatencyStats {
        mean_us,
        p50_us,
        p95_us,
        throughput_qps,
    }
}

// ── Per-variant run ──────────────────────────────────────────────────────────

struct RunResult {
    name: &'static str,
    insert_ms: f64,
    insert_throughput: f64,
    merges: usize,
    recall: f32,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    qps: f64,
    memory_kb: usize,
    graph_size: usize,
    wal_size: usize,
    acceptance: bool,
}

fn run_variant<G: MergeGate>(
    name: &'static str,
    dataset: &[(u64, Vec<f32>)],
    queries: &[Vec<f32>],
    ground_truth: &[Vec<u64>],
    gate: G,
    dims: usize,
    k: usize,
) -> RunResult {
    let wal_capacity = dataset.len() + 1; // Overflow guard larger than dataset.
    let mut idx = WalAnnIndex::new(dims, wal_capacity, gate);

    // ── Insert phase ────────────────────────────────────────────────────────
    let t_insert = Instant::now();
    for (_, v) in dataset {
        idx.insert(v.clone());
    }
    // Final flush so all vectors are in the graph for recall measurement.
    idx.flush_wal();
    let insert_ms = t_insert.elapsed().as_secs_f64() * 1000.0;
    let insert_throughput = dataset.len() as f64 / (insert_ms / 1000.0);

    let merges = idx.merge_count;
    let memory_kb = idx.memory_bytes() / 1024;
    let graph_size = idx.graph_size();
    let wal_size = idx.wal_size();

    // ── Query phase ─────────────────────────────────────────────────────────
    let mut latencies: Vec<Duration> = Vec::with_capacity(queries.len());
    let mut total_recall = 0.0f32;

    for (qi, q) in queries.iter().enumerate() {
        let t = Instant::now();
        let results = idx.search(q, k);
        latencies.push(t.elapsed());

        let result_ids: Vec<u64> = results.iter().map(|r| r.id).collect();
        total_recall += recall_at_k(&ground_truth[qi], &result_ids, k);
    }

    let recall = total_recall / queries.len() as f32;
    let stats = latency_stats(&mut latencies);

    // Acceptance: recall@10 ≥ 0.70 and mean search latency < 5ms
    let acceptance = recall >= 0.70 && stats.mean_us < 5_000.0;

    RunResult {
        name,
        insert_ms,
        insert_throughput,
        merges,
        recall,
        mean_us: stats.mean_us,
        p50_us: stats.p50_us,
        p95_us: stats.p95_us,
        qps: stats.throughput_qps,
        memory_kb,
        graph_size,
        wal_size,
        acceptance,
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::from_env();

    // ── Header ───────────────────────────────────────────────────────────────
    println!("════════════════════════════════════════════════════════════════");
    println!("  ruvector-wal-ann  ·  Streaming WAL-ANN Benchmark");
    println!("════════════════════════════════════════════════════════════════");
    println!("  OS         : {}", std::env::consts::OS);
    println!("  Arch       : {}", std::env::consts::ARCH);
    println!("  Dataset    : {} vectors × {} dims", args.n, args.dims);
    println!("  Queries    : {} × k={}", args.queries, args.k);
    println!("  Build      : release");
    println!("════════════════════════════════════════════════════════════════");
    println!();

    // ── Data generation ───────────────────────────────────────────────────
    print!("  Generating dataset ({} × {}) ... ", args.n, args.dims);
    let t_gen = Instant::now();
    let dataset = make_dataset(args.n, args.dims, 0xFEED_BEEF);
    let queries = make_queries(args.queries, args.dims, 0xFEED_BEEF);
    println!("done in {:.1}ms", t_gen.elapsed().as_secs_f64() * 1000.0);

    // ── Ground truth (brute force) ────────────────────────────────────────
    print!("  Computing ground truth (brute-force) ... ");
    let t_gt = Instant::now();
    let ground_truth: Vec<Vec<u64>> = queries
        .iter()
        .map(|q| brute_force_search(&dataset, q, args.k))
        .collect();
    println!("done in {:.1}ms", t_gt.elapsed().as_secs_f64() * 1000.0);
    println!();

    // ── Variants ─────────────────────────────────────────────────────────
    let results = vec![
        run_variant(
            "EagerMerge",
            &dataset,
            &queries,
            &ground_truth,
            EagerGate { threshold: 32 },
            args.dims,
            args.k,
        ),
        run_variant(
            "LazyMerge",
            &dataset,
            &queries,
            &ground_truth,
            LazyGate { threshold: 512 },
            args.dims,
            args.k,
        ),
        run_variant(
            "CoherenceGatedMerge",
            &dataset,
            &queries,
            &ground_truth,
            // min_coherence = 0.08: in 64-dim Gaussian space, average isolation
            // distance is ~7.0 → coherence ≈ 1/(1+7) = 0.125. Threshold 0.08
            // fires when WAL vectors are genuinely MORE isolated than typical
            // (isolation > 11.5), acting as a quality signal. Hard cap of 256.
            CoherenceGate {
                min_coherence: 0.08,
                max_wal_size: 256,
            },
            args.dims,
            args.k,
        ),
    ];

    // ── Results table ─────────────────────────────────────────────────────
    println!("  INSERT THROUGHPUT");
    println!(
        "  {:<24} {:>12} {:>12} {:>10} {:>10}",
        "Variant", "Total(ms)", "Vecs/sec", "Merges", "GraphSz"
    );
    println!("  {}", "-".repeat(74));
    for r in &results {
        println!(
            "  {:<24} {:>12.1} {:>12.0} {:>10} {:>10}",
            r.name, r.insert_ms, r.insert_throughput, r.merges, r.graph_size
        );
    }

    println!();
    println!("  QUERY PERFORMANCE  (k={})", args.k);
    println!(
        "  {:<24} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Variant", "Recall@k", "Mean(µs)", "p50(µs)", "p95(µs)", "QPS", "Mem(KB)"
    );
    println!("  {}", "-".repeat(90));
    for r in &results {
        println!(
            "  {:<24} {:>10.3} {:>10.1} {:>10.1} {:>10.1} {:>10.0} {:>10}",
            r.name, r.recall, r.mean_us, r.p50_us, r.p95_us, r.qps, r.memory_kb
        );
    }

    // ── Acceptance ────────────────────────────────────────────────────────
    println!();
    println!(
        "  ACCEPTANCE CRITERIA  (recall@{k} ≥ 0.70  AND  mean latency < 5ms)",
        k = args.k
    );
    println!("  {}", "-".repeat(50));
    let mut all_pass = true;
    for r in &results {
        let tag = if r.acceptance { "PASS" } else { "FAIL" };
        println!(
            "  {:<24}  recall={:.3}  mean={:.1}µs  → {}",
            r.name, r.recall, r.mean_us, tag
        );
        if !r.acceptance {
            all_pass = false;
        }
    }

    // ── Merge behaviour summary ───────────────────────────────────────────
    println!();
    println!("  MERGE BEHAVIOUR SUMMARY");
    for r in &results {
        let merges_per_k = if r.merges == 0 {
            0.0
        } else {
            args.n as f64 / r.merges as f64
        };
        println!(
            "  {:<24}  {} merges  ({:.0} vectors/merge avg)",
            r.name, r.merges, merges_per_k
        );
    }

    // ── Memory math ──────────────────────────────────────────────────────
    println!();
    println!("  MEMORY ESTIMATE (post-flush)");
    let vec_bytes = args.n * args.dims * 4;
    let adj_bytes = args.n * 12 * 4; // avg 12 neighbours × 4 bytes per u32
    let id_bytes = args.n * 8;
    let total_est = vec_bytes + adj_bytes + id_bytes;
    println!(
        "  Vectors: {}K  Adjacency: {}K  IDs: {}K  → Total est: {}K",
        vec_bytes / 1024,
        adj_bytes / 1024,
        id_bytes / 1024,
        total_est / 1024
    );
    for r in &results {
        println!("  {:<24}  measured: {}KB", r.name, r.memory_kb);
    }

    println!();
    println!("════════════════════════════════════════════════════════════════");
    if all_pass {
        println!("  BENCHMARK RESULT: ALL VARIANTS PASS");
    } else {
        println!("  BENCHMARK RESULT: ONE OR MORE VARIANTS FAILED ACCEPTANCE");
    }
    println!("════════════════════════════════════════════════════════════════");

    if !all_pass {
        std::process::exit(1);
    }
}
