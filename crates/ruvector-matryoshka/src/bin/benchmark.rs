//! Matryoshka coarse-to-fine benchmark.
//!
//! Measures recall@10, mean latency, p50 latency, p95 latency, throughput, and
//! estimated memory for three ANN search variants:
//!
//!   1. FullDimHNSW  – standard HNSW at full dimension (baseline)
//!   2. TwoStage     – coarse HNSW + full-dim rerank
//!   3. ThreeStage   – coarse HNSW + mid-dim filter + full-dim rerank
//!
//! All numbers come from real cargo runs. No numbers are invented.
//!
//! Usage:
//!   cargo run --release -p ruvector-matryoshka --bin benchmark
//!   cargo run --release -p ruvector-matryoshka --bin benchmark -- --n 5000 --dim 256

use ruvector_matryoshka::{
    brute_force_knn, dataset::generate_matryoshka_dataset, recall_at_k, FullDimIndex,
    MatryoshkaConfig, Searcher, ThreeStageIndex, TwoStageIndex,
};
use std::time::Instant;

// ─── CLI arg parsing (no external crate) ──────────────────────────────────────

struct Args {
    n: usize,
    n_queries: usize,
    full_dim: usize,
    k: usize,
    ef: usize,
    seed: u64,
}

impl Args {
    fn parse() -> Self {
        let mut a = Args {
            n: 3000,
            n_queries: 200,
            full_dim: 128,
            k: 10,
            ef: 64,
            seed: 42,
        };
        let raw: Vec<String> = std::env::args().collect();
        let mut i = 1;
        while i < raw.len() {
            match raw[i].as_str() {
                "--n" => {
                    a.n = raw[i + 1].parse().unwrap();
                    i += 2;
                }
                "--queries" => {
                    a.n_queries = raw[i + 1].parse().unwrap();
                    i += 2;
                }
                "--dim" => {
                    a.full_dim = raw[i + 1].parse().unwrap();
                    i += 2;
                }
                "--k" => {
                    a.k = raw[i + 1].parse().unwrap();
                    i += 2;
                }
                "--ef" => {
                    a.ef = raw[i + 1].parse().unwrap();
                    i += 2;
                }
                "--seed" => {
                    a.seed = raw[i + 1].parse().unwrap();
                    i += 2;
                }
                _ => {
                    i += 1;
                }
            }
        }
        a
    }
}

// ─── Timing helpers ──────────────────────────────────────────────────────────

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

struct Stats {
    mean_us: f64,
    p50_us: u64,
    p95_us: u64,
    qps: f64,
    recall: f32,
}

fn run_variant<S: Searcher>(
    cfg: &MatryoshkaConfig,
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    ground_truth: &[Vec<usize>],
    k: usize,
    ef: usize,
) -> (Stats, S) {
    let idx = S::build(cfg, vectors);

    let mut latencies_ns: Vec<u64> = Vec::with_capacity(queries.len());
    let mut total_recall = 0.0f32;

    for (qi, q) in queries.iter().enumerate() {
        let t0 = Instant::now();
        let result = idx.search(q, k, ef);
        let elapsed = t0.elapsed().as_nanos() as u64;
        latencies_ns.push(elapsed);
        total_recall += recall_at_k(&result, &ground_truth[qi]);
    }

    latencies_ns.sort_unstable();
    let mean_ns = latencies_ns.iter().sum::<u64>() as f64 / latencies_ns.len() as f64;
    let p50 = percentile(&latencies_ns, 50.0);
    let p95 = percentile(&latencies_ns, 95.0);

    let total_elapsed_s = latencies_ns.iter().sum::<u64>() as f64 / 1e9;
    let qps = queries.len() as f64 / total_elapsed_s;

    let stats = Stats {
        mean_us: mean_ns / 1000.0,
        p50_us: p50 / 1000,
        p95_us: p95 / 1000,
        qps,
        recall: total_recall / queries.len() as f32,
    };
    (stats, idx)
}

// ─── Memory estimation ────────────────────────────────────────────────────────

fn hnsw_memory_kb(n: usize, dim: usize, m: usize) -> usize {
    let vec_bytes = n * dim * 4;
    // Rough estimate: each node has ~2M neighbours on avg across layers
    let graph_bytes = n * m * 2 * 4;
    (vec_bytes + graph_bytes) / 1024
}

fn two_stage_memory_kb(n: usize, full_dim: usize, coarse_dim: usize, m: usize) -> usize {
    let coarse_kb = hnsw_memory_kb(n, coarse_dim, m);
    let full_vec_kb = n * full_dim * 4 / 1024;
    coarse_kb + full_vec_kb
}

fn three_stage_memory_kb(
    n: usize,
    full_dim: usize,
    mid_dim: usize,
    coarse_dim: usize,
    m: usize,
) -> usize {
    let coarse_kb = hnsw_memory_kb(n, coarse_dim, m);
    let mid_vec_kb = n * mid_dim * 4 / 1024;
    let full_vec_kb = n * full_dim * 4 / 1024;
    coarse_kb + mid_vec_kb + full_vec_kb
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    // ── Environment info ──
    println!("═══════════════════════════════════════════════════════════");
    println!("  RuVector Matryoshka Coarse-to-Fine Benchmark");
    println!("═══════════════════════════════════════════════════════════");
    println!("  OS:        {}", std::env::consts::OS);
    println!("  Arch:      {}", std::env::consts::ARCH);
    println!("  Rust:      {}", rustc_version());
    println!("  Dataset:   {} vectors × {} dims", args.n, args.full_dim);
    println!("  Queries:   {}", args.n_queries);
    println!("  k:         {}", args.k);
    println!("  ef search: {}", args.ef);
    println!("  Seed:      {}", args.seed);
    println!();

    // ── Config ──
    let cfg = MatryoshkaConfig {
        full_dim: args.full_dim,
        coarse_dim: (args.full_dim / 4).max(8),
        mid_dim: (args.full_dim / 2).max(16),
        m: 16,
        ef_construction: 100,
        two_stage_candidates: (args.k * 10).max(50),
        three_stage_coarse_candidates: (args.k * 15).max(75),
        three_stage_mid_candidates: (args.k * 5).max(25),
    };

    println!(
        "  Coarse dim: {}   Mid dim: {}   Full dim: {}",
        cfg.coarse_dim, cfg.mid_dim, cfg.full_dim
    );
    println!("  TwoStage candidates:       {}", cfg.two_stage_candidates);
    println!(
        "  ThreeStage candidates: {}/{}",
        cfg.three_stage_coarse_candidates, cfg.three_stage_mid_candidates
    );
    println!();

    // ── Dataset ──
    println!("Generating Matryoshka-structured dataset …");
    let t0 = Instant::now();
    let (vectors, queries) = generate_matryoshka_dataset(
        args.n,
        args.n_queries,
        args.full_dim,
        cfg.coarse_dim,
        args.seed,
    );
    println!("  done in {:.1} ms", t0.elapsed().as_millis());

    // ── Ground truth ──
    println!("Computing brute-force ground truth …");
    let t0 = Instant::now();
    let ground_truth: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| brute_force_knn(&vectors, q, args.k, args.full_dim))
        .collect();
    println!("  done in {:.1} ms", t0.elapsed().as_millis());
    println!();

    // ── Variant 1: FullDimHNSW ──
    println!("Building FullDimHNSW …");
    let t0 = Instant::now();
    let (s1, _idx1) =
        run_variant::<FullDimIndex>(&cfg, &vectors, &queries, &ground_truth, args.k, args.ef);
    println!("  done in {:.1} ms", t0.elapsed().as_millis());
    let m1_kb = hnsw_memory_kb(args.n, args.full_dim, cfg.m);

    // ── Variant 2: TwoStage ──
    println!("Building TwoStageIndex …");
    let t0 = Instant::now();
    let (s2, _idx2) =
        run_variant::<TwoStageIndex>(&cfg, &vectors, &queries, &ground_truth, args.k, args.ef);
    println!("  done in {:.1} ms", t0.elapsed().as_millis());
    let m2_kb = two_stage_memory_kb(args.n, args.full_dim, cfg.coarse_dim, cfg.m);

    // ── Variant 3: ThreeStage ──
    println!("Building ThreeStageIndex …");
    let t0 = Instant::now();
    let (s3, _idx3) =
        run_variant::<ThreeStageIndex>(&cfg, &vectors, &queries, &ground_truth, args.k, args.ef);
    println!("  done in {:.1} ms", t0.elapsed().as_millis());
    let m3_kb = three_stage_memory_kb(args.n, args.full_dim, cfg.mid_dim, cfg.coarse_dim, cfg.m);

    // ── Results table ──
    println!();
    println!("─────────────────────────────────────────────────────────────────────────────────");
    println!(
        "{:<16} {:>12} {:>10} {:>10} {:>10} {:>12} {:>10}",
        "Variant", "Recall@k", "Mean(μs)", "p50(μs)", "p95(μs)", "QPS", "Mem(KB)"
    );
    println!("─────────────────────────────────────────────────────────────────────────────────");
    print_row("FullDimHNSW", &s1, m1_kb);
    print_row("TwoStage", &s2, m2_kb);
    print_row("ThreeStage", &s3, m3_kb);
    println!("─────────────────────────────────────────────────────────────────────────────────");

    // ── Acceptance tests ──
    println!();
    println!("Acceptance tests:");
    let full_pass = s1.recall >= 0.80;
    let two_pass = s2.recall >= 0.75;
    let three_pass = s3.recall >= 0.70;
    let latency_ratio = if s1.mean_us > 0.0 {
        s2.mean_us / s1.mean_us
    } else {
        99.0
    };

    println!(
        "  [{}] FullDimHNSW  recall@{} = {:.3}  (threshold ≥ 0.80)",
        if full_pass { "PASS" } else { "FAIL" },
        args.k,
        s1.recall
    );
    println!(
        "  [{}] TwoStage     recall@{} = {:.3}  (threshold ≥ 0.75)",
        if two_pass { "PASS" } else { "FAIL" },
        args.k,
        s2.recall
    );
    println!(
        "  [{}] ThreeStage   recall@{} = {:.3}  (threshold ≥ 0.70)",
        if three_pass { "PASS" } else { "FAIL" },
        args.k,
        s3.recall
    );
    println!(
        "  [INFO] TwoStage latency ratio vs FullDim = {:.2}x",
        latency_ratio
    );

    let dim_ratio = cfg.coarse_dim as f32 / cfg.full_dim as f32;
    println!(
        "  [INFO] Coarse-dim reduction = {:.0}% of full dim",
        dim_ratio * 100.0
    );

    println!();
    let all_pass = full_pass && two_pass && three_pass;
    if all_pass {
        println!("RESULT: ALL ACCEPTANCE TESTS PASSED");
    } else {
        println!("RESULT: SOME ACCEPTANCE TESTS FAILED — see rows above");
        std::process::exit(1);
    }
}

fn print_row(name: &str, s: &Stats, mem_kb: usize) {
    println!(
        "{:<16} {:>12.3} {:>10.1} {:>10} {:>10} {:>12.0} {:>10}",
        name, s.recall, s.mean_us, s.p50_us, s.p95_us, s.qps, mem_kb
    );
}

fn rustc_version() -> String {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}
