//! Matryoshka + RaBitQ cascade benchmark.
//!
//! Measures recall@10, latency, throughput, and memory for three ANN
//! variants that share the same coarse-then-rerank funnel shape:
//!
//!   1. FullDimHNSW   - single HNSW at full dimension (baseline)
//!   2. TwoStage      - HNSW coarse stage + full-dim rerank (candidate A, from ruvector-matryoshka)
//!   3. RaBitQCascade - 1-bit XNOR-popcount coarse stage + full-dim rerank (candidate B, this crate)
//!
//! Then runs a small bounded parameter sweep over RaBitQCascade's coarse
//! candidate-pool size (generations=1, candidates_per_generation=3,
//! max_promotions=1 - reduced from the harness's default generations=3
//! budget because each point here is a full N=3000 benchmark run, not a
//! cheap synthetic eval) and scores each point with the nightly harness's
//! declared fitness formula.
//!
//! All numbers come from real cargo runs. No numbers are invented.
//!
//! Usage:
//!   cargo run --release -p ruvector-matryoshka-rabitq --bin benchmark
//!   cargo run --release -p ruvector-matryoshka-rabitq --bin benchmark -- --n 5000 --dim 256

use ruvector_matryoshka_rabitq::{
    brute_force_knn, dataset::generate_matryoshka_dataset, recall_at_k, FullDimIndex,
    MatryoshkaConfig, RaBitQCascadeIndex, Searcher, TwoStageIndex,
};
use std::time::Instant;

// ─── CLI arg parsing (no external crate, matches sibling ruvector-matryoshka) ─

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

fn rustc_version() -> String {
    std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown (see `rustc --version`)".into())
}

// ─── Timing helpers ──────────────────────────────────────────────────────────

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[derive(Clone)]
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

fn full_vecs_kb(n: usize, dim: usize) -> f64 {
    (n * dim * 4) as f64 / 1024.0
}

/// HNSW-coarse memory: raw coarse-dim f32 vectors + graph edges (same
/// formula ruvector-matryoshka's own benchmark uses for TwoStage).
fn hnsw_coarse_kb(n: usize, coarse_dim: usize, m: usize) -> f64 {
    let vec_bytes = n * coarse_dim * 4;
    let graph_bytes = n * m * 2 * 4;
    (vec_bytes + graph_bytes) as f64 / 1024.0
}

// ─── Darwin-style bounded sweep over RaBitQCascade's candidate pool size ─────

struct SweepPoint {
    candidates: usize,
    stats: Stats,
    mem_kb: f64,
}

fn fitness(point: &SweepPoint, points: &[SweepPoint], recall_threshold: f32) -> f64 {
    let min_max = |get: fn(&SweepPoint) -> f64, invert: bool| -> f64 {
        let vals: Vec<f64> = points.iter().map(get).collect();
        let (lo, hi) = (
            vals.iter().cloned().fold(f64::INFINITY, f64::min),
            vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        );
        let v = get(point);
        if (hi - lo).abs() < 1e-9 {
            1.0
        } else if invert {
            // lower is better (latency, memory): invert so lower -> higher score
            1.0 - (v - lo) / (hi - lo)
        } else {
            (v - lo) / (hi - lo)
        }
    };

    let normalized_quality = min_max(|p| p.stats.recall as f64, false);
    let normalized_latency = min_max(|p| p.stats.mean_us, true);
    let normalized_throughput = min_max(|p| p.stats.qps, false);
    let normalized_memory = min_max(|p| p.mem_kb, true);
    let correctness = if point.stats.recall >= recall_threshold {
        1.0
    } else {
        0.0
    };
    let reproducibility = 1.0; // deterministic seeded dataset + seeded rotation

    0.30 * normalized_quality
        + 0.25 * normalized_latency
        + 0.15 * normalized_throughput
        + 0.10 * normalized_memory
        + 0.10 * correctness
        + 0.10 * reproducibility
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    println!("===============================================================");
    println!("  RuVector Matryoshka + RaBitQ Cascade Benchmark");
    println!("===============================================================");
    println!("  OS:        {}", std::env::consts::OS);
    println!("  Arch:      {}", std::env::consts::ARCH);
    println!("  Rust:      {}", rustc_version());
    println!("  Dataset:   {} vectors x {} dims", args.n, args.full_dim);
    println!("  Queries:   {}", args.n_queries);
    println!("  k:         {}", args.k);
    println!("  ef search: {}", args.ef);
    println!("  Seed:      {}", args.seed);
    println!();

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
        "  Coarse dim: {}   Full dim: {}   Coarse candidates: {}",
        cfg.coarse_dim, cfg.full_dim, cfg.two_stage_candidates
    );
    println!();

    println!("Generating Matryoshka-structured dataset ...");
    let t0 = Instant::now();
    let (vectors, queries) = generate_matryoshka_dataset(
        args.n,
        args.n_queries,
        args.full_dim,
        cfg.coarse_dim,
        args.seed,
    );
    println!("  done in {:.1} ms", t0.elapsed().as_millis());

    println!("Computing brute-force ground truth ...");
    let t0 = Instant::now();
    let ground_truth: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| brute_force_knn(&vectors, q, args.k, args.full_dim))
        .collect();
    println!("  done in {:.1} ms", t0.elapsed().as_millis());
    println!();

    // ── Variant 1: FullDimHNSW (baseline) ──
    println!("Building FullDimHNSW (baseline) ...");
    let t0 = Instant::now();
    let (s1, _idx1) =
        run_variant::<FullDimIndex>(&cfg, &vectors, &queries, &ground_truth, args.k, args.ef);
    println!("  done in {:.1} ms", t0.elapsed().as_millis());
    let m1_kb = hnsw_coarse_kb(args.n, args.full_dim, cfg.m); // full-dim graph, no separate rerank store

    // ── Variant 2: TwoStage / HNSW-coarse (candidate A) ──
    println!("Building TwoStageIndex (candidate A, HNSW coarse) ...");
    let t0 = Instant::now();
    let (s2, _idx2) =
        run_variant::<TwoStageIndex>(&cfg, &vectors, &queries, &ground_truth, args.k, args.ef);
    println!("  done in {:.1} ms", t0.elapsed().as_millis());
    let two_stage_coarse_kb = hnsw_coarse_kb(args.n, cfg.coarse_dim, cfg.m);
    let m2_kb = two_stage_coarse_kb + full_vecs_kb(args.n, args.full_dim);

    // ── Variant 3: RaBitQCascade (candidate B) ──
    println!("Building RaBitQCascadeIndex (candidate B, 1-bit coarse scan) ...");
    let t0 = Instant::now();
    let (s3, idx3) =
        run_variant::<RaBitQCascadeIndex>(&cfg, &vectors, &queries, &ground_truth, args.k, args.ef);
    println!("  done in {:.1} ms", t0.elapsed().as_millis());
    let cascade_coarse_kb = idx3.coarse_stage_bytes() as f64 / 1024.0;
    let m3_kb = cascade_coarse_kb + full_vecs_kb(args.n, args.full_dim);

    println!();
    println!(
        "-------------------------------------------------------------------------------------"
    );
    println!(
        "{:<16} {:>12} {:>10} {:>10} {:>10} {:>12} {:>12} {:>14}",
        "Variant",
        "Recall@k",
        "Mean(us)",
        "p50(us)",
        "p95(us)",
        "QPS",
        "TotalMem(KB)",
        "CoarseMem(KB)"
    );
    println!(
        "-------------------------------------------------------------------------------------"
    );
    print_row("FullDimHNSW", &s1, m1_kb, m1_kb);
    print_row("TwoStage", &s2, m2_kb, two_stage_coarse_kb);
    print_row("RaBitQCascade", &s3, m3_kb, cascade_coarse_kb);
    println!(
        "-------------------------------------------------------------------------------------"
    );

    // ── Pre-declared acceptance thresholds (fixed before this run; see the
    //    research README's hypothesis section — never adjusted post hoc) ──
    println!();
    println!("Acceptance tests (pre-declared thresholds):");
    let full_sane = s1.recall >= 0.80;
    let two_sane = s2.recall >= 0.70;
    let recall_within_margin = s3.recall >= s2.recall - 0.02;
    let mem_win = cascade_coarse_kb <= two_stage_coarse_kb * 0.5;
    let latency_ok = s3.mean_us <= s2.mean_us * 2.0;

    println!(
        "  [{}] FullDimHNSW sanity   recall@{} = {:.3}  (>= 0.80)",
        tag(full_sane),
        args.k,
        s1.recall
    );
    println!(
        "  [{}] TwoStage sanity      recall@{} = {:.3}  (>= 0.70)",
        tag(two_sane),
        args.k,
        s2.recall
    );
    println!(
        "  [{}] RaBitQCascade recall margin   {:.3} >= {:.3} - 0.02 = {:.3}",
        tag(recall_within_margin),
        s3.recall,
        s2.recall,
        s2.recall - 0.02
    );
    println!(
        "  [{}] RaBitQCascade coarse memory   {:.1} KB <= 50% of {:.1} KB = {:.1} KB",
        tag(mem_win),
        cascade_coarse_kb,
        two_stage_coarse_kb,
        two_stage_coarse_kb * 0.5
    );
    println!(
        "  [{}] RaBitQCascade latency bound   {:.1} us <= 2x of {:.1} us = {:.1} us",
        tag(latency_ok),
        s3.mean_us,
        s2.mean_us,
        s2.mean_us * 2.0
    );

    let accept = full_sane && two_sane && recall_within_margin && mem_win && latency_ok;
    println!();
    if !full_sane || !two_sane {
        println!("ACCEPTANCE: INCONCLUSIVE (baseline/candidate-A sanity check failed - cannot evaluate candidate B fairly)");
    } else if accept {
        println!("ACCEPTANCE: ACCEPT (RaBitQCascade matches candidate-A recall within margin, uses <=50% of its coarse-stage memory, and stays within 2x its latency)");
    } else {
        println!(
            "ACCEPTANCE: REJECT (one or more pre-declared thresholds failed - see rows above)"
        );
    }

    // ── Bounded Darwin-style sweep over RaBitQCascade's coarse candidate pool ──
    println!();
    println!("===============================================================");
    println!("  Bounded evolution: RaBitQCascade coarse-candidate-pool sweep");
    println!("  (generations=1, candidates_per_generation=3, max_promotions=1)");
    println!("===============================================================");

    let parent_candidates = cfg.two_stage_candidates;
    let sweep_candidates = [
        (args.k * 5).max(25),
        parent_candidates,
        (args.k * 20).max(100),
    ];

    let mut sweep_points = Vec::new();
    for &c in &sweep_candidates {
        let mut sweep_cfg = cfg.clone();
        sweep_cfg.two_stage_candidates = c;
        let (stats, idx) = run_variant::<RaBitQCascadeIndex>(
            &sweep_cfg,
            &vectors,
            &queries,
            &ground_truth,
            args.k,
            args.ef,
        );
        let mem_kb = idx.coarse_stage_bytes() as f64 / 1024.0 + full_vecs_kb(args.n, args.full_dim);
        println!(
            "  candidates={:>4}  recall@{}={:.3}  mean={:.1}us  qps={:.0}  mem={:.1}KB",
            c, args.k, stats.recall, stats.mean_us, stats.qps, mem_kb
        );
        sweep_points.push(SweepPoint {
            candidates: c,
            stats,
            mem_kb,
        });
    }

    let fitnesses: Vec<f64> = sweep_points
        .iter()
        .map(|p| fitness(p, &sweep_points, s2.recall - 0.02))
        .collect();
    let (winner_idx, &winner_fitness) = fitnesses
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    let parent_idx = sweep_points
        .iter()
        .position(|p| p.candidates == parent_candidates)
        .unwrap();
    let parent_fitness = fitnesses[parent_idx];

    println!();
    for (i, p) in sweep_points.iter().enumerate() {
        println!(
            "  fitness(candidates={:>4}) = {:.4}{}",
            p.candidates,
            fitnesses[i],
            if i == parent_idx { "  [parent]" } else { "" }
        );
    }

    println!();
    if winner_idx != parent_idx && winner_fitness > parent_fitness + 0.01 {
        println!(
            "DARWIN RESULT: promote candidates={} (fitness {:.4} > parent {:.4} by > 0.01 margin)",
            sweep_points[winner_idx].candidates, winner_fitness, parent_fitness
        );
    } else {
        println!(
            "DARWIN RESULT: keep parent candidates={} (fitness {:.4}; best alternative {:.4} did not clear the promotion margin)",
            parent_candidates, parent_fitness, winner_fitness
        );
    }

    if !accept || !full_sane || !two_sane {
        std::process::exit(1);
    }
}

fn tag(pass: bool) -> &'static str {
    if pass {
        "PASS"
    } else {
        "FAIL"
    }
}

fn print_row(name: &str, s: &Stats, total_mem_kb: f64, coarse_mem_kb: f64) {
    println!(
        "{:<16} {:>12.3} {:>10.1} {:>10} {:>10} {:>12.0} {:>12.1} {:>14.1}",
        name, s.recall, s.mean_us, s.p50_us, s.p95_us, s.qps, total_mem_kb, coarse_mem_kb
    );
}
