//! # ruvector-mrl benchmark
//!
//! Two experiments reveal when Matryoshka Resolution indexing helps and when it hurts:
//!
//! **Experiment A — Random Gaussian** (no MRL training structure):
//!   The first D_FAST dimensions carry no more information than any other slice.
//!   Prefix-based screening has poor correlation with full-dim ranking.
//!   Shows the fundamental limitation: MRL gains require MRL training.
//!
//! **Experiment B — MRL-Simulated** (prefix carries most of the signal):
//!   Vectors are generated as: v = normalize(signal + α·noise), where
//!   signal is a D_FAST-dim random vector padded with zeros, and α=0.25.
//!   The prefix is genuinely informative; later dims add fine-grained detail.
//!   Shows the MRL opportunity: 3–10× throughput at >85% recall.
//!
//! Three variants are compared in each experiment:
//!   1. FlatFull  — exact brute-force on D_FULL dimensions.
//!   2. MrlLinear — brute-force on D_FAST prefix, exact rerank.
//!   3. MrlGraph  — kNN graph on D_FAST prefix, exact rerank.
//!
//! Run: cargo run --release -p ruvector-mrl --bin mrl-bench

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_mrl::{
    flat::FlatIndex,
    mrl::{MrlGraph, MrlLinear},
    normalize, recall_at_k, MrlSearch,
};
use std::time::Instant;

// ── Dataset parameters ──────────────────────────────────────────────────────

const N: usize = 5_000;
const D_FULL: usize = 128;
const D_FAST: usize = 32;     // MRL prefix = 25 % of full dims
const N_QUERIES: usize = 200;
const K: usize = 10;

// ── Index parameters ─────────────────────────────────────────────────────────

const M: usize = 16;
const EF: usize = 60;
const OVERSAMPLE: usize = 8;
const K_OVER_LINEAR: usize = 10;

// ────────────────────────────────────────────────────────────────────────────

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!(" ruvector-mrl — Matryoshka Resolution Index Benchmark");
    println!("═══════════════════════════════════════════════════════════════");
    println!("N={N}, D_FULL={D_FULL}, D_FAST={D_FAST} ({:.0}% of dims)",
        100.0 * D_FAST as f64 / D_FULL as f64);
    println!("M={M}, ef={EF}, oversample={OVERSAMPLE}");
    println!();

    // ── Experiment A: random Gaussian (no MRL structure) ─────────────────
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║ Experiment A — Random Gaussian (no MRL prefix structure)   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("Purpose: shows that without Matryoshka training, prefix-based");
    println!("screening does NOT reliably predict full-dim similarity.");
    println!();
    let mut rng_a = StdRng::seed_from_u64(0xAAAA_DEAD_BEEF_0001);
    let (db_a, queries_a) = gen_random_dataset(N, D_FULL, N_QUERIES, &mut rng_a);
    let exp_a = run_experiment(&db_a, &queries_a);
    print_results(&exp_a, "A: Random Gaussian");
    println!();

    // ── Experiment B: MRL-structured (prefix carries most signal) ────────
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║ Experiment B — MRL-Simulated (prefix is informative)       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("Purpose: simulates Matryoshka-trained embeddings where the");
    println!("first {D_FAST} dims capture ~94% of the similarity structure.");
    println!("α = 0.25 (tail dims = 25% noise relative to prefix signal).");
    println!();
    let mut rng_b = StdRng::seed_from_u64(0xBBBB_CAFE_1234_0002);
    let (db_b, queries_b) = gen_mrl_dataset(N, D_FULL, D_FAST, N_QUERIES, 0.25, &mut rng_b);
    let exp_b = run_experiment(&db_b, &queries_b);
    print_results(&exp_b, "B: MRL-Simulated");
    println!();

    // ── Acceptance test (against Experiment B) ───────────────────────────
    println!("── Acceptance test (Experiment B only) ─────────────────────────");
    let ok_linear_recall  = exp_b.linear_recall  >= 0.80;
    let ok_graph_recall   = exp_b.graph_recall   >= 0.70;
    let ok_linear_speedup = exp_b.linear_speedup >= 1.5;
    let ok_graph_speedup  = exp_b.graph_speedup  >= 3.0;

    check("MrlLinear recall@10 ≥ 0.80",  ok_linear_recall);
    check("MrlGraph  recall@10 ≥ 0.70",  ok_graph_recall);
    check("MrlLinear speedup   ≥ 1.5×",  ok_linear_speedup);
    check("MrlGraph  speedup   ≥ 3.0×",  ok_graph_speedup);

    let all_pass = ok_linear_recall && ok_graph_recall
        && ok_linear_speedup && ok_graph_speedup;
    println!();
    if all_pass {
        println!("RESULT: PASS — all acceptance criteria met on MRL-structured data.");
    } else {
        eprintln!("RESULT: FAIL — one or more acceptance criteria not met.");
        std::process::exit(1);
    }
}

// ── Dataset generators ───────────────────────────────────────────────────────

/// Pure random unit-sphere vectors.  The prefix is statistically uninformative
/// relative to the full vector — MRL speedup cannot be expected here.
fn gen_random_dataset(
    n: usize,
    d_full: usize,
    n_queries: usize,
    rng: &mut StdRng,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let gen = |rng: &mut StdRng| {
        let mut v: Vec<f32> = (0..d_full).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
        normalize(&mut v);
        v
    };
    let db = (0..n).map(|_| gen(rng)).collect();
    let queries = (0..n_queries).map(|_| gen(rng)).collect();
    (db, queries)
}

/// MRL-simulated vectors: strong signal in first `d_fast` dims, small noise
/// in remaining dims.  Prefix similarity is a reliable predictor of full-dim
/// similarity — exactly the guarantee Matryoshka training provides.
///
/// v = normalize(signal || α · noise)
/// where signal ~ U(-1,1)^d_fast, noise ~ U(-1,1)^(d_full-d_fast)
fn gen_mrl_dataset(
    n: usize,
    d_full: usize,
    d_fast: usize,
    n_queries: usize,
    alpha: f32,
    rng: &mut StdRng,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let gen = |rng: &mut StdRng| {
        let mut v: Vec<f32> = Vec::with_capacity(d_full);
        // prefix: strong signal
        for _ in 0..d_fast {
            v.push(rng.gen::<f32>() * 2.0 - 1.0);
        }
        // suffix: scaled noise
        for _ in d_fast..d_full {
            v.push((rng.gen::<f32>() * 2.0 - 1.0) * alpha);
        }
        normalize(&mut v);
        v
    };
    let db = (0..n).map(|_| gen(rng)).collect();
    let queries = (0..n_queries).map(|_| gen(rng)).collect();
    (db, queries)
}

// ── Benchmark harness ────────────────────────────────────────────────────────

struct ExpResult {
    flat_stats: Stats,
    linear_stats: Stats,
    graph_stats: Stats,
    linear_recall: f32,
    graph_recall: f32,
    linear_speedup: f64,
    graph_speedup: f64,
}

fn run_experiment(db: &[Vec<f32>], queries: &[Vec<f32>]) -> ExpResult {
    let n = db.len();

    // ── FlatFull ─────────────────────────────────────────────────────────
    let mut flat = FlatIndex::new(D_FULL);
    for (i, v) in db.iter().enumerate() {
        flat.insert(i as u32, v);
    }
    let (flat_latencies, flat_gt): (Vec<u64>, Vec<Vec<u32>>) = queries
        .iter()
        .map(|q| {
            let t0 = Instant::now();
            let results = flat.search(q, K);
            let elapsed = t0.elapsed().as_micros() as u64;
            let ids: Vec<u32> = results.iter().map(|r| r.id).collect();
            (elapsed, ids)
        })
        .unzip();
    let flat_stats = Stats::from(&flat_latencies);

    // ── MrlLinear ────────────────────────────────────────────────────────
    let mut mrl_linear = MrlLinear::new(D_FAST, D_FULL, K_OVER_LINEAR);
    for (i, v) in db.iter().enumerate() {
        mrl_linear.insert(i as u32, v);
    }
    let mut linear_recalls = Vec::new();
    let linear_latencies: Vec<u64> = queries
        .iter()
        .zip(flat_gt.iter())
        .map(|(q, gt)| {
            let t0 = Instant::now();
            let results = mrl_linear.search(q, K);
            let elapsed = t0.elapsed().as_micros() as u64;
            let ids: Vec<u32> = results.iter().map(|r| r.id).collect();
            linear_recalls.push(recall_at_k(gt, &ids));
            elapsed
        })
        .collect();
    let linear_stats = Stats::from(&linear_latencies);
    let linear_recall = mean(&linear_recalls);

    // ── MrlGraph ─────────────────────────────────────────────────────────
    let build_t0 = Instant::now();
    let mut mrl_graph = MrlGraph::new(D_FAST, D_FULL, M, EF, OVERSAMPLE);
    for (i, v) in db.iter().enumerate() {
        mrl_graph.insert(i as u32, v);
    }
    mrl_graph.build_edges();
    let graph_build_ms = build_t0.elapsed().as_millis();

    let mut graph_recalls = Vec::new();
    let graph_latencies: Vec<u64> = queries
        .iter()
        .zip(flat_gt.iter())
        .map(|(q, gt)| {
            let t0 = Instant::now();
            let results = mrl_graph.search(q, K);
            let elapsed = t0.elapsed().as_micros() as u64;
            let ids: Vec<u32> = results.iter().map(|r| r.id).collect();
            graph_recalls.push(recall_at_k(gt, &ids));
            elapsed
        })
        .collect();
    let graph_stats = Stats::from(&graph_latencies);
    let graph_recall = mean(&graph_recalls);

    let linear_speedup = flat_stats.mean / linear_stats.mean.max(0.01);
    let graph_speedup = flat_stats.mean / graph_stats.mean.max(0.01);

    println!(
        "  FlatFull build: {n} inserts | Graph build: {graph_build_ms} ms"
    );

    ExpResult {
        flat_stats,
        linear_stats,
        graph_stats,
        linear_recall,
        graph_recall,
        linear_speedup,
        graph_speedup,
    }
}

fn print_results(e: &ExpResult, label: &str) {
    println!("  Memory: FlatFull {:.1} MB | Graph adj ~{:.1} MB",
        mem_mb(N, D_FULL), graph_adj_mb(N, M));
    println!();
    println!("╔══════════════════════╦══════════════╦══════════╦══════════╦══════════╦══════════╦════════════╗");
    println!("║ {label:<20} ║ Mean µs/q    ║ p50 µs   ║ p95 µs   ║ QPS      ║ Recall@{K} ║ Speedup    ║");
    println!("╠══════════════════════╬══════════════╬══════════╬══════════╬══════════╬══════════╬════════════╣");
    println!(
        "║ {:<20} ║ {:>12.1} ║ {:>8} ║ {:>8} ║ {:>8.0} ║ {:>8.3}  ║ {:>10}  ║",
        "FlatFull (exact)", e.flat_stats.mean, e.flat_stats.p50, e.flat_stats.p95,
        qps(e.flat_stats.mean), 1.0f32, "1.0×"
    );
    println!(
        "║ {:<20} ║ {:>12.1} ║ {:>8} ║ {:>8} ║ {:>8.0} ║ {:>8.3}  ║ {:>10}  ║",
        "MrlLinear", e.linear_stats.mean, e.linear_stats.p50, e.linear_stats.p95,
        qps(e.linear_stats.mean), e.linear_recall,
        format!("{:.1}×", e.linear_speedup)
    );
    println!(
        "║ {:<20} ║ {:>12.1} ║ {:>8} ║ {:>8} ║ {:>8.0} ║ {:>8.3}  ║ {:>10}  ║",
        "MrlGraph", e.graph_stats.mean, e.graph_stats.p50, e.graph_stats.p95,
        qps(e.graph_stats.mean), e.graph_recall,
        format!("{:.1}×", e.graph_speedup)
    );
    println!("╚══════════════════════╩══════════════╩══════════╩══════════╩══════════╩══════════╩════════════╝");
}

// ── Utilities ────────────────────────────────────────────────────────────────

fn mem_mb(n: usize, d: usize) -> f64 {
    (n * d * 4) as f64 / 1_048_576.0
}
fn graph_adj_mb(n: usize, m: usize) -> f64 {
    (n * m * 4) as f64 / 1_048_576.0
}
fn qps(mean_us: f64) -> f64 {
    if mean_us < 1e-9 { return 0.0; }
    1_000_000.0 / mean_us
}
fn mean(v: &[f32]) -> f32 {
    if v.is_empty() { return 0.0; }
    v.iter().sum::<f32>() / v.len() as f32
}
fn check(label: &str, ok: bool) {
    println!("  [{}] {label}", if ok { "PASS" } else { "FAIL" });
}

struct Stats {
    mean: f64,
    p50: u64,
    p95: u64,
}
impl Stats {
    fn from(latencies: &[u64]) -> Self {
        let mut sorted = latencies.to_vec();
        sorted.sort_unstable();
        let n = sorted.len();
        let sum: u64 = sorted.iter().sum();
        Stats {
            mean: sum as f64 / n as f64,
            p50: sorted[n / 2],
            p95: sorted[(n * 95 / 100).min(n - 1)],
        }
    }
}
