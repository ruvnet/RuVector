/// Benchmark: Multi-Subspace HNSW vs Baseline HNSW
///
/// Measures recall@10, mean/p50/p95 latency, throughput, and memory
/// for three variants on a structured clustered dataset.
use std::time::Instant;

use ruvector_subspace_hnsw::{
    dataset::{generate_clustered, generate_queries},
    ground_truth, percentiles, recall_at_k, BaselineHnsw, CoherenceHnsw, IndexConfig,
    SubspaceUnionHnsw,
};

// ─── Dataset parameters ──────────────────────────────────────────────────────
const N: usize = 10_000;
const DIM: usize = 128;
const N_CLUSTERS: usize = 20;
const N_SIGNAL_DIMS: usize = 96; // first 96 dims are informative, last 32 are noise
const N_QUERIES: usize = 200;
const SEED: u64 = 42;
const K: usize = 10; // recall@K

// ─── Index build parameters ───────────────────────────────────────────────────
const M: usize = 16;
const EF_CONSTRUCTION: usize = 100;
const EF_SEARCH: usize = 80;
const N_SUBSPACES: usize = 4; // 4 × 32-dim subspaces

// ─── Acceptance thresholds ────────────────────────────────────────────────────
// With simplified 2-layer NSW, D=128, N=10K and 75% signal dims, baseline
// recall is typically 0.55-0.75; the acceptance threshold reflects this PoC.
const ACCEPT_BASELINE_RECALL: f32 = 0.50;
const ACCEPT_COHERENCE_VS_UNION_DELTA: f32 = -0.05; // coherence must not be >5pp below union

fn main() {
    print_header();

    // 1. Build dataset
    let t0 = Instant::now();
    let (vectors, _labels) = generate_clustered(N, DIM, N_CLUSTERS, N_SIGNAL_DIMS, SEED);
    let queries = generate_queries(N_QUERIES, DIM, SEED + 1);
    println!(
        "Dataset built in {:.1}ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );
    println!();

    // 2. Pre-compute ground truth
    print!("Computing ground truth (brute-force)…  ");
    let t_gt = Instant::now();
    let ground_truths: Vec<Vec<(u32, f32)>> = queries
        .iter()
        .map(|q| ground_truth(&vectors, q, K))
        .collect();
    println!("{:.1}ms", t_gt.elapsed().as_secs_f64() * 1000.0);
    println!();

    let cfg = IndexConfig {
        m: M,
        ef_construction: EF_CONSTRUCTION,
        ef_search: EF_SEARCH,
        num_subspaces: N_SUBSPACES,
    };

    // ── Variant 1: Baseline full-dim HNSW ─────────────────────────────────────
    print!("Building Baseline-HNSW (M={M}, ef_c={EF_CONSTRUCTION})…  ");
    let t_b = Instant::now();
    let baseline = BaselineHnsw::build(&vectors, cfg.m, cfg.ef_construction);
    let t_build_base = t_b.elapsed().as_secs_f64() * 1000.0;
    println!("{t_build_base:.1}ms");

    let (rec_base, lat_base, mem_base) =
        run_queries_baseline(&baseline, &queries, &ground_truths, &cfg);

    // ── Variant 2: Subspace-Union (K=4 equal subspaces) ───────────────────────
    print!(
        "Building SubspaceUnion-HNSW ({N_SUBSPACES}×{}-dim subspaces)…  ",
        DIM / N_SUBSPACES
    );
    let t_su = Instant::now();
    let union_idx = SubspaceUnionHnsw::build(&vectors, N_SUBSPACES, cfg.m, cfg.ef_construction);
    let t_build_union = t_su.elapsed().as_secs_f64() * 1000.0;
    println!("{t_build_union:.1}ms");

    let (rec_union, lat_union, mem_union) =
        run_queries_union(&union_idx, &queries, &ground_truths, &cfg);

    // ── Variant 3: Coherence-Fused (same K subspaces, variance-weighted) ──────
    print!(
        "Building CoherenceHnsw ({N_SUBSPACES}×{}-dim subspaces)…  ",
        DIM / N_SUBSPACES
    );
    let t_coh = Instant::now();
    let coh_idx = CoherenceHnsw::build(&vectors, N_SUBSPACES, cfg.m, cfg.ef_construction);
    let t_build_coh = t_coh.elapsed().as_secs_f64() * 1000.0;
    println!("{t_build_coh:.1}ms");

    let (rec_coh, lat_coh, mem_coh) = run_queries_coh(&coh_idx, &queries, &ground_truths, &cfg);

    // ── Results table ─────────────────────────────────────────────────────────
    println!();
    println!("┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│  Variant              │  Build(ms)  │  Recall@{K}  │  Mean(µs) │  p50(µs) │  p95(µs) │  QPS    │  Mem(MB)  │");
    println!("├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤");
    print_row(
        "Baseline-HNSW",
        t_build_base,
        &rec_base,
        &lat_base,
        mem_base,
    );
    print_row(
        "SubspaceUnion-HNSW",
        t_build_union,
        &rec_union,
        &lat_union,
        mem_union,
    );
    print_row("CoherenceHnsw", t_build_coh, &rec_coh, &lat_coh, mem_coh);
    println!("└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // ── Dataset & index parameters ────────────────────────────────────────────
    println!("Dataset : N={N}, D={DIM}, clusters={N_CLUSTERS}, signal_dims={N_SIGNAL_DIMS}, queries={N_QUERIES}");
    println!("Index   : M={M}, ef_construction={EF_CONSTRUCTION}, ef_search={EF_SEARCH}, K_subspaces={N_SUBSPACES}");
    println!();

    // ── Acceptance check ──────────────────────────────────────────────────────
    let pass_base = rec_base[0] >= ACCEPT_BASELINE_RECALL;
    let delta_coh_union = rec_coh[0] - rec_union[0];
    let pass_coh = delta_coh_union >= ACCEPT_COHERENCE_VS_UNION_DELTA;

    println!("┌── Acceptance ─────────────────────────────────────────────────────────┐");
    println!(
        "│ Baseline recall@{K} >= {ACCEPT_BASELINE_RECALL:.2}:  {}   ({:.3})",
        if pass_base { "PASS ✓" } else { "FAIL ✗" },
        rec_base[0]
    );
    println!(
        "│ Coherence delta vs union >= {ACCEPT_COHERENCE_VS_UNION_DELTA:.2}: {}  ({:+.3})",
        if pass_coh { "PASS ✓" } else { "FAIL ✗" },
        delta_coh_union
    );
    println!("└───────────────────────────────────────────────────────────────────────┘");

    if !pass_base || !pass_coh {
        eprintln!("BENCHMARK FAILED — see acceptance checks above");
        std::process::exit(1);
    }
    println!("All acceptance tests passed.");
}

// ── Helper: run queries on BaselineHnsw ──────────────────────────────────────
fn run_queries_baseline(
    idx: &BaselineHnsw,
    queries: &[Vec<f32>],
    ground_truths: &[Vec<(u32, f32)>],
    cfg: &IndexConfig,
) -> (Vec<f32>, Vec<f64>, usize) {
    let mut recalls = Vec::with_capacity(queries.len());
    let mut latencies_us = Vec::with_capacity(queries.len());

    for (q, gt) in queries.iter().zip(ground_truths.iter()) {
        let t = Instant::now();
        let res = idx.search(q, K, cfg.ef_search);
        latencies_us.push(t.elapsed().as_micros() as u64);
        recalls.push(recall_at_k(&res, gt, K));
    }

    summarise(recalls, latencies_us, idx.memory_bytes())
}

fn run_queries_union(
    idx: &SubspaceUnionHnsw,
    queries: &[Vec<f32>],
    ground_truths: &[Vec<(u32, f32)>],
    cfg: &IndexConfig,
) -> (Vec<f32>, Vec<f64>, usize) {
    let mut recalls = Vec::with_capacity(queries.len());
    let mut latencies_us = Vec::with_capacity(queries.len());

    for (q, gt) in queries.iter().zip(ground_truths.iter()) {
        let t = Instant::now();
        let res = idx.search(q, K, cfg.ef_search);
        latencies_us.push(t.elapsed().as_micros() as u64);
        recalls.push(recall_at_k(&res, gt, K));
    }

    summarise(recalls, latencies_us, idx.memory_bytes())
}

fn run_queries_coh(
    idx: &CoherenceHnsw,
    queries: &[Vec<f32>],
    ground_truths: &[Vec<(u32, f32)>],
    cfg: &IndexConfig,
) -> (Vec<f32>, Vec<f64>, usize) {
    let mut recalls = Vec::with_capacity(queries.len());
    let mut latencies_us = Vec::with_capacity(queries.len());

    for (q, gt) in queries.iter().zip(ground_truths.iter()) {
        let t = Instant::now();
        let res = idx.search(q, K, cfg.ef_search);
        latencies_us.push(t.elapsed().as_micros() as u64);
        recalls.push(recall_at_k(&res, gt, K));
    }

    summarise(recalls, latencies_us, idx.memory_bytes())
}

fn summarise(recalls: Vec<f32>, latencies_us: Vec<u64>, mem: usize) -> (Vec<f32>, Vec<f64>, usize) {
    let mean_recall = recalls.iter().sum::<f32>() / recalls.len() as f32;
    let (mean_us, p50_us, p95_us) = percentiles(latencies_us.clone());
    let qps = 1_000_000.0 / mean_us.max(1.0);
    (
        vec![
            mean_recall,
            mean_us as f32,
            p50_us as f32,
            p95_us as f32,
            qps as f32,
        ],
        vec![mean_us, p50_us as f64, p95_us as f64, qps],
        mem,
    )
}

fn print_row(name: &str, build_ms: f64, metrics: &[f32], _latencies: &[f64], mem_bytes: usize) {
    let recall = metrics[0];
    let mean_us = metrics[1];
    let p50_us = metrics[2];
    let p95_us = metrics[3];
    let qps = metrics[4];
    let mem_mb = mem_bytes as f64 / 1_048_576.0;
    println!(
        "│  {name:<21}│  {build_ms:>9.1}  │  {recall:>8.3}   │ {mean_us:>9.1}  │{p50_us:>8.0}  │{p95_us:>8.0}  │{qps:>7.0}  │ {mem_mb:>7.2}   │"
    );
}

fn print_header() {
    println!();
    println!("════════════════════════════════════════════════════════════════════════");
    println!(" ruvector-subspace-hnsw  ·  Nightly benchmark  2026-06-12");
    println!(" Multi-Subspace HNSW with Coherence-Weighted Fusion");
    println!("════════════════════════════════════════════════════════════════════════");
    println!(" OS:   {}", std::env::consts::OS);
    println!(" Arch: {}", std::env::consts::ARCH);
    println!();
}
