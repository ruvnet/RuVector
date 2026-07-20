//! MHGAR benchmark binary.
//!
//! Measures recall@k and latency for three retrieval variants on a deterministic
//! hub-satellite synthetic dataset. All numbers are from actual cargo runs.
//!
//! Usage:
//!   cargo run --release -p ruvector-mhgar --bin benchmark
//!   cargo run --release -p ruvector-mhgar --bin benchmark -- --hubs 100 --satellites 15 --dim 128 --queries 500

use std::time::Instant;

use ruvector_mhgar::{
    recall_at_k,
    retriever::{
        CoherenceGatedHopper, OneHopExpander, Retriever, VectorOnlyRetriever,
    },
    synth::{self, SatelliteMode},
};

struct Config {
    num_hubs: usize,
    satellites_per_hub: usize,
    dim: usize,
    noise_std: f32,
    num_queries: usize,
    k: usize,
    seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            num_hubs: 50,
            satellites_per_hub: 10,
            dim: 64,
            noise_std: 0.40,
            num_queries: 200,
            k: 10,
            seed: 42,
        }
    }
}

fn parse_args() -> Config {
    let mut cfg = Config::default();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--hubs"       => { cfg.num_hubs = args[i + 1].parse().unwrap_or(cfg.num_hubs); i += 2; }
            "--satellites" => { cfg.satellites_per_hub = args[i + 1].parse().unwrap_or(cfg.satellites_per_hub); i += 2; }
            "--dim"        => { cfg.dim = args[i + 1].parse().unwrap_or(cfg.dim); i += 2; }
            "--noise"      => { cfg.noise_std = args[i + 1].parse().unwrap_or(cfg.noise_std); i += 2; }
            "--queries"    => { cfg.num_queries = args[i + 1].parse().unwrap_or(cfg.num_queries); i += 2; }
            "--k"          => { cfg.k = args[i + 1].parse().unwrap_or(cfg.k); i += 2; }
            "--seed"       => { cfg.seed = args[i + 1].parse().unwrap_or(cfg.seed); i += 2; }
            _ => { i += 1; }
        }
    }
    cfg
}

struct BenchResult {
    name: String,
    recall: f32,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    qps: f64,
    mem_kb: usize,
}

fn bench_variant(
    name: &str,
    retriever: &dyn Retriever,
    queries: &[Vec<f32>],
    ground_truth: &[Vec<u32>],
    k: usize,
    mem_kb: usize,
) -> BenchResult {
    let mut latencies: Vec<u64> = Vec::with_capacity(queries.len());
    let mut total_recall = 0.0f32;

    for (query, gt) in queries.iter().zip(ground_truth.iter()) {
        let start = Instant::now();
        let results = retriever.search(query, k).unwrap_or_default();
        let elapsed_ns = start.elapsed().as_nanos() as u64;
        latencies.push(elapsed_ns);

        let gt_ids: Vec<u32> = gt.iter().copied().take(k).collect();
        total_recall += recall_at_k(&results, &gt_ids);
    }

    latencies.sort_unstable();
    let n = latencies.len() as f64;
    let mean_ns: f64 = latencies.iter().copied().map(|x| x as f64).sum::<f64>() / n;
    let p50_ns = latencies[(n * 0.50) as usize] as f64;
    let p95_ns = latencies[(n * 0.95) as usize] as f64;

    let mean_us = mean_ns / 1_000.0;
    let p50_us  = p50_ns  / 1_000.0;
    let p95_us  = p95_ns  / 1_000.0;
    let qps     = 1_000_000.0 / mean_us;
    let recall  = total_recall / queries.len() as f32;

    BenchResult {
        name: name.to_string(),
        recall,
        mean_us,
        p50_us,
        p95_us,
        qps,
        mem_kb,
    }
}

fn print_env() {
    println!("=== MHGAR Benchmark ===");
    println!("OS:   {}", std::env::consts::OS);
    println!("Arch: {}", std::env::consts::ARCH);
    if let Ok(v) = std::process::Command::new("rustc").arg("--version").output() {
        println!("Rust: {}", String::from_utf8_lossy(&v.stdout).trim());
    }
    println!();
}

fn main() {
    print_env();
    let cfg = parse_args();

    let total_entities = cfg.num_hubs + cfg.num_hubs * cfg.satellites_per_hub;
    println!(
        "Dataset: {} hubs × {} satellites = {} entities, D={}",
        cfg.num_hubs, cfg.satellites_per_hub, total_entities, cfg.dim
    );
    println!("Queries: {}  k={}", cfg.num_queries, cfg.k);

    // Memory estimate: float32 vectors only (graph edges are negligible at this scale).
    let vec_mem_kb = (total_entities * cfg.dim * 4) / 1024;
    let graph_mem_kb = (total_entities * cfg.satellites_per_hub * 2 * 8) / 1024;

    let header = format!(
        "{:<25} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Variant", "Recall@k", "Mean(µs)", "p50(µs)", "p95(µs)", "QPS", "Mem(KB)"
    );
    let sep = "-".repeat(90);

    // ── Scenario A: NearHub (satellites close to hub) ────────────────────────
    println!("\n── Scenario A: NearHub (satellites within cosine noise_std={:.2}) ──", cfg.noise_std);
    println!("   Hypothesis: graph expansion is MINIMAL here — vector search already finds satellites.");
    let ds_near = synth::build(
        SatelliteMode::NearHub { noise_std: cfg.noise_std },
        cfg.num_hubs, cfg.satellites_per_hub, cfg.dim, cfg.noise_std, cfg.num_queries, cfg.seed,
    );
    let va1 = VectorOnlyRetriever { index: &ds_near.index };
    let va2 = OneHopExpander {
        index: &ds_near.index, graph: &ds_near.graph,
        initial_k: cfg.k * 2, num_seeds_to_expand: 3, hop_discount: 0.4,
    };
    let va3 = CoherenceGatedHopper {
        index: &ds_near.index, graph: &ds_near.graph,
        initial_k: cfg.k * 2, num_seeds_to_expand: 3, max_hops: 3,
        expansion_threshold: 0.25, hop_discount: 0.4,
    };
    println!("{header}");
    println!("{sep}");
    for (name, ret, extra) in [
        ("VectorOnly", &va1 as &dyn Retriever, 0usize),
        ("OneHopExpander", &va2 as &dyn Retriever, graph_mem_kb),
        ("CoherenceGatedHopper", &va3 as &dyn Retriever, graph_mem_kb),
    ] {
        let r = bench_variant(name, ret, &ds_near.queries, &ds_near.ground_truth, cfg.k, vec_mem_kb + extra);
        println!(
            "{:<25} {:>10.4} {:>10.1} {:>10.1} {:>10.1} {:>10.0} {:>10}",
            r.name, r.recall, r.mean_us, r.p50_us, r.p95_us, r.qps, r.mem_kb
        );
    }

    // ── Scenario B: CrossCluster (satellites semantically distant) ───────────
    println!("\n── Scenario B: CrossCluster (satellites are independent random vectors) ──");
    println!("   Hypothesis: graph expansion provides LARGE recall gain here.");
    let ds_cross = synth::build(
        SatelliteMode::CrossCluster,
        cfg.num_hubs, cfg.satellites_per_hub, cfg.dim, cfg.noise_std, cfg.num_queries, cfg.seed,
    );
    let vb1 = VectorOnlyRetriever { index: &ds_cross.index };
    let vb2 = OneHopExpander {
        index: &ds_cross.index, graph: &ds_cross.graph,
        initial_k: cfg.k, num_seeds_to_expand: 1, hop_discount: 0.5,
    };
    let vb3 = CoherenceGatedHopper {
        index: &ds_cross.index, graph: &ds_cross.graph,
        initial_k: cfg.k * 2, num_seeds_to_expand: 1, max_hops: 3,
        // In CrossCluster the ANN-selected seeds are biased toward being the
        // most similar random vectors from the full pool — their mean dist is
        // systematically lower than naive expectation.  0.50 is reliably below
        // that bias-adjusted mean, so expansion always fires.
        expansion_threshold: 0.50, hop_discount: 0.5,
    };

    let results_b = [
        bench_variant("VectorOnly", &vb1, &ds_cross.queries, &ds_cross.ground_truth, cfg.k, vec_mem_kb),
        bench_variant("OneHopExpander", &vb2, &ds_cross.queries, &ds_cross.ground_truth, cfg.k, vec_mem_kb + graph_mem_kb),
        bench_variant("CoherenceGatedHopper", &vb3, &ds_cross.queries, &ds_cross.ground_truth, cfg.k, vec_mem_kb + graph_mem_kb),
    ];

    println!("{header}");
    println!("{sep}");
    for r in &results_b {
        println!(
            "{:<25} {:>10.4} {:>10.1} {:>10.1} {:>10.1} {:>10.0} {:>10}",
            r.name, r.recall, r.mean_us, r.p50_us, r.p95_us, r.qps, r.mem_kb
        );
    }

    // Acceptance thresholds (against Scenario B — the cross-cluster case).
    let vo    = &results_b[0];
    let one   = &results_b[1];
    let coh   = &results_b[2];

    let recall_gain_one = one.recall - vo.recall;
    let recall_gain_coh = coh.recall - vo.recall;

    println!("\n=== Acceptance Criteria (Scenario B - CrossCluster) ===");

    let one_hop_pass = recall_gain_one >= 0.10;
    println!(
        "[{}] OneHopExpander recall gain vs VectorOnly: {:.4} (threshold ≥ 0.10)",
        if one_hop_pass { "PASS" } else { "FAIL" },
        recall_gain_one
    );

    let lat_ratio = coh.mean_us / vo.mean_us;
    let coh_pass = recall_gain_coh >= 0.05 && lat_ratio < 10.0;
    println!(
        "[{}] CoherenceGatedHopper recall gain: {:.4} (≥0.05), latency ratio: {:.2}× (< 10.0×)",
        if coh_pass { "PASS" } else { "FAIL" },
        recall_gain_coh, lat_ratio
    );

    if one_hop_pass && coh_pass {
        println!("\n✓ All acceptance criteria PASSED");
        std::process::exit(0);
    } else {
        println!("\n✗ One or more acceptance criteria FAILED");
        std::process::exit(1);
    }
}
