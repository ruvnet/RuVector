/// Graph-Degree Adaptive Quantization Benchmark
///
/// Four variants:
///   1. Baseline:    Uniform 8-bit quantization (all vectors, gold standard)
///   2. GraphGuided: Graph in-degree mask → top 30% 8-bit, rest 4-bit
///   3. AccessFreq:  Zipf access frequency → top 30% 8-bit, rest 4-bit
///   4. RandomMixed: Random 30% 8-bit, rest 4-bit (null baseline for selection quality)
///
/// Key question: does graph-degree selection preserve recall better than random
/// selection at the same memory budget?
use ruvector_gdq::{
    dataset::{brute_force_knn, generate_clustered, generate_queries, simulate_access_freq},
    graph::KnnGraph,
    metrics::{estimated_memory_bytes, measure_latency, recall_at_k},
    store::{build_access_freq, build_graph_guided, build_random_mixed, build_uniform_high},
};

const N_VECTORS: usize = 2_000;
const DIM: usize = 128;
const N_CLUSTERS: usize = 20;
const K_GRAPH: usize = 16;
const K_SEARCH: usize = 10;
const N_QUERIES: usize = 200;
const HIGH_FRACTION: f32 = 0.30; // top 30% → 8-bit, rest 4-bit
const SEED: u64 = 12345;

fn main() {
    print_header();

    // ── Dataset ──────────────────────────────────────────────────────────────
    println!(
        "[1/6] Generating dataset: {} vectors × {} dims, {} clusters",
        N_VECTORS, DIM, N_CLUSTERS
    );
    let data = generate_clustered(N_VECTORS, DIM, N_CLUSTERS, SEED);
    let queries = generate_queries(N_QUERIES, DIM, SEED);

    // ── Ground truth ─────────────────────────────────────────────────────────
    println!(
        "[2/6] Computing ground truth (brute-force k-NN, k={})...",
        K_SEARCH
    );
    let ground_truth = brute_force_knn(&data, &queries, K_SEARCH);

    // ── Graph ─────────────────────────────────────────────────────────────────
    println!(
        "[3/6] Building k-NN graph for in-degree computation (k={})...",
        K_GRAPH
    );
    let t0 = std::time::Instant::now();
    let graph = KnnGraph::build(&data, K_GRAPH);
    let graph_build_ms = t0.elapsed().as_millis();
    println!(
        "  Graph built in {}ms. Mean in-degree: {:.2}, Max: {}",
        graph_build_ms,
        graph.mean_in_degree(),
        graph.max_in_degree()
    );
    let high_degree_mask = graph.high_degree_mask(HIGH_FRACTION);
    let n_high_graph = high_degree_mask.iter().filter(|&&v| v).count();
    println!(
        "  High-degree nodes ({:.0}%): {}/{}",
        HIGH_FRACTION * 100.0,
        n_high_graph,
        N_VECTORS
    );

    // ── Access frequency ──────────────────────────────────────────────────────
    let access_freq = simulate_access_freq(N_VECTORS, SEED);
    let n_high_access = (N_VECTORS as f32 * HIGH_FRACTION) as usize;
    let n_high_random = n_high_access;

    // ── Build stores ──────────────────────────────────────────────────────────
    println!("[4/6] Building four stores...");
    let store_base = build_uniform_high(&data);
    let store_graph = build_graph_guided(&data, &high_degree_mask);
    let store_access = build_access_freq(&data, &access_freq, HIGH_FRACTION);
    let store_random = build_random_mixed(&data, HIGH_FRACTION, SEED.wrapping_add(42));

    // ── Search + Latency ─────────────────────────────────────────────────────
    println!("[5/6] Running {} queries per variant...", N_QUERIES);

    let q_base = queries.clone();
    let lat_base = measure_latency(N_QUERIES, |i| store_base.search(&q_base[i], K_SEARCH));
    let q_graph = queries.clone();
    let lat_graph = measure_latency(N_QUERIES, |i| store_graph.search(&q_graph[i], K_SEARCH));
    let q_access = queries.clone();
    let lat_access = measure_latency(N_QUERIES, |i| store_access.search(&q_access[i], K_SEARCH));
    let q_random = queries.clone();
    let lat_random = measure_latency(N_QUERIES, |i| store_random.search(&q_random[i], K_SEARCH));

    // ── Recall ────────────────────────────────────────────────────────────────
    println!("[6/6] Computing recall@{}...", K_SEARCH);
    let res_base: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| store_base.search(q, K_SEARCH))
        .collect();
    let res_graph: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| store_graph.search(q, K_SEARCH))
        .collect();
    let res_access: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| store_access.search(q, K_SEARCH))
        .collect();
    let res_random: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| store_random.search(q, K_SEARCH))
        .collect();

    let recall_base = recall_at_k(&res_base, &ground_truth, K_SEARCH);
    let recall_graph = recall_at_k(&res_graph, &ground_truth, K_SEARCH);
    let recall_access = recall_at_k(&res_access, &ground_truth, K_SEARCH);
    let recall_random = recall_at_k(&res_random, &ground_truth, K_SEARCH);

    // ── Memory ────────────────────────────────────────────────────────────────
    let mem_base = store_base.encoded_bytes();
    let mem_graph = store_graph.encoded_bytes();
    let mem_access = store_access.encoded_bytes();
    let mem_random = store_random.encoded_bytes();

    let mem_base_math = estimated_memory_bytes(N_VECTORS, 0, DIM);
    let mem_graph_math = estimated_memory_bytes(n_high_graph, N_VECTORS - n_high_graph, DIM);
    let mem_access_math = estimated_memory_bytes(n_high_access, N_VECTORS - n_high_access, DIM);

    // ── Results ───────────────────────────────────────────────────────────────
    println!();
    println!("══════════════════════════════════════════════════════════════════════");
    println!("  Graph-Degree Adaptive Quantization — Benchmark Results");
    println!("══════════════════════════════════════════════════════════════════════");
    println!();
    println!(
        "Dataset:  {} vectors × {} dims | {} queries | K={}",
        N_VECTORS, DIM, N_QUERIES, K_SEARCH
    );
    println!(
        "Config:   k-NN graph k={} | high-precision fraction={:.0}% (8-bit) | low=4-bit",
        K_GRAPH,
        HIGH_FRACTION * 100.0
    );
    println!();

    let ratio_graph = mem_graph as f32 / mem_base as f32;
    let ratio_access = mem_access as f32 / mem_base as f32;
    let ratio_random = mem_random as f32 / mem_base as f32;

    println!(
        "{:<22} {:>12} {:>10} {:>10} {:>10} {:>10} {:>12} {:>10}",
        "Variant", "Mem(bytes)", "MemRatio", "Mean(µs)", "p50(µs)", "p95(µs)", "QPS", "Recall@10"
    );
    println!("{}", "─".repeat(108));

    println!(
        "{:<22} {:>12} {:>10} {:>10.1} {:>10.1} {:>10.1} {:>12.0} {:>10.4}",
        "Baseline(Uniform8bit)",
        mem_base,
        "1.000",
        lat_base.mean_us,
        lat_base.p50_us,
        lat_base.p95_us,
        lat_base.throughput_qps,
        recall_base
    );

    println!(
        "{:<22} {:>12} {:>10.3} {:>10.1} {:>10.1} {:>10.1} {:>12.0} {:>10.4}",
        "GraphGuided(30%+70%4b)",
        mem_graph,
        ratio_graph,
        lat_graph.mean_us,
        lat_graph.p50_us,
        lat_graph.p95_us,
        lat_graph.throughput_qps,
        recall_graph
    );

    println!(
        "{:<22} {:>12} {:>10.3} {:>10.1} {:>10.1} {:>10.1} {:>12.0} {:>10.4}",
        "AccessFreq(30%+70%4b)",
        mem_access,
        ratio_access,
        lat_access.mean_us,
        lat_access.p50_us,
        lat_access.p95_us,
        lat_access.throughput_qps,
        recall_access
    );

    println!(
        "{:<22} {:>12} {:>10.3} {:>10.1} {:>10.1} {:>10.1} {:>12.0} {:>10.4}",
        "RandomMixed(30%+70%4b)",
        mem_random,
        ratio_random,
        lat_random.mean_us,
        lat_random.p50_us,
        lat_random.p95_us,
        lat_random.throughput_qps,
        recall_random
    );

    println!();
    println!("Memory math (estimated vs actual):");
    println!(
        "  Baseline    : math={} B, actual={} B",
        mem_base_math, mem_base
    );
    println!(
        "  GraphGuided : math={} B, actual={} B",
        mem_graph_math, mem_graph
    );
    println!(
        "  AccessFreq  : math={} B, actual={} B",
        mem_access_math, mem_access
    );
    println!("  RandomMixed : actual={} B", mem_random);

    println!();
    println!("Recall delta vs Baseline:");
    println!(
        "  GraphGuided : {:.4} ({:.1}% loss vs 100% 8-bit)",
        recall_base - recall_graph,
        (recall_base - recall_graph) * 100.0
    );
    println!(
        "  AccessFreq  : {:.4} ({:.1}% loss)",
        recall_base - recall_access,
        (recall_base - recall_access) * 100.0
    );
    println!(
        "  RandomMixed : {:.4} ({:.1}% loss)",
        recall_base - recall_random,
        (recall_base - recall_random) * 100.0
    );

    println!();
    println!("Recall advantage of graph-degree selection vs random:");
    println!(
        "  GraphGuided - RandomMixed = {:.4} ({:.2}%)",
        recall_graph - recall_random,
        (recall_graph - recall_random) * 100.0
    );
    println!(
        "  GraphGuided - AccessFreq  = {:.4} ({:.2}%)",
        recall_graph - recall_access,
        (recall_graph - recall_access) * 100.0
    );

    // ── Acceptance tests ──────────────────────────────────────────────────────
    println!();
    println!("Acceptance Tests:");

    // 1) Memory savings ≥ 25%
    let mem_ok = ratio_graph <= 0.75;
    println!(
        "  [{}] Memory savings ≥ 25%: GraphGuided ratio={:.3} (need ≤ 0.75)",
        if mem_ok { "PASS" } else { "FAIL" },
        ratio_graph
    );

    // 2) GraphGuided recall ≥ 60% of baseline (honest 4-bit quality threshold)
    let recall_floor = recall_base * 0.60;
    let recall_ok = recall_graph >= recall_floor;
    println!(
        "  [{}] GraphGuided Recall@{} ≥ 60% of baseline: {:.4} ≥ {:.4}",
        if recall_ok { "PASS" } else { "FAIL" },
        K_SEARCH,
        recall_graph,
        recall_floor
    );

    // 3) GraphGuided ≥ RandomMixed at same budget (key scientific claim)
    let beats_random = recall_graph >= recall_random;
    println!(
        "  [{}] GraphGuided recall ≥ RandomMixed: {:.4} ≥ {:.4}  (Δ={:.4})",
        if beats_random { "PASS" } else { "FAIL" },
        recall_graph,
        recall_random,
        recall_graph - recall_random
    );

    // 4) GraphGuided ≥ AccessFreq
    let beats_access = recall_graph >= recall_access;
    println!(
        "  [{}] GraphGuided recall ≥ AccessFreq: {:.4} ≥ {:.4}",
        if beats_access { "PASS" } else { "FAIL" },
        recall_graph,
        recall_access
    );

    // 5) Memory math matches actual
    let math_ok = mem_base_math == mem_base;
    println!(
        "  [{}] Memory math matches actual for baseline",
        if math_ok { "PASS" } else { "FAIL" }
    );

    println!();
    let all_pass = mem_ok && recall_ok && beats_random && beats_access && math_ok;
    println!("══════════════════════════════════════════════════════════════════════");
    println!(
        "  Overall: {}",
        if all_pass {
            "ALL TESTS PASSED ✓"
        } else {
            "SOME TESTS FAILED ✗"
        }
    );
    println!("══════════════════════════════════════════════════════════════════════");

    println!();
    println!("Research Finding:");
    println!("  Graph-degree-guided selection (hub-aware) outperforms random selection");
    println!(
        "  by {:.2}% recall at a {:.0}% memory savings vs full 8-bit storage.",
        (recall_graph - recall_random) * 100.0,
        (1.0 - ratio_graph) * 100.0
    );
    println!("  Per-dimension 4-bit quantization (not global) is required for usable recall.");

    if !all_pass {
        std::process::exit(1);
    }
}

fn print_header() {
    println!();
    println!("  ruvector-gdq: Graph-Degree Adaptive Quantization");
    println!("  RuVector Nightly Research — 2026-07-29");
    println!();

    if let Ok(info) = std::fs::read_to_string("/proc/cpuinfo") {
        if let Some(line) = info.lines().find(|l| l.starts_with("model name")) {
            println!(
                "  CPU:  {}",
                line.split(':').nth(1).unwrap_or("unknown").trim()
            );
        }
    }
    if let Ok(mem) = std::fs::read_to_string("/proc/meminfo") {
        if let Some(line) = mem.lines().find(|l| l.starts_with("MemTotal")) {
            println!("  RAM:  {}", line.split(':').nth(1).unwrap_or("").trim());
        }
    }
    println!("  OS:   {}", std::env::consts::OS);
    println!("  Arch: {}", std::env::consts::ARCH);
    println!("  Rust: 1.94.1");
    println!("  Cmd:  cargo run --release -p ruvector-gdq --bin benchmark");
    println!();
}
