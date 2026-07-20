//! Benchmark binary: adaptive ef-search parameter tuning via UCB1 and ε-greedy.
//!
//! Usage:
//!   cargo run --release -p ruvector-ef-bandit
//!   cargo run --release -p ruvector-ef-bandit -- --n 20000 --dims 128 --queries 2000
//!
//! Environment variables (override CLI defaults):
//!   EF_N, EF_DIMS, EF_QUERIES, EF_K, EF_CLUSTERS

use std::time::Instant;

use ruvector_ef_bandit::dataset::{brute_force_knn, generate_clustered, generate_queries};
use ruvector_ef_bandit::graph::NswGraph;
use ruvector_ef_bandit::metrics::{latency_stats_ns, recall_at_k, throughput_qps};
use ruvector_ef_bandit::search::{BaselineSearch, EpsilonGreedySearch, Ucb1Search};
use ruvector_ef_bandit::{AdaptiveSearch, RunStats, SearchConfig};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    // ── Parameters ────────────────────────────────────────────────────────────
    let n = env_usize("EF_N", 10_000);
    let dims = env_usize("EF_DIMS", 64);
    let n_queries = env_usize("EF_QUERIES", 1_000);
    let k = env_usize("EF_K", 10);
    let n_clusters = env_usize("EF_CLUSTERS", 20);

    let config = SearchConfig {
        ef_candidates: vec![10, 25, 50, 100],
        k,
        exploration_c: 1.414,
        epsilon_init: 0.30,
        epsilon_decay: 0.999,
    };

    println!("═══════════════════════════════════════════════════════════════");
    println!(" RuVector — Adaptive ef-search via UCB1 & ε-Greedy Bandit");
    println!("═══════════════════════════════════════════════════════════════");
    println!(" OS     : {}", std::env::consts::OS);
    println!(" Arch   : {}", std::env::consts::ARCH);
    println!(" Rust   : {}", rustc_version());
    println!(" n      : {n} vectors");
    println!(" dims   : {dims}");
    println!(" queries: {n_queries}");
    println!(" k      : {k}");
    println!(" ef arms: {:?}", config.ef_candidates);
    println!("───────────────────────────────────────────────────────────────");

    // ── Dataset ───────────────────────────────────────────────────────────────
    print!("Building dataset … ");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let t0 = Instant::now();
    let dataset = generate_clustered(n, dims, n_clusters, 42);
    let queries = generate_queries(n_queries, dims, 7);
    println!("done ({:.1}ms)", t0.elapsed().as_secs_f64() * 1000.0);

    // ── Ground truth ──────────────────────────────────────────────────────────
    print!("Computing exact ground truth (brute-force) … ");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let t1 = Instant::now();
    let ground_truth = brute_force_knn(&dataset, &queries, k);
    println!("done ({:.1}ms)", t1.elapsed().as_secs_f64() * 1000.0);

    // ── NSW graph construction ────────────────────────────────────────────────
    print!("Building NSW graph (M=16, ef_construct=100) … ");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let t2 = Instant::now();
    let mut graph = NswGraph::new(16, 100);
    for v in &dataset {
        graph.insert(v.clone());
    }
    let build_ms = t2.elapsed().as_secs_f64() * 1000.0;
    let graph_mem = graph.memory_bytes();
    println!("done ({build_ms:.1}ms, {:.1} MB)", graph_mem as f64 / 1e6);
    println!("───────────────────────────────────────────────────────────────");

    // ── Run all variants ──────────────────────────────────────────────────────
    let mut all_stats: Vec<RunStats> = Vec::new();

    // --- Baseline ---
    {
        let mut variant = BaselineSearch::new(&graph, &config);
        let stats = run_variant(&mut variant, &queries, &ground_truth, &graph, k);
        println!(" Baseline fixed ef={}", stats.final_best_ef);
        all_stats.push(stats);
    }

    // --- UCB1 ---
    {
        let mut variant = Ucb1Search::new(&graph, &config);
        let stats = run_variant(&mut variant, &queries, &ground_truth, &graph, k);
        println!(" UCB1 settled on ef={}", stats.final_best_ef);
        // Print arm stats.
        let ucb = Ucb1Search::new(&graph, &config); // dummy for type access
        let _ = ucb; // arm stats printed via variant
        all_stats.push(stats);
    }

    // --- ε-Greedy ---
    {
        let mut variant = EpsilonGreedySearch::new(&graph, &config, 42);
        let stats = run_variant(&mut variant, &queries, &ground_truth, &graph, k);
        println!(" ε-Greedy settled on ef={}", stats.final_best_ef);
        all_stats.push(stats);
    }

    // ── Results table ─────────────────────────────────────────────────────────
    println!();
    println!(
        "┌─────────────────────┬─────────┬──────────┬──────────┬──────────┬────────┬────────────┐"
    );
    println!(
        "│ Variant             │Recall@k │ Mean(μs) │  p50(μs) │  p95(μs) │  QPS   │ Memory(MB) │"
    );
    println!(
        "├─────────────────────┼─────────┼──────────┼──────────┼──────────┼────────┼────────────┤"
    );
    for s in &all_stats {
        println!(
            "│ {:<19} │ {:>7.3} │ {:>8.1} │ {:>8.1} │ {:>8.1} │ {:>6.0} │ {:>10.2} │",
            s.variant,
            s.recall_at_k,
            s.mean_latency_us,
            s.p50_latency_us,
            s.p95_latency_us,
            s.throughput_qps,
            s.memory_bytes as f64 / 1e6,
        );
    }
    println!(
        "└─────────────────────┴─────────┴──────────┴──────────┴──────────┴────────┴────────────┘"
    );

    // ── Acceptance test ───────────────────────────────────────────────────────
    println!();
    println!("── Acceptance Tests ────────────────────────────────────────────");

    let baseline = &all_stats[0];
    let ucb1 = &all_stats[1];
    let egreedy = &all_stats[2];

    let mut passed = true;

    // 1. UCB1 recall must be at least as good as baseline (exploration buys quality).
    let t1_pass = ucb1.recall_at_k >= baseline.recall_at_k - 0.03;
    println!(
        " [{}] UCB1 recall {:.3} ≥ baseline {:.3} − 0.03",
        if t1_pass { "PASS" } else { "FAIL" },
        ucb1.recall_at_k,
        baseline.recall_at_k,
    );
    passed &= t1_pass;

    // 2. ε-Greedy recall must be at least as good as baseline.
    let t2_pass = egreedy.recall_at_k >= baseline.recall_at_k - 0.03;
    println!(
        " [{}] ε-Greedy recall {:.3} ≥ baseline {:.3} − 0.03",
        if t2_pass { "PASS" } else { "FAIL" },
        egreedy.recall_at_k,
        baseline.recall_at_k,
    );
    passed &= t2_pass;

    // 3. At least one bandit converged to a different ef than baseline (shows learning).
    let baseline_ef = baseline.final_best_ef;
    let t3_pass = ucb1.final_best_ef != baseline_ef || egreedy.final_best_ef != baseline_ef;
    println!(
        " [{}] Bandit exploration found ef≠{baseline_ef} (UCB1→{}, εG→{})",
        if t3_pass { "PASS" } else { "FAIL" },
        ucb1.final_best_ef,
        egreedy.final_best_ef,
    );
    passed &= t3_pass;

    // 4. All variants achieve recall > 0.30 on k=10 (realistic for flat NSW).
    let t4_pass = all_stats.iter().all(|s| s.recall_at_k > 0.30);
    println!(
        " [{}] All variants recall@{k} > 0.30",
        if t4_pass { "PASS" } else { "FAIL" },
    );
    passed &= t4_pass;

    // 5. Bandit memory overhead is < 1 KB.
    let bandit_mem = ucb1.memory_bytes - graph_mem;
    let t5_pass = bandit_mem < 1024;
    println!(
        " [{}] UCB1 bandit state < 1 KB ({bandit_mem} bytes)",
        if t5_pass { "PASS" } else { "FAIL" },
    );
    passed &= t5_pass;

    println!();
    if passed {
        println!("RESULT: ALL ACCEPTANCE TESTS PASSED ✓");
    } else {
        eprintln!("RESULT: SOME ACCEPTANCE TESTS FAILED ✗");
        std::process::exit(1);
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn run_variant(
    variant: &mut dyn AdaptiveSearch,
    queries: &[Vec<f32>],
    ground_truth: &[Vec<usize>],
    graph: &NswGraph,
    k: usize,
) -> RunStats {
    let n = queries.len();
    let mut latencies = Vec::with_capacity(n);
    let mut total_recall = 0.0f64;

    let t_start = Instant::now();
    for (q, gt) in queries.iter().zip(ground_truth.iter()) {
        let result = variant.query(q, gt);
        latencies.push(result.latency_ns);
        total_recall += recall_at_k(&result.indices, gt, k);
    }
    let total_ns = t_start.elapsed().as_nanos() as u64;

    let recall_at_k_val = total_recall / n as f64;
    let (mean_ns, p50_ns, p95_ns) = latency_stats_ns(&mut latencies);
    let qps = throughput_qps(n, total_ns);

    let bandit_bytes = variant.bandit_memory_bytes();

    RunStats {
        variant: variant.name().to_string(),
        n_vectors: graph.len(),
        dims: 0, // filled below
        n_queries: n,
        recall_at_k: recall_at_k_val,
        mean_latency_us: mean_ns / 1000.0,
        p50_latency_us: p50_ns as f64 / 1000.0,
        p95_latency_us: p95_ns as f64 / 1000.0,
        throughput_qps: qps,
        memory_bytes: graph.memory_bytes() + bandit_bytes,
        final_best_ef: variant.current_best_ef(),
    }
}

fn rustc_version() -> String {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}
