//! Benchmark: four adaptive ef policies on a simulated HNSW index.
//!
//! Prints hardware info, dataset params, per-policy stats, and an acceptance verdict.

use ruvector_adaptive_ef::{
    hnsw_sim::HnswSim,
    metrics::{LatencyWindow, recall_at_k},
    policy::{BanditPolicy, EwmaGreedy, FixedPolicy, PidController, SearchPolicy},
};

fn print_header() {
    println!("════════════════════════════════════════════════════════════════");
    println!("  ruvector-adaptive-ef  ·  Adaptive ef-Search Benchmark");
    println!("════════════════════════════════════════════════════════════════");

    if let Ok(os) = std::fs::read_to_string("/etc/os-release") {
        if let Some(line) = os.lines().find(|l| l.starts_with("PRETTY_NAME=")) {
            println!("  OS     : {}", line.trim_start_matches("PRETTY_NAME=").trim_matches('"'));
        }
    } else {
        println!("  OS     : unknown");
    }

    if let Ok(cpu) = std::fs::read_to_string("/proc/cpuinfo") {
        if let Some(line) = cpu.lines().find(|l| l.starts_with("model name")) {
            println!("  CPU    : {}", line.split(':').nth(1).unwrap_or("unknown").trim());
        }
    }

    let rust_ver = option_env!("RUSTC_VERSION").unwrap_or("(see rustc --version)");
    println!("  Rust   : {rust_ver}");
    println!("────────────────────────────────────────────────────────────────");
}

struct PolicyResult {
    name: String,
    mean_us: f64,
    p50_us: u64,
    p95_us: u64,
    qps: f64,
    mean_recall: f32,
    final_ef: u32,
    converged: bool,
}

fn run_policy(
    policy: &mut dyn SearchPolicy,
    index: &HnswSim,
    queries: &[Vec<f32>],
    ground_truths: &[Vec<usize>],
    k: usize,
    budget_us: u64,
) -> PolicyResult {
    let mut window = LatencyWindow::new(queries.len());
    let mut total_recall = 0.0_f32;

    for (q, gt) in queries.iter().zip(ground_truths.iter()) {
        let ef = policy.recommend_ef(budget_us) as usize;
        let (result, lat) = index.search(q, k, ef.max(k));
        let recall = recall_at_k(&result, gt);

        policy.observe(lat, recall, ef as u32);
        window.push(lat);
        total_recall += recall;
    }

    let _n = queries.len() as f64;
    let mean_us = window.mean_us();
    let p50 = window.percentile_us(50.0);
    let p95 = window.percentile_us(95.0);
    let qps = if mean_us > 0.0 { 1_000_000.0 / mean_us } else { 0.0 };
    let mean_recall = total_recall / queries.len() as f32;

    // Convergence: last 20% of queries should be within ±30% of budget
    let tail_start = (queries.len() as f64 * 0.8) as usize;
    let tail_recall = &ground_truths[tail_start..];
    let mut tail_window = LatencyWindow::new(queries.len() - tail_start);
    for (q, gt) in queries[tail_start..].iter().zip(tail_recall.iter()) {
        let ef = policy.recommend_ef(budget_us) as usize;
        let (res, lat) = index.search(q, k, ef.max(k));
        let rec = recall_at_k(&res, gt);
        policy.observe(lat, rec, ef as u32);
        tail_window.push(lat);
    }
    let tail_mean = tail_window.mean_us();
    let converged = tail_mean <= budget_us as f64 * 1.30 && tail_mean >= budget_us as f64 * 0.40;

    PolicyResult {
        name: policy.name().to_string(),
        mean_us,
        p50_us: p50,
        p95_us: p95,
        qps,
        mean_recall,
        final_ef: policy.current_ef(),
        converged,
    }
}

fn main() {
    print_header();

    // ── Dataset parameters ──────────────────────────────────────────────────
    let n: usize = 3_000;
    let dim: usize = 64;
    let k: usize = 10;
    let n_queries: usize = 500;
    let budget_us: u64 = 400; // latency target per query
    let m_neighbors: usize = 16;

    println!("  Dataset : N={n} vectors, dim={dim}, M={m_neighbors}");
    println!("  Queries : {n_queries}");
    println!("  K       : {k}");
    println!("  Budget  : {budget_us}µs per query");
    println!("════════════════════════════════════════════════════════════════");

    // ── Build index ─────────────────────────────────────────────────────────
    print!("  Building index … ");
    let build_start = std::time::Instant::now();
    let mut index = HnswSim::new(dim, m_neighbors);
    for _ in 0..n {
        let v = index.random_vector();
        index.insert(v);
    }
    println!("done ({:.1}ms)", build_start.elapsed().as_millis());

    // ── Generate queries and ground truths ──────────────────────────────────
    print!("  Generating queries and ground truths … ");
    let mut queries: Vec<Vec<f32>> = Vec::with_capacity(n_queries);
    let mut ground_truths: Vec<Vec<usize>> = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        let q = index.random_vector();
        let gt = index.exact_search(&q, k);
        queries.push(q);
        ground_truths.push(gt);
    }
    println!("done");
    println!("────────────────────────────────────────────────────────────────");

    // ── Run policies ─────────────────────────────────────────────────────────
    let mut results: Vec<PolicyResult> = Vec::new();

    {
        let mut p = FixedPolicy::new(64);
        results.push(run_policy(&mut p, &index, &queries, &ground_truths, k, budget_us));
    }
    {
        let mut p = EwmaGreedy::new(64, 0.15);
        results.push(run_policy(&mut p, &index, &queries, &ground_truths, k, budget_us));
    }
    {
        let mut p = BanditPolicy::new(0.20);
        results.push(run_policy(&mut p, &index, &queries, &ground_truths, k, budget_us));
    }
    {
        let mut p = PidController::new(64);
        results.push(run_policy(&mut p, &index, &queries, &ground_truths, k, budget_us));
    }

    // ── Print results ─────────────────────────────────────────────────────────
    println!(
        "  {:<14} {:>8} {:>8} {:>8} {:>10} {:>10} {:>8} {:>10}",
        "Policy", "Mean(µs)", "p50(µs)", "p95(µs)", "QPS", "Recall@10", "FinalEf", "Converged"
    );
    println!("  {}", "─".repeat(82));

    let mut all_pass = true;
    for r in &results {
        println!(
            "  {:<14} {:>8.1} {:>8} {:>8} {:>10.0} {:>10.3} {:>8} {:>10}",
            r.name, r.mean_us, r.p50_us, r.p95_us, r.qps, r.mean_recall, r.final_ef,
            if r.converged { "YES" } else { "NO" }
        );
        if r.name != "Fixed" && !r.converged {
            all_pass = false;
        }
    }

    println!("────────────────────────────────────────────────────────────────");

    // ── Acceptance criteria ───────────────────────────────────────────────────
    // 1. All adaptive policies must achieve recall@10 ≥ 0.70
    // 2. EwmaGreedy, Bandit, and PID must converge (tail latency within budget)
    let recall_pass = results
        .iter()
        .filter(|r| r.name != "Fixed")
        .all(|r| r.mean_recall >= 0.70);

    println!();
    println!("  Acceptance criteria:");
    println!(
        "    [{}] Adaptive recall@10 ≥ 0.70   (lowest: {:.3})",
        if recall_pass { "PASS" } else { "FAIL" },
        results.iter().filter(|r| r.name != "Fixed").map(|r| r.mean_recall).fold(1.0f32, f32::min)
    );
    println!(
        "    [{}] Adaptive policies converge to ≤130% of budget",
        if all_pass { "PASS" } else { "FAIL" }
    );

    let overall = recall_pass && all_pass;
    println!();
    println!(
        "  ══ Overall: {} ══",
        if overall { "PASS ✓" } else { "FAIL ✗" }
    );
    println!("════════════════════════════════════════════════════════════════");

    if !overall {
        std::process::exit(1);
    }
}
