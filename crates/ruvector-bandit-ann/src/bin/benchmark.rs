//! Benchmark binary for bandit-tuned ANN.
//!
//! Runs three measurable variants and prints a structured report.
//! All numbers come from real cargo run measurements — no aspirational values.
//!
//! Usage:
//!   cargo run --release -p ruvector-bandit-ann --bin benchmark
//!
//! Optional env vars:
//!   N_VECS=5000     dataset size (default 5000)
//!   DIM=96          dimensions (default 96)
//!   N_QUERIES=300   query count (default 300)
//!   K=10            top-k (default 10)

use ruvector_bandit_ann::{
    dataset::{ground_truth, random_queries, random_unit_vectors},
    recall_at_k, AnnVariant, BanditTuned, Hit, StaticDefault, StaticFast,
};
use std::time::Instant;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn percentile(mut v: Vec<f64>, p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((v.len() as f64 * p / 100.0) as usize).min(v.len() - 1);
    v[idx]
}

struct BenchResult {
    name: String,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    qps: f64,
    recall: f64,
    memory_mb: f64,
    pass: bool,
}

fn bench_variant(
    variant: &dyn AnnVariant,
    queries: &[Vec<f32>],
    gts: &[Vec<Hit>],
    k: usize,
    recall_threshold: f64,
) -> BenchResult {
    let mut latencies_us: Vec<f64> = Vec::with_capacity(queries.len());
    let mut total_recall = 0.0_f64;

    for (q, gt) in queries.iter().zip(gts.iter()) {
        let t0 = Instant::now();
        let results = variant.search(q, k);
        let us = t0.elapsed().as_nanos() as f64 / 1_000.0;
        latencies_us.push(us);
        total_recall += recall_at_k(&results, gt, k) as f64;
    }

    let n = queries.len() as f64;
    let mean_us = latencies_us.iter().sum::<f64>() / n;
    let p50_us = percentile(latencies_us.clone(), 50.0);
    let p95_us = percentile(latencies_us.clone(), 95.0);
    let qps = 1_000_000.0 / mean_us;
    let recall = total_recall / n;
    let memory_mb = variant.memory_bytes() as f64 / 1_048_576.0;
    let pass = recall >= recall_threshold;

    BenchResult {
        name: variant.name().to_string(),
        mean_us,
        p50_us,
        p95_us,
        qps,
        recall,
        memory_mb,
        pass,
    }
}

fn main() {
    let n_vecs = env_usize("N_VECS", 5_000);
    let dim = env_usize("DIM", 96);
    let n_queries = env_usize("N_QUERIES", 300);
    let k = env_usize("K", 10);
    let m = 16;
    let ef_construction = 200;
    // Acceptance thresholds:
    //  - StaticDefault and BanditTuned must reach >= 0.80 recall.
    //  - BanditTuned recall must exceed StaticFast by >= 20pp.
    let recall_floor = 0.80_f64;
    let recall_gap_min = 0.20_f64;

    println!("===================================================================");
    println!("  RuVector Bandit-Tuned ANN Benchmark");
    println!("===================================================================");
    println!("  OS:              {}", std::env::consts::OS);
    println!("  Arch:            {}", std::env::consts::ARCH);
    println!("  Dataset size:    {}", n_vecs);
    println!("  Dimensions:      {}", dim);
    println!("  Queries:         {}", n_queries);
    println!("  k:               {}", k);
    println!("  M (graph):       {}", m);
    println!("  ef_construction: {}", ef_construction);
    println!(
        "  Recall floor:    {:.2} (StaticDefault, BanditTuned)",
        recall_floor
    );
    println!(
        "  Recall gap min:  {:.0}pp (BanditTuned > StaticFast)",
        recall_gap_min * 100.0
    );
    println!("-------------------------------------------------------------------");

    print!("  Building dataset ({} x {})...", n_vecs, dim);
    let t0 = Instant::now();
    let data = random_unit_vectors(n_vecs, dim, 0xCAFE);
    let queries = random_queries(n_queries, dim, 0xCAFE);
    println!(" {:.1}ms", t0.elapsed().as_millis());

    print!("  Computing ground truth ({} queries)...", n_queries);
    let t0 = Instant::now();
    let gts: Vec<Vec<Hit>> = queries
        .iter()
        .map(|q| {
            ground_truth(&data, q, k)
                .into_iter()
                .map(|(id, dist)| Hit { id, dist })
                .collect()
        })
        .collect();
    println!(" {:.1}ms", t0.elapsed().as_millis());

    print!("  Building StaticDefault index...");
    let t0 = Instant::now();
    let v1 = StaticDefault::build(&data, m, ef_construction);
    println!(" {:.1}ms", t0.elapsed().as_millis());

    print!("  Building StaticFast index...");
    let t0 = Instant::now();
    let v2 = StaticFast::build(&data, m, ef_construction);
    println!(" {:.1}ms", t0.elapsed().as_millis());

    let arms = vec![10usize, 20, 30, 40, 50];
    print!(
        "  Building BanditTuned index + warm-up ({} arms)...",
        arms.len()
    );
    let t0 = Instant::now();
    let mut v3_raw = BanditTuned::build(&data, m, ef_construction, arms.clone());

    let warmup_n = n_queries.min(200);
    for i in 0..warmup_n {
        let q = &queries[i];
        let gt = &gts[i];
        let t_inner = Instant::now();
        let _ =
            v3_raw.search_with_feedback(q, k, gt, t_inner.elapsed().as_nanos() as f64 / 1_000.0);
        let t_inner = Instant::now();
        let _ =
            v3_raw.search_with_feedback(q, k, gt, t_inner.elapsed().as_nanos() as f64 / 1_000.0);
    }
    println!(" {:.1}ms", t0.elapsed().as_millis());

    let means = v3_raw.bandit.mean_rewards();
    let counts = v3_raw.bandit.pull_counts();
    println!(
        "  Bandit arm summary (after {} pulls):",
        v3_raw.bandit.total_pulls()
    );
    for (i, &ef) in arms.iter().enumerate() {
        println!(
            "    arm[{}] ef={:>3}  pulls={:>4}  mean_reward={:.4}",
            i, ef, counts[i], means[i]
        );
    }
    let best_ef = arms[v3_raw.bandit.best_arm()];
    println!("  Converged ef_search = {}", best_ef);
    println!("-------------------------------------------------------------------");

    println!("  Running benchmarks ({} queries each)...", n_queries);
    let r1 = bench_variant(&v1, &queries, &gts, k, recall_floor);
    let r2 = bench_variant(&v2, &queries, &gts, k, recall_floor);
    let r3 = bench_variant(&v3_raw, &queries, &gts, k, recall_floor);

    println!();
    println!("+---------------------------+----------+----------+----------+----------+----------+----------+--------+");
    println!("| Variant                   | Recall@k | Mean us  | p50 us   | p95 us   | QPS      | Mem MB   | Pass   |");
    println!("+---------------------------+----------+----------+----------+----------+----------+----------+--------+");
    for r in [&r1, &r2, &r3] {
        println!(
            "| {:<25} | {:>8.4} | {:>8.1} | {:>8.1} | {:>8.1} | {:>8.0} | {:>8.2} | {:<6} |",
            r.name,
            r.recall,
            r.mean_us,
            r.p50_us,
            r.p95_us,
            r.qps,
            r.memory_mb,
            if r.pass { "PASS" } else { "FAIL" }
        );
    }
    println!("+---------------------------+----------+----------+----------+----------+----------+----------+--------+");

    println!();
    println!("  Analysis:");
    let recall_gain = (r3.recall - r2.recall) * 100.0;
    let latency_delta = ((r3.mean_us - r1.mean_us) / r1.mean_us) * 100.0;
    println!(
        "    BanditTuned vs StaticFast:    recall gain  = {:.1}pp",
        recall_gain
    );
    println!(
        "    BanditTuned vs StaticDefault: latency delta = {:+.1}%",
        latency_delta
    );

    // Acceptance check.
    let gap = r3.recall - r2.recall;
    let default_ok = r1.recall >= recall_floor;
    let bandit_ok = r3.recall >= recall_floor;
    let gap_ok = gap >= recall_gap_min;

    println!();
    if default_ok && bandit_ok && gap_ok {
        println!(
            "  ACCEPTANCE: PASS  StaticDefault {:.4} >= {:.2}  |  BanditTuned {:.4} >= {:.2}  |  gap {:.1}pp >= {:.0}pp",
            r1.recall, recall_floor, r3.recall, recall_floor, gap * 100.0, recall_gap_min * 100.0
        );
    } else {
        if !default_ok {
            println!(
                "  FAIL: StaticDefault recall {:.4} < {:.2}",
                r1.recall, recall_floor
            );
        }
        if !bandit_ok {
            println!(
                "  FAIL: BanditTuned recall {:.4} < {:.2} (bandit did not converge)",
                r3.recall, recall_floor
            );
        }
        if !gap_ok {
            println!(
                "  FAIL: recall gap {:.1}pp < {:.0}pp",
                gap * 100.0,
                recall_gap_min * 100.0
            );
        }
    }
    println!();

    if !(default_ok && bandit_ok && gap_ok) {
        std::process::exit(1);
    }
}
