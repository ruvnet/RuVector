//! Namespace-Merge MinCut benchmark binary.
//!
//! Measures three namespace routing strategies on a 5-namespace clustered dataset:
//!   1. AllSearch      – brute-force scan of all namespaces (ground truth)
//!   2. CentroidFilter – skip namespaces below cosine threshold (heuristic)
//!   3. MinCutRoute    – S-T mincut partition on the namespace graph (principled)
//!
//! Run:
//!   cargo run --release -p ruvector-namespace-merge --bin benchmark
//!
//! Environment overrides:
//!   PER_NS=1000 DIMS=64 N_QUERIES=200 THRESHOLD=0.4

use ruvector_namespace_merge::{
    dataset::{Dataset, DatasetConfig},
    recall_at_k,
    router::{AllSearch, CentroidFilter, MinCutRoute, NamespaceRouter},
    Hit,
};
use std::time::Instant;

fn per_ns() -> usize {
    std::env::var("PER_NS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500)
}
fn dims() -> usize {
    std::env::var("DIMS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64)
}
fn n_queries() -> usize {
    std::env::var("N_QUERIES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300)
}
fn threshold() -> f32 {
    std::env::var("THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.35)
}

const K: usize = 10;
const SEED: u64 = 0xF00D_CAFE_1234_5678;

// ─── acceptance criteria ─────────────────────────────────────────────────────

const MIN_RECALL_CENTROID: f32 = 0.80;
const MIN_RECALL_MINCUT: f32 = 0.80;
const MAX_DIST_OPS_CENTROID_FRAC: f64 = 0.70; // ≤70% of AllSearch dist ops
const MAX_DIST_OPS_MINCUT_FRAC: f64 = 0.60; // ≤60% of AllSearch dist ops

// ─── stat helpers ─────────────────────────────────────────────────────────────

fn percentile(sorted: &[u128], p: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn mean(vals: &[u128]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.iter().sum::<u128>() as f64 / vals.len() as f64
}

// ─── run one variant ─────────────────────────────────────────────────────────

struct Stats {
    name: String,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    qps: f64,
    recall: f64,
    avg_ns_searched: f64,
    avg_dist_ops: f64,
    memory_kb: usize,
    pass: bool,
}

fn run_variant(
    router: &dyn NamespaceRouter,
    dataset: &Dataset,
    queries: &[Vec<f32>],
    gt: &[Vec<Hit>],
    min_recall: Option<f32>,
    max_dist_frac: Option<(f64, f64)>, // (numerator frac, all_search_avg_ops)
) -> Stats {
    let mut latencies: Vec<u128> = Vec::with_capacity(queries.len());
    let mut recalls: Vec<f32> = Vec::with_capacity(queries.len());
    let mut ns_searched_sum = 0usize;
    let mut dist_ops_sum = 0usize;

    for (q, truth) in queries.iter().zip(gt.iter()) {
        let t0 = Instant::now();
        let res = router.search(dataset, q, K);
        latencies.push(t0.elapsed().as_micros());
        recalls.push(recall_at_k(&res.hits, truth, K));
        ns_searched_sum += res.ns_searched;
        dist_ops_sum += res.dist_ops;
    }

    latencies.sort_unstable();

    let mean_us = mean(&latencies);
    let p50_us = percentile(&latencies, 0.50) as f64;
    let p95_us = percentile(&latencies, 0.95) as f64;
    let total_s = latencies.iter().sum::<u128>() as f64 / 1_000_000.0;
    let qps = queries.len() as f64 / total_s.max(1e-9);
    let recall = recalls.iter().sum::<f32>() as f64 / recalls.len() as f64;
    let avg_ns = ns_searched_sum as f64 / queries.len() as f64;
    let avg_ops = dist_ops_sum as f64 / queries.len() as f64;

    let mut pass = true;
    if let Some(mr) = min_recall {
        if recall < mr as f64 {
            pass = false;
        }
    }
    if let Some((frac, all_ops)) = max_dist_frac {
        if avg_ops > all_ops * frac {
            pass = false;
        }
    }

    Stats {
        name: router.name().to_string(),
        mean_us,
        p50_us,
        p95_us,
        qps,
        recall,
        avg_ns_searched: avg_ns,
        avg_dist_ops: avg_ops,
        memory_kb: (router.memory_bytes() + 1023) / 1024,
        pass,
    }
}

// ─── print ────────────────────────────────────────────────────────────────────

fn print_header() {
    println!(
        "{:<20} {:>10} {:>10} {:>10} {:>10} {:>8} {:>10} {:>11} {:>9} {:>6}",
        "Variant",
        "Mean(µs)",
        "p50(µs)",
        "p95(µs)",
        "QPS",
        "Recall",
        "NS searched",
        "Dist ops",
        "Mem(KB)",
        "Pass?"
    );
    println!("{}", "-".repeat(110));
}

fn print_row(s: &Stats) {
    println!(
        "{:<20} {:>10.1} {:>10.0} {:>10.0} {:>10.0} {:>8.4} {:>10.2} {:>11.0} {:>9} {:>6}",
        s.name,
        s.mean_us,
        s.p50_us,
        s.p95_us,
        s.qps,
        s.recall,
        s.avg_ns_searched,
        s.avg_dist_ops,
        s.memory_kb,
        if s.pass { "PASS" } else { "FAIL" }
    );
}

// ─── main ─────────────────────────────────────────────────────────────────────

fn main() {
    // ── system info ──────────────────────────────────────────────────────────
    println!("=== Namespace-Merge MinCut Benchmark ===");
    println!("OS:           {}", std::env::consts::OS);
    println!("Arch:         {}", std::env::consts::ARCH);
    println!("Rust version: (check via `rustc --version`)");
    println!();

    // ── dataset ──────────────────────────────────────────────────────────────
    let per_ns = per_ns();
    let dims = dims();
    let n_queries = n_queries();
    let threshold = threshold();

    println!("Dataset:");
    println!("  Namespaces:   5 (groups A×2, B×2, C×1)");
    println!("  Vectors/NS:   {per_ns}");
    println!("  Total vecs:   {}", 5 * per_ns);
    println!("  Dimensions:   {dims}");
    println!("  Queries:      {n_queries} (targeted at group A)");
    println!("  k:            {K}");
    println!("  CF threshold: {threshold:.2}");
    println!();

    let cfg = DatasetConfig {
        per_ns,
        dims,
        seed: SEED,
        noise: 0.30,
    };
    let dataset = Dataset::generate(&cfg);
    let queries = dataset.group_a_queries(n_queries, SEED ^ 0xABCD);

    // ── ground truth (AllSearch) ──────────────────────────────────────────────
    let all_search = AllSearch;
    let gt: Vec<Vec<Hit>> = queries
        .iter()
        .map(|q| all_search.search(&dataset, q, K).hits)
        .collect();

    // ── run variants ─────────────────────────────────────────────────────────
    // Measure AllSearch avg dist ops for the fraction check
    let all_stats = run_variant(&all_search, &dataset, &queries, &gt, None, None);
    let all_ops = all_stats.avg_dist_ops;

    let cf = CentroidFilter::new(threshold);
    let cf_stats = run_variant(
        &cf,
        &dataset,
        &queries,
        &gt,
        Some(MIN_RECALL_CENTROID),
        Some((MAX_DIST_OPS_CENTROID_FRAC, all_ops)),
    );

    let mc = MinCutRoute::new(&dataset);
    let mc_stats = run_variant(
        &mc,
        &dataset,
        &queries,
        &gt,
        Some(MIN_RECALL_MINCUT),
        Some((MAX_DIST_OPS_MINCUT_FRAC, all_ops)),
    );

    // ── results ───────────────────────────────────────────────────────────────
    println!("Results:");
    print_header();
    print_row(&all_stats);
    print_row(&cf_stats);
    print_row(&mc_stats);
    println!();

    // ── acceptance summary ───────────────────────────────────────────────────
    println!("Acceptance criteria:");
    println!(
        "  CentroidFilter recall ≥ {:.0}%:  {:>8.4} → {}",
        MIN_RECALL_CENTROID * 100.0,
        cf_stats.recall,
        if cf_stats.recall >= MIN_RECALL_CENTROID as f64 {
            "PASS"
        } else {
            "FAIL"
        }
    );
    println!(
        "  MinCutRoute    recall ≥ {:.0}%:  {:>8.4} → {}",
        MIN_RECALL_MINCUT * 100.0,
        mc_stats.recall,
        if mc_stats.recall >= MIN_RECALL_MINCUT as f64 {
            "PASS"
        } else {
            "FAIL"
        }
    );
    println!(
        "  CentroidFilter dist ops ≤ {:.0}% of AllSearch: {:>6.0} / {:>6.0} → {}",
        MAX_DIST_OPS_CENTROID_FRAC * 100.0,
        cf_stats.avg_dist_ops,
        all_ops,
        if cf_stats.avg_dist_ops <= all_ops * MAX_DIST_OPS_CENTROID_FRAC {
            "PASS"
        } else {
            "FAIL"
        }
    );
    println!(
        "  MinCutRoute    dist ops ≤ {:.0}% of AllSearch: {:>6.0} / {:>6.0} → {}",
        MAX_DIST_OPS_MINCUT_FRAC * 100.0,
        mc_stats.avg_dist_ops,
        all_ops,
        if mc_stats.avg_dist_ops <= all_ops * MAX_DIST_OPS_MINCUT_FRAC {
            "PASS"
        } else {
            "FAIL"
        }
    );
    println!();

    let overall = cf_stats.pass && mc_stats.pass;
    println!(
        "Overall: {}",
        if overall {
            "ALL ACCEPTANCE CRITERIA PASSED"
        } else {
            "ONE OR MORE CRITERIA FAILED"
        }
    );

    if !overall {
        std::process::exit(1);
    }
}
