//! Benchmark runner: latency statistics and recall measurement.

use std::time::Instant;

use crate::{recall_at_k, AnnVariant, Hit};

/// One benchmark result row.
pub struct BenchResult {
    pub variant: &'static str,
    pub n: usize,
    pub dim: usize,
    pub nqueries: usize,
    pub k: usize,
    pub mean_us: f64,
    pub p50_us: f64,
    pub p95_us: f64,
    pub qps: f64,
    pub mem_bytes: usize,
    pub mean_recall: f64,
}

/// Run `nqueries` searches and collect latency + recall statistics.
pub fn run_bench(
    variant: &dyn AnnVariant,
    queries: &[Vec<f32>],
    k: usize,
    ground_truth: &[Vec<Hit>],
    n: usize,
    dim: usize,
) -> BenchResult {
    let nq = queries.len();
    assert_eq!(nq, ground_truth.len());

    let mut latencies_us: Vec<f64> = Vec::with_capacity(nq);
    let mut total_recall = 0.0f64;

    for (q, gt) in queries.iter().zip(ground_truth.iter()) {
        let t0 = Instant::now();
        let hits = variant.search(q, k);
        let elapsed = t0.elapsed();
        latencies_us.push(elapsed.as_secs_f64() * 1_000_000.0);
        total_recall += recall_at_k(&hits, gt) as f64;
    }

    latencies_us.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_us = latencies_us.iter().sum::<f64>() / nq as f64;
    let p50_us = percentile(&latencies_us, 50.0);
    let p95_us = percentile(&latencies_us, 95.0);
    let qps = 1_000_000.0 / mean_us;

    BenchResult {
        variant: variant.name(),
        n,
        dim,
        nqueries: nq,
        k,
        mean_us,
        p50_us,
        p95_us,
        qps,
        mem_bytes: variant.mem_bytes(),
        mean_recall: total_recall / nq as f64,
    }
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((pct / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Print a single result row.
pub fn print_header() {
    println!(
        "{:<18} {:>8} {:>6} {:>6} {:>10} {:>10} {:>10} {:>10} {:>12} {:>8}",
        "Variant", "Mean µs", "p50 µs", "p95 µs", "QPS", "Memory", "Recall@K", "", "", ""
    );
    println!("{}", "-".repeat(100));
}

pub fn print_row(r: &BenchResult) {
    println!(
        "{:<18} {:>8.1} {:>6.1} {:>6.1} {:>10.0} {:>12} {:>8.3}",
        r.variant,
        r.mean_us,
        r.p50_us,
        r.p95_us,
        r.qps,
        format_bytes(r.mem_bytes),
        r.mean_recall,
    );
}

pub fn format_bytes(b: usize) -> String {
    if b >= 1_048_576 {
        format!("{:.1} MB", b as f64 / 1_048_576.0)
    } else if b >= 1024 {
        format!("{:.1} KB", b as f64 / 1024.0)
    } else {
        format!("{} B", b)
    }
}

/// Acceptance gate: all variants must exceed `min_recall`.
pub fn acceptance_gate(results: &[BenchResult], min_recall: f64) -> bool {
    results.iter().all(|r| {
        let pass = r.mean_recall >= min_recall;
        if !pass {
            eprintln!(
                "FAIL: {} recall {:.3} < threshold {:.3}",
                r.variant, r.mean_recall, min_recall
            );
        }
        pass
    })
}
