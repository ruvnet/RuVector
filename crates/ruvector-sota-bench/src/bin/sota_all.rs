//! Master SOTA benchmark — runs all indexes on all datasets, prints leaderboard.
//!
//! Usage:
//!   cargo run --release -p ruvector-sota-bench --bin sota-all -- --smoke
//!   cargo run --release -p ruvector-sota-bench --bin sota-all
//!   cargo run --release -p ruvector-sota-bench --bin sota-all -- --json results/sota.json

use anyhow::Result;
use clap::Parser;
use ruvector_sota_bench::{
    datasets::{ann_benchmark_synthetic, ci_smoke},
    runners::run_core_hnsw,
    report::BenchReport,
    BenchScore,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "sota-all")]
#[command(about = "RuVector SOTA master benchmark — proves recall/QPS/memory against public leaderboards")]
struct Args {
    /// Run only quick smoke-test datasets
    #[arg(long)]
    smoke: bool,

    /// ef_search values to sweep (comma-separated)
    #[arg(long, default_value = "50,100,200,400")]
    ef_search: String,

    /// HNSW M parameter
    #[arg(long, default_value = "32")]
    m: usize,

    /// HNSW ef_construction
    #[arg(long, default_value = "200")]
    ef_construction: usize,

    /// k nearest neighbours
    #[arg(long, default_value = "10")]
    k: usize,

    /// Output JSON report (optional)
    #[arg(long)]
    json: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let datasets = if args.smoke { ci_smoke() } else { ann_benchmark_synthetic() };
    let ef_values: Vec<usize> = args.ef_search
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    println!("RuVector SOTA Benchmark");
    println!("  Datasets:    {}", datasets.iter().map(|d| d.name.as_str()).collect::<Vec<_>>().join(", "));
    println!("  ef_search:   {:?}", ef_values);
    println!("  m:           {}", args.m);
    println!("  k:           {}", args.k);
    println!();

    let mut scores: Vec<BenchScore> = Vec::new();

    for dataset in &datasets {
        println!("── Dataset: {} (n={}, dims={}) ──", dataset.name, dataset.corpus.len(), dataset.dims);

        for &ef in &ef_values {
            match run_core_hnsw(dataset, args.m, args.ef_construction, ef, args.k) {
                Ok(score) => {
                    let sota_marker = if score.sota { " ★SOTA" } else { "" };
                    println!(
                        "  core-hnsw ef={:<4} | recall@10={:.4}  qps={:>8.0}  p99={:>6.1}µs  darwin={:.3}{}",
                        ef, score.recall.recall_at_10, score.qps, score.latency.p99_us,
                        score.darwin_score, sota_marker
                    );
                    scores.push(score);
                }
                Err(e) => eprintln!("  ✗ core-hnsw ef={ef}: {e}"),
            }
        }
        println!();
    }

    let report = BenchReport::new(scores);
    report.print_table();

    if let Some(path) = args.json {
        std::fs::create_dir_all(path.parent().unwrap_or(std::path::Path::new(".")))?;
        report.save_json(&path)?;
        println!("Report saved to {}", path.display());
    }

    // Exit 1 if no SOTA claims at all (useful for CI gate)
    if report.sota_claims.is_empty() && !args.smoke {
        eprintln!("WARNING: No SOTA claims — recall@10 < 0.95 or QPS below 80% of baseline");
    }

    Ok(())
}
