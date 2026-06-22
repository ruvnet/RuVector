//! ANN-Benchmarks sweep: recall@10 vs QPS Pareto front.
use anyhow::Result;
use clap::Parser;
use ruvector_sota_bench::{datasets::ann_benchmark_synthetic, runners::run_core_hnsw};

#[derive(Parser)]
#[command(name = "sota-ann")]
struct Args {
    #[arg(long, default_value = "32")]
    m: usize,
    #[arg(long, default_value = "200")]
    ef_construction: usize,
    #[arg(long, default_value = "10,20,50,100,200,400,800")]
    ef_search: String,
    #[arg(long, default_value = "10")]
    k: usize,
    #[arg(long)]
    smoke: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let datasets = if args.smoke {
        ruvector_sota_bench::smoke_test_datasets()
    } else {
        ann_benchmark_synthetic()
    };
    let ef_values: Vec<usize> = args
        .ef_search
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    println!("System,Dataset,ef_search,recall@10,qps,p50_us,p99_us,memory_mb,darwin_score");
    for d in &datasets {
        for &ef in &ef_values {
            if let Ok(s) = run_core_hnsw(d, args.m, args.ef_construction, ef, args.k) {
                println!(
                    "core-hnsw,{},{},{:.5},{:.1},{:.1},{:.1},{:.1},{:.4}",
                    d.name,
                    ef,
                    s.recall.recall_at_10,
                    s.qps,
                    s.latency.p50_us,
                    s.latency.p99_us,
                    s.memory_mb,
                    s.darwin_score
                );
            }
        }
    }
    Ok(())
}
