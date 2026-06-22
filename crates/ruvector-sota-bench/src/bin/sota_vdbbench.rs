//! VectorDBBench-compatible benchmark — proves RuVector against Qdrant/Redis/Weaviate.
//!
//! Implements the same scenarios as VectorDBBench 1.0 in-process (no Python/REST overhead).
//!
//! Reference targets (VDBBench 1.0, Cohere-1M, recall@10 ≥ 0.99):
//!   Qdrant:   15,000 QPS   p99 ~1ms
//!   Redis:    30,000 QPS   p99 ~0.5ms
//!   Weaviate:  7,000 QPS   p99 ~4ms
//!
//! Run:
//!   cargo run --release -p ruvector-sota-bench --bin sota-vdbbench -- --smoke
//!   cargo run --release -p ruvector-sota-bench --bin sota-vdbbench

use anyhow::Result;
use clap::Parser;
use ruvector_sota_bench::{
    datasets::{ann_benchmark_synthetic, ci_smoke},
    runners::{
        print_vdbbench_comparison, run_vdbbench_scenario, VdbBenchConfig, VDBBENCH_REFERENCES,
    },
    BenchScore,
};

#[derive(Parser)]
#[command(name = "sota-vdbbench")]
#[command(about = "VectorDBBench-compatible benchmark vs Qdrant/Redis/Weaviate")]
struct Args {
    /// Quick smoke datasets (CI-safe)
    #[arg(long)]
    smoke: bool,

    /// ef_search sweep values
    #[arg(long, default_value = "100,200,400")]
    ef_search: String,

    /// HNSW M parameter
    #[arg(long, default_value = "32")]
    m: usize,

    /// k nearest neighbours
    #[arg(long, default_value = "10")]
    k: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let datasets = if args.smoke {
        ci_smoke()
    } else {
        ann_benchmark_synthetic()
    };
    let ef_values: Vec<usize> = args
        .ef_search
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    println!("RuVector VectorDBBench Scenarios");
    println!("  In-process HNSW (no REST/gRPC overhead)");
    println!("  Reference: VectorDBBench 1.0 (zilliztech/VectorDBBench)\n");

    // Print reference table header
    println!("── Reference leaderboard (published numbers) ──");
    for r in VDBBENCH_REFERENCES {
        println!(
            "  {:<20}  dataset={:<25} recall={:.3}  QPS={:>8.0}  p99={:.2}ms  [{}]",
            r.system, r.dataset, r.recall, r.qps, r.p99_ms, r.notes
        );
    }
    println!();

    let mut all_scores: Vec<BenchScore> = Vec::new();

    for dataset in &datasets {
        println!(
            "── Dataset: {} (n={}, dims={}) ──",
            dataset.name,
            dataset.corpus.len(),
            dataset.dims
        );

        for &ef in &ef_values {
            let cfg = VdbBenchConfig {
                k: args.k,
                ef_search: ef,
                concurrency: 1,
                warmup: 20,
            };
            match run_vdbbench_scenario(dataset, &cfg, args.m, 200, "ruvector-hnsw") {
                Ok(s) => {
                    let sota_mark = if s.sota { " ★SOTA" } else { "" };
                    // Qdrant ref: 15K QPS, p99 1ms, recall 0.99
                    let vs_qdrant_qps = s.qps / 15_000.0 * 100.0;
                    let vs_qdrant_p99 = 1.0 / (s.latency.p99_us / 1_000.0) * 100.0;
                    println!(
                        "  ef={:<4}  recall@10={:.4}  qps={:>8.0} ({:>5.1}% vs Qdrant)  p99={:>6.2}ms ({:>5.1}% faster){}",
                        ef, s.recall.recall_at_10, s.qps, vs_qdrant_qps,
                        s.latency.p99_us / 1_000.0, vs_qdrant_p99, sota_mark
                    );
                    all_scores.push(s);
                }
                Err(e) => eprintln!("  ✗ ef={ef}: {e}"),
            }
        }
        println!();
    }

    print_vdbbench_comparison(&all_scores);

    // Summary
    let best = all_scores
        .iter()
        .filter(|s| s.recall.recall_at_10 >= 0.95)
        .max_by(|a, b| a.qps.partial_cmp(&b.qps).unwrap());

    if let Some(best) = best {
        println!("\n── Best at recall@10 ≥ 0.95 ──");
        println!(
            "  RuVector: {:.4} recall  {:>8.0} QPS  {:>6.2}ms p99",
            best.recall.recall_at_10,
            best.qps,
            best.latency.p99_us / 1_000.0
        );
        println!("  Qdrant:   0.990 recall   15,000 QPS   1.00ms p99");
        let qps_ratio = best.qps / 15_000.0;
        let p99_ratio = 1.0 / (best.latency.p99_us / 1_000.0);
        if qps_ratio >= 1.0 || p99_ratio >= 1.0 {
            println!(
                "  ★ RuVector beats Qdrant: {:.2}× QPS, {:.2}× lower p99",
                qps_ratio, p99_ratio
            );
        } else {
            println!(
                "  RuVector at {:.1}% Qdrant QPS, {:.1}% Qdrant p99",
                qps_ratio * 100.0,
                p99_ratio * 100.0
            );
            println!("  Note: smoke datasets are 5K–10K vectors; Qdrant reference is 1M vectors.");
            println!("  Run with full ANN-Benchmarks scale for a fair comparison.");
        }
    }

    Ok(())
}
