//! BigANN Streaming track benchmark: recall during active insertions.
//!
//! Models the NeurIPS'23 streaming track winner (0.887 averaged recall).
//! Target: match or beat 0.887 recall on the LSM-ANN FullLsm variant.
//!
//! Run: cargo run --release -p ruvector-sota-bench --bin sota-streaming

use anyhow::Result;
use clap::Parser;
use ruvector_sota_bench::{
    datasets::{ann_benchmark_synthetic, ci_smoke},
    runners::{run_lsm_ann, run_lsm_streaming},
};

#[derive(Parser)]
#[command(name = "sota-streaming")]
struct Args {
    #[arg(long)]
    smoke: bool,
    #[arg(long, default_value = "10")]
    k: usize,
    #[arg(long, default_value = "1000")]
    l0_max: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let datasets = if args.smoke {
        ci_smoke()
    } else {
        ann_benchmark_synthetic()
    };

    println!("RuVector — BigANN Streaming Track Benchmark");
    println!("  NeurIPS'23 target: 0.887 averaged recall during active inserts");
    println!("  RuVector using FullLsm (MemTable + L1 NSW segments + L2 merged)\n");

    let mut total_recall = 0.0f64;
    let mut n_checkpoints = 0usize;

    for dataset in &datasets {
        println!(
            "── {} (n={}, dims={}) ──",
            dataset.name,
            dataset.corpus.len(),
            dataset.dims
        );

        // 1. Streaming checkpoints (recall at 25% / 50% / 100% fill)
        println!("  Streaming recall during insertion:");
        match run_lsm_streaming(dataset, args.k) {
            Ok(checkpoints) => {
                for (fill_pct, recall, mem_mb) in &checkpoints {
                    let status = if *recall >= 0.887 {
                        "✓ beats NeurIPS target"
                    } else {
                        "✗ below target"
                    };
                    println!(
                        "    fill={:5.1}%  recall@10={:.4}  mem={:.1}MB  {}",
                        fill_pct, recall, mem_mb, status
                    );
                    total_recall += recall;
                    n_checkpoints += 1;
                }
            }
            Err(e) => eprintln!("  ✗ streaming: {e}"),
        }

        // 2. Full build + query (post-compaction)
        println!("  Post-compaction (static) performance:");
        match run_lsm_ann(dataset, args.k, args.l0_max) {
            Ok(s) => println!(
                "    {}  recall@10={:.4}  qps={:.0}  mem={:.1}MB{}",
                s.index,
                s.recall.recall_at_10,
                s.qps,
                s.memory_mb,
                if s.sota { "  ★SOTA" } else { "" }
            ),
            Err(e) => eprintln!("  ✗ lsm static: {e}"),
        }
        println!();
    }

    if n_checkpoints > 0 {
        let avg = total_recall / n_checkpoints as f64;
        println!("Averaged recall across all checkpoints: {:.4}", avg);
        if avg >= 0.887 {
            println!("★ BEATS NeurIPS'23 streaming track target (0.887)");
        } else {
            println!("  Below NeurIPS'23 target — increase ef_construction or l0_max");
        }
    }

    Ok(())
}
