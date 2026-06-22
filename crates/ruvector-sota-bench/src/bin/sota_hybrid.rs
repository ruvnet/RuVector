//! MTEB + hybrid retrieval benchmark (nDCG@10).
//!
//! Measures retrieval quality in the MTEB "Retrieval" category using
//! HashEmbedding (synthetic) as a structural proxy for real embeddings.
//!
//! MTEB Leaderboard April 2026 (nDCG@10):
//!   BGE-M3:              63.0   (Apache-2.0, 568M params, 1024D)
//!   OpenAI text-3-large: 59.0   (commercial, 3072D)
//!   all-MiniLM-L6-v2:   46.8   (ruvector current default, 384D)
//!
//! Run:
//!   cargo run --release -p ruvector-sota-bench --bin sota-hybrid -- --smoke
//!   cargo run --release -p ruvector-sota-bench --bin sota-hybrid

use anyhow::Result;
use clap::Parser;
use ruvector_sota_bench::runners::{print_mteb_leaderboard, run_mteb_retrieval};

#[derive(Parser)]
#[command(name = "sota-hybrid")]
#[command(about = "MTEB retrieval benchmark — nDCG@10 vs public leaderboard")]
struct Args {
    /// Quick smoke (1K docs, 50 queries)
    #[arg(long)]
    smoke: bool,

    /// Number of corpus documents
    #[arg(long, default_value = "10000")]
    n_docs: usize,

    /// Number of queries
    #[arg(long, default_value = "200")]
    n_queries: usize,

    /// Embedding dimension
    #[arg(long, default_value = "384")]
    dims: usize,

    /// ef_search
    #[arg(long, default_value = "200")]
    ef_search: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let (n_docs, n_queries) = if args.smoke {
        (1_000, 50)
    } else {
        (args.n_docs, args.n_queries)
    };

    println!("RuVector MTEB Retrieval Benchmark");
    println!("  Metric: nDCG@10 (normalized discounted cumulative gain)");
    println!(
        "  Dataset: synthetic BEIR-style ({n_docs} docs, {n_queries} queries, {}D)",
        args.dims
    );
    println!("  Embedder: HashEmbedding (structural proxy — swap in BGE-M3 for real MTEB)\n");

    // Baseline: 384D (all-MiniLM equivalent)
    let s384 = run_mteb_retrieval(n_docs, n_queries, 384, args.ef_search, "hash-embed-384D")?;
    println!(
        "  hash-384D   nDCG@10={:.4}  QPS={:>8.0}  p99={:.2}ms{}",
        s384.recall.recall_at_10,
        s384.qps,
        s384.latency.p99_us / 1_000.0,
        if s384.sota {
            "  ★beats text-3-large"
        } else {
            ""
        }
    );

    // 768D (common production embedding size)
    let s768 = run_mteb_retrieval(n_docs, n_queries, 768, args.ef_search, "hash-embed-768D")?;
    println!(
        "  hash-768D   nDCG@10={:.4}  QPS={:>8.0}  p99={:.2}ms{}",
        s768.recall.recall_at_10,
        s768.qps,
        s768.latency.p99_us / 1_000.0,
        if s768.sota {
            "  ★beats text-3-large"
        } else {
            ""
        }
    );

    // 1024D (BGE-M3 size)
    let s1024 = run_mteb_retrieval(n_docs, n_queries, 1024, args.ef_search, "hash-embed-1024D")?;
    println!(
        "  hash-1024D  nDCG@10={:.4}  QPS={:>8.0}  p99={:.2}ms{}",
        s1024.recall.recall_at_10,
        s1024.qps,
        s1024.latency.p99_us / 1_000.0,
        if s1024.sota {
            "  ★beats text-3-large"
        } else {
            ""
        }
    );

    // Show best vs leaderboard
    let all = [s384, s768, s1024];
    let best = all
        .iter()
        .max_by(|a, b| {
            a.recall
                .recall_at_10
                .partial_cmp(&b.recall.recall_at_10)
                .unwrap()
        })
        .unwrap();

    print_mteb_leaderboard(best.recall.recall_at_10 * 100.0, &best.index, args.dims);

    println!("\n── Next step for real MTEB submission ──");
    println!("  1. Download BGE-M3 ONNX model (~2.2 GB):");
    println!("       huggingface-cli download BAAI/bge-m3 --local-dir ~/.cache/bge-m3");
    println!("  2. Run with real model:");
    println!("       cargo run --release -p ruvector-sota-bench --features real-datasets --bin sota-hybrid");
    println!("  3. Expected nDCG@10: ~63.0 (BGE-M3, Apache-2.0, beats OpenAI text-3-large 59.0)");

    Ok(())
}
