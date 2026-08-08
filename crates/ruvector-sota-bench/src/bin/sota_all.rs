//! Master SOTA benchmark — runs all available runners on all datasets.
//!
//! Runners included:
//!   1. core-hnsw       — ruvector-core HNSW at multiple ef_search values
//!   2. matryoshka      — FullDim + TwoStage coarse-to-fine funnel
//!   3. hybrid-rrf/rsf  — BM25 + ANN with RRF / RSF / score-fusion
//!
//! Usage:
//!   cargo run --release -p ruvector-sota-bench --bin sota-all -- --smoke
//!   cargo run --release -p ruvector-sota-bench --bin sota-all -- --json results/sota.json

use anyhow::{bail, Result};
use clap::Parser;
use ruvector_sota_bench::{
    datasets::{
        ann_benchmark_synthetic_with_seed, ci_smoke_with_seed, load_ann_dataset, ANN_DATASETS,
    },
    report::BenchReport,
    runners::{
        run_core_hnsw, run_hybrid_suite, run_lsm_ann, run_matryoshka_suite, run_rabitq_suite,
    },
    BenchScore,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "sota-all")]
#[command(
    about = "RuVector SOTA master benchmark — proves recall/QPS/memory vs public leaderboards"
)]
struct Args {
    /// Quick smoke-test datasets only (CI-safe, < 30s)
    #[arg(long)]
    smoke: bool,

    /// Load the named ANN-Benchmarks HDF5 dataset instead of a synthetic
    /// distribution. Requires the `real-datasets` feature and --dataset.
    #[arg(long, requires = "dataset")]
    real: bool,

    /// Cache for real ANN-Benchmarks datasets.
    #[arg(long, default_value = "data/ann-benchmarks")]
    dataset_cache: PathBuf,

    /// Corpus cap for real dataset runs.
    #[arg(long, default_value = "100000")]
    max_corpus: usize,

    /// Query cap for real dataset runs.
    #[arg(long, default_value = "1000")]
    max_queries: usize,

    /// Idle window used by the external harness to sample process baseline RSS
    /// before dataset allocation begins.
    #[arg(long, default_value = "0")]
    measurement_start_delay_ms: u64,

    /// Deterministic dataset seed. Confirmation and frozen-anchor suites must
    /// use disjoint seed sets.
    #[arg(long, default_value = "0")]
    seed: u64,

    /// Run one named dataset instead of the complete selected dataset family.
    #[arg(long)]
    dataset: Option<String>,

    /// HNSW ef_search values to sweep
    #[arg(long, default_value = "50,100,200,400")]
    ef_search: String,

    /// HNSW M parameter
    #[arg(long, default_value = "32")]
    m: usize,

    /// HNSW ef_construction
    #[arg(long, default_value = "200")]
    ef_construction: usize,

    /// k nearest neighbours to retrieve
    #[arg(long, default_value = "10")]
    k: usize,

    /// Skip matryoshka runners (faster, focuses on core-hnsw)
    #[arg(long)]
    no_matryoshka: bool,

    /// Skip hybrid runners (BM25+ANN)
    #[arg(long)]
    no_hybrid: bool,

    /// Skip LSM-ANN streaming runner
    #[arg(long)]
    no_lsm: bool,

    /// Skip RaBitQ 1-bit compressed runners
    #[arg(long)]
    no_rabitq: bool,

    /// Output JSON report path
    #[arg(long)]
    json: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.measurement_start_delay_ms > 0 {
        std::thread::sleep(std::time::Duration::from_millis(
            args.measurement_start_delay_ms,
        ));
    }

    let mut datasets = if args.real {
        let name = args.dataset.as_deref().expect("clap requires dataset");
        let Some(spec) = ANN_DATASETS.iter().find(|spec| spec.name == name) else {
            bail!("unknown real ANN-Benchmarks dataset {name:?}");
        };
        let mut dataset =
            load_ann_dataset(spec, &args.dataset_cache, args.max_corpus, args.max_queries)?;
        // Seeded query rotation creates paired, reproducible query samples
        // without changing the immutable real corpus or its ground truth.
        if !dataset.queries.is_empty() {
            let offset = args.seed as usize % dataset.queries.len();
            dataset.queries.rotate_left(offset);
            dataset.ground_truth.rotate_left(offset);
        }
        vec![dataset]
    } else if args.smoke {
        ci_smoke_with_seed(args.seed)
    } else {
        ann_benchmark_synthetic_with_seed(args.seed)
    };
    if !args.real {
        if let Some(name) = &args.dataset {
            datasets.retain(|dataset| dataset.name == *name);
            if datasets.is_empty() {
                bail!("unknown dataset {name:?} for the selected benchmark mode");
            }
        }
    }
    let ef_values: Vec<usize> = args
        .ef_search
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    println!("RuVector SOTA Benchmark");
    println!(
        "  Mode:      {}",
        if args.real {
            "real ANN-Benchmarks"
        } else if args.smoke {
            "smoke (synthetic, fast)"
        } else {
            "full (synthetic ANN-Benchmarks scale)"
        }
    );
    println!(
        "  Datasets:  {}",
        datasets
            .iter()
            .map(|d| d.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("  ef_search: {:?}", ef_values);
    println!("  Seed:      {}", args.seed);
    println!();

    let mut scores: Vec<BenchScore> = Vec::new();

    for dataset in &datasets {
        println!(
            "── Dataset: {} (n={}, dims={}) ──",
            dataset.name,
            dataset.corpus.len(),
            dataset.dims
        );

        // 1. core-hnsw sweep
        for &ef in &ef_values {
            match run_core_hnsw(dataset, args.m, args.ef_construction, ef, args.k) {
                Ok(s) => {
                    println!("  core-hnsw ef={:<4} | recall@10={:.4}  qps={:>8.0}  p99={:>6.1}µs  darwin={:.3}{}",
                        ef, s.recall.recall_at_10, s.qps, s.latency.p99_us,
                        s.darwin_score, if s.sota { " ★SOTA" } else { "" });
                    scores.push(s);
                }
                Err(e) => eprintln!("  ✗ core-hnsw ef={ef}: {e}"),
            }
        }

        // 2. matryoshka funnel — MRL-structured dataset (fixes #597)
        if !args.no_matryoshka {
            let ef = *ef_values.last().unwrap_or(&400);
            for s in run_matryoshka_suite(
                &dataset.name,
                dataset.corpus.len(),
                dataset.dims,
                args.k,
                ef,
            ) {
                println!(
                    "  {:<26} | recall@10={:.4}  qps={:>8.0}  p99={:>6.1}µs  darwin={:.3}{}",
                    s.index,
                    s.recall.recall_at_10,
                    s.qps,
                    s.latency.p99_us,
                    s.darwin_score,
                    if s.sota { " ★SOTA" } else { "" }
                );
                scores.push(s);
            }
        }

        // 3. RaBitQ 1-bit compressed ANN (primary SOTA claim vs IVF-PQ)
        if !args.no_rabitq {
            for s in run_rabitq_suite(dataset, args.k) {
                match s {
                    Ok(s) => {
                        println!("  {:<26} | recall@10={:.4}  qps={:>8.0}  p99={:>6.1}µs  darwin={:.3}{}",
                            s.index, s.recall.recall_at_10, s.qps, s.latency.p99_us,
                            s.darwin_score, if s.sota { " ★SOTA" } else { "" });
                        scores.push(s);
                    }
                    Err(e) => eprintln!("  ✗ rabitq: {e}"),
                }
            }
        }

        // 4. LSM-ANN streaming index
        if !args.no_lsm {
            match run_lsm_ann(dataset, args.k, 500) {
                Ok(s) => {
                    println!(
                        "  {:<26} | recall@10={:.4}  qps={:>8.0}  p99={:>6.1}µs  darwin={:.3}{}",
                        s.index,
                        s.recall.recall_at_10,
                        s.qps,
                        s.latency.p99_us,
                        s.darwin_score,
                        if s.sota { " ★SOTA" } else { "" }
                    );
                    scores.push(s);
                }
                Err(e) => eprintln!("  ✗ lsm-ann: {e}"),
            }
        }

        // 4. hybrid (BM25 + ANN fusion)
        if !args.no_hybrid {
            for s in run_hybrid_suite(dataset, args.k) {
                println!(
                    "  {:<26} | recall@10={:.4}  qps={:>8.0}  p99={:>6.1}µs  darwin={:.3}{}",
                    s.index,
                    s.recall.recall_at_10,
                    s.qps,
                    s.latency.p99_us,
                    s.darwin_score,
                    if s.sota { " ★SOTA" } else { "" }
                );
                scores.push(s);
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

    // Print Darwin score summary — highest score first
    let mut darwin_ranked: Vec<_> = report.scores.iter().collect();
    darwin_ranked.sort_by(|a, b| b.darwin_score.partial_cmp(&a.darwin_score).unwrap());
    if !darwin_ranked.is_empty() {
        println!("\n── Darwin Mode Score Ranking (higher = better for evolution) ──");
        for (i, s) in darwin_ranked.iter().take(5).enumerate() {
            println!(
                "  #{} {:<30} darwin={:.4}  recall@10={:.4}  qps={:.0}",
                i + 1,
                format!("{} on {}", s.index, s.dataset),
                s.darwin_score,
                s.recall.recall_at_10,
                s.qps
            );
        }
    }

    Ok(())
}
