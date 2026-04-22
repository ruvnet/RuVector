//! RuVix vs Linux comprehensive comparison benchmark.
//!
//! Runs all benchmarks and generates a markdown report.
//!
//! Usage: cargo run --bin ruvix-vs-linux -- [OPTIONS]
//!
//! Options:
//!   --iterations <N>    Number of iterations (default: 10000)
//!   --warmup <N>        Number of warmup iterations (default: 100)
//!   --output <FILE>     Output markdown file (default: stdout)
//!   --json              Output as JSON instead of markdown

use clap::Parser;
use std::fs::File;
use std::io::Write;
use std::time::Duration;

use ruvix_bench::{
    comparison::{self, generate_memory_comparisons, ComparisonSummary},
    linux::LinuxBenchConfig,
    report::{generate_json_report, generate_markdown_report, print_console_report},
    ruvix::{self, BenchConfig},
    targets::{spec_for, TargetSummary, TargetVerification},
};

#[derive(Parser, Debug)]
#[command(name = "ruvix-vs-linux")]
#[command(about = "Comprehensive RuVix vs Linux syscall benchmarks")]
struct Args {
    /// Number of measurement iterations per benchmark.
    #[arg(short, long, default_value = "10000")]
    iterations: usize,

    /// Number of warmup iterations.
    #[arg(short, long, default_value = "100")]
    warmup: usize,

    /// Output file for the report (markdown or json).
    #[arg(short, long)]
    output: Option<String>,

    /// Output as JSON instead of markdown.
    #[arg(long)]
    json: bool,

    /// Print results to console with colors.
    #[arg(long)]
    console: bool,

    /// Run quick benchmark (fewer iterations).
    #[arg(long)]
    quick: bool,
}

fn main() {
    let args = Args::parse();

    let (iterations, warmup) = if args.quick {
        (1000, 50)
    } else {
        (args.iterations, args.warmup)
    };

    println!("RuVix vs Linux Benchmark Suite");
    println!("==============================");
    println!("Iterations: {}", iterations);
    println!("Warmup: {}", warmup);
    println!();

    let ruvix_config = BenchConfig {
        warmup_iterations: warmup,
        measure_iterations: iterations,
    };

    let linux_config = LinuxBenchConfig {
        warmup_iterations: warmup,
        measure_iterations: iterations,
    };

    // Run RuVix benchmarks
    println!("Running RuVix syscall benchmarks...");
    let syscall_results = ruvix::bench_all_syscalls(&ruvix_config);
    println!("  Completed {} syscall benchmarks", syscall_results.len());

    // Run comparison benchmarks
    #[cfg(unix)]
    let comparisons = {
        println!("Running Linux comparison benchmarks...");
        let comps = comparison::run_all_comparisons(&ruvix_config, &linux_config);
        println!("  Completed {} comparisons", comps.len());
        comps
    };

    #[cfg(not(unix))]
    let comparisons = Vec::new();

    // Generate memory comparisons
    println!("Generating memory overhead comparisons...");
    let memory_comparisons = generate_memory_comparisons();
    println!(
        "  Generated {} memory comparisons",
        memory_comparisons.len()
    );

    // Build target verification summary
    println!("Verifying ADR-087 targets...");
    let mut target_summary = TargetSummary::new();
    for result in &syscall_results {
        if let Some(spec) = spec_for(&result.operation) {
            let verification =
                TargetVerification::new(Duration::from_nanos(result.p95_ns as u64), spec.target);
            target_summary.add(&result.operation, verification);
        }
    }
    println!(
        "  {} / {} targets met",
        target_summary.passing, target_summary.total
    );

    // Generate summary
    let comp_summary = ComparisonSummary::from_comparisons(&comparisons, &memory_comparisons);

    // Print key findings
    println!();
    println!("Key Findings:");
    println!("  RuVix faster in: {} operations", comp_summary.ruvix_wins);
    println!("  Linux faster in: {} operations", comp_summary.linux_wins);
    println!("  Average speedup: {:.1}x", comp_summary.avg_speedup);
    println!("  Max speedup: {:.1}x", comp_summary.max_speedup);
    println!(
        "  ADR-087 pass rate: {:.0}%",
        target_summary.pass_rate() * 100.0
    );
    println!();

    // Generate report
    if args.console {
        print_console_report(&syscall_results, &comparisons, &memory_comparisons);
    } else {
        let report = if args.json {
            generate_json_report(&syscall_results, &comparisons, &memory_comparisons)
        } else {
            generate_markdown_report(
                &syscall_results,
                &comparisons,
                &memory_comparisons,
                &target_summary,
            )
        };

        match args.output {
            Some(path) => {
                let mut file = File::create(&path).expect("Failed to create output file");
                file.write_all(report.as_bytes())
                    .expect("Failed to write report");
                println!("Report written to: {}", path);
            }
            None => {
                println!("{}", report);
            }
        }
    }
}
