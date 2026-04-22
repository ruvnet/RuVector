//! RuVix syscall benchmark runner.
//!
//! Benchmarks all 12 RuVix syscalls against ADR-087 targets.
//!
//! Usage: cargo run --bin syscall-bench -- [OPTIONS]

use clap::Parser;
use std::time::Duration;
use tabled::{Table, Tabled};

use ruvix_bench::{
    ruvix::{self, BenchConfig},
    targets::{spec_for, TargetSummary, TargetVerification, TARGET_SPECS},
};

#[derive(Parser, Debug)]
#[command(name = "syscall-bench")]
#[command(about = "Benchmark RuVix syscalls against ADR-087 targets")]
struct Args {
    /// Number of measurement iterations.
    #[arg(short, long, default_value = "10000")]
    iterations: usize,

    /// Number of warmup iterations.
    #[arg(short, long, default_value = "100")]
    warmup: usize,

    /// Run quick benchmark.
    #[arg(long)]
    quick: bool,

    /// Verbose output with detailed statistics.
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Tabled)]
struct ResultRow {
    #[tabled(rename = "Syscall")]
    syscall: String,
    #[tabled(rename = "Mean")]
    mean: String,
    #[tabled(rename = "P50")]
    p50: String,
    #[tabled(rename = "P95")]
    p95: String,
    #[tabled(rename = "P99")]
    p99: String,
    #[tabled(rename = "Target")]
    target: String,
    #[tabled(rename = "Margin")]
    margin: String,
    #[tabled(rename = "Status")]
    status: String,
}

fn format_duration(ns: f64) -> String {
    if ns >= 1_000_000.0 {
        format!("{:.2}ms", ns / 1_000_000.0)
    } else if ns >= 1_000.0 {
        format!("{:.2}us", ns / 1_000.0)
    } else {
        format!("{:.0}ns", ns)
    }
}

fn main() {
    let args = Args::parse();

    let (iterations, warmup) = if args.quick {
        (1000, 50)
    } else {
        (args.iterations, args.warmup)
    };

    println!("RuVix Syscall Benchmark");
    println!("=======================");
    println!();
    println!("Configuration:");
    println!("  Iterations: {}", iterations);
    println!("  Warmup: {}", warmup);
    println!();

    // Print ADR-087 targets
    println!("ADR-087 Target Latencies:");
    for spec in TARGET_SPECS.iter() {
        let target = if spec.target.as_millis() > 0 {
            format!("{}ms", spec.target.as_millis())
        } else if spec.target.as_micros() > 0 {
            format!("{}us", spec.target.as_micros())
        } else {
            format!("{}ns", spec.target.as_nanos())
        };
        let tier = spec
            .proof_tier
            .map(|t| format!(" [{}]", t.name()))
            .unwrap_or_default();
        println!("  {}: {}{} - {}", spec.name, target, tier, spec.notes);
    }
    println!();

    // Run benchmarks
    let config = BenchConfig {
        warmup_iterations: warmup,
        measure_iterations: iterations,
    };

    println!("Running benchmarks...");
    println!();

    let results = ruvix::bench_all_syscalls(&config);

    // Build result table
    let mut rows = Vec::new();
    let mut summary = TargetSummary::new();

    for result in &results {
        let target_ns = result.target_ns.unwrap_or(0.0);
        let margin = if target_ns > 0.0 {
            result.p95_ns / target_ns
        } else {
            0.0
        };

        let status = if result.meets_target { "PASS" } else { "FAIL" };

        if let Some(spec) = spec_for(&result.operation) {
            let verification =
                TargetVerification::new(Duration::from_nanos(result.p95_ns as u64), spec.target);
            summary.add(&result.operation, verification);
        }

        rows.push(ResultRow {
            syscall: result.operation.clone(),
            mean: format_duration(result.mean_ns),
            p50: format_duration(result.p50_ns),
            p95: format_duration(result.p95_ns),
            p99: format_duration(result.p99_ns),
            target: if target_ns > 0.0 {
                format_duration(target_ns)
            } else {
                "-".to_string()
            },
            margin: if target_ns > 0.0 {
                format!("{:.0}%", margin * 100.0)
            } else {
                "-".to_string()
            },
            status: status.to_string(),
        });
    }

    println!("{}", Table::new(&rows));
    println!();

    // Print summary
    println!("Summary:");
    println!("  Total syscalls: {}", summary.total);
    println!(
        "  Passing: {} ({:.0}%)",
        summary.passing,
        summary.pass_rate() * 100.0
    );
    println!("  Failing: {}", summary.failing);
    println!();

    if args.verbose {
        println!("Detailed Statistics:");
        for result in &results {
            println!();
            println!("  {}:", result.operation);
            println!("    Iterations: {}", result.iterations);
            println!("    Mean: {:.1}ns", result.mean_ns);
            println!("    Std Dev: {:.1}ns", result.std_dev_ns);
            println!("    Min: {:.1}ns", result.min_ns);
            println!("    Max: {:.1}ns", result.max_ns);
            println!("    P50: {:.1}ns", result.p50_ns);
            println!("    P95: {:.1}ns", result.p95_ns);
            println!("    P99: {:.1}ns", result.p99_ns);
        }
    }

    // Exit code based on pass/fail
    let exit_code = if summary.all_passing() { 0 } else { 1 };
    std::process::exit(exit_code);
}
