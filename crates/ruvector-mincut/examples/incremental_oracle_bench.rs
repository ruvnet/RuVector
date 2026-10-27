//! Real, measured benchmark for the 2026-10-27
//! `mincut-incremental-boundary-search` nightly.
//!
//! Reproduces the exact ring k-NN graph shape and sizes used by
//! `ruvector-agent-memory`'s `mincut_scaling_probe` example (ADR-345/346's
//! own reproduction harness for the `RuVectorGraphAnalyzer::partition()`
//! latency problem), and times the same `partition()` call end-to-end
//! through three `CutSearchStrategy` variants:
//!
//! - `Legacy`        — unmodified pre-existing behavior (the baseline).
//! - `IncrementalOracle` — incremental-boundary oracle, legacy budget×seed
//!   loop structure kept as-is.
//! - `IncrementalOracleCollapsedBudget` — incremental-boundary oracle,
//!   collapsed single-pass-per-seed loop (candidate B).
//!
//! Every number this prints comes from a real `Instant::now()`-timed call
//! into the production `RuVectorGraphAnalyzer::partition()` path (via
//! `RuVectorGraphAnalyzer::with_factory` for the two candidates, `new` for
//! the baseline) — no synthetic/estimated numbers.
//!
//! Run with:
//!   cargo run --release -p ruvector-mincut --example incremental_oracle_bench

use ruvector_mincut::{BoundedInstance, CutSearchStrategy, RuVectorGraphAnalyzer};
use std::time::{Duration, Instant};

/// Ring k-NN graph: vertex i connects to the next k vertices mod n. Exact
/// shape `mincut_scaling_probe` uses.
fn ring_knn(n: usize, k: usize) -> Vec<(usize, Vec<(usize, f64)>)> {
    (0..n)
        .map(|i| {
            let nbrs: Vec<(usize, f64)> = (1..=k)
                .map(|d| ((i + d) % n, 0.1 + (d as f64) * 0.01))
                .collect();
            (i, nbrs)
        })
        .collect()
}

fn time_partition(
    neighbors: &[(usize, Vec<(usize, f64)>)],
    strategy: CutSearchStrategy,
) -> (Duration, Option<usize>) {
    let mut analyzer = match strategy {
        CutSearchStrategy::Legacy => RuVectorGraphAnalyzer::from_knn(neighbors),
        other => {
            let graph = std::sync::Arc::new(ruvector_mincut::DynamicGraph::new());
            for &(vertex, ref nn_list) in neighbors {
                for &(neighbor, distance) in nn_list {
                    let weight = if distance > 0.0 { 1.0 / distance } else { 1.0 };
                    let _ = graph.insert_edge(vertex as u64, neighbor as u64, weight);
                }
            }
            RuVectorGraphAnalyzer::with_factory(graph, move |_g, min, max| {
                Box::new(BoundedInstance::with_strategy(min, max, other))
            })
        }
    };

    let t0 = Instant::now();
    let partition = analyzer.partition();
    let elapsed = t0.elapsed();
    (elapsed, partition.map(|(a, _)| a.len()))
}

fn main() {
    let sizes = [19usize, 50, 100, 200, 400];
    let k = 8usize;

    println!("n,strategy,partition_ms,side_a_len");
    let mut baseline_ms: Vec<(usize, f64)> = Vec::new();
    let mut collapsed_ms: Vec<(usize, f64)> = Vec::new();

    for &n in &sizes {
        let neighbors = ring_knn(n, k);

        for (label, strategy) in [
            ("Legacy", CutSearchStrategy::Legacy),
            ("IncrementalOracle", CutSearchStrategy::IncrementalOracle),
            (
                "IncrementalOracleCollapsedBudget",
                CutSearchStrategy::IncrementalOracleCollapsedBudget,
            ),
        ] {
            let (elapsed, side_a_len) = time_partition(&neighbors, strategy);
            let ms = elapsed.as_secs_f64() * 1000.0;
            println!("{n},{label},{ms:.3},{}", side_a_len.unwrap_or(0));

            if label == "Legacy" {
                baseline_ms.push((n, ms));
            } else if label == "IncrementalOracleCollapsedBudget" {
                collapsed_ms.push((n, ms));
            }
        }
    }

    println!();
    println!("n,baseline_ms,collapsed_budget_ms,speedup_x");
    for ((n, base), (_, collapsed)) in baseline_ms.iter().zip(collapsed_ms.iter()) {
        let speedup = if *collapsed > 0.0 {
            base / collapsed
        } else {
            f64::INFINITY
        };
        println!("{n},{base:.3},{collapsed:.3},{speedup:.2}");
    }
}
