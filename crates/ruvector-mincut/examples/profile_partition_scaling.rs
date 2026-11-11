//! 2026-11-11 nightly: attributes `RuVectorGraphAnalyzer::partition()`
//! latency (ADR-345's ~77ms@n=50 -> ~11.4s@n=400 scaling, re-measured below)
//! to its actual call sites, to test the 2026-09-11 nightly's Next Research
//! item 2 ("evaluate `connectivity::polylog::PolylogConnectivity` ... as a
//! replacement backend" for `BoundedInstance`).
//!
//! Reuses the exact ring k-NN corpus (k=8, n in {19,50,100,200,400}) from
//! `ruvector-agent-memory/examples/mincut_scaling_probe.rs` for direct
//! comparability with ADR-345's numbers. Requires `--features
//! profile-counters` to see non-zero counters; without it this still runs
//! and times `partition()`, it just can't attribute the time (counters read
//! 0).
//!
//! Run: cargo run --release -p ruvector-mincut --features profile-counters
//!      --example profile_partition_scaling

use ruvector_mincut::profiling;
use std::time::Instant;

/// Independently-measured BFS connectivity check cost at size `n`, using the
/// same ring adjacency `is_connected()` walks internally, but instrumented
/// from the *outside* so its wall-clock cost is a directly measured number,
/// not inferred from the call count alone.
fn measure_single_bfs_cost(n: usize, k: usize) -> std::time::Duration {
    use std::collections::{HashMap, HashSet, VecDeque};

    let mut adjacency: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        for d in 1..=k {
            let j = (i + d) % n;
            adjacency.entry(i).or_default().push(j);
            adjacency.entry(j).or_default().push(i);
        }
    }

    let t0 = Instant::now();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(0usize);
    visited.insert(0usize);
    while let Some(cur) = queue.pop_front() {
        if let Some(neighbors) = adjacency.get(&cur) {
            for &nb in neighbors {
                if visited.insert(nb) {
                    queue.push_back(nb);
                }
            }
        }
    }
    assert_eq!(visited.len(), n, "ring graph must be fully connected");
    t0.elapsed()
}

fn main() {
    let sizes = [19usize, 50, 100, 200, 400];
    let k = 8usize;

    println!(
        "{:<6} {:>12} {:>12} {:>10} {:>10} {:>10} {:>14} {:>10}",
        "n",
        "build_ms",
        "partition_ms",
        "oracle_calls",
        "conn_calls",
        "inst_tried",
        "conn_attrib_us",
        "conn_pct"
    );

    for &n in &sizes {
        let neighbors: Vec<(usize, Vec<(usize, f64)>)> = (0..n)
            .map(|i| {
                let nbrs: Vec<(usize, f64)> = (1..=k)
                    .map(|d| ((i + d) % n, 0.1 + (d as f64) * 0.01))
                    .collect();
                (i, nbrs)
            })
            .collect();

        profiling::reset();

        let t0 = Instant::now();
        let mut analyzer = ruvector_mincut::RuVectorGraphAnalyzer::from_knn(&neighbors);
        let build_elapsed = t0.elapsed();

        let t1 = Instant::now();
        let _ = analyzer.partition();
        let partition_elapsed = t1.elapsed();

        let snap = profiling::snapshot();

        // Independently measured single-BFS cost at this n, multiplied by
        // the observed call count, gives a directly measured (not merely
        // counted) estimate of time attributable to connectivity checks.
        let single_bfs = measure_single_bfs_cost(n, k);
        let conn_attrib = single_bfs.saturating_mul(snap.connectivity_check_calls.max(1) as u32);
        let conn_pct = if partition_elapsed.as_nanos() > 0 {
            100.0 * conn_attrib.as_secs_f64() / partition_elapsed.as_secs_f64()
        } else {
            0.0
        };

        println!(
            "{:<6} {:>12.3} {:>12.3} {:>10} {:>10} {:>10} {:>14.1} {:>9.3}%",
            n,
            build_elapsed.as_secs_f64() * 1000.0,
            partition_elapsed.as_secs_f64() * 1000.0,
            snap.oracle_search_calls,
            snap.connectivity_check_calls,
            snap.instances_tried,
            conn_attrib.as_secs_f64() * 1_000_000.0,
            conn_pct,
        );
    }
}
