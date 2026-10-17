//! Nightly research probe (2026-10-07,
//! `docs/research/nightly/2026-10-07-mincut-polylog-connectivity-backend/`):
//! paired A/B measurement of `RuVectorGraphAnalyzer::partition()` latency
//! under `ConnectivityBackend::EulerTour` (baseline, default) vs.
//! `ConnectivityBackend::Polylog`.
//!
//! This is the direct successor to the 2026-09-05 nightly's
//! `mincut_scaling_probe.rs` (`crates/ruvector-agent-memory/examples/`),
//! which established the latency/scaling problem this run investigates.
//! Unlike that probe, this one:
//!
//! - builds each graph **once** per (size, seed) and times both backends
//!   against the *exact same* `DynamicGraph` (one independent variable:
//!   the connectivity backend; see ADR-282 ×3's methodology discipline),
//!   instead of comparing across separately-constructed graphs;
//! - uses randomized (not perfectly regular ring) connected graphs, so the
//!   measurement is not dominated by one adversarial topology's tie-breaking
//!   behavior;
//! - repeats multiple seeds per size (`REPS` below) for a mean/stdev/worst
//!   instead of a single sample.
//!
//! `n=19` gets only 1 repetition per backend: the `<20`-vertex path is
//! `BoundedInstance::brute_force_min_cut`, an O(2^n) exhaustive subset
//! enumeration whose cost is dominated by `2^19 ≈ 524,288` enumerated
//! subsets almost independently of topology (confirmed by ad hoc
//! instrumentation during this run's root-cause pass, not included here —
//! see the README's "Root cause" section) — repeating it 5x per backend
//! would cost several minutes for no additional evidentiary value, since
//! the dominant cost is enumeration count, not graph shape.
//!
//! Run:
//!   cargo run --release -p ruvector-mincut --example connectivity_backend_partition_probe

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_mincut::graph::DynamicGraph;
use ruvector_mincut::{ConnectivityBackend, RuVectorGraphAnalyzer};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// (size, seeds-to-use, label) — `n=19` intentionally gets one seed; see
/// the module doc above.
const CASES: &[(usize, &[u64])] = &[
    (19, &[11]),
    (50, &[11, 22, 33, 44, 55]),
    (84, &[11, 22, 33, 44, 55]),
    (100, &[11, 22, 33, 44, 55]),
    (200, &[11, 22, 33, 44, 55]),
];

const K: usize = 8;

/// Build a random connected graph on `n` vertices: a random spanning path
/// (guarantees connectivity) plus `k * n / 2` additional random edges for
/// density comparable to a k-NN graph (`k` neighbors/vertex).
fn random_connected_graph(n: usize, k: usize, seed: u64) -> DynamicGraph {
    let mut rng = StdRng::seed_from_u64(seed);
    let graph = DynamicGraph::new();

    let mut order: Vec<u64> = (0..n as u64).collect();
    // Fisher-Yates shuffle for the spanning path order.
    for i in (1..order.len()).rev() {
        let j = rng.gen_range(0..=i);
        order.swap(i, j);
    }
    let mut edge_id = 0u64;
    for w in order.windows(2) {
        let _ = graph.insert_edge(w[0], w[1], 1.0);
        edge_id += 1;
    }

    let target_extra_edges = (k * n) / 2;
    let mut added = 0usize;
    let mut attempts = 0usize;
    while added < target_extra_edges && attempts < target_extra_edges * 20 {
        attempts += 1;
        let u = rng.gen_range(0..n as u64);
        let v = rng.gen_range(0..n as u64);
        if u == v {
            continue;
        }
        if graph.insert_edge(u, v, 1.0).is_ok() {
            added += 1;
            edge_id += 1;
        }
    }
    let _ = edge_id;
    graph
}

struct Stats {
    mean_ms: f64,
    stdev_ms: f64,
    worst_ms: f64,
    samples_ms: Vec<f64>,
}

fn summarize(samples_ms: &[f64]) -> Stats {
    let n = samples_ms.len() as f64;
    let mean = samples_ms.iter().sum::<f64>() / n;
    let var = samples_ms.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let worst = samples_ms.iter().cloned().fold(0.0_f64, f64::max);
    Stats {
        mean_ms: mean,
        stdev_ms: var.sqrt(),
        worst_ms: worst,
        samples_ms: samples_ms.to_vec(),
    }
}

fn time_partition(graph: &Arc<DynamicGraph>, backend: ConnectivityBackend) -> Duration {
    let t0 = Instant::now();
    let mut analyzer = RuVectorGraphAnalyzer::new_with_backend(Arc::clone(graph), backend);
    let _ = analyzer.partition();
    t0.elapsed()
}

fn main() {
    println!("Connectivity backend A/B partition() latency probe");
    println!("rustc target: release build assumed (run with --release)\n");
    println!(
        "{:<6} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "n", "seeds", "euler_ms", "euler_sd", "euler_max", "poly_ms", "poly_sd", "poly_max"
    );
    println!("{}", "-".repeat(90));

    for &(n, seeds) in CASES {
        let mut euler_samples = Vec::new();
        let mut poly_samples = Vec::new();

        for &seed in seeds {
            let graph = Arc::new(random_connected_graph(n, K, seed));

            let euler_elapsed = time_partition(&graph, ConnectivityBackend::EulerTour);
            euler_samples.push(euler_elapsed.as_secs_f64() * 1000.0);

            let poly_elapsed = time_partition(&graph, ConnectivityBackend::Polylog);
            poly_samples.push(poly_elapsed.as_secs_f64() * 1000.0);
        }

        let euler = summarize(&euler_samples);
        let poly = summarize(&poly_samples);

        println!(
            "{:<6} {:>8} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
            n,
            seeds.len(),
            euler.mean_ms,
            euler.stdev_ms,
            euler.worst_ms,
            poly.mean_ms,
            poly.stdev_ms,
            poly.worst_ms,
        );
        println!(
            "       euler raw (ms): {:?}",
            euler
                .samples_ms
                .iter()
                .map(|x| format!("{x:.3}"))
                .collect::<Vec<_>>()
        );
        println!(
            "       poly  raw (ms): {:?}",
            poly.samples_ms
                .iter()
                .map(|x| format!("{x:.3}"))
                .collect::<Vec<_>>()
        );
    }

    println!(
        "\nNote: both backends are compared against the *same* graph per \
         (size, seed) pair; the only independent variable is \
         ConnectivityBackend. See the 2026-10-07 nightly README for the \
         root-cause explanation of why these are expected to be \
         statistically indistinguishable."
    );
}
