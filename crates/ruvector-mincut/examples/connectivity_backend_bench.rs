//! 2026-11-11 nightly: standalone three-way bench answering "would swapping
//! `BoundedInstance::is_connected()`'s naive per-call BFS for
//! `connectivity::DynamicConnectivity` (already in-tree, used elsewhere by
//! `MinCutWrapper`) or `connectivity::polylog::PolylogConnectivity`
//! (suggested by the 2026-09-11 nightly's Next Research item 2) make a
//! single whole-graph connectivity check faster?"
//!
//! baseline     = the exact BFS `BoundedInstance::is_connected()` runs today
//! candidate_a  = build a `DynamicConnectivity` from the same edge list, then `.is_connected()`
//! candidate_b  = build a `PolylogConnectivity` from the same edge list, then `.is_connected()`
//!
//! Each variant pays the *full* build-from-scratch cost every call, matching
//! how `BoundedInstance` would have to use them: it has no persistent
//! connectivity structure today, and `profile_partition_scaling.rs`
//! confirms `is_connected()` is called at most once per `query()`, so a
//! persistent structure would need separate insert/delete wiring through
//! `BoundedInstance::insert`/`delete` to ever pay off — out of scope here;
//! this bench isolates the narrower question of per-call cost as asked.
//!
//! Run: cargo run --release -p ruvector-mincut --example connectivity_backend_bench

use ruvector_mincut::connectivity::polylog::PolylogConnectivity;
use ruvector_mincut::connectivity::DynamicConnectivity;
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

fn ring_edges(n: usize, k: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::with_capacity(n * k);
    for i in 0..n {
        for d in 1..=k {
            edges.push((i, (i + d) % n));
        }
    }
    edges
}

fn baseline_bfs(n: usize, edges: &[(usize, usize)]) -> bool {
    let mut adjacency: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(u, v) in edges {
        adjacency.entry(u).or_default().push(v);
        adjacency.entry(v).or_default().push(u);
    }
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
    visited.len() == n
}

fn candidate_a_euler_tour(edges: &[(usize, usize)]) -> bool {
    let mut dc = DynamicConnectivity::new();
    for &(u, v) in edges {
        dc.insert_edge(u as u64, v as u64);
    }
    dc.is_connected()
}

fn candidate_b_polylog(edges: &[(usize, usize)]) -> bool {
    let mut pc = PolylogConnectivity::new();
    for &(u, v) in edges {
        pc.insert_edge(u as u64, v as u64);
    }
    pc.is_connected()
}

fn time_it<F: FnMut() -> bool>(mut f: F, reps: usize) -> (Duration, bool) {
    // Warm up once (outside the measured loop), matching this crate's own
    // benches (e.g. `boundary_incremental_bench.rs`).
    let warmup_result = f();
    let t0 = Instant::now();
    let mut last = warmup_result;
    for _ in 0..reps {
        last = f();
    }
    (t0.elapsed() / reps as u32, last)
}

fn main() {
    let sizes = [19usize, 50, 100, 200, 400, 800];
    let k = 8usize;
    let reps = 20usize;

    println!(
        "{:<6} {:>14} {:>14} {:>14}",
        "n", "baseline_us", "euler_tour_us", "polylog_us"
    );

    for &n in &sizes {
        let edges = ring_edges(n, k);

        let (baseline_t, baseline_ok) = time_it(|| baseline_bfs(n, &edges), reps);
        let (a_t, a_ok) = time_it(|| candidate_a_euler_tour(&edges), reps);
        let (b_t, b_ok) = time_it(|| candidate_b_polylog(&edges), reps);

        assert!(baseline_ok, "baseline must find the ring connected");
        assert_eq!(
            baseline_ok, a_ok,
            "candidate_a disagreed with baseline at n={n}"
        );
        assert_eq!(
            baseline_ok, b_ok,
            "candidate_b disagreed with baseline at n={n}"
        );

        println!(
            "{:<6} {:>14.2} {:>14.2} {:>14.2}",
            n,
            baseline_t.as_secs_f64() * 1_000_000.0,
            a_t.as_secs_f64() * 1_000_000.0,
            b_t.as_secs_f64() * 1_000_000.0,
        );
    }
}
