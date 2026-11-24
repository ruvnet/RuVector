//! Nightly research (ADR-361): three-variant comparison of the λ-instance
//! sweep inside `MinCutWrapper::process_instances` for small (n < 20)
//! graphs, where `BoundedInstance::query()` falls back to
//! `brute_force_min_cut()` — a computation that is *independent* of the
//! instance's own `[lambda_min, lambda_max]` bounds.
//!
//! - `baseline`: the original sweep. Visits every λ-instance from 0 up to
//!   the first one whose range contains the true value, recomputing
//!   `brute_force_min_cut()` from scratch on each one.
//! - `candidate_b`: an exponential-doubling-then-binary-search probe over
//!   instance indices. Does not exploit the fact that the computed value is
//!   exact and bound-independent — only that `AboveRange`/`ValueInRange` is
//!   a monotone predicate in the index — so it still pays one full
//!   recomputation per probe, but needs O(log target) probes instead of
//!   O(target).
//! - `candidate_a`: the shipped fix (`exact_value_hint` / `seed_known_value`
//!   in `ProperCutInstance`, wired into `MinCutWrapper::process_instances`).
//!   Exploits that the brute-force value is exact and bound-independent to
//!   jump directly to the correct instance in O(1) recomputations. Measured
//!   here via the real, production `MinCutWrapper` — not a reimplementation.
//!
//! All three variants are fed the *same* canonical edge list, read back
//! from a single `DynamicGraph` built once per case
//! (`DynamicGraph::insert_edge` canonicalizes `(u, v)` and rejects a
//! reciprocal duplicate — ring constructions with `2*k >= n` produce such
//! duplicates, e.g. vertex 0's forward edge to vertex 6 and vertex 6's
//! "forward" edge back to vertex 0 are the same undirected edge). Deriving
//! all three variants' edges from one graph, instead of three independent
//! constructions, is deliberate: an earlier version of this harness built
//! `baseline`/`candidate_b`'s edges by hand and `candidate_a`'s through
//! `DynamicGraph`, which silently diverged on exactly this case — a bug in
//! the *comparison harness*, not in `ProperCutInstance`/`MinCutWrapper`
//! (`wrapper::tests::test_skip_ahead_elides_intermediate_instances_k6` and
//! the other ADR-361 regression tests, which build one graph and query it
//! one way, caught no such issue). Left in as the `assert_eq!` calls below,
//! which now hold: this is the harness actually proving the three variants
//! agree, not just asserting it.
//!
//! Not itself part of the shipped research artifact (like
//! `ruvector-agent-memory/examples/mincut_scaling_probe.rs`, whose ring
//! construction this reuses) — a throwaway comparison harness whose printed
//! output is pasted, unedited, into
//! `docs/research/nightly/2026-11-24_mincut-redundant-sweep/README.md`.

use ruvector_mincut::{BoundedInstance, DynamicGraph, EdgeId, InstanceResult, ProperCutInstance};
use ruvector_mincut::{MinCutWrapper, VertexId};
use std::sync::Arc;
use std::time::Instant;

const RANGE_FACTOR: f64 = 1.2;
const MAX_INSTANCES: usize = 100;

fn lambda_bounds(i: usize) -> (u64, u64) {
    let lo = (RANGE_FACTOR.powi(i as i32)).floor() as u64;
    let hi = (RANGE_FACTOR.powi((i + 1) as i32)).floor() as u64;
    (lo.max(1), hi.max(1))
}

/// Build the fixed-degree ring k-NN graph used by `mincut_scaling_probe.rs`
/// (vertex i connects to the next k vertices mod n), then read back its
/// canonical, de-duplicated edge list — the single source of truth fed to
/// every variant below.
fn build_canonical_edges(
    n: usize,
    k: usize,
) -> (Arc<DynamicGraph>, Vec<(EdgeId, VertexId, VertexId)>) {
    let graph = Arc::new(DynamicGraph::new());
    for i in 0..n {
        for d in 1..=k {
            let _ = graph.insert_edge(i as u64, ((i + d) % n) as u64, 1.0);
        }
    }
    let edges: Vec<_> = graph
        .edges()
        .iter()
        .map(|e| (e.id, e.source, e.target))
        .collect();
    (graph, edges)
}

/// Variant: baseline. Linear sweep, one fresh `BoundedInstance` per index,
/// recomputing `brute_force_min_cut()` on every one.
fn baseline(edges: &[(EdgeId, VertexId, VertexId)]) -> (Option<u64>, usize, std::time::Duration) {
    let t0 = Instant::now();
    let mut calls = 0usize;
    for i in 0..MAX_INSTANCES {
        let (lo, hi) = lambda_bounds(i);
        let mut inst = BoundedInstance::new(lo, hi);
        inst.apply_inserts(edges);
        calls += 1;
        if let InstanceResult::ValueInRange { value, .. } = inst.query() {
            return (Some(value), calls, t0.elapsed());
        }
    }
    (None, calls, t0.elapsed())
}

/// Variant B: exponential-doubling probe over instance indices, then binary
/// search between the last `AboveRange` probe and the first `ValueInRange`
/// one. Does not use `exact_value_hint` — only the monotone
/// `AboveRange`/`ValueInRange` signal — so each probe still pays a full
/// `brute_force_min_cut()` recomputation.
fn candidate_b(
    edges: &[(EdgeId, VertexId, VertexId)],
) -> (Option<u64>, usize, std::time::Duration) {
    let t0 = Instant::now();
    let mut calls = 0usize;

    let probe = |i: usize, calls: &mut usize| -> InstanceResult {
        let (lo, hi) = lambda_bounds(i);
        let mut inst = BoundedInstance::new(lo, hi);
        inst.apply_inserts(edges);
        *calls += 1;
        inst.query()
    };

    let mut lo_idx = 0usize;
    let mut have_lo = false;
    let mut step = 1usize;
    let mut i = 0usize;

    loop {
        if i >= MAX_INSTANCES {
            return (None, calls, t0.elapsed());
        }
        match probe(i, &mut calls) {
            InstanceResult::ValueInRange { .. } => {
                let mut hi_idx = i;
                let mut lo = if have_lo { lo_idx } else { 0 };
                // Binary search strictly between (lo, hi_idx] for the first
                // in-range index, to land on the same target index
                // `candidate_a` would (an apples-to-apples call count).
                while hi_idx - lo > 1 {
                    let mid = lo + (hi_idx - lo) / 2;
                    match probe(mid, &mut calls) {
                        InstanceResult::ValueInRange { .. } => hi_idx = mid,
                        InstanceResult::AboveRange => lo = mid,
                    }
                }
                let (_, final_hi) = lambda_bounds(hi_idx);
                let mut final_inst = BoundedInstance::new(1, final_hi);
                final_inst.apply_inserts(edges);
                calls += 1;
                if let InstanceResult::ValueInRange { value, .. } = final_inst.query() {
                    return (Some(value), calls, t0.elapsed());
                }
                return (None, calls, t0.elapsed());
            }
            InstanceResult::AboveRange => {
                lo_idx = i;
                have_lo = true;
                step *= 2;
                i += step;
            }
        }
    }
}

/// Variant A (shipped fix): the real, production `MinCutWrapper` (the same
/// engine `RuVectorGraphAnalyzer::partition()` — exercised by
/// `mincut_scaling_probe.rs` — delegates to). Visit count is read off
/// `num_instances()`, which — per
/// `wrapper::tests::test_skip_ahead_elides_intermediate_instances_k6` — is
/// exactly the number of `query()` calls made on a cold wrapper. Unlike
/// `baseline`/`candidate_b`, not every visit here is a real
/// `brute_force_min_cut()` recomputation: the *seeded* (jumped-to) instance
/// answers from `MinCutWrapper`'s own cached-witness fast path in O(1)
/// instead of recomputing — confirmed separately by instrumenting
/// `brute_force_min_cut()` directly, which logs exactly one real
/// recomputation for this n=19 case, not two.
fn candidate_a(
    graph: &Arc<DynamicGraph>,
    edges: &[(EdgeId, VertexId, VertexId)],
) -> (Option<u64>, usize, std::time::Duration) {
    let t0 = Instant::now();
    let mut wrapper = MinCutWrapper::new(Arc::clone(graph));
    for &(id, u, v) in edges {
        wrapper.insert_edge(id, u, v);
    }
    let result = wrapper.query();
    (Some(result.value()), wrapper.num_instances(), t0.elapsed())
}

fn run_case(label: &str, n: usize, k: usize) {
    println!("=== {label} (n={n}, k={k}) ===");

    let (graph, edges) = build_canonical_edges(n, k);
    println!("canonical edge count: {}", edges.len());

    let (base_value, base_calls, base_elapsed) = baseline(&edges);
    let (b_value, b_calls, b_elapsed) = candidate_b(&edges);
    let (a_value, a_calls, a_elapsed) = candidate_a(&graph, &edges);

    println!(
        "baseline:     value={base_value:?} instance_visits={base_calls:<4} elapsed={:>10.3}ms",
        base_elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "candidate_b:  value={b_value:?} instance_visits={b_calls:<4} elapsed={:>10.3}ms",
        b_elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "candidate_a:  value={a_value:?} instance_visits={a_calls:<4} elapsed={:>10.3}ms  (shipped fix, via MinCutWrapper)",
        a_elapsed.as_secs_f64() * 1000.0
    );
    assert_eq!(
        base_value, a_value,
        "candidate_a must match baseline's value"
    );
    assert_eq!(b_value, a_value, "candidate_b must match baseline's value");
    println!();
}

fn main() {
    run_case(
        "19-vertex regular ring (the documented 69.3s/83.4s outlier)",
        19,
        8,
    );
    run_case("10-vertex regular ring, secondary data point", 10, 6);
}
