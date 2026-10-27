//! Differential equivalence suite: `CutSearchStrategy::IncrementalOracle`
//! and `CutSearchStrategy::IncrementalOracleCollapsedBudget` must return
//! the identical `(cut_value, witness vertex membership)` as
//! `CutSearchStrategy::Legacy` on every graph here, backing the
//! correctness "subject to" clause in the 2026-10-27
//! `mincut-incremental-boundary-search` nightly's hypothesis.
//!
//! Two independent checks:
//! 1. `n < 20` graphs: `BoundedInstance` takes the brute-force path
//!    regardless of strategy (identical for all three), so these are a
//!    sanity floor — strategy choice must be a no-op where it never
//!    engages `search_for_cuts` at all.
//! 2. `n >= 20` graphs: all three strategies exercise `search_for_cuts`
//!    (the oracle path this nightly changed). These are the real
//!    equivalence test: baseline vs. both candidates, same edges, same
//!    `[lambda_min, lambda_max]`.

use ruvector_mincut::{BoundedInstance, CutSearchStrategy, InstanceResult, ProperCutInstance};
use std::collections::HashSet;

/// Small deterministic LCG so graph generation needs no new dependency and
/// is reproducible across runs without pulling in `rand`.
struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn next_range(&mut self, n: u64) -> u64 {
        self.next_u64() % n.max(1)
    }
}

/// Build a random connected **simple** graph on `n` vertices: a random
/// spanning tree (guarantees connectivity) plus up to `extra_edges`
/// additional random edges, each between a vertex pair that does not
/// already have an edge.
///
/// Deliberately simple (one edge per unordered vertex pair): `BoundedInstance`'s
/// `search_for_cuts_legacy` rebuilds a temporary `DynamicGraph` from
/// `self.edges` each call and silently discards the `Result` of
/// `insert_edge` (`let _ = graph.insert_edge(u, v, 1.0)`), and
/// `DynamicGraph::insert_edge` errors (`EdgeExists`) on a second edge
/// between a pair already connected — so a multigraph with duplicate
/// same-pair edges makes the legacy oracle silently search a *simple*
/// graph while `self.adjacency` (what the new `IncrementalLocalKCut` path
/// searches directly, with no rebuild) still reflects every edge. That
/// pre-existing inconsistency is orthogonal to this nightly's change — it
/// would surface identically if two *legacy* instances were compared on a
/// multigraph, since the simplification happens inside `search_for_cuts`'s
/// own `DynamicGraph` rebuild — so this generator avoids it rather than
/// asserting equivalence through it. Documented as a Limitation/Next
/// Research item in this nightly's README.
fn random_connected_graph(seed: u64, n: u64, extra_edges: u64) -> Vec<(u64, u64, u64)> {
    let mut rng = Lcg(seed);
    let mut edges = Vec::new();
    let mut edge_id = 0u64;
    let mut present: HashSet<(u64, u64)> = HashSet::new();

    let mut add = |u: u64,
                   v: u64,
                   edges: &mut Vec<(u64, u64, u64)>,
                   edge_id: &mut u64,
                   present: &mut HashSet<(u64, u64)>| {
        let key = (u.min(v), u.max(v));
        if present.insert(key) {
            edges.push((*edge_id, u, v));
            *edge_id += 1;
        }
    };

    // Random spanning tree: attach vertex i (1..n) to a uniformly chosen
    // earlier vertex in 0..i.
    for i in 1..n {
        let parent = rng.next_range(i);
        add(parent, i, &mut edges, &mut edge_id, &mut present);
    }

    // Extra random edges, skipped if the pair already has one.
    for _ in 0..extra_edges {
        let u = rng.next_range(n);
        let mut v = rng.next_range(n);
        if v == u {
            v = (v + 1) % n;
        }
        add(u, v, &mut edges, &mut edge_id, &mut present);
    }

    edges
}

/// Ring k-NN graph: vertex i connects to the next k vertices mod n. The
/// exact shape used by `mincut_scaling_probe` (ADR-345/346's own
/// reproduction harness) and this nightly's benchmark.
fn ring_knn_graph(n: u64, k: u64) -> Vec<(u64, u64, u64)> {
    let mut edges = Vec::new();
    let mut edge_id = 0u64;
    for i in 0..n {
        for d in 1..=k {
            edges.push((edge_id, i, (i + d) % n));
            edge_id += 1;
        }
    }
    edges
}

fn run_strategy(
    edges: &[(u64, u64, u64)],
    lambda_min: u64,
    lambda_max: u64,
    strategy: CutSearchStrategy,
) -> InstanceResult {
    let mut instance = BoundedInstance::with_strategy(lambda_min, lambda_max, strategy);
    instance.apply_inserts(edges);
    instance.query()
}

/// Extract `(matched, value, membership)` for comparison: `membership` is
/// the sorted vertex list the witness contains, restricted to the graph's
/// vertex range, so two witnesses over the same graph compare by content
/// rather than by object identity.
fn describe(result: &InstanceResult, all_vertices: &[u64]) -> (bool, Option<u64>, Vec<u64>) {
    match result {
        InstanceResult::ValueInRange { value, witness } => {
            let membership: Vec<u64> = all_vertices
                .iter()
                .copied()
                .filter(|&v| witness.contains(v))
                .collect();
            (true, Some(*value), membership)
        }
        InstanceResult::AboveRange => (false, None, Vec::new()),
    }
}

fn assert_strategies_agree(
    edges: &[(u64, u64, u64)],
    lambda_min: u64,
    lambda_max: u64,
    label: &str,
) {
    let all_vertices: Vec<u64> = {
        let mut vs: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for &(_, u, v) in edges {
            vs.insert(u);
            vs.insert(v);
        }
        let mut vs: Vec<u64> = vs.into_iter().collect();
        vs.sort_unstable();
        vs
    };

    let legacy = run_strategy(edges, lambda_min, lambda_max, CutSearchStrategy::Legacy);
    let incremental = run_strategy(
        edges,
        lambda_min,
        lambda_max,
        CutSearchStrategy::IncrementalOracle,
    );
    let collapsed = run_strategy(
        edges,
        lambda_min,
        lambda_max,
        CutSearchStrategy::IncrementalOracleCollapsedBudget,
    );

    let legacy_d = describe(&legacy, &all_vertices);
    let incremental_d = describe(&incremental, &all_vertices);
    let collapsed_d = describe(&collapsed, &all_vertices);

    assert_eq!(
        legacy_d,
        incremental_d,
        "{label}: IncrementalOracle disagrees with Legacy (n={}, lambda=[{},{}])",
        all_vertices.len(),
        lambda_min,
        lambda_max
    );
    assert_eq!(
        legacy_d,
        collapsed_d,
        "{label}: IncrementalOracleCollapsedBudget disagrees with Legacy (n={}, lambda=[{},{}])",
        all_vertices.len(),
        lambda_min,
        lambda_max
    );
}

#[test]
fn small_random_graphs_brute_force_path_agrees() {
    // n < 20: all three strategies take the brute-force path untouched by
    // this nightly. This is the floor: strategy selection must be a no-op
    // here.
    for n in [5u64, 8, 12, 15, 19] {
        for seed in [1u64, 2, 3, 42] {
            let edges = random_connected_graph(seed, n, n); // n extra edges for some density
            assert_strategies_agree(&edges, 0, 1_000, &format!("small n={n} seed={seed}"));
        }
    }
}

#[test]
fn oracle_path_random_graphs_agree() {
    // n >= 20: exercises search_for_cuts, the code path this nightly
    // changed. Several densities (extra_edges) and several seeds.
    for n in [20u64, 25, 40] {
        for extra in [0u64, n / 2, n] {
            for seed in [7u64, 99, 123] {
                let edges = random_connected_graph(seed, n, extra);
                assert_strategies_agree(
                    &edges,
                    0,
                    50,
                    &format!("oracle n={n} extra={extra} seed={seed}"),
                );
            }
        }
    }
}

#[test]
fn ring_knn_graphs_agree_at_benchmark_sizes() {
    // The exact shape mincut_scaling_probe (ADR-345/346) and this
    // nightly's benchmark use, at a representative subset of its sizes
    // (400 is covered separately by the latency benchmark, not the test
    // suite, to keep `cargo test` fast).
    for &n in &[19u64, 20, 50, 100] {
        let edges = ring_knn_graph(n, 8);
        assert_strategies_agree(&edges, 0, 60, &format!("ring n={n} k=8"));
    }
}

#[test]
fn disconnected_graph_agrees_on_cut_value() {
    // Two disjoint triangles: min cut is 0 for every strategy. Cut *value*
    // only, not witness membership: the disconnected fast path in
    // `ProperCutInstance::query()` (shared by every strategy — it runs
    // before `search_for_cuts` is ever reached) seeds its single-vertex
    // witness from `self.vertices.iter().next()` on a `HashSet<VertexId>`,
    // whose iteration order is randomly seeded per `HashSet` instance (the
    // same root cause the 2026-09-11 nightly fixed in `DynamicGraph`'s
    // `vertices()`/`edges()`, but not here) — so two *separately
    // constructed* `BoundedInstance`s over byte-identical edges can
    // legitimately report different (but equally valid, cut-value-0)
    // isolated vertices regardless of strategy. Documented as a Limitation
    // in this nightly's README rather than silently worked around.
    let edges = vec![
        (0, 0, 1),
        (1, 1, 2),
        (2, 2, 0),
        (3, 10, 11),
        (4, 11, 12),
        (5, 12, 10),
    ];
    let legacy = run_strategy(&edges, 0, 10, CutSearchStrategy::Legacy);
    let incremental = run_strategy(&edges, 0, 10, CutSearchStrategy::IncrementalOracle);
    let collapsed = run_strategy(
        &edges,
        0,
        10,
        CutSearchStrategy::IncrementalOracleCollapsedBudget,
    );
    for (label, result) in [
        ("IncrementalOracle", &incremental),
        ("IncrementalOracleCollapsedBudget", &collapsed),
    ] {
        match (&legacy, result) {
            (
                InstanceResult::ValueInRange { value: lv, .. },
                InstanceResult::ValueInRange { value: rv, .. },
            ) => {
                assert_eq!(lv, rv, "{label}: cut value disagrees on disconnected graph");
            }
            _ => panic!("{label}: expected both ValueInRange(0) on a disconnected graph"),
        }
    }
}

#[test]
fn path_and_cycle_graphs_agree() {
    assert_strategies_agree(&[(0, 0, 1), (1, 1, 2)], 0, 10, "path");
    assert_strategies_agree(&[(0, 0, 1), (1, 1, 2), (2, 2, 0)], 0, 10, "cycle");
}

#[test]
fn narrow_range_instances_agree() {
    // Narrow / offset [lambda_min, lambda_max] windows exercise the
    // "cut value outside this instance's range" (AboveRange) path on both
    // sides, which is exactly where the collapsed-budget loop's
    // early-return behavior most needs to match the legacy per-integer
    // budget sweep.
    let edges = ring_knn_graph(30, 6);
    for &(lo, hi) in &[(0u64, 2), (3, 5), (6, 8), (0, 1), (20, 30)] {
        assert_strategies_agree(&edges, lo, hi, &format!("narrow range [{lo},{hi}]"));
    }
}
