//! Nightly research 2026-10-20-polylog-mincut-connectivity-backend.
//!
//! Tests the "Next Research" item left by
//! `docs/research/nightly/2026-09-11-mincut-partition-determinism/README.md`:
//! root-cause `RuVectorGraphAnalyzer::partition()`'s latency scaling
//! (ADR-345: ~77ms at n=50 to ~11.4s at n=400) by evaluating
//! `connectivity::polylog::PolylogConnectivity`, already in-tree but unused
//! by any `ProperCutInstance`, as a replacement backend.
//!
//! Isolates exactly one variable: `BoundedInstanceImpl`'s disconnection
//! fast-path, which `query()` evaluates unconditionally on every call
//! (before its witness cache is even consulted — see `bounded.rs`). Three
//! variants share 100% of the surrounding code (LocalKCut search, brute
//! force, caching, certificate tracking):
//!
//!   baseline   = `BoundedInstance`         (BFS-from-scratch, pre-existing)
//!   candidate_A = `PolylogBoundedInstance`  (arXiv:2510.08297)
//!   candidate_B = `EttBoundedInstance`      (Euler Tour Tree + union-find,
//!                                            already used one layer up by
//!                                            `MinCutWrapper` itself)
//!
//! Two phases per graph size, reusing ADR-345's own ring k-NN generator
//! (`mincut_scaling_probe.rs` in ruvector-agent-memory) so the "cold"
//! numbers below are directly comparable to the ADR-345 table:
//!
//!   cold:  one fresh wrapper, all edges inserted, one `query()` call
//!          (dominated by `search_for_cuts`'s graph rebuild + budget loop —
//!          expected near-identical across variants; this is the control).
//!   warm:  repeats steady-state `query()` calls on the *same* wrapper with
//!          no graph mutation in between, meant to isolate the disconnection
//!          fast-path's cost (it runs unconditionally on every call, before
//!          the witness cache is even consulted). It does NOT end up doing
//!          that cleanly: `BoundedInstanceImpl::query()` only caches
//!          `ValueInRange` results, never `AboveRange` ones, so every
//!          instance below the one that finally matches re-runs the full
//!          `search_for_cuts()` (graph rebuild + budget/seed loop) on every
//!          single warm call too, not just the cold one. This was
//!          discovered empirically (n=50's first run: warm_mean ~27ms,
//!          roughly on par with a per-instance share of cold_query, not the
//!          microseconds a pure connectivity-check difference would
//!          predict) and is reported as a finding in its own right, not
//!          smoothed over. `WARM_REPEATS` is kept small as a result — this
//!          workload does not amortize the way its name suggests.
//!
//! Acceptance threshold (fixed before the n=50 run that revealed the above):
//! ACCEPT if a candidate's mean per-call `warm` latency is at least 5x lower
//! than baseline's at the largest size reached, AND cut_value/partition
//! membership match baseline exactly at every n, AND cut_value is identical
//! across `DETERMINISM_REBUILDS` independent rebuilds at `DETERMINISM_SIZES`
//! (building on the 2026-09-11 determinism fix). REJECT otherwise. Both
//! `cold` and `warm` numbers are reported regardless of outcome.

use ruvector_mincut::graph::{DynamicGraph, VertexId};
use ruvector_mincut::wrapper::MinCutResult;
use ruvector_mincut::{
    BoundedInstance, EttBoundedInstance, MinCutWrapper, PolylogBoundedInstance, ProperCutInstance,
};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Kept small: each "warm" call costs close to a full `search_for_cuts()`
/// pass per untried instance, not microseconds -- see module docs.
const WARM_REPEATS: usize = 5;
const DETERMINISM_REBUILDS: usize = 5;
/// Determinism rebuilds repeat the *cold* (expensive) path, so they only
/// run at this size, not across the whole sweep -- see module docs.
const DETERMINISM_SIZES: &[usize] = &[50];

/// Same ring k-NN generator as
/// `ruvector-agent-memory/examples/mincut_scaling_probe.rs`, reproduced
/// here (not imported — that crate depends on this one, not the reverse)
/// so n=50/100/200/400 are directly comparable to the ADR-345 table.
fn ring_graph(n: usize, k: usize) -> Vec<(usize, Vec<(usize, f64)>)> {
    (0..n)
        .map(|i| {
            let nbrs: Vec<(usize, f64)> = (1..=k)
                .map(|d| ((i + d) % n, 0.1 + (d as f64) * 0.01))
                .collect();
            (i, nbrs)
        })
        .collect()
}

fn build_dynamic_graph(neighbors: &[(usize, Vec<(usize, f64)>)]) -> Arc<DynamicGraph> {
    let graph = Arc::new(DynamicGraph::new());
    for &(vertex, ref nn_list) in neighbors {
        for &(neighbor, distance) in nn_list {
            let weight = if distance > 0.0 { 1.0 / distance } else { 1.0 };
            let _ = graph.insert_edge(vertex as u64, neighbor as u64, weight);
        }
    }
    graph
}

fn wrapper_for<F>(graph: &Arc<DynamicGraph>, factory: F) -> MinCutWrapper
where
    F: Fn(&DynamicGraph, u64, u64) -> Box<dyn ProperCutInstance> + Send + Sync + 'static,
{
    let mut wrapper = MinCutWrapper::with_factory(Arc::clone(graph), factory);
    for edge in graph.edges() {
        wrapper.insert_edge(edge.id, edge.source, edge.target);
    }
    wrapper
}

/// Mirrors `RuVectorGraphAnalyzer::partition()`'s materialization exactly
/// (see `integration/mod.rs`), reimplemented here because that type does
/// not expose `MinCutWrapper::with_factory` injection.
fn partition_from(
    graph: &DynamicGraph,
    result: &MinCutResult,
) -> (u64, Vec<VertexId>, Vec<VertexId>) {
    match result {
        MinCutResult::Disconnected => (0, Vec::new(), Vec::new()),
        MinCutResult::Value { witness, .. } => {
            let value = result.value();
            let mut side_a = Vec::new();
            let mut side_b = Vec::new();
            for v in graph.vertices() {
                if witness.contains(v) {
                    side_a.push(v);
                } else {
                    side_b.push(v);
                }
            }
            side_a.sort_unstable();
            side_b.sort_unstable();
            (value, side_a, side_b)
        }
    }
}

struct VariantResult {
    name: &'static str,
    cold_build: Duration,
    cold_query: Duration,
    warm_mean: Duration,
    warm_min: Duration,
    warm_max: Duration,
    cut_value: u64,
    partition: (Vec<VertexId>, Vec<VertexId>),
    /// `None` when not checked at this size (see `DETERMINISM_SIZES`).
    determinism_stable: Option<bool>,
}

fn run_variant<F>(
    name: &'static str,
    neighbors: &[(usize, Vec<(usize, f64)>)],
    factory: F,
    check_determinism: bool,
) -> VariantResult
where
    F: Fn(&DynamicGraph, u64, u64) -> Box<dyn ProperCutInstance> + Send + Sync + Clone + 'static,
{
    // Cold phase.
    let t0 = Instant::now();
    let graph = build_dynamic_graph(neighbors);
    let mut wrapper = wrapper_for(&graph, factory.clone());
    let cold_build = t0.elapsed();

    let t1 = Instant::now();
    let result = wrapper.query();
    let cold_query = t1.elapsed();
    let (cut_value, side_a, side_b) = partition_from(&graph, &result);

    // Warm phase: repeated query() on the SAME wrapper, no mutation.
    let mut durations = Vec::with_capacity(WARM_REPEATS);
    for _ in 0..WARM_REPEATS {
        let t = Instant::now();
        let _ = wrapper.query();
        durations.push(t.elapsed());
    }
    let total: Duration = durations.iter().sum();
    let warm_mean = total / (WARM_REPEATS as u32);
    let warm_min = *durations.iter().min().unwrap();
    let warm_max = *durations.iter().max().unwrap();

    // Determinism: fresh rebuild, repeated, same input -> same cut_value.
    // Only run at DETERMINISM_SIZES: this repeats the expensive cold path,
    // so running it at every swept size would be prohibitively slow at
    // n>=200 (see ADR-345's own scaling numbers).
    let determinism_stable = if check_determinism {
        let mut values = Vec::with_capacity(DETERMINISM_REBUILDS);
        for _ in 0..DETERMINISM_REBUILDS {
            let g = build_dynamic_graph(neighbors);
            let mut w = wrapper_for(&g, factory.clone());
            values.push(w.query().value());
        }
        Some(values.iter().all(|&v| v == values[0]))
    } else {
        None
    };

    VariantResult {
        name,
        cold_build,
        cold_query,
        warm_mean,
        warm_min,
        warm_max,
        cut_value,
        partition: (side_a, side_b),
        determinism_stable,
    }
}

fn fmt_us(d: Duration) -> String {
    format!("{:.3}us", d.as_secs_f64() * 1_000_000.0)
}

fn fmt_ms(d: Duration) -> String {
    format!("{:.3}ms", d.as_secs_f64() * 1_000.0)
}

fn main() {
    use std::io::Write;
    let out = std::io::stdout();

    println!("ruvector-mincut: connectivity-backend bench (baseline vs PolylogConnectivity vs DynamicConnectivity)");
    println!("WARM_REPEATS={WARM_REPEATS} DETERMINISM_REBUILDS={DETERMINISM_REBUILDS} DETERMINISM_SIZES={DETERMINISM_SIZES:?}\n");
    let _ = out.lock().flush();

    // Matches 4 of 5 points in ADR-345's own scaling table (n=50,100,200,400).
    // n=19 is deliberately excluded: at n<20, `BoundedInstanceImpl::query()`
    // takes the brute-force path (`vertices.len() < 20`) regardless of
    // which lambda-range instance is asking, so `MinCutWrapper` ends up
    // recomputing the *same* O(2^n * m) global brute-force search once per
    // lambda-range instance tried before reaching the one whose range
    // covers the true cut value -- with RANGE_FACTOR=1.2 (slow growth),
    // that was observed to be 10+ repeated full recomputations, independent
    // of the connectivity-backend variable this experiment isolates. That
    // is itself a real finding (see the nightly README's "Next Research"),
    // but it is orthogonal to this experiment's hypothesis and exceeded this
    // run's time budget, so n=19 is left out of the swept sizes here.
    let sizes = [50usize, 100, 200, 400];
    let k = 8usize;

    let mut all_agree = true;

    for &n in &sizes {
        let neighbors = ring_graph(n, k);
        let check_determinism = DETERMINISM_SIZES.contains(&n);

        let baseline = run_variant(
            "baseline(Bfs)",
            &neighbors,
            |g, min, max| Box::new(BoundedInstance::init(g, min, max)),
            check_determinism,
        );
        let candidate_a = run_variant(
            "candidate_A(Polylog)",
            &neighbors,
            |g, min, max| Box::new(PolylogBoundedInstance::init(g, min, max)),
            check_determinism,
        );
        let candidate_b = run_variant(
            "candidate_B(Ett)",
            &neighbors,
            |g, min, max| Box::new(EttBoundedInstance::init(g, min, max)),
            check_determinism,
        );

        println!("=== n={n} (k={k}) ===");
        for v in [&baseline, &candidate_a, &candidate_b] {
            let determinism_label = match v.determinism_stable {
                Some(true) => "stable",
                Some(false) => "UNSTABLE",
                None => "not-checked",
            };
            println!(
                "{:<22} cold_build={:>10} cold_query={:>10} warm_mean={:>12} warm_min={:>12} warm_max={:>12} cut={} determinism={}",
                v.name,
                fmt_ms(v.cold_build),
                fmt_ms(v.cold_query),
                fmt_us(v.warm_mean),
                fmt_us(v.warm_min),
                fmt_us(v.warm_max),
                v.cut_value,
                determinism_label,
            );
        }
        let _ = out.lock().flush();

        let agree_a = candidate_a.cut_value == baseline.cut_value
            && candidate_a.partition == baseline.partition;
        let agree_b = candidate_b.cut_value == baseline.cut_value
            && candidate_b.partition == baseline.partition;
        if !agree_a || !agree_b {
            all_agree = false;
            println!(
                "  CORRECTNESS MISMATCH at n={n}: candidate_A agrees={agree_a} candidate_B agrees={agree_b}"
            );
        }

        let speedup_a =
            baseline.warm_mean.as_secs_f64() / candidate_a.warm_mean.as_secs_f64().max(1e-12);
        let speedup_b =
            baseline.warm_mean.as_secs_f64() / candidate_b.warm_mean.as_secs_f64().max(1e-12);
        let cold_speedup_a =
            baseline.cold_query.as_secs_f64() / candidate_a.cold_query.as_secs_f64().max(1e-12);
        println!(
            "  warm speedup: candidate_A={speedup_a:.2}x candidate_B={speedup_b:.2}x | cold(end-to-end) speedup: candidate_A={cold_speedup_a:.2}x"
        );
        println!();
    }

    println!(
        "all_variants_agree_with_baseline={all_agree} (correctness invariant across all sizes)"
    );
}
