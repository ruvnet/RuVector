//! Nightly research benchmark (2026-11-04): `StaticCutAnalyzer` vs
//! `RuVectorGraphAnalyzer` on one-shot (build-once, query-once) minimum-cut
//! queries over a static graph snapshot.
//!
//! Background: ADR-345 (2026-09-05) measured `RuVectorGraphAnalyzer::partition()`
//! -- the fully-dynamic paper algorithm behind `MinCutWrapper` -- costing
//! ~77ms at n=50 vertices and ~11.4s at n=400 on a single cold query, and
//! rejected a `ruvector-agent-memory` integration partly on that basis.
//! ADR-346 (2026-09-11) separately fixed a correctness bug (non-deterministic
//! degenerate partitions) in the same code path but left the latency
//! unaddressed. This benchmark tests whether the already-implemented,
//! previously disconnected `canonical::CactusGraph` (Stoer-Wagner) engine,
//! wrapped here as `StaticCutAnalyzer`, is a faster drop-in for the one-shot
//! case while producing an identical cut value.
//!
//! Run with:
//!   cargo run --release -p ruvector-mincut --features canonical,integration \
//!     --example static_vs_dynamic_cut_bench
//!
//! Pre-registered acceptance thresholds (set before this benchmark was run;
//! see docs/research/nightly/2026-11-04-canonical-mincut-static-fast-path/):
//!   1. Cut value from StaticCutAnalyzer::min_cut() must equal the cut value
//!      from RuVectorGraphAnalyzer::min_cut() on every tested graph. Any
//!      mismatch is an automatic REJECT (correctness is non-negotiable).
//!   2. StaticCutAnalyzer::partition() must be deterministic (byte-identical
//!      partition + cut value) across >=20 fresh rebuilds of the same graph.
//!   3. StaticCutAnalyzer must be at least 20x faster than
//!      RuVectorGraphAnalyzer at n=200 and n=400 on the ring k-NN graph.
//! Gate 3 not being met at a given size is reported as a scaling-curve data
//! point, not silently dropped -- see the "crossover" note in the summary.

use ruvector_mincut::{RuVectorGraphAnalyzer, StaticCutAnalyzer};
use std::collections::HashSet;
use std::time::Instant;

/// Fixed-degree ring k-NN graph -- identical generator to
/// `crates/ruvector-agent-memory/examples/mincut_scaling_probe.rs`, reused
/// here (not re-derived) so the new numbers are directly comparable to the
/// ADR-345 baseline figures quoted above.
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

/// Deterministic pseudo-random sparse k-NN graph (xorshift64, fixed seed) --
/// a second topology so the comparison is not reporting a result that only
/// holds for one convenient, highly regular shape.
fn random_knn(n: usize, k: usize, seed: u64) -> Vec<(usize, Vec<(usize, f64)>)> {
    let mut state = seed.max(1);
    let mut next_u64 = move || {
        // xorshift64*
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    (0..n)
        .map(|i| {
            let mut nbrs = Vec::with_capacity(k);
            for _ in 0..k {
                let j = (next_u64() as usize) % n;
                if j != i {
                    let w = 0.05 + ((next_u64() % 1000) as f64) / 1000.0;
                    nbrs.push((j, w));
                }
            }
            (i, nbrs)
        })
        .collect()
}

/// Exact two-clique-plus-bridge 19-vertex topology from
/// `crates/ruvector-agent-memory/examples/mincut_determinism_probe.rs`,
/// reused unmodified for the determinism / correctness cross-check.
fn two_clique_bridge_19() -> Vec<(usize, Vec<(usize, f64)>)> {
    fn normalize3(v: [f32; 3]) -> Vec<f32> {
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        vec![v[0] / n, v[1] / n, v[2] / n]
    }
    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < 1e-9 || nb < 1e-9 {
            0.0
        } else {
            (dot / (na * nb)).clamp(-1.0, 1.0)
        }
    }

    let mut entries: Vec<Vec<f32>> = Vec::new();
    for axis in 0..2 {
        let plain = if axis == 0 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let gateway = if axis == 0 {
            normalize3([1.0, 0.0, 0.5])
        } else {
            normalize3([0.0, 1.0, 0.5])
        };
        for _ in 0..8 {
            entries.push(plain.to_vec());
        }
        entries.push(gateway);
    }
    entries.push(vec![0.0, 0.0, 1.0]);
    let n = entries.len();

    let k = 8usize;
    let min_sim = 0.05f32;
    (0..n)
        .map(|i| {
            let mut sims: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, cosine_sim(&entries[i], &entries[j])))
                .filter(|&(_, s)| s >= min_sim)
                .collect();
            sims.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            sims.truncate(k);
            let dists = sims
                .into_iter()
                .map(|(j, s)| (j, (1.0 - s).max(1e-4) as f64))
                .collect();
            (i, dists)
        })
        .collect()
}

fn time_baseline(neighbors: &[(usize, Vec<(usize, f64)>)]) -> (f64, u64) {
    let t0 = Instant::now();
    let mut analyzer = RuVectorGraphAnalyzer::from_knn(neighbors);
    let cut = analyzer.min_cut();
    (t0.elapsed().as_secs_f64() * 1000.0, cut)
}

fn time_candidate(neighbors: &[(usize, Vec<(usize, f64)>)]) -> (f64, u64) {
    let t0 = Instant::now();
    let analyzer = StaticCutAnalyzer::from_knn(neighbors);
    let cut = analyzer.min_cut();
    (t0.elapsed().as_secs_f64() * 1000.0, cut)
}

fn run_scaling_table(
    label: &str,
    gen: impl Fn(usize) -> Vec<(usize, Vec<(usize, f64)>)>,
    sizes: &[usize],
) {
    println!("\n=== Scaling: {label} ===");
    println!(
        "{:>6} {:>14} {:>14} {:>10} {:>12} {:>12}",
        "n", "baseline(ms)", "candidate(ms)", "speedup", "cut(base)", "cut(cand)"
    );
    for &n in sizes {
        let neighbors = gen(n);
        let (base_ms, base_cut) = time_baseline(&neighbors);
        let (cand_ms, cand_cut) = time_candidate(&neighbors);
        let speedup = if cand_ms > 0.0 {
            base_ms / cand_ms
        } else {
            f64::INFINITY
        };
        let match_mark = if base_cut == cand_cut {
            ""
        } else {
            "  <-- MISMATCH"
        };
        println!(
            "{:>6} {:>14.3} {:>14.3} {:>9.1}x {:>12} {:>12}{}",
            n, base_ms, cand_ms, speedup, base_cut, cand_cut, match_mark
        );
    }
}

fn run_determinism_check(trials: usize) {
    println!("\n=== Determinism + correctness cross-check: two-clique-bridge (n=19), {trials} fresh rebuilds ===");
    let neighbors = two_clique_bridge_19();

    let (_, trusted_cut) = time_baseline(&neighbors);

    let mut cuts = HashSet::new();
    let mut partitions = HashSet::new();
    let mut mismatches = 0usize;
    let t0 = Instant::now();
    for _ in 0..trials {
        let analyzer = StaticCutAnalyzer::from_knn(&neighbors);
        let cut = analyzer.min_cut();
        if cut != trusted_cut {
            mismatches += 1;
        }
        cuts.insert(cut);

        if let Some((mut a, mut b)) = analyzer.partition() {
            a.sort_unstable();
            b.sort_unstable();
            if a.len() > b.len() || (a.len() == b.len() && a > b) {
                std::mem::swap(&mut a, &mut b);
            }
            partitions.insert((a, b));
        }
    }
    let elapsed = t0.elapsed();

    println!(
        "trials={trials} elapsed={:.3}s avg_per_call={:.3}ms",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1000.0 / trials as f64
    );
    println!("trusted cut value (RuVectorGraphAnalyzer): {trusted_cut}");
    println!(
        "distinct cut values across {trials} fresh StaticCutAnalyzer rebuilds: {} {}",
        cuts.len(),
        if cuts.len() == 1 {
            "(deterministic)"
        } else {
            "(NON-DETERMINISTIC)"
        }
    );
    println!(
        "distinct partitions across {trials} fresh rebuilds: {} {}",
        partitions.len(),
        if partitions.len() == 1 {
            "(deterministic)"
        } else {
            "(NON-DETERMINISTIC)"
        }
    );
    println!("cut-value mismatches vs trusted baseline: {mismatches}/{trials}");
}

/// Hand-verifiable sanity check: two weight-10 triangles joined by a single
/// weight-1 bridge edge. The unique global minimum cut is the bridge itself
/// (value 1): any other cut must sever at least one weight-10 triangle edge.
/// Run first, before the harder-to-verify k-NN topologies, to tell a real
/// implementation bug apart from a weight-convention mismatch between the
/// two engines.
fn run_sanity_check() {
    println!(
        "\n=== Sanity check: two weight-10 triangles + weight-1 bridge (expected min cut = 1) ==="
    );
    use ruvector_mincut::DynamicGraph;
    use std::sync::Arc;

    let graph = Arc::new(DynamicGraph::new());
    let tri_edges = [(0u64, 1u64), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)];
    for (u, v) in tri_edges {
        let _ = graph.insert_edge(u, v, 10.0);
    }
    let _ = graph.insert_edge(2, 3, 1.0);

    let mut baseline = RuVectorGraphAnalyzer::new(Arc::clone(&graph));
    let base_cut = baseline.min_cut();
    let base_partition = baseline.partition();

    let candidate = StaticCutAnalyzer::new(Arc::clone(&graph));
    let cand_cut = candidate.min_cut();
    let cand_partition = candidate.partition();

    println!("expected: 1");
    println!("baseline (RuVectorGraphAnalyzer): cut={base_cut} partition={base_partition:?}");
    println!("candidate (StaticCutAnalyzer):    cut={cand_cut} partition={cand_partition:?}");
}

fn main() {
    run_sanity_check();
    run_determinism_check(30);

    let ring_sizes = [19usize, 30, 50, 100, 200, 400];
    run_scaling_table("ring k-NN (k=8)", |n| ring_knn(n, 8), &ring_sizes);

    let random_sizes = [19usize, 30, 50, 100, 200, 400];
    run_scaling_table(
        "random sparse k-NN (k=6, seed=0x5EED)",
        |n| random_knn(n, 6, 0x5EED),
        &random_sizes,
    );

    println!("\n=== Candidate-only extended scaling (baseline skipped above n=400 -- ADR-345 already measured ~11.4s there; not re-measured here) ===");
    for &n in &[800usize, 1600] {
        let neighbors = ring_knn(n, 8);
        let (cand_ms, cand_cut) = time_candidate(&neighbors);
        println!("n={n:<6} candidate={cand_ms:>12.3}ms cut={cand_cut}");
    }
}
