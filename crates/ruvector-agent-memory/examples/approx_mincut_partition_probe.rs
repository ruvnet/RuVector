//! Minimal, executable correctness probe for the nightly 2026-09-19
//! follow-up to ADR-345: does `ruvector_mincut::ApproxMinCut`'s reported
//! `partition` actually correspond to the minimum cut it computed?
//!
//! Two graphs, run in order because the first result changed this
//! experiment's design (see the "balanced" case below):
//!
//! 1. **Balanced**: the textbook "two triangles joined by a bridge" (also a
//!    unit test in `ruvector-mincut`'s own
//!    `algorithm::mod::tests::test_bridge_graph`, which only checks the cut
//!    *value*, not the partition) — two equal-size (3-vertex) sides.
//! 2. **Unbalanced**: a 9-vertex clique joined by one weak bridge edge to a
//!    3-vertex clique — sides of very different sizes.
//!
//! `compute_partition` (`crates/ruvector-mincut/src/algorithm/approximate.rs`)
//! does not use the cut it just computed at all: it BFS-walks from an
//! arbitrary start vertex and stops once it has visited exactly half the
//! *total* vertex count, regardless of where the graph's actual weak edges
//! are. On the balanced graph this coincidentally reproduces the correct
//! split (BFS from inside one triangle visits exactly 3 vertices — that
//! triangle — before needing to cross the bridge, and 3 is also "half" of 6).
//! It cannot coincidentally succeed on the unbalanced graph, where "half of
//! 12" is 6 — a number that does not correspond to either true side (9 or 3)
//! — so a real disconnect between the reported partition and the reported
//! cut can only be demonstrated unambiguously there. Run both, in this
//! order, so the balanced case's misleadingly clean result is not reported
//! in isolation.
//!
//! Run:
//!   cargo run --release -p ruvector-agent-memory --example approx_mincut_partition_probe --features mincut-forget

use ruvector_mincut::ApproxMinCut;
use std::collections::HashSet;

fn clique_edges(vertices: &[u64], weight: f64) -> Vec<(u64, u64, f64)> {
    let mut edges = Vec::new();
    for i in 0..vertices.len() {
        for j in (i + 1)..vertices.len() {
            edges.push((vertices[i], vertices[j], weight));
        }
    }
    edges
}

/// Runs one probe: builds `edges`, queries `ApproxMinCut`, and checks the
/// reported partition against `expected_a`/`expected_b` (either assignment
/// of sides counts as a match). Returns whether it matched.
fn probe(label: &str, edges: &[(u64, u64, f64)], expected_a: &[u64], expected_b: &[u64]) -> bool {
    let mut approx = ApproxMinCut::default();
    for &(u, v, w) in edges {
        approx.insert_edge(u, v, w);
    }
    let result = approx.min_cut();

    println!("{label}");
    println!("  vertices           : {}", approx.vertex_count());
    println!("  edges              : {}", edges.len());
    println!("  reported cut value : {:.3}", result.value);
    println!(
        "  bounds             : [{:.3}, {:.3}]",
        result.lower_bound, result.upper_bound
    );

    let expected_a_set: HashSet<u64> = expected_a.iter().copied().collect();
    let expected_b_set: HashSet<u64> = expected_b.iter().copied().collect();

    let matches = match &result.partition {
        Some((a, b)) => {
            let a_set: HashSet<u64> = a.iter().copied().collect();
            let b_set: HashSet<u64> = b.iter().copied().collect();
            let mut a_sorted = a.clone();
            let mut b_sorted = b.clone();
            a_sorted.sort_unstable();
            b_sorted.sort_unstable();
            println!(
                "  reported partition : {a_sorted:?} | {b_sorted:?} (sizes {}/{})",
                a_sorted.len(),
                b_sorted.len()
            );
            (a_set == expected_a_set && b_set == expected_b_set)
                || (a_set == expected_b_set && b_set == expected_a_set)
        }
        None => {
            println!("  reported partition : None");
            false
        }
    };
    println!(
        "  expected partition : {expected_a:?} | {expected_b:?} (sizes {}/{})",
        expected_a.len(),
        expected_b.len()
    );
    println!(
        "  partition matches the true min cut        : {}",
        if matches { "YES" } else { "NO" }
    );
    println!();
    matches
}

fn main() {
    // 1. Balanced: triangle 1 (1-2-3), bridge 3-4 (weight 1, the unique
    // minimum cut), triangle 2 (4-5-6).
    let mut balanced_edges = clique_edges(&[1, 2, 3], 2.0);
    balanced_edges.push((3, 4, 1.0));
    balanced_edges.extend(clique_edges(&[4, 5, 6], 2.0));
    let balanced_match = probe(
        "1. BALANCED graph (two 3-vertex triangles + bridge)",
        &balanced_edges,
        &[1, 2, 3],
        &[4, 5, 6],
    );

    // 2. Unbalanced: a 9-vertex clique (1..=9) joined by one weak bridge
    // (9-10, weight 1) to a 3-vertex clique (10, 11, 12). "Half of 12" is 6,
    // which is neither 9 nor 3 — no BFS-half-split can match this cut.
    let big: Vec<u64> = (1..=9).collect();
    let small: Vec<u64> = vec![10, 11, 12];
    let mut unbalanced_edges = clique_edges(&big, 5.0);
    unbalanced_edges.push((9, 10, 1.0));
    unbalanced_edges.extend(clique_edges(&small, 5.0));
    let unbalanced_match = probe(
        "2. UNBALANCED graph (9-vertex clique + bridge + 3-vertex clique)",
        &unbalanced_edges,
        &big,
        &small,
    );

    if balanced_match && unbalanced_match {
        println!(
            "=> ApproxMinCut::compute_partition correctly reflects its own min_cut() value on \
             both graphs. (If this differs from the nightly 2026-09-19 finding — balanced YES, \
             unbalanced NO — the upstream implementation has likely been fixed; see \
             docs/research/nightly/2026-09-19-approx-mincut-forgetting/README.md.)"
        );
    } else if balanced_match && !unbalanced_match {
        println!(
            "=> CONFIRMED: ApproxMinCut's reported cut *value* is credible (Stoer-Wagner: exact \
             for graphs of <=50 edges, which both probe graphs are), but its reported \
             *partition* is a plain BFS walk that stops once half of the *total* vertex count \
             has been visited (`compute_partition` in \
             crates/ruvector-mincut/src/algorithm/approximate.rs), independent of the cut value. \
             On the balanced graph this coincidentally reproduces the correct split (half of 6 \
             is 3, which is also each side's true size). It cannot coincidentally succeed on the \
             unbalanced graph (half of 12 is 6, matching neither true side of 9 or 3), which is \
             exactly what was measured. The partition is not a general-purpose structural \
             boundary/bridge-detection signal."
        );
    } else {
        println!("=> UNEXPECTED: see the per-graph results above; the research doc should be updated to match.");
    }
}
