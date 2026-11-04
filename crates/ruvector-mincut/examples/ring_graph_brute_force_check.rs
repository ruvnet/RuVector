//! Throwaway diagnostic (not part of the shipped research artifact):
//! `brute_force_cross_check` validated `StaticCutAnalyzer` against small
//! *generic* random weighted graphs and found exact agreement. This probe
//! adds a second, independent weighted-graph family -- a circulant ring
//! with heavy, exact tie-degeneracy (every vertex is structurally
//! identical, so isolating *any* single vertex yields the *same* minimum
//! cut value) -- to check whether `StaticCutAnalyzer` still agrees with
//! ground truth under that much higher degree of cut-value degeneracy.
//!
//! IMPORTANT: edges here carry their final weight directly (`insert_edge`
//! is called with the formula's output as-is). This is deliberately
//! *not* the same graph as `static_vs_dynamic_cut_bench`'s `ring_knn` +
//! `StaticCutAnalyzer::from_knn`, whose `from_knn` constructor additionally
//! inverts each input number as `1.0 / distance` before building edges
//! (matching `RuVectorGraphAnalyzer::from_knn`'s existing convention). The
//! two are structurally related (same offsets) but numerically different
//! graphs; do not compare this file's cut values against that benchmark's
//! "cut(cand)" column directly -- each is internally consistent and
//! correct for the graph it actually builds, not for the other's.
use ruvector_mincut::{DynamicGraph, StaticCutAnalyzer};
use std::sync::Arc;

fn ring_knn_edges(n: usize, k: usize) -> Vec<(usize, usize, f64)> {
    let mut edges = Vec::new();
    for i in 0..n {
        for d in 1..=k {
            let j = (i + d) % n;
            let w = 0.1 + (d as f64) * 0.01;
            edges.push((i, j, w));
        }
    }
    edges
}

fn brute_force_min_cut(n: usize, edges: &[(usize, usize, f64)]) -> f64 {
    let mut adj = vec![vec![0.0f64; n]; n];
    for &(u, v, w) in edges {
        adj[u][v] += w;
        adj[v][u] += w;
    }
    let mut best = f64::INFINITY;
    for mask in 1u32..(1u32 << (n - 1)) {
        let mut cut = 0.0;
        for u in 0..n {
            for v in (u + 1)..n {
                let u_in = (mask >> u) & 1 == 1;
                let v_in = (mask >> v) & 1 == 1;
                if u_in != v_in {
                    cut += adj[u][v];
                }
            }
        }
        if cut < best {
            best = cut;
        }
    }
    best
}

fn single_vertex_isolation_cost(n: usize, k: usize) -> f64 {
    // Every vertex in this circulant graph has identical total incident
    // weight: k outgoing edges (weights 0.11..0.1+0.01k) plus k incoming
    // edges from predecessors (same weight formula), so this is a
    // closed-form upper bound on the true minimum cut, independent of n.
    let one_way: f64 = (1..=k).map(|d| 0.1 + (d as f64) * 0.01).sum();
    2.0 * one_way
}

fn main() {
    println!(
        "{:>6} {:>14} {:>16} {:>16}",
        "n", "k", "brute_force", "static_cut_analyzer"
    );
    for &(n, k) in &[
        (8usize, 3usize),
        (10, 4),
        (12, 5),
        (13, 6),
        (14, 6),
        (19, 8),
        (17, 8),
        (20, 8),
    ] {
        let edges = ring_knn_edges(n, k);
        let truth = brute_force_min_cut(n, &edges);
        let bound = single_vertex_isolation_cost(n, k);

        let graph = Arc::new(DynamicGraph::new());
        for &(u, v, w) in &edges {
            let _ = graph.insert_edge(u as u64, v as u64, w);
        }
        let got = StaticCutAnalyzer::new(graph).min_cut();

        println!(
            "{n:>6} {k:>14} {truth:>16.3} {got:>16} (single-vertex-isolation bound: {bound:.3})"
        );
    }
}
