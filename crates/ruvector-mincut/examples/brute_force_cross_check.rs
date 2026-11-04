//! Throwaway validator (not part of the shipped research artifact):
//! cross-checks `StaticCutAnalyzer::min_cut()` against an exhaustive,
//! independently-written brute-force weighted minimum cut (enumerate all
//! 2^(n-1) non-trivial vertex subsets, sum crossing edge weights, take the
//! min) on small random weighted graphs. This does not depend on
//! `RuVectorGraphAnalyzer` at all, so it is a correctness oracle that is not
//! itself in question -- unlike the baseline engine, which this nightly run
//! found computes an unweighted cut.
use ruvector_mincut::{DynamicGraph, StaticCutAnalyzer};
use std::sync::Arc;

/// Computes the brute-force weighted min cut over exactly the vertex set
/// that appears in `edges` -- matching `DynamicGraph`'s own behavior, where
/// a vertex index that never appears in an `insert_edge` call is never
/// registered as a graph vertex at all (not an isolated, trivially-cuttable
/// vertex). An earlier version of this check assumed every index in
/// `0..n_requested` was present and wrongly flagged `StaticCutAnalyzer` as
/// buggy on a graph where the random generator happened to skip one vertex
/// index; this is the corrected version.
fn brute_force_min_cut(edges: &[(usize, usize, f64)]) -> f64 {
    let mut ids: Vec<usize> = edges.iter().flat_map(|&(u, v, _)| [u, v]).collect();
    ids.sort_unstable();
    ids.dedup();
    let n = ids.len();
    let idx = |v: usize| ids.binary_search(&v).unwrap();

    let mut adj = vec![vec![0.0f64; n]; n];
    for &(u, v, w) in edges {
        let (iu, iv) = (idx(u), idx(v));
        adj[iu][iv] += w;
        adj[iv][iu] += w;
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

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn random_weighted_graph(n: usize, m: usize, seed: u64) -> Vec<(usize, usize, f64)> {
    let mut state = seed.max(1);
    let mut edges = Vec::with_capacity(m);
    let mut seen = std::collections::HashSet::new();
    while edges.len() < m {
        let u = (xorshift(&mut state) as usize) % n;
        let v = (xorshift(&mut state) as usize) % n;
        if u == v {
            continue;
        }
        let key = (u.min(v), u.max(v));
        if !seen.insert(key) {
            continue;
        }
        let w = 0.5 + ((xorshift(&mut state) % 1000) as f64) / 100.0; // 0.5..10.5
        edges.push((key.0, key.1, w));
    }
    edges
}

fn main() {
    // Explicit disconnected case: two disjoint weighted triangles, no edges
    // between them. True min cut is 0.
    {
        let graph = Arc::new(DynamicGraph::new());
        for (u, v) in [(0u64, 1u64), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)] {
            let _ = graph.insert_edge(u, v, 3.0);
        }
        let candidate = StaticCutAnalyzer::new(graph);
        let got = candidate.min_cut();
        println!(
            "disconnected-graph case: candidate={got} {}",
            if got == 0 { "PASS" } else { "FAIL" }
        );
    }

    let mut all_pass = true;
    for (trial, &(n, m, seed)) in [
        (6usize, 8usize, 1u64),
        (8, 12, 2),
        (10, 16, 3),
        (12, 20, 4),
        (12, 30, 5),
        (13, 25, 6),
    ]
    .iter()
    .enumerate()
    {
        let edges = random_weighted_graph(n, m, seed);
        let truth = brute_force_min_cut(&edges);

        let graph = Arc::new(DynamicGraph::new());
        for &(u, v, w) in &edges {
            let _ = graph.insert_edge(u as u64, v as u64, w);
        }
        let candidate = StaticCutAnalyzer::new(graph);
        let got = candidate.min_cut() as f64;

        let pass = (got - truth).abs() < 0.5; // StaticCutAnalyzer rounds to nearest u64
        all_pass &= pass;
        println!(
            "trial {trial}: n={n} m={m} seed={seed} brute_force_truth={truth:.3} candidate={got:.3} {}",
            if pass { "PASS" } else { "FAIL" }
        );
    }
    println!(
        "\noverall: {}",
        if all_pass { "ALL PASS" } else { "SOME FAILED" }
    );
}
