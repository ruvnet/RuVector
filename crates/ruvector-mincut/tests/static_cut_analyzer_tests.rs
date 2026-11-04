//! Regression tests for `StaticCutAnalyzer` (nightly research 2026-11-04,
//! ADR-355). Pins the acceptance criteria measured in
//! `docs/research/nightly/2026-11-04-canonical-mincut-static-fast-path/`:
//! weighted correctness (against an independent brute-force oracle, not
//! `RuVectorGraphAnalyzer` -- that engine was found by this nightly run to
//! ignore edge weight), disconnected-graph handling, and determinism across
//! fresh rebuilds.

#![cfg(feature = "canonical")]

use ruvector_mincut::{DynamicGraph, StaticCutAnalyzer};
use std::sync::Arc;

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

fn graph_from_edges(edges: &[(usize, usize, f64)]) -> Arc<DynamicGraph> {
    let graph = Arc::new(DynamicGraph::new());
    for &(u, v, w) in edges {
        let _ = graph.insert_edge(u as u64, v as u64, w);
    }
    graph
}

#[test]
fn matches_brute_force_on_small_random_weighted_graphs() {
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
            let w = 0.5 + ((xorshift(&mut state) % 1000) as f64) / 100.0;
            edges.push((key.0, key.1, w));
        }
        edges
    }

    for &(n, m, seed) in &[
        (6usize, 8usize, 1u64),
        (8, 12, 2),
        (10, 16, 3),
        (12, 20, 4),
        (12, 30, 5),
        (13, 25, 6),
    ] {
        let edges = random_weighted_graph(n, m, seed);
        let truth = brute_force_min_cut(&edges);
        let got = StaticCutAnalyzer::new(graph_from_edges(&edges)).min_cut_f64();
        assert!(
            (got - truth).abs() < 1e-6,
            "n={n} m={m} seed={seed}: truth={truth} got={got}"
        );
    }
}

#[test]
fn weight_actually_changes_the_result() {
    // Two weight-1 triangles joined by a weight-100 bridge: a weight-aware
    // engine must prefer the 2-edge, weight-2 single-vertex isolation over
    // the 1-edge, weight-100 bridge cut.
    let edges: Vec<(usize, usize, f64)> = vec![
        (0, 1, 1.0),
        (1, 2, 1.0),
        (2, 0, 1.0),
        (3, 4, 1.0),
        (4, 5, 1.0),
        (5, 3, 1.0),
        (2, 3, 100.0),
    ];
    let analyzer = StaticCutAnalyzer::new(graph_from_edges(&edges));
    assert_eq!(analyzer.min_cut(), 2);
}

#[test]
fn disconnected_graph_reports_zero() {
    // Two disjoint weighted triangles, no edge between them.
    let edges: Vec<(usize, usize, f64)> = vec![
        (0, 1, 3.0),
        (1, 2, 3.0),
        (2, 0, 3.0),
        (3, 4, 3.0),
        (4, 5, 3.0),
        (5, 3, 3.0),
    ];
    let analyzer = StaticCutAnalyzer::new(graph_from_edges(&edges));
    assert_eq!(analyzer.min_cut(), 0);
    assert_eq!(analyzer.partition(), Some((Vec::new(), Vec::new())));
}

#[test]
fn trivial_graphs_report_zero() {
    let empty = Arc::new(DynamicGraph::new());
    assert_eq!(StaticCutAnalyzer::new(empty).min_cut(), 0);

    let single = Arc::new(DynamicGraph::new());
    single.add_vertex(0);
    assert_eq!(StaticCutAnalyzer::new(single).min_cut(), 0);
}

#[test]
fn deterministic_across_fresh_rebuilds() {
    // Two-clique-plus-bridge topology (19 vertices), reused from
    // crates/ruvector-agent-memory/examples/mincut_determinism_probe.rs,
    // across 30 independent fresh builds (not repeated calls on one
    // already-built object).
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
    let edges: Vec<(usize, usize, f64)> = (0..n)
        .flat_map(|i| {
            let mut sims: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, cosine_sim(&entries[i], &entries[j])))
                .filter(|&(_, s)| s >= min_sim)
                .collect();
            sims.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            sims.truncate(k);
            sims.into_iter()
                .map(move |(j, s)| (i, j, (1.0 - s).max(1e-4) as f64))
        })
        .collect();

    let mut cuts = std::collections::HashSet::new();
    let mut partitions = std::collections::HashSet::new();
    for _ in 0..30 {
        let analyzer = StaticCutAnalyzer::new(graph_from_edges(&edges));
        cuts.insert(analyzer.min_cut());
        if let Some((mut a, mut b)) = analyzer.partition() {
            a.sort_unstable();
            b.sort_unstable();
            if a.len() > b.len() || (a.len() == b.len() && a > b) {
                std::mem::swap(&mut a, &mut b);
            }
            partitions.insert((a, b));
        }
    }
    assert_eq!(
        cuts.len(),
        1,
        "min_cut() was not deterministic across fresh rebuilds"
    );
    assert_eq!(
        partitions.len(),
        1,
        "partition() was not deterministic across fresh rebuilds"
    );
}
