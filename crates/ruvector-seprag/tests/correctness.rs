//! M0 correctness gate — exit criteria from
//! `docs/plans/seprag-cch-retrieval/M0-correctness-gate.md`.
//!
//! 1. SepRAG k-NN == brute-force Dijkstra oracle (the gate).
//! 2. Pruned == unpruned (pruning never drops a true top-k).
//! 3. Pruning reduces search space (region pruning fires).
//! 4. Determinism.
//! 5. Blowup ratio is bounded on road-like synthetic graphs.

use ruvector_seprag::graph::{cmp_dist_id, Graph, NodeId};
use ruvector_seprag::query::{knn_exhaustive, KnnIndex, QueryStats};
use ruvector_seprag::{contraction, customize, gen, order, SepRag};

const TOL: f64 = 1e-9;

/// Assert two result lists are equal: same nodes in order, distances within TOL.
fn assert_results_eq(got: &[(NodeId, f64)], want: &[(NodeId, f64)], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch\n got={got:?}\nwant={want:?}");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g.1 - w.1).abs() < TOL,
            "{ctx}: distance mismatch at {i}: got {g:?} want {w:?}"
        );
        assert_eq!(g.0, w.0, "{ctx}: node mismatch at {i}: got {g:?} want {w:?}");
    }
}

fn check_graph_against_oracle(g: Graph, pois: &[NodeId], srcs: &[NodeId], label: &str) {
    let ord = order::nested_dissection(&g);
    // Sanity: ordering is a permutation of 0..n.
    {
        let mut seen = vec![false; g.n];
        assert_eq!(ord.order.len(), g.n, "{label}: order length");
        for &v in &ord.order {
            assert!(!seen[v as usize], "{label}: duplicate in order");
            seen[v as usize] = true;
        }
    }
    let topo = contraction::contract(&g, &ord.order);
    let metric = customize::customize(&topo);
    let idx = KnnIndex::build(&topo, &metric, pois);

    for &src in srcs {
        for &k in &[1usize, 5, 10, 50] {
            let oracle = g.knn_oracle(src, pois, k);
            let exhaustive = knn_exhaustive(&topo, &metric, src, pois, k);
            let mut s = QueryStats::default();
            let pruned = idx.knn(src, k, true, &mut s);
            let unpruned = idx.knn(src, k, false, &mut QueryStats::default());

            let ctx = format!("{label} src={src} k={k}");
            // Exhaustive CCH validates order+contraction+customization vs ground truth.
            assert_results_eq(&exhaustive, &oracle, &format!("{ctx} [exhaustive vs oracle]"));
            // Bucket index (unpruned) must equal exhaustive.
            assert_results_eq(&unpruned, &exhaustive, &format!("{ctx} [bucket vs exhaustive]"));
            // Pruning must not change the answer.
            assert_results_eq(&pruned, &unpruned, &format!("{ctx} [pruned vs unpruned]"));
        }
    }
}

#[test]
fn sbm_matches_oracle() {
    for seed in [1u64, 7, 42, 1000] {
        let g = gen::sbm(4, 25, 0.30, 0.01, seed);
        let pois = gen::sample_pois(g.n, 40, seed);
        let srcs = gen::sample_pois(g.n, 6, seed ^ 0xABCD);
        check_graph_against_oracle(g, &pois, &srcs, &format!("sbm[seed={seed}]"));
    }
}

#[test]
fn grid_matches_oracle() {
    for seed in [3u64, 99] {
        let g = gen::grid(16, 16, seed); // 256 vertices, ~16-wide separators
        let pois = gen::sample_pois(g.n, 50, seed);
        let srcs = gen::sample_pois(g.n, 6, seed ^ 0x55);
        check_graph_against_oracle(g, &pois, &srcs, &format!("grid[seed={seed}]"));
    }
}

#[test]
fn path_matches_oracle() {
    // Degenerate: size-1 separators, deep elimination tree.
    let g = gen::path(120, 5);
    let pois = gen::sample_pois(g.n, 30, 5);
    let srcs = vec![0, 17, 60, 119];
    check_graph_against_oracle(g, &pois, &srcs, "path");
}

#[test]
fn clique_matches_oracle() {
    // Degenerate worst case: full fill-in, no layer separator (leaf fallback).
    let g = gen::clique(24, 11);
    let pois = gen::sample_pois(g.n, 20, 11);
    let srcs = vec![0, 5, 23];
    check_graph_against_oracle(g, &pois, &srcs, "clique");
}

#[test]
fn pruning_reduces_search_space() {
    // On a clean SBM, pruning should fire and scan fewer bucket entries.
    let g = gen::sbm(6, 40, 0.25, 0.004, 77); // 240 vertices, well-separated
    let pois = gen::sample_pois(g.n, 120, 77);
    let sr = SepRag::build(g);
    let idx = sr.index(&pois);

    let mut total_pruned = 0usize;
    let mut pruned_scans = 0usize;
    let mut unpruned_scans = 0usize;
    let srcs = gen::sample_pois(sr.graph.n, 20, 0x1234);
    for &src in &srcs {
        let (_r, sp) = idx.query_with_stats(src, 10);
        total_pruned += sp.ancestors_pruned;
        pruned_scans += sp.bucket_entries_scanned;
        unpruned_scans += unpruned_entry_count(&sr, &pois, src, 10);
    }
    assert!(total_pruned > 0, "expected region pruning to fire on a clean SBM");
    assert!(
        pruned_scans < unpruned_scans,
        "pruned scans ({pruned_scans}) should be < unpruned ({unpruned_scans})"
    );
}

/// Helper: bucket entries scanned with pruning disabled.
fn unpruned_entry_count(sr: &SepRag, pois: &[NodeId], src: NodeId, k: usize) -> usize {
    let idx = KnnIndex::build(&sr.topo, &sr.metric, pois);
    let mut s = QueryStats::default();
    let _ = idx.knn(src, k, false, &mut s);
    s.bucket_entries_scanned
}

#[test]
fn deterministic_across_runs() {
    let build = || {
        let g = gen::sbm(4, 30, 0.28, 0.01, 2024);
        let pois = gen::sample_pois(g.n, 50, 2024);
        let sr = SepRag::build(g);
        let idx = sr.index(&pois);
        let mut all = Vec::new();
        for src in [0u32, 11, 55, 119] {
            all.push(idx.query(src, 7));
        }
        all
    };
    let a = build();
    let b = build();
    for (qa, qb) in a.iter().zip(b.iter()) {
        assert_results_eq(qa, qb, "determinism");
    }
}

#[test]
fn blowup_ratio_is_bounded() {
    // Road-like synthetic graphs should not explode under contraction.
    let grid = SepRag::build(gen::grid(20, 20, 1));
    let sbm = SepRag::build(gen::sbm(5, 40, 0.22, 0.005, 1));
    let n_grid = grid.graph.n as f64;
    let n_sbm = sbm.graph.n as f64;
    // Sanity bound: |G+| should stay well below a complete graph.
    assert!(grid.blowup_ratio() < n_grid, "grid blowup unbounded: {}", grid.blowup_ratio());
    assert!(sbm.blowup_ratio() < n_sbm, "sbm blowup unbounded: {}", sbm.blowup_ratio());
}

#[test]
fn results_are_canonically_sorted() {
    let sr = SepRag::build(gen::grid(12, 12, 9));
    let pois = gen::sample_pois(sr.graph.n, 40, 9);
    let idx = sr.index(&pois);
    let r = idx.query(0, 10);
    for w in r.windows(2) {
        assert!(cmp_dist_id(w[0], w[1]) != std::cmp::Ordering::Greater, "not sorted: {r:?}");
    }
}
