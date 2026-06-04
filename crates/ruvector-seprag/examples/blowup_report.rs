//! M0→M1 diagnostic: print the metrics that become M1's go/no-go signal
//! (ADR-199 §4) on synthetic graphs — shortcut-blowup ratio, elimination-tree
//! height, and pruned-vs-unpruned search space.
//!
//! Run: `cargo run -p ruvector-seprag --example blowup_report`

use ruvector_seprag::query::{elim_depth, KnnIndex, QueryStats};
use ruvector_seprag::{gen, Graph, SepRag};

fn report(name: &str, g: Graph) {
    let n = g.n;
    let m = g.edges().count();
    let pois: Vec<u32> = gen::sample_pois(n, (n / 2).max(1), 1);
    let srcs = gen::sample_pois(n, 32.min(n), 2);

    let sr = SepRag::build(g);
    let max_depth = (0..n as u32).map(|r| elim_depth(&sr.topo, r)).max().unwrap_or(0);
    let idx = KnnIndex::build(&sr.topo, &sr.metric, &pois);

    let (mut pruned, mut unpruned, mut anc_vis, mut anc_prune) = (0usize, 0usize, 0usize, 0usize);
    for &src in &srcs {
        let mut sp = QueryStats::default();
        let _ = idx.knn(src, 10, true, &mut sp);
        let mut su = QueryStats::default();
        let _ = idx.knn(src, 10, false, &mut su);
        pruned += sp.bucket_entries_scanned;
        unpruned += su.bucket_entries_scanned;
        anc_vis += sp.ancestors_visited;
        anc_prune += sp.ancestors_pruned;
    }
    let q = srcs.len().max(1);
    println!(
        "{name:<14} n={n:<5} m={m:<6} blowup={:>5.2}x  elim_h={max_depth:<4} \
         scans/q: pruned={:<5} unpruned={:<5} ({:.0}% saved)  anc_vis/q={} pruned/q={}",
        sr.blowup_ratio(),
        pruned / q,
        unpruned / q,
        100.0 * (1.0 - pruned as f64 / unpruned.max(1) as f64),
        anc_vis / q,
        anc_prune / q,
    );
}

fn main() {
    println!("SepRAG M0 diagnostic — synthetic graphs (lower blowup + more pruning = more road-like)\n");
    report("grid-20x20", gen::grid(20, 20, 1));
    report("grid-40x40", gen::grid(40, 40, 1));
    report("sbm-clean", gen::sbm(8, 50, 0.25, 0.003, 1));
    report("sbm-dense", gen::sbm(8, 50, 0.25, 0.05, 1));
    report("path-1000", gen::path(1000, 1));
    println!("\nNote: synthetic only. The real go/no-go is M1 on ogbn-arxiv (ADR-199).");
}
