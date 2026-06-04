//! M1 decisive thesis test (ADR-197/199): does the embedding *manifold* have
//! smaller separators than the citation topology?
//!
//! Builds an α-pruned kNN graph (DiskANN/Vamana-style RobustPrune) over real
//! ogbn-arxiv 128-d node features, then runs the SepRAG hierarchy and reports
//! the same go/no-go metrics as `m1_arxiv`. Compare blowup / elim-tree height
//! against the road control (~7.6× / ~136) and the citation graph (~23.8× / ~941).
//!
//! Run:
//!   gunzip -c arxiv/raw/node-feat.csv.gz | head -2000 > node-feat-2000.csv
//!   cargo run --release -p ruvector-seprag --example m1_manifold -- <feat.csv> <N> <k> <alpha>

use ruvector_seprag::graph::{Graph, NodeId};
use ruvector_seprag::query::{elim_depth, KnnIndex, QueryStats};
use ruvector_seprag::{gen, SepRag};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "target/m1-data/node-feat-2000.csv".into());
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1500);
    let k: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
    let alpha: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1.2);

    eprintln!("[manifold] reading features from {path}");
    let feats = read_features(&path, n);
    let n = feats.len();
    let dim = feats.first().map_or(0, Vec::len);
    eprintln!("[manifold] {n} nodes x {dim} dims; building k={k} graph, alpha-prune alpha={alpha}");

    let norms: Vec<f64> = feats.iter().map(|v| v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12)).collect();
    let dist = |i: usize, j: usize| -> f64 {
        let dot: f64 = feats[i].iter().zip(&feats[j]).map(|(a, b)| a * b).sum();
        (1.0 - dot / (norms[i] * norms[j])).max(1e-6) // cosine distance, kept positive
    };

    // Exact kNN per node (brute force; fine at this scale).
    let t = Instant::now();
    let mut knn: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for i in 0..n {
        let mut cand: Vec<(usize, f64)> = (0..n).filter(|&j| j != i).map(|j| (j, dist(i, j))).collect();
        cand.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        cand.truncate(k.max(8) * 2); // keep a candidate pool for the prune step
        knn[i] = cand;
    }
    eprintln!("[manifold] kNN built in {:.1}s", t.elapsed().as_secs_f64());

    // α-prune (Vamana RobustPrune): keep q unless a closer kept r dominates it.
    let mut g = Graph::new(n);
    let mut kept_edges = 0usize;
    for i in 0..n {
        let mut kept: Vec<(usize, f64)> = Vec::new();
        for &(q, dq) in &knn[i] {
            let dominated = kept.iter().any(|&(r, _)| alpha * dist(r, q) <= dq);
            if !dominated {
                kept.push((q, dq));
                if kept.len() >= k {
                    break;
                }
            }
        }
        for (q, dq) in kept {
            g.add_edge(i as NodeId, q as NodeId, dq);
            kept_edges += 1;
        }
    }

    let sr = SepRag::build(g);
    let max_h = (0..sr.graph.n as u32).map(|r| elim_depth(&sr.topo, r)).max().unwrap_or(0);
    let avg_deg = 2.0 * sr.graph.edges().count() as f64 / n as f64;

    println!("\n=== M1 manifold test: ogbn-arxiv feature kNN (k={k}, alpha={alpha}) ===");
    println!("nodes              {n}");
    println!("base edges |G_nav| {}  (avg degree {avg_deg:.1}, directed kept {kept_edges})", sr.graph.edges().count());
    println!("chordal arcs |G+|  {}", sr.topo.arc_count());
    println!("BLOWUP RATIO       {:.2}x   (road ~7.6x, citation ~23.8x)", sr.blowup_ratio());
    println!("elim-tree height   {max_h}   (road ~136, citation ~941; sqrt(n)~{:.0})", (n as f64).sqrt());

    // Recall sanity vs Dijkstra oracle on the manifold graph.
    let pois = gen::sample_pois(n, n / 2, 7);
    let srcs = gen::sample_pois(n, 50, 13);
    let idx = KnnIndex::build(&sr.topo, &sr.metric, &pois);
    let (mut ok, mut pr, mut un) = (0usize, 0usize, 0usize);
    for &src in &srcs {
        let oracle = sr.graph.knn_oracle(src, &pois, 10);
        let mut sp = QueryStats::default();
        let got = idx.knn(src, 10, true, &mut sp);
        let mut su = QueryStats::default();
        let _ = idx.knn(src, 10, false, &mut su);
        pr += sp.bucket_entries_scanned;
        un += su.bucket_entries_scanned;
        if multiset_eq(&got, &oracle) {
            ok += 1;
        }
    }
    let q = srcs.len();
    println!("recall sanity      {ok}/{q} match Dijkstra oracle");
    println!("search space       pruned {} vs unpruned {} scans/query ({:.0}% saved)", pr / q, un / q, 100.0 * (1.0 - pr as f64 / un.max(1) as f64));
}

fn read_features(path: &str, n: usize) -> Vec<Vec<f64>> {
    let data = std::fs::read_to_string(path).expect("read features");
    data.lines()
        .take(n)
        .map(|line| line.split(',').filter_map(|s| s.trim().parse::<f64>().ok()).collect())
        .filter(|v: &Vec<f64>| !v.is_empty())
        .collect()
}

fn multiset_eq(a: &[(NodeId, f64)], b: &[(NodeId, f64)]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut da: Vec<f64> = a.iter().map(|x| x.1).collect();
    let mut db: Vec<f64> = b.iter().map(|x| x.1).collect();
    da.sort_by(|x, y| x.partial_cmp(y).unwrap());
    db.sort_by(|x, y| x.partial_cmp(y).unwrap());
    da.iter().zip(&db).all(|(x, y)| (x - y).abs() < 1e-9)
}
