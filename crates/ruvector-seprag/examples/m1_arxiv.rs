//! M1 first-pass (ADR-199): SepRAG on a real ogbn-arxiv citation subgraph.
//!
//! Measures the go/no-go signal — shortcut-blowup ratio, elimination-tree
//! height, build time — plus a sampled Dijkstra-oracle recall check, on a
//! connected BFS-ball subgraph of the real citation network.
//!
//! Scope honesty: this uses (a) the citation graph only — α-pruned kNN over node
//! features is the next pass — and (b) the M0 BFS separator, which M0 showed
//! degenerates on low-diameter graphs. So a high blowup here is expected to be a
//! *separator-quality* artifact, not a verdict on SepRAG; the verdict needs
//! ruvector-mincut balanced separators at full scale. Treat this as pipeline
//! validation + a first real-data data point.
//!
//! Run:
//!   gunzip -kc target/m1-data/arxiv/raw/edge.csv.gz > .../edge.csv   # once
//!   cargo run --release -p ruvector-seprag --example m1_arxiv -- <edge.csv> <N> <seed>

use ruvector_seprag::graph::{cmp_dist_id, Graph, NodeId};
use ruvector_seprag::query::{elim_depth, KnnIndex, QueryStats};
use ruvector_seprag::{gen, SepRag, SeparatorKind};
use std::collections::VecDeque;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| {
        "target/m1-data/arxiv/raw/edge.csv".to_string()
    });
    let n_target: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(6000);
    let seed_node: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    // arg4: max degree (0 = no backbone sparsification). arg5: "bal" | "layer".
    let max_degree: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
    let kind = match args.get(5).map(String::as_str) {
        Some("layer") => SeparatorKind::BfsLayer,
        _ => SeparatorKind::Balanced,
    };

    eprintln!("[m1] reading {path}");
    let t = Instant::now();
    let (adj, max_id) = read_edges(&path);
    eprintln!(
        "[m1] full graph: {} nodes, {} undirected edges, read in {:.1}s",
        max_id + 1,
        adj.iter().map(Vec::len).sum::<usize>() / 2,
        t.elapsed().as_secs_f64()
    );

    // Connected induced subgraph via BFS ball from `seed_node`.
    let (g0, orig_ids) = bfs_ball(&adj, seed_node, n_target);
    let g = if max_degree > 0 { degree_bound(&g0, max_degree) } else { g0 };
    eprintln!(
        "[m1] subgraph: {} nodes, {} edges (BFS ball from orig id {seed_node}); \
         backbone max_degree={max_degree} ({:?} separator)",
        g.n,
        g.edges().count(),
        kind,
    );

    let t = Instant::now();
    let sr = SepRag::build_with(g, kind);
    let build_s = t.elapsed().as_secs_f64();
    let max_h = (0..sr.graph.n as u32).map(|r| elim_depth(&sr.topo, r)).max().unwrap_or(0);

    println!("\n=== M1 first-pass: ogbn-arxiv citation subgraph ===");
    println!("nodes              {}", sr.graph.n);
    println!("base edges |G_nav| {}", sr.graph.edges().count());
    println!("chordal arcs |G+|  {}", sr.topo.arc_count());
    println!("BLOWUP RATIO       {:.2}x   (ADR-199 gate; target <=3-5x)", sr.blowup_ratio());
    println!("elim-tree height   {max_h}   (sublinear vs n={} is the goal)", sr.graph.n);
    println!("build time         {build_s:.2}s");

    // Sampled-oracle recall: SepRAG k-NN vs Dijkstra over the subgraph.
    let pois = gen::sample_pois(sr.graph.n, sr.graph.n / 2, 7);
    let srcs = gen::sample_pois(sr.graph.n, 50, 13);
    let idx = KnnIndex::build(&sr.topo, &sr.metric, &pois);
    let (mut ok, mut pruned_scans, mut unpruned_scans) = (0usize, 0usize, 0usize);
    for &src in &srcs {
        let oracle = sr.graph.knn_oracle(src, &pois, 10);
        let mut sp = QueryStats::default();
        let got = idx.knn(src, 10, true, &mut sp);
        let mut su = QueryStats::default();
        let _ = idx.knn(src, 10, false, &mut su);
        pruned_scans += sp.bucket_entries_scanned;
        unpruned_scans += su.bucket_entries_scanned;
        if dist_multiset_eq(&got, &oracle) {
            ok += 1;
        }
    }
    let q = srcs.len();
    println!("\nrecall sanity      {ok}/{q} queries match Dijkstra oracle (distance multiset)");
    println!(
        "search space       pruned {} vs unpruned {} bucket scans/query ({:.0}% saved)",
        pruned_scans / q,
        unpruned_scans / q,
        100.0 * (1.0 - pruned_scans as f64 / unpruned_scans.max(1) as f64)
    );
    let _ = orig_ids;
}

/// Read "src,dst" edge CSV → undirected adjacency (dense ids) + max id.
fn read_edges(path: &str) -> (Vec<Vec<u32>>, usize) {
    let data = std::fs::read_to_string(path).expect("read edge csv");
    let mut edges: Vec<(u32, u32)> = Vec::new();
    let mut max_id = 0u32;
    for line in data.lines() {
        // Skip SNAP-style comment lines; accept comma/tab/space separators.
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let mut it = line.split(|c: char| matches!(c, ',' | '\t' | ' ')).filter(|s| !s.is_empty());
        if let (Some(a), Some(b)) = (it.next(), it.next()) {
            if let (Ok(u), Ok(v)) = (a.trim().parse::<u32>(), b.trim().parse::<u32>()) {
                max_id = max_id.max(u).max(v);
                edges.push((u, v));
            }
        }
    }
    let mut adj = vec![Vec::new(); max_id as usize + 1];
    for (u, v) in edges {
        if u != v {
            adj[u as usize].push(v);
            adj[v as usize].push(u);
        }
    }
    (adj, max_id as usize)
}

/// Induced connected subgraph: BFS from `seed` collecting up to `n_target` nodes.
/// Unit edge weights (hop distance). Returns the graph + original-id map.
fn bfs_ball(adj: &[Vec<u32>], seed: u32, n_target: usize) -> (Graph, Vec<u32>) {
    let mut order = Vec::new();
    let mut seen = vec![false; adj.len()];
    let mut q = VecDeque::from([seed]);
    seen[seed as usize] = true;
    while let Some(u) = q.pop_front() {
        order.push(u);
        if order.len() >= n_target {
            break;
        }
        for &v in &adj[u as usize] {
            if !seen[v as usize] {
                seen[v as usize] = true;
                q.push_back(v);
            }
        }
    }
    let mut remap = vec![u32::MAX; adj.len()];
    for (new, &old) in order.iter().enumerate() {
        remap[old as usize] = new as u32;
    }
    let mut g = Graph::new(order.len());
    for &old in &order {
        let nu = remap[old as usize];
        for &v in &adj[old as usize] {
            let nv = remap[v as usize];
            if nv != u32::MAX && nu < nv {
                g.add_edge(nu, nv, 1.0);
            }
        }
    }
    (g, order)
}

/// Degree-bound backbone sparsification (ADR-197): keep, per node, edges to its
/// `d` lowest-degree neighbours (hub-dampening), unioned undirected. A cheap
/// stand-in for α-pruning when no vector metric is loaded yet.
fn degree_bound(g: &Graph, d: usize) -> Graph {
    let deg: Vec<usize> = g.adj.iter().map(Vec::len).collect();
    let mut out = Graph::new(g.n);
    for u in 0..g.n {
        let mut nb = g.adj[u].clone();
        nb.sort_by(|a, b| deg[a.0 as usize].cmp(&deg[b.0 as usize]).then(a.0.cmp(&b.0)));
        for &(v, w) in nb.iter().take(d) {
            out.add_edge(u as NodeId, v, w);
        }
    }
    out
}

fn dist_multiset_eq(a: &[(NodeId, f64)], b: &[(NodeId, f64)]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut da: Vec<f64> = a.iter().map(|x| x.1).collect();
    let mut db: Vec<f64> = b.iter().map(|x| x.1).collect();
    da.sort_by(|x, y| x.partial_cmp(y).unwrap());
    db.sort_by(|x, y| x.partial_cmp(y).unwrap());
    da.iter().zip(&db).all(|(x, y)| (x - y).abs() < 1e-9)
        && { let _ = cmp_dist_id; true }
}
