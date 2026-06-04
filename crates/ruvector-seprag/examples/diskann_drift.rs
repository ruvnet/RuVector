//! BET 1 on the PRODUCTION index (ADR-200 next step): re-run the re-weight-vs-
//! rebuild test on `ruvector-diskann`'s real Vamana graph, not the lite
//! reference Vamana. This (a) confirms the result on the shipping index and
//! (b) firms the rebuild baseline (the lite Vamana showed build variance).
//!
//! The reuse trick is native to `VamanaGraph`: the graph stores only topology;
//! `greedy_search(vectors, query, beam)` takes the vectors externally. So drift
//! = pass the *transformed* vectors to a graph built on the *original* ones.
//!
//! Run: cargo run --release -p ruvector-seprag --example diskann_drift -- <feat.csv> <N>

use ruvector_diskann::distance::FlatVectors;
use ruvector_diskann::graph::VamanaGraph;
use ruvector_seprag::ann::{
    apply_linear, brute_topk, identity, l2, lerp_mat, read_vectors, recall, target_rot, Rng, Vec32,
};
use std::time::Instant;

const R: usize = 32;
const BUILD_BEAM: usize = 64;
const SEARCH_BEAM: usize = 64;
const ALPHA: f32 = 1.2;
const K: usize = 10;

fn flat(vecs: &[Vec32], dim: usize) -> FlatVectors {
    let mut f = FlatVectors::with_capacity(dim, vecs.len());
    for v in vecs {
        f.push(v);
    }
    f
}

fn build_graph(vecs: &[Vec32], dim: usize) -> VamanaGraph {
    let f = flat(vecs, dim);
    let mut g = VamanaGraph::new(vecs.len(), R, BUILD_BEAM, ALPHA);
    g.build(&f).expect("vamana build");
    g
}

/// Top-k from a graph search over `vecs`, re-ranked by exact distance to the query.
fn topk(g: &VamanaGraph, vecs: &[Vec32], f: &FlatVectors, q: usize) -> Vec<u32> {
    let (cands, _) = g.greedy_search(f, &vecs[q], SEARCH_BEAM);
    let mut scored: Vec<(f32, u32)> = cands.iter().map(|&c| (l2(&vecs[c as usize], &vecs[q]), c)).collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0));
    scored.into_iter().filter(|&(_, c)| c as usize != q).take(K).map(|(_, c)| c).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "target/m1-data/node-feat-100k.csv".into());
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20000);
    let vecs = read_vectors(&path, n);
    let n = vecs.len();
    let dim = vecs[0].len();

    eprintln!("[diskann] n={n} dim={dim}; ruvector-diskann Vamana R={R} L={BUILD_BEAM} alpha={ALPHA}");
    let t0 = Instant::now();
    let g0 = build_graph(&vecs, dim);
    eprintln!("[diskann] base graph built in {:.1}s\n", t0.elapsed().as_secs_f64());

    let id = identity(dim);
    let rot = target_rot(dim, &mut Rng::new(54321));

    // ---- Part 1: global rotational drift ----
    println!("=== diskann BET 1: GLOBAL rotational drift (recall@{K}) ===");
    println!("{:>5} {:>7} | {:>8} {:>8} | {:>9}", "t", "churn", "A reuse", "B rebld", "B build s");
    println!("{}", "-".repeat(46));
    let mut qrng = Rng::new(999);
    let queries: Vec<usize> = (0..100).map(|_| qrng.below(n)).collect();
    let base_truth: Vec<Vec<u32>> = queries.iter().map(|&q| brute_topk(&vecs, q, K)).collect();

    for &t in &[0.0f32, 0.25, 0.5, 1.0] {
        let vt = apply_linear(&lerp_mat(&id, &rot, t), &vecs, dim);
        let ft = flat(&vt, dim);
        let truth: Vec<Vec<u32>> = queries.iter().map(|&q| brute_topk(&vt, q, K)).collect();
        let churn: f64 = truth.iter().zip(&base_truth).map(|(a, b)| 1.0 - recall(a, b)).sum::<f64>() / queries.len() as f64;

        let ra: f64 = queries.iter().zip(&truth).map(|(&q, tr)| recall(&topk(&g0, &vt, &ft, q), tr)).sum::<f64>() / queries.len() as f64;

        let tb = Instant::now();
        let gt = build_graph(&vt, dim);
        let bt = tb.elapsed().as_secs_f64();
        let rb: f64 = queries.iter().zip(&truth).map(|(&q, tr)| recall(&topk(&gt, &vt, &ft, q), tr)).sum::<f64>() / queries.len() as f64;

        println!("{:>5.2} {:>6.0}% | {:>7.1}% {:>7.1}% | {:>9.2}", t, churn * 100.0, ra * 100.0, rb * 100.0, bt);
    }

    // ---- Part 2: region-local drift (does the lite-Vamana t=0.25 dip reproduce?) ----
    println!("\n=== diskann BET 1: REGION-LOCAL drift (warp 15% cluster, recall@{K}) ===");
    let region_frac = 0.15f32;
    let mut rng = Rng::new(2024);
    let centre = vecs[rng.below(n)].clone();
    let mut by_dist: Vec<(f32, usize)> = (0..n).map(|i| (l2(&vecs[i], &centre), i)).collect();
    by_dist.sort_by(|a, b| a.0.total_cmp(&b.0));
    let region_size = (n as f32 * region_frac) as usize;
    let mut in_region = vec![false; n];
    for &(_, i) in by_dist.iter().take(region_size) {
        in_region[i] = true;
    }
    let region_ids: Vec<usize> = (0..n).filter(|&i| in_region[i]).collect();
    let outside_ids: Vec<usize> = (0..n).filter(|&i| !in_region[i]).collect();
    let mut qr = Rng::new(77);
    let q_in: Vec<usize> = (0..100).map(|_| region_ids[qr.below(region_ids.len())]).collect();
    let q_out: Vec<usize> = (0..100).map(|_| outside_ids[qr.below(outside_ids.len())]).collect();

    println!("{:>5} | {:>7} {:>7} {:>7} | {:>7} {:>7}", "t", "chrnIn", "A_in", "B_in", "A_out", "B_out");
    println!("{}", "-".repeat(54));
    for &t in &[0.0f32, 0.25, 0.5, 1.0] {
        let a = lerp_mat(&id, &rot, t);
        let mut vt = vecs.clone();
        for &i in &region_ids {
            vt[i] = (0..dim).map(|r| { let row = &a[r * dim..(r + 1) * dim]; row.iter().zip(&vecs[i]).map(|(x, y)| x * y).sum() }).collect();
        }
        let ft = flat(&vt, dim);
        let gt = build_graph(&vt, dim);

        let eval = |qs: &[usize]| -> (f64, f64, f64) {
            let (mut churn, mut ra, mut rb) = (0.0, 0.0, 0.0);
            for &q in qs {
                let truth = brute_topk(&vt, q, K);
                let truth0 = brute_topk(&vecs, q, K);
                churn += 1.0 - recall(&truth, &truth0);
                ra += recall(&topk(&g0, &vt, &ft, q), &truth);
                rb += recall(&topk(&gt, &vt, &ft, q), &truth);
            }
            let m = qs.len() as f64;
            (churn / m * 100.0, ra / m * 100.0, rb / m * 100.0)
        };
        let (ci, ai, bi) = eval(&q_in);
        let (_co, ao, bo) = eval(&q_out);
        println!("{:>5.2} | {:>6.0}% {:>6.1}% {:>6.1}% | {:>6.1}% {:>6.1}%", t, ci, ai, bi, ao, bo);
    }

    println!("\nGate: A within 2% of B (overall and in-region). Production-index confirmation of ADR-200.");
}
