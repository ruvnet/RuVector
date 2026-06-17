//! BET 1 scale test (ADR-200 next step): does the re-weight-vs-rebuild win hold
//! at n≥10⁵, and how big is the rebuild-cost gap?
//!
//! For each N, build a base graph, apply the adversarial ROTATIONAL drift
//! (t=0.5, the ~36%-churn point), then compare:
//!   A re-weight : reuse base graph under the drifted metric (rebuild cost: 0)
//!   B rebuild   : rebuild under the drifted metric (rebuild cost: measured)
//! Recall@10 vs brute-force truth under the drifted metric. The B-build-seconds
//! column is the rebuild-cost curve; A's update cost is ~0 (topology reused).
//!
//! Pre-registered gate: recall(A) within 2% of recall(B) at every N, AND rebuild
//! cost grows with N (so the saved cost grows). Win = scale-robust + large gap.
//!
//! Run: cargo run --release -p ruvector-seprag --example scale_drift -- <feat.csv> [Ns...]

use ruvector_seprag::ann::*;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "target/m1-data/node-feat-100k.csv".into());
    let ns: Vec<usize> = if args.len() > 2 {
        args[2..].iter().filter_map(|s| s.parse().ok()).collect()
    } else {
        vec![5000, 10000, 25000, 50000, 100000]
    };
    let max_n = *ns.iter().max().unwrap();

    eprintln!("[scale] loading up to {max_n} vectors from {path}");
    let all = read_vectors(&path, max_n);
    let dim = all[0].len();
    eprintln!("[scale] loaded {} vectors x {dim} dims\n", all.len());

    let p = AnnParams { r: 24, l: 64, alpha: 1.2, k: 10 };
    let id = identity(dim);
    let rot = target_rot(dim, &mut Rng::new(54321));
    let drift = lerp_mat(&id, &rot, 0.5); // adversarial ~36%-churn point

    println!("=== BET 1 @ scale: rotational drift (t=0.5), recall@{} ===", p.k);
    println!("{:>8} | {:>8} {:>8} {:>6} | {:>9} {:>10} | {:>7} {:>7}",
        "N", "A rewt", "B rebld", "churn", "B build s", "A update s", "A ev/q", "B ev/q");
    println!("{}", "-".repeat(80));

    for &n in &ns {
        if n > all.len() { continue; }
        let vecs: Vec<Vec32> = all[..n].to_vec();
        let vt = apply_linear(&drift, &vecs, dim);

        // queries + ground truth under the drifted metric.
        let mut qrng = Rng::new(999);
        let queries: Vec<usize> = (0..100).map(|_| qrng.below(n)).collect();
        let truth: Vec<Vec<u32>> = queries.iter().map(|&q| brute_topk(&vt, q, p.k)).collect();
        let truth0: Vec<Vec<u32>> = queries.iter().map(|&q| brute_topk(&vecs, q, p.k)).collect();
        let churn: f64 = truth.iter().zip(&truth0).map(|(a, b)| 1.0 - recall(a, b)).sum::<f64>() / queries.len() as f64;

        // Base graph (built once under the original metric; this IS the cost A avoids re-paying).
        let g0 = build(&vecs, &p, 7);

        // A: reuse g0 under the drifted metric. "Update cost" = recompute medoid only.
        let ta = Instant::now();
        let medt_for_a = medoid(&vt); // A needs an entry point in the new metric; O(N), cheap
        let a_update = ta.elapsed().as_secs_f64();
        let (mut ra, mut a_ev) = (0.0f64, 0usize);
        for (&q, tr) in queries.iter().zip(&truth) {
            let (got, _, ev) = search(&g0, &vt, medt_for_a, &vt[q], p.l, p.k);
            ra += recall(&got, tr);
            a_ev += ev;
        }

        // B: full rebuild under the drifted metric (the cost we are trying to avoid).
        let tb = Instant::now();
        let gt = build(&vt, &p, 7);
        let b_build = tb.elapsed().as_secs_f64();
        let medt = medoid(&vt);
        let (mut rb, mut b_ev) = (0.0f64, 0usize);
        for (&q, tr) in queries.iter().zip(&truth) {
            let (got, _, ev) = search(&gt, &vt, medt, &vt[q], p.l, p.k);
            rb += recall(&got, tr);
            b_ev += ev;
        }

        let nq = queries.len() as f64;
        println!("{:>8} | {:>7.1}% {:>7.1}% {:>5.0}% | {:>9.2} {:>10.3} | {:>7.0} {:>7.0}",
            n, ra / nq * 100.0, rb / nq * 100.0, churn * 100.0, b_build, a_update, a_ev as f64 / nq, b_ev as f64 / nq);
    }

    println!("\nGate: WIN if A within 2% of B at every N AND rebuild cost grows with N.");
    println!("'A update s' = re-weight cost (medoid recompute only); B build s = rebuild cost avoided.");
}
