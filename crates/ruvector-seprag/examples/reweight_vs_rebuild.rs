//! BET 1 (ADR-200): does a FIXED ANN topology + recomputed distances absorb
//! metric drift as well as a full rebuild? Three drift modes (diagonal,
//! rotational, non-linear), recall@10 + per-query cost vs full rebuild, with a
//! stale-index negative control. Shared engine in `ruvector_seprag::ann`.
//!
//! Run: cargo run --release -p ruvector-seprag --example reweight_vs_rebuild -- <feat.csv> <N>

use ruvector_seprag::ann::*;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "target/m1-data/node-feat-2000.csv".into());
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let vecs = read_vectors(&path, n);
    let n = vecs.len();
    let dim = vecs[0].len();
    let p = AnnParams { r: 24, l: 64, alpha: 1.2, k: 10 };

    let mut qrng = Rng::new(999);
    let queries: Vec<usize> = (0..100.min(n)).map(|_| qrng.below(n)).collect();

    eprintln!("[bet1] {n} vectors x {dim} dims; Vamana R={} L={} alpha={} k={}", p.r, p.l, p.alpha, p.k);
    let t0 = Instant::now();
    let g0 = build(&vecs, &p, 7);
    let med0 = medoid(&vecs);
    eprintln!("[bet1] base graph built once in {:.2}s\n", t0.elapsed().as_secs_f64());

    let id = identity(dim);
    let diag = target_diag(dim, &mut Rng::new(12345));
    let rot = target_rot(dim, &mut Rng::new(54321));
    let warp = random_rotation(dim, &mut Rng::new(7));
    let beta = 4.0f32;

    run_mode("DIAGONAL drift (per-axis rescale)", &g0, med0, &queries, &p, dim,
        |t| apply_linear(&lerp_mat(&id, &diag, t), &vecs, dim));
    run_mode("ROTATIONAL drift (anisotropic scale on rotated axes — adversarial linear)", &g0, med0, &queries, &p, dim,
        |t| apply_linear(&lerp_mat(&id, &rot, t), &vecs, dim));
    run_mode("NON-LINEAR drift (residual tanh warp — adversarial non-linear)", &g0, med0, &queries, &p, dim,
        |t| apply_nonlin(&warp, &vecs, t * beta, dim));

    println!("\nGate: WIN if A within 2% of B across the sweep; KILL if A drops >2% below B.");
    println!("A's rebuild cost is 0 (topology reused); B pays a full rebuild per drift step.");
}

#[allow(clippy::too_many_arguments)]
fn run_mode<F: Fn(f32) -> Vec<Vec32>>(label: &str, g0: &[Vec<u32>], med0: u32, queries: &[usize], p: &AnnParams, _dim: usize, vt_of: F) {
    let v0 = vt_of(0.0);
    let truth0: Vec<Vec<u32>> = queries.iter().map(|&q| brute_topk(&v0, q, p.k)).collect();

    println!("=== BET 1: {label} ===");
    println!("{:>5} {:>7} | {:>8} {:>8} {:>8} | {:>8} | {:>7} {:>7}", "t", "churn", "A rewt", "B rebld", "C stale", "B bld s", "A ev/q", "B ev/q");
    println!("{}", "-".repeat(74));

    for &t in &[0.0f32, 0.1, 0.25, 0.5, 0.75, 1.0] {
        let vt = vt_of(t);
        let truth: Vec<Vec<u32>> = queries.iter().map(|&q| brute_topk(&vt, q, p.k)).collect();
        let churn: f64 = truth.iter().zip(&truth0).map(|(a, b)| 1.0 - recall(a, b)).sum::<f64>() / queries.len() as f64;

        let (mut ra, mut a_ev) = (0.0f64, 0usize);
        for (&q, tr) in queries.iter().zip(&truth) {
            let (got, _, ev) = search(g0, &vt, med0, &vt[q], p.l, p.k);
            ra += recall(&got, tr);
            a_ev += ev;
        }

        let tb = Instant::now();
        let gt = build(&vt, p, 7);
        let bt = tb.elapsed().as_secs_f64();
        let medt = medoid(&vt);
        let (mut rb, mut b_ev) = (0.0f64, 0usize);
        for (&q, tr) in queries.iter().zip(&truth) {
            let (got, _, ev) = search(&gt, &vt, medt, &vt[q], p.l, p.k);
            rb += recall(&got, tr);
            b_ev += ev;
        }

        let rc: f64 = queries.iter().zip(&truth).map(|(&q, tr)| {
            let (got, _, _) = search(g0, &v0, med0, &v0[q], p.l, p.k);
            recall(&got, tr)
        }).sum::<f64>() / queries.len() as f64;

        let nq = queries.len() as f64;
        println!("{:>5.2} {:>6.0}% | {:>7.1}% {:>7.1}% {:>7.1}% | {:>8.2} | {:>7.0} {:>7.0}",
            t, churn * 100.0, ra / nq * 100.0, rb / nq * 100.0, rc * 100.0, bt, a_ev as f64 / nq, b_ev as f64 / nq);
    }
    println!();
}
