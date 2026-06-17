//! BET 1 adversarial test (ADR-200): REGION-LOCAL metric drift.
//!
//! All prior drift was global (one transform for every point). The realistic
//! harder case for a self-learning system is *local*: the metric/embedding for
//! ONE region of the space changes a lot while the rest is stationary (e.g. the
//! GNN re-learns structure for one topic). This is the scenario most likely to
//! strand a reused topology *locally*.
//!
//! Method: pick a local cluster R (the nearest `region_frac` of points to a
//! random centre); apply a strong rotational warp to ONLY those vectors. Then
//! compare reuse (A) vs rebuild (B), **reporting recall separately for queries
//! inside R (the drifted region) vs outside** — a global average would hide a
//! local failure.
//!
//! Gate: WIN if A within 2% of B for IN-region queries (not just overall). KILL
//! if A_in drops >2% below B_in → reuse fails locally → need local/periodic rebuild.
//!
//! Run: cargo run --release -p ruvector-seprag --example region_drift -- <feat.csv> <N>

use ruvector_seprag::ann::*;
use std::time::Instant;

fn matvec(a: &[f32], v: &[f32], dim: usize) -> Vec<f32> {
    (0..dim).map(|i| {
        let row = &a[i * dim..(i + 1) * dim];
        row.iter().zip(v).map(|(x, y)| x * y).sum()
    }).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "target/m1-data/node-feat-100k.csv".into());
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20000);
    let region_frac = 0.15f32;

    let vecs = read_vectors(&path, n);
    let n = vecs.len();
    let dim = vecs[0].len();
    let p = AnnParams { r: 24, l: 64, alpha: 1.2, k: 10 };

    // Region R = the nearest `region_frac` of points to a random centre.
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

    // Query sets: 100 inside R (the stressed region), 100 outside.
    let mut qr = Rng::new(77);
    let q_in: Vec<usize> = (0..100).map(|_| region_ids[qr.below(region_ids.len())]).collect();
    let q_out: Vec<usize> = (0..100).map(|_| outside_ids[qr.below(outside_ids.len())]).collect();

    eprintln!("[region] n={n} dim={dim}; region R = {region_size} pts ({:.0}%) warped, rest stationary", region_frac * 100.0);
    let t0 = Instant::now();
    let g0 = build(&vecs, &p, 7);
    eprintln!("[region] base graph built in {:.1}s\n", t0.elapsed().as_secs_f64());

    let id = identity(dim);
    let rot = target_rot(dim, &mut Rng::new(54321));

    println!("=== BET 1: REGION-LOCAL drift ({:.0}% of space warped) ===", region_frac * 100.0);
    println!("recall@{} split by query location; gate = A_in within 2% of B_in\n", p.k);
    println!("{:>5} | {:>7} {:>7} {:>7} | {:>7} {:>7} {:>7} | {:>8}",
        "t", "churnIn", "A_in", "B_in", "chrnOut", "A_out", "B_out", "B bld s");
    println!("{}", "-".repeat(72));

    for &t in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
        let a = lerp_mat(&id, &rot, t);
        // Warp ONLY region-R vectors; everything else stays put.
        let mut vt = vecs.clone();
        for &i in &region_ids {
            vt[i] = matvec(&a, &vecs[i], dim);
        }

        let med0 = medoid(&vt); // entry point in the (mostly stationary) drifted space
        let tb = Instant::now();
        let gt = build(&vt, &p, 7);
        let bt = tb.elapsed().as_secs_f64();
        let medt = medoid(&vt);

        let eval = |qs: &[usize]| -> (f64, f64, f64) {
            let mut churn = 0.0;
            let mut ra = 0.0;
            let mut rb = 0.0;
            for &q in qs {
                let truth = brute_topk(&vt, q, p.k);
                let truth0 = brute_topk(&vecs, q, p.k);
                churn += 1.0 - recall(&truth, &truth0);
                let (ga, _, _) = search(&g0, &vt, med0, &vt[q], p.l, p.k);
                ra += recall(&ga, &truth);
                let (gb, _, _) = search(&gt, &vt, medt, &vt[q], p.l, p.k);
                rb += recall(&gb, &truth);
            }
            let m = qs.len() as f64;
            (churn / m * 100.0, ra / m * 100.0, rb / m * 100.0)
        };

        let (ci, ai, bi) = eval(&q_in);
        let (co, ao, bo) = eval(&q_out);
        println!("{:>5.2} | {:>6.0}% {:>6.1}% {:>6.1}% | {:>6.0}% {:>6.1}% {:>6.1}% | {:>8.2}",
            t, ci, ai, bi, co, ao, bo, bt);
    }

    println!("\nA_in/B_in = recall for queries INSIDE the drifted region (the stress test).");
    println!("A_out/B_out = queries outside it (should stay ~unchanged).");
}
