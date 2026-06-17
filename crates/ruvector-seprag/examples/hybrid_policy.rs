//! BET 1 → operating policy (ADR-200): the hybrid re-weight + periodic/triggered
//! rebuild strategy, the shippable answer to "the recall gap widens with drift."
//!
//! A self-learning system drifts its metric a little every step. Per step you can
//! RE-WEIGHT (reuse the graph under the new metric, ~0 cost) or REBUILD (expensive).
//! We simulate a drift *trajectory* and compare four policies on the production
//! `ruvector-diskann` Vamana:
//!   - always   : rebuild every step          (recall ceiling, max cost)
//!   - never    : build once, reuse forever    (min cost, recall decays)
//!   - periodic : rebuild every K steps
//!   - triggered: rebuild when drift-since-last-rebuild exceeds τ (cheap monitor)
//!
//! Win: a hybrid matches `always` recall within ~2% using a small fraction of the
//! rebuilds — turning the proven finding into a usable operating point.
//!
//! Run: cargo run --release -p ruvector-seprag --example hybrid_policy -- <feat.csv> <N> <T>

use ruvector_diskann::distance::FlatVectors;
use ruvector_diskann::graph::VamanaGraph;
use ruvector_seprag::ann::{apply_linear, brute_topk, gaussian, identity, l2, read_vectors, recall, Rng, Vec32};
use std::time::Instant;

const R: usize = 32;
const BUILD_BEAM: usize = 64;
const SEARCH_BEAM: usize = 64;
const ALPHA: f32 = 1.2;
const K: usize = 10;
const EPS: f32 = 0.3; // per-step random-walk drift magnitude

/// Cumulative-transform step: A' = (I + eps·G/√dim) · A  — a random-walk metric
/// drift (fresh direction each step), more adversarial than a straight path.
fn small_warp(dim: usize, rng: &mut Rng, eps: f32) -> Vec<f32> {
    let scale = eps / (dim as f32).sqrt();
    let mut m = identity(dim);
    for i in 0..dim {
        for j in 0..dim {
            m[i * dim + j] += scale * gaussian(rng);
        }
    }
    m
}

fn matmul(a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for k in 0..dim {
            let aik = a[i * dim + k];
            if aik == 0.0 { continue; }
            for j in 0..dim {
                c[i * dim + j] += aik * b[k * dim + j];
            }
        }
    }
    c
}

fn frob(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt()
}

fn flat(vecs: &[Vec32], dim: usize) -> FlatVectors {
    let mut f = FlatVectors::with_capacity(dim, vecs.len());
    for v in vecs { f.push(v); }
    f
}

fn build_graph(vecs: &[Vec32], dim: usize) -> VamanaGraph {
    let f = flat(vecs, dim);
    let mut g = VamanaGraph::new(vecs.len(), R, BUILD_BEAM, ALPHA);
    g.build(&f).expect("build");
    g
}

fn topk(g: &VamanaGraph, vecs: &[Vec32], f: &FlatVectors, q: usize) -> Vec<u32> {
    let (cands, _) = g.greedy_search(f, &vecs[q], SEARCH_BEAM);
    let mut s: Vec<(f32, u32)> = cands.iter().map(|&c| (l2(&vecs[c as usize], &vecs[q]), c)).collect();
    s.sort_by(|a, b| a.0.total_cmp(&b.0));
    s.into_iter().filter(|&(_, c)| c as usize != q).take(K).map(|(_, c)| c).collect()
}

struct Step { vt: Vec<Vec32>, ft: FlatVectors, truth: Vec<Vec<u32>>, a: Vec<f32> }

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "target/m1-data/node-feat-100k.csv".into());
    let n: usize = args.get(2).and_then(|x| x.parse().ok()).unwrap_or(5000);
    let steps_n: usize = args.get(3).and_then(|x| x.parse().ok()).unwrap_or(24);
    let vecs = read_vectors(&path, n);
    let n = vecs.len();
    let dim = vecs[0].len();

    let mut qrng = Rng::new(999);
    let queries: Vec<usize> = (0..100).map(|_| qrng.below(n)).collect();

    eprintln!("[hybrid] n={n} dim={dim} steps={steps_n}; precomputing random-walk drift (eps={EPS})…");
    // Precompute each step's drifted vectors + ground truth (shared across policies).
    // Drift is a compounding random walk: A_t = (I+eps·G)·A_{t-1}.
    let mut acc = identity(dim);
    let mut wrng = Rng::new(2);
    let steps: Vec<Step> = (0..steps_n).map(|t| {
        if t > 0 {
            let w = small_warp(dim, &mut wrng, EPS);
            acc = matmul(&w, &acc, dim);
        }
        let a = acc.clone();
        let vt = apply_linear(&a, &vecs, dim);
        let ft = flat(&vt, dim);
        let truth = queries.iter().map(|&q| brute_topk(&vt, q, K)).collect();
        Step { vt, ft, truth, a }
    }).collect();
    // Calibrate trigger thresholds from the mean per-step drift.
    let d_step: f32 = (1..steps_n).map(|t| frob(&steps[t].a, &steps[t - 1].a)).sum::<f32>() / (steps_n - 1).max(1) as f32;
    eprintln!("[hybrid] mean per-step drift (Frobenius) = {d_step:.2}");

    // One representative rebuild cost (for the cost column).
    let t0 = Instant::now();
    let _ = build_graph(&steps[0].vt, dim);
    let build_s = t0.elapsed().as_secs_f64();
    eprintln!("[hybrid] one rebuild ≈ {build_s:.2}s\n");

    println!("=== BET 1 hybrid policy: drift trajectory, {steps_n} steps, recall@{K} (diskann) ===");
    println!("{:>12} | {:>9} {:>9} {:>9} | {:>8} {:>10}", "policy", "mean rec", "min rec", "end rec", "rebuilds", "rebuild s");
    println!("{}", "-".repeat(72));

    // policy = closure(step_idx, frob_drift_since_last_rebuild) -> should_rebuild
    let run = |name: &str, should_rebuild: &dyn Fn(usize, f32) -> bool| {
        let mut g = build_graph(&steps[0].vt, dim); // t=0 always builds
        let mut last = 0usize;
        let mut builds = 1usize;
        let mut recalls = Vec::with_capacity(steps_n);
        for (t, st) in steps.iter().enumerate() {
            if t > 0 {
                let drift = frob(&st.a, &steps[last].a);
                if should_rebuild(t, drift) {
                    g = build_graph(&st.vt, dim);
                    last = t;
                    builds += 1;
                }
            }
            let r: f64 = queries.iter().zip(&st.truth).map(|(&q, tr)| recall(&topk(&g, &st.vt, &st.ft, q), tr)).sum::<f64>() / queries.len() as f64;
            recalls.push(r);
        }
        let mean = recalls.iter().sum::<f64>() / recalls.len() as f64;
        let min = recalls.iter().cloned().fold(1.0, f64::min);
        let end = *recalls.last().unwrap();
        println!("{:>14} | {:>8.1}% {:>8.1}% {:>8.1}% | {:>8} {:>10.2}", name, mean * 100.0, min * 100.0, end * 100.0, builds, builds as f64 * build_s);
    };

    let tau_a = 3.0 * d_step; // rebuild after ~3 steps of drift
    let tau_b = 6.0 * d_step; // rebuild after ~6 steps of drift
    run("always", &|_, _| true);
    run("never", &|_, _| false);
    run("periodic-4", &|t, _| t % 4 == 0);
    run("periodic-8", &|t, _| t % 8 == 0);
    run("triggered~3", &|_, d| d >= tau_a);
    run("triggered~6", &|_, d| d >= tau_b);

    println!("\nWin: a hybrid matches 'always' mean recall within ~2% at a fraction of the rebuilds.");
    println!("'never' shows the decay reuse-only suffers as drift accumulates; the gap is what hybrids close.");
}
