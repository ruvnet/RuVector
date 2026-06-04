//! M3 — the frozen-gate run: tuned ACORN vs region-pruned IVF over a selectivity × ρ grid.
//!
//! Compares **cost at matched recall** (the honest framing): contender A's exact B&B has
//! recall 1.0 ≥ ACORN, so we tune A's probe cap down until its recall ≈ ACORN's, then
//! compare distance-evals/query. Reports the ratio against the pre-registered gate
//! (≥5× at sel≤1%, ≥2× at sel=5%, ρ≥0.7), the ρ=0 kill control, and wall-clock (the
//! honesty guard — a distance-eval win that reverses on wall-clock is "inconclusive").
//!
//! Run: cargo run --release -p ruvector-filtered-bench --example sweep -- [N] [Q] [nclusters] [ef] [seed]

use ruvector_acorn::graph::exact_filtered_knn;
use ruvector_acorn::search::acorn_search_counted;
use ruvector_filtered_bench::contenders::{recall, Acorn};
use ruvector_filtered_bench::data::{Dataset, FEAT_100K};
use ruvector_filtered_bench::predicate;
use ruvector_filtered_bench::prune::RegionPruneIvf;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::path::Path;
use std::time::Instant;

const K: usize = 10;
const GATE: f64 = 0.02; // recall match tolerance

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let q_count: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let nclusters: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(128);
    let ef: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(512);
    let seed: u64 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(7);

    if !Path::new(FEAT_100K).exists() {
        eprintln!("data not extracted ({FEAT_100K}); skipping.");
        return;
    }

    let ds = Dataset::load_arxiv(n);
    let n = ds.len();
    eprintln!("[sweep] n={n} Q={q_count} nclusters={nclusters} ef={ef}");
    eprintln!("[sweep] building ACORN-γ2 + region-prune IVF…");
    let t = Instant::now();
    let acorn = Acorn::build(&ds.feats, 2, ef);
    let ivf = RegionPruneIvf::build(&ds.feats, nclusters, 15, seed);
    eprintln!("[sweep] built in {:.1}s (ivf nclusters={})", t.elapsed().as_secs_f64(), ivf.nclusters);

    let mut rng = StdRng::seed_from_u64(seed);
    let queries: Vec<usize> = (0..q_count).map(|_| rng.gen_range(0..n)).collect();

    let sels = [0.001_f64, 0.005, 0.01, 0.05, 0.10, 0.30];
    let rhos = [0.0_f64, 0.3, 0.5, 0.7, 1.0];
    let probe_caps = [1usize, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128];

    println!("\n=== M3 sweep (n={n}, k={K}, ACORN γ2 ef={ef}, IVF nclusters={}) ===", ivf.nclusters);
    println!(
        "{:>4} {:>6} {:>7} | {:>7} {:>8} | {:>7} {:>8} | {:>7} {:>8} {:>6} {:>6} | verdict",
        "ρ", "sel", "#match", "B_rec", "B_evals", "Aex_rec", "Aex_ev", "Am_rec", "Am_evals", "ev×", "wc×"
    );
    println!("{}", "-".repeat(104));

    for &rho in &rhos {
        for &sel in &sels {
            let pred = predicate::correlated(&ds.labels, sel, rho, 0, &mut rng);
            if pred.n_match < K {
                continue;
            }
            let pf = pred.as_fn();

            // Truth per query (exclude self to avoid trivial distance-0 inflation).
            let truths: Vec<Vec<u32>> = queries
                .iter()
                .map(|&qi| {
                    exact_filtered_knn(&ds.feats, &ds.feats[qi], K + 1, pf)
                        .into_iter()
                        .filter(|&id| id as usize != qi)
                        .take(K)
                        .collect()
                })
                .collect();

            // ACORN (B).
            let (b_rec, b_ev, b_ms) = measure(&queries, &truths, |qi| {
                let (got, ev) = acorn_search_counted(&acorn.graph, &ds.feats[qi], K, ef, pf);
                (got.into_iter().map(|(id, _)| id).collect(), ev)
            });

            // A exact (B&B, recall ~1.0).
            let (aex_rec, aex_ev, _) = measure(&queries, &truths, |qi| {
                let r = ivf.search(&ds.feats, &ds.feats[qi], K, pf, None);
                (r.ids, r.evals)
            });

            // A matched: smallest probe cap with recall >= b_rec - GATE.
            let mut am_rec = aex_rec;
            let mut am_ev = aex_ev;
            let mut am_ms = 0.0;
            for &cap in &probe_caps {
                let (r, ev, ms) = measure(&queries, &truths, |qi| {
                    let res = ivf.search(&ds.feats, &ds.feats[qi], K, pf, Some(cap));
                    (res.ids, res.evals)
                });
                if r >= b_rec - GATE {
                    am_rec = r;
                    am_ev = ev;
                    am_ms = ms;
                    break;
                }
            }

            let ratio = if am_ev > 0 { b_ev as f64 / am_ev as f64 } else { 0.0 };
            // Wall-clock honesty guard: a distance-eval win that reverses on the clock is
            // not a real win. wc_ratio > 1 means A is also faster in wall time.
            let wc_ratio = if am_ms > 0.0 { b_ms / am_ms } else { 0.0 };
            let target = if sel <= 0.01 { 5.0 } else if sel <= 0.05 { 2.0 } else { 0.0 };
            let verdict = if rho >= 0.7 && target > 0.0 {
                if ratio >= target { "WIN" } else { "miss" }
            } else if rho <= 0.3 {
                // graceful-degradation guard: A must not lose by >1.5x
                if ratio >= 1.0 / 1.5 { "ok(ctrl)" } else { "DEGRADE" }
            } else {
                "—"
            };

            println!(
                "{rho:>4.1} {sel:>6.3} {:>7} | {:>6.1}% {:>8} | {:>6.1}% {:>8} | {:>6.1}% {:>8} {:>5.1}× {:>5.1}× | {verdict}",
                pred.n_match,
                100.0 * b_rec,
                b_ev,
                100.0 * aex_rec,
                aex_ev,
                100.0 * am_rec,
                am_ev,
                ratio,
                wc_ratio,
            );
        }
        println!();
    }
}

/// Mean recall, mean distance-evals, mean wall-clock(µs) over the query set.
fn measure(
    queries: &[usize],
    truths: &[Vec<u32>],
    mut run: impl FnMut(usize) -> (Vec<u32>, u64),
) -> (f64, u64, f64) {
    let mut rec = 0.0;
    let mut ev = 0u64;
    let t = Instant::now();
    for (&qi, truth) in queries.iter().zip(truths) {
        let (ids, e) = run(qi);
        let ids: Vec<u32> = ids.into_iter().filter(|&id| id as usize != qi).collect();
        rec += recall(truth, &ids);
        ev += e;
    }
    let nq = queries.len() as f64;
    (rec / nq, ev / queries.len() as u64, t.elapsed().as_secs_f64() * 1e6 / nq)
}
