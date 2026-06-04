//! M3 adversarial check (protocol rule #5) — does predicate-aware-entry ACORN (contender D)
//! erase region-pruning's win in its own regime?
//!
//! For the win cells (ρ≥0.7, sel≤5%) it reports vanilla ACORN (B), the best
//! predicate-aware-entry ACORN (D, over a probe-budget sweep, tuned to match B's recall at
//! fewest evals), and contender A matched to the same recall. The headline ratio is A vs the
//! **cheaper** ACORN variant — so the win must survive the strongest ACORN we can build.
//!
//! Run: cargo run --release -p ruvector-filtered-bench --example adversarial -- [N] [Q] [nclusters]

use ruvector_acorn::graph::exact_filtered_knn;
use ruvector_acorn::search::acorn_search_counted;
use ruvector_filtered_bench::contenders::{recall, Acorn};
use ruvector_filtered_bench::data::{Dataset, FEAT_100K};
use ruvector_filtered_bench::predicate;
use ruvector_filtered_bench::prune::RegionPruneIvf;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::path::Path;

const K: usize = 10;
const GATE: f64 = 0.02;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let q_count: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let nclusters: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let ef = 512;
    let seed = 7;

    if !Path::new(FEAT_100K).exists() {
        eprintln!("data not extracted ({FEAT_100K}); skipping.");
        return;
    }

    let ds = Dataset::load_arxiv(n);
    let n = ds.len();
    let acorn = Acorn::build(&ds.feats, 2, ef);
    let ivf = RegionPruneIvf::build(&ds.feats, nclusters, 15, seed);
    let mut rng = StdRng::seed_from_u64(seed);
    let queries: Vec<usize> = (0..q_count).map(|_| rng.gen_range(0..n)).collect();

    let probe_budgets = [256usize, 1024, 4096, 16384];
    let a_caps = [1usize, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128];

    println!("\n=== M3 adversarial: A vs best-of(vanilla-B, predicate-entry-D) (n={n}, nclusters={}) ===", ivf.nclusters);
    println!(
        "{:>4} {:>6} | {:>7} {:>7} | {:>7} {:>7} {:>7} | {:>7} {:>7} | {:>6} {:>6} | verdict",
        "ρ", "sel", "B_rec", "B_ev", "D_rec", "D_ev", "D_pb", "Am_rec", "Am_ev", "vsB", "vsBest"
    );
    println!("{}", "-".repeat(100));

    for &rho in &[0.7_f64, 1.0] {
        for &sel in &[0.001_f64, 0.005, 0.01, 0.05] {
            let pred = predicate::correlated(&ds.labels, sel, rho, 0, &mut rng);
            if pred.n_match < K {
                continue;
            }
            let pf = pred.as_fn();
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

            // B — vanilla ACORN.
            let (b_rec, b_ev) = mean(&queries, &truths, |qi| {
                let (g, e) = acorn_search_counted(&acorn.graph, &ds.feats[qi], K, ef, pf);
                (g.into_iter().map(|(id, _)| id).collect(), e)
            });

            // D — predicate-aware entry; pick cheapest probe budget reaching B's recall.
            let (mut d_rec, mut d_ev, mut d_pb) = (0.0, u64::MAX, 0usize);
            for &pb in &probe_budgets {
                let (r, e) = mean(&queries, &truths, |qi| {
                    let res = acorn.search_predicate_entry(&ds.feats[qi], K, pf, pb, 4);
                    (res.ids, res.evals)
                });
                if r >= b_rec - GATE && e < d_ev {
                    d_rec = r;
                    d_ev = e;
                    d_pb = pb;
                }
            }
            if d_ev == u64::MAX {
                // none matched B's recall; report the highest-budget point.
                let (r, e) = mean(&queries, &truths, |qi| {
                    let res = acorn.search_predicate_entry(&ds.feats[qi], K, pf, 16384, 4);
                    (res.ids, res.evals)
                });
                d_rec = r;
                d_ev = e;
                d_pb = 16384;
            }

            // A — matched to vanilla B's recall.
            let (mut am_rec, mut am_ev) = (1.0, u64::MAX);
            for &cap in &a_caps {
                let (r, e) = mean(&queries, &truths, |qi| {
                    let res = ivf.search(&ds.feats, &ds.feats[qi], K, pf, Some(cap));
                    (res.ids, res.evals)
                });
                if r >= b_rec - GATE {
                    am_rec = r;
                    am_ev = e;
                    break;
                }
            }

            let best_acorn = b_ev.min(d_ev);
            let vs_b = b_ev as f64 / am_ev as f64;
            let vs_best = best_acorn as f64 / am_ev as f64;
            let target = if sel <= 0.01 { 5.0 } else { 2.0 };
            let verdict = if vs_best >= target { "WIN" } else { "miss" };

            println!(
                "{rho:>4.1} {sel:>6.3} | {:>6.1}% {:>7} | {:>6.1}% {:>7} {:>7} | {:>6.1}% {:>7} | {:>5.1}× {:>5.1}× | {verdict}",
                100.0 * b_rec, b_ev,
                100.0 * d_rec, d_ev, d_pb,
                100.0 * am_rec, am_ev,
                vs_b, vs_best,
            );
        }
    }
}

fn mean(queries: &[usize], truths: &[Vec<u32>], mut run: impl FnMut(usize) -> (Vec<u32>, u64)) -> (f64, u64) {
    let mut rec = 0.0;
    let mut ev = 0u64;
    for (&qi, truth) in queries.iter().zip(truths) {
        let (ids, e) = run(qi);
        let ids: Vec<u32> = ids.into_iter().filter(|&id| id as usize != qi).collect();
        rec += recall(truth, &ids);
        ev += e;
    }
    (rec / queries.len() as f64, ev / queries.len() as u64)
}
