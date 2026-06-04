//! M1 — find ACORN's *tuned* operating point (rule #2: beat the incumbent tuned).
//!
//! Sweeps ef × γ for filtered recall@10 at a representative low selectivity (ρ=1), so the
//! later head-to-head compares against ACORN at its best, not an under-tuned strawman.
//!
//! Run: cargo run --release -p ruvector-filtered-bench --example acorn_tune -- [N] [Q] [sel]

use ruvector_acorn::graph::exact_filtered_knn;
use ruvector_filtered_bench::contenders::{recall, Acorn};
use ruvector_filtered_bench::data::{Dataset, FEAT_100K};
use ruvector_filtered_bench::predicate;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let q_count: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let sel: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.01);

    if !Path::new(FEAT_100K).exists() {
        eprintln!("data not extracted ({FEAT_100K}); skipping.");
        return;
    }

    let k = 10;
    let ds = Dataset::load_arxiv(n);
    let n = ds.len();
    let mut rng = StdRng::seed_from_u64(7);
    let pred = predicate::correlated(&ds.labels, sel, 1.0, 0, &mut rng);
    let pf = pred.as_fn();
    let queries: Vec<usize> = (0..q_count).map(|_| rng.gen_range(0..n)).collect();

    // Precompute truth once per query (independent of ef/γ).
    let truths: Vec<Vec<u32>> = queries
        .iter()
        .map(|&qi| {
            exact_filtered_knn(&ds.feats, &ds.feats[qi], k + 1, pf)
                .into_iter()
                .filter(|&id| id as usize != qi)
                .take(k)
                .collect()
        })
        .collect();

    println!(
        "\n=== ACORN tuning: filtered recall@{k} (n={n}, sel={sel}, #match={}, Q={q_count}) ===",
        pred.n_match
    );
    println!("{:>5} {:>6} | {:>10} {:>11}", "γ", "ef", "recall", "evals/q");
    println!("{}", "-".repeat(40));

    for &gamma in &[2usize, 3] {
        let acorn = Acorn::build(&ds.feats, gamma, 64); // ef field unused; we pass ef below
        for &ef in &[64usize, 128, 256, 512, 1024] {
            let (mut rec, mut ev) = (0.0, 0u64);
            for (qi, truth) in queries.iter().zip(&truths) {
                let (got, evals) =
                    ruvector_acorn::search::acorn_search_counted(&acorn.graph, &ds.feats[*qi], k, ef, pf);
                let got: Vec<u32> = got
                    .into_iter()
                    .map(|(id, _)| id)
                    .filter(|&id| id as usize != *qi)
                    .collect();
                rec += recall(truth, &got);
                ev += evals;
            }
            let nq = queries.len() as f64;
            println!(
                "{gamma:>5} {ef:>6} | {:>9.1}% {:>11}",
                100.0 * rec / nq,
                ev / queries.len() as u64
            );
        }
    }
}
