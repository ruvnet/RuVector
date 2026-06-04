//! M1 — "the benchmark has teeth."
//!
//! Before claiming contender A beats ACORN, we must show the problem is real: at low
//! selectivity, the classic **post-filter** baseline (retrieve top-`pool` ignoring the
//! predicate, then filter) collapses, while ACORN's predicate-agnostic search holds recall.
//! Both run on the *same* ACORN-γ graph, so the only variable is the traversal policy —
//! isolating post-filter as the cause of the collapse (not graph density).
//!
//! This is the negative-control analogue of ADR-200's stale-index control: if post-filter
//! did *not* collapse, the benchmark would be insensitive and any later "win" meaningless.
//!
//! Run: cargo run --release -p ruvector-filtered-bench --example teeth -- [N] [Q] [seed]

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
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(7);

    if !Path::new(FEAT_100K).exists() {
        eprintln!("data not extracted ({FEAT_100K}); see src/data.rs header. skipping.");
        return;
    }

    let k = 10;
    let ef = 512; // tuned operating point (see acorn_tune: ~92% recall at sel=1%, n=20k)
    let pool = 512; // post-filter retrieval pool == ef (generous; not a strawman k-only pool)
    let gamma = 2;

    eprintln!("[teeth] loading arxiv slice n={n}…");
    let ds = Dataset::load_arxiv(n);
    let n = ds.len();
    eprintln!("[teeth] building ACORN-γ (γ={gamma}, {} edges/node, ef={ef})…", 16 * gamma);
    let t0 = std::time::Instant::now();
    let acorn = Acorn::build(&ds.feats, gamma, ef);
    eprintln!("[teeth] graph built in {:.1}s", t0.elapsed().as_secs_f64());

    let mut rng = StdRng::seed_from_u64(seed);
    let queries: Vec<usize> = (0..q_count).map(|_| rng.gen_range(0..n)).collect();

    println!("\n=== M1 teeth: post-filter collapse vs ACORN-agnostic (ρ=1, n={n}, k={k}, Q={q_count}) ===");
    println!(
        "{:>7} {:>8} | {:>10} {:>10} | {:>11} {:>11}",
        "sel", "#match", "B_recall", "C_recall", "B_evals", "C_evals"
    );
    println!("{}", "-".repeat(66));

    for &sel in &[0.001_f64, 0.005, 0.01, 0.05, 0.10, 0.30] {
        let pred = predicate::correlated(&ds.labels, sel, 1.0, 0, &mut rng);
        if pred.n_match < k {
            println!("{sel:>7.3} {:>8} | (skipped: #match < k)", pred.n_match);
            continue;
        }
        let pf = pred.as_fn();

        let (mut b_rec, mut c_rec, mut b_ev, mut c_ev) = (0.0, 0.0, 0u64, 0u64);
        for &qi in &queries {
            let q = &ds.feats[qi];
            // Exclude the query's own id so the trivial self-match (distance 0) can't
            // inflate either contender.
            let truth: Vec<u32> = exact_filtered_knn(&ds.feats, q, k + 1, pf)
                .into_iter()
                .filter(|&id| id as usize != qi)
                .take(k)
                .collect();

            let b = acorn.search(q, k, pf);
            let c = acorn.postfilter(q, k, pool, pf);
            let strip = |ids: Vec<u32>| ids.into_iter().filter(|&id| id as usize != qi).collect::<Vec<_>>();

            b_rec += recall(&truth, &strip(b.ids));
            c_rec += recall(&truth, &strip(c.ids));
            b_ev += b.evals;
            c_ev += c.evals;
        }
        let nq = queries.len() as f64;
        println!(
            "{sel:>7.3} {:>8} | {:>9.1}% {:>9.1}% | {:>11} {:>11}",
            pred.n_match,
            100.0 * b_rec / nq,
            100.0 * c_rec / nq,
            b_ev / queries.len() as u64,
            c_ev / queries.len() as u64,
        );
    }
    println!("\nExpected (teeth): C_recall falls sharply as sel→0 while B_recall stays high.");
}
