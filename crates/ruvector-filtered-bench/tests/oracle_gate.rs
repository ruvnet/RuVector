//! M0 gate — "trust the oracle."
//!
//! Every contender (A/B/C/D) is scored against `ruvector-acorn::exact_filtered_knn`. If that
//! oracle is wrong, every downstream recall number is meaningless. This test cross-checks it
//! against a **fully independent** brute-force filtered k-NN (separate distance code, separate
//! sort) on a real ogbn-arxiv slice, exercising the whole data → predicate → oracle path.
//!
//! Skips cleanly when the arxiv data isn't extracted (CI without the dataset).

use ruvector_filtered_bench::data::{Dataset, FEAT_100K};
use ruvector_filtered_bench::exact_filtered_knn;
use ruvector_filtered_bench::predicate;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::path::Path;

/// Independent brute force: no shared code with the oracle. Plain scalar L2, stable sort by
/// (distance, id) so ties (which don't occur on real float embeddings) are still deterministic.
fn independent_filtered_knn(
    feats: &[Vec<f32>],
    labels_mask: &dyn Fn(u32) -> bool,
    query: &[f32],
    k: usize,
) -> Vec<u32> {
    let mut scored: Vec<(f64, u32)> = (0..feats.len() as u32)
        .filter(|&id| labels_mask(id))
        .map(|id| {
            let d: f64 = feats[id as usize]
                .iter()
                .zip(query)
                .map(|(a, b)| {
                    let diff = (*a - *b) as f64;
                    diff * diff
                })
                .sum();
            (d, id)
        })
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    scored.into_iter().take(k).map(|(_, id)| id).collect()
}

#[test]
fn oracle_matches_independent_brute_force() {
    if !Path::new(FEAT_100K).exists() {
        eprintln!("skip: arxiv data not extracted ({FEAT_100K})");
        return;
    }
    let ds = Dataset::load_arxiv(3000);
    let k = 10;
    let mut rng = StdRng::seed_from_u64(42);

    // Sweep a few selectivities; each must keep #matches >= k (the M0 selectivity floor).
    for &sel in &[0.02_f64, 0.05, 0.20] {
        for &rho in &[0.0_f64, 1.0] {
            let pred = predicate::correlated(&ds.labels, sel, rho, 0, &mut rng);
            assert!(
                pred.n_match >= k,
                "selectivity floor violated: sel={sel} ρ={rho} → only {} matches < k={k}",
                pred.n_match
            );
            let pf = pred.as_fn();

            // 8 random queries drawn from the corpus.
            for _ in 0..8 {
                let qi = rng.gen_range(0..ds.len());
                let q = &ds.feats[qi];

                let oracle = exact_filtered_knn(&ds.feats, q, k, pf);
                let truth = independent_filtered_knn(&ds.feats, &pf, q, k);

                assert_eq!(
                    oracle, truth,
                    "oracle disagrees with independent brute force (sel={sel} ρ={rho} q={qi})"
                );
                // Every returned id must actually satisfy the predicate.
                assert!(
                    oracle.iter().all(|&id| pf(id)),
                    "oracle returned a non-matching id (sel={sel} ρ={rho})"
                );
            }
        }
    }
}
