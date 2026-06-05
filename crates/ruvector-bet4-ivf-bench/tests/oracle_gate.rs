//! M0 gate: full-budget `BnBIvf` must be **exact** — its top-10 must match the brute-force
//! oracle (recall ≈ 1.0) on a real arxiv slice. This certifies the branch-and-bound invariant
//! (ascending-LB order + `break` when `LB ≥ τ`) on real data before any matched-recall claim.

use ruvector_bet4_ivf_bench::data::load_feat_csv;
use ruvector_bet4_ivf_bench::kernel::BnBIvf;
use ruvector_bet4_ivf_bench::oracle::{brute_force_topk, recall_at_k};

/// Repo-root-relative path to the gitignored arxiv feature slice.
const DATA: &str = "../../target/m1-data/node-feat-2000.csv";

#[test]
fn bnb_full_budget_is_exact() {
    let corpus = match load_feat_csv(DATA, 2000) {
        Ok(c) if c.len() >= 500 => c,
        _ => {
            eprintln!("skipping bnb_full_budget_is_exact: {DATA} not available");
            return;
        }
    };
    let k = 10;
    let idx = BnBIvf::build(&corpus, 64, 25, 42);
    let nq = 100;
    let mut acc = 0.0;
    for q in 0..nq {
        let truth = brute_force_topk(&corpus, &corpus[q], k);
        let (res, _evals, _probed) = idx.search(&corpus[q], k, None); // None = full budget = exact
        let got: Vec<usize> = res.iter().map(|r| r.id).collect();
        acc += recall_at_k(&truth, &got, k);
    }
    let recall = acc / nq as f64;
    assert!(
        recall >= 0.999,
        "full-budget B&B must be exact (B&B invariant broken): recall@10={recall:.4}"
    );
}

#[test]
fn capped_probe_reduces_member_evals() {
    let corpus = match load_feat_csv(DATA, 2000) {
        Ok(c) if c.len() >= 500 => c,
        _ => {
            eprintln!("skipping capped_probe_reduces_member_evals: {DATA} not available");
            return;
        }
    };
    let idx = BnBIvf::build(&corpus, 64, 25, 42);
    let (_r_full, evals_full, _p) = idx.search(&corpus[0], 10, None);
    let (_r_cap, evals_cap, probed_cap) = idx.search(&corpus[0], 10, Some(4));
    assert!(probed_cap <= 4, "cap must bound clusters probed");
    assert!(
        evals_cap <= evals_full,
        "capped probe should not cost more member-evals than full budget"
    );
}
