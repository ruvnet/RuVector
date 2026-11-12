//! `TrainerEvaluator` tests: unit checks on the paired-rank encoding, and a
//! full campaign over a small learnable KG where a well-tuned arm is promoted
//! over a deliberately crippled baseline, the receipt chain verifies, and the
//! test split is scored exactly twice.

use super::{mrr_of, pair_ranks, TrainerEvaluator};
use crate::data::{Split4, TripleStore};
use crate::optimize::{Campaign, CampaignSpec, HpoGrid, KgeReceiptLog, Knobs, Loss, Optimizer};
use crate::{ScorerKind, Triple};
use ruvector_typesafe_core::loop_gate::GateDecision;

#[test]
fn pair_ranks_encodes_rank_improvement() {
    // candidate better (lower rank) -> champion win; worse -> baseline win;
    // equal -> concordant. Length follows the shorter of the two (empty
    // incumbent = no evidence, the baseline's own first eval).
    let inc = [5usize, 3, 8, 2];
    let cand = [2usize, 9, 8, 1];
    assert_eq!(
        pair_ranks(&inc, &cand),
        vec![(false, true), (true, false), (false, false), (false, true)]
    );
    assert!(pair_ranks(&[], &cand).is_empty());
}

#[test]
fn mrr_of_is_mean_reciprocal_rank() {
    assert_eq!(mrr_of(&[]), 0.0);
    assert_eq!(mrr_of(&[1, 2]), (1.0 + 0.5) / 2.0);
}

/// A small symmetric KG: `clusters` clusters of `per` entities, three symmetric
/// relations (ring neighbour, two-step neighbour, cluster tag). Symmetric so a
/// bilinear scorer can fit it well; dense enough that a 70% train split covers
/// every entity.
fn synthetic_kg(clusters: u32, per: u32) -> TripleStore {
    let mut triples: Vec<Triple> = Vec::new();
    for c in 0..clusters {
        let base = c * per;
        for a in 0..per {
            let b = base + (a + 1) % per;
            triples.push(Triple::new(base + a, 0, b));
            triples.push(Triple::new(b, 0, base + a));
            let d = base + (a + 2) % per;
            triples.push(Triple::new(base + a, 1, d));
            triples.push(Triple::new(d, 1, base + a));
            triples.push(Triple::new(base + a, 2, base));
            triples.push(Triple::new(base, 2, base + a));
        }
    }
    let ne = (clusters * per) as usize;
    TripleStore::with_counts(triples, Some(ne), Some(3)).unwrap()
}

/// A frozen 4-way split by index stride, keeping every relation in `train`
/// (so, unlike `split4`'s whole-relation holdout, `transfer` MRR is a real,
/// stable number rather than noise on untrained relations — see ADR-004 §gate).
fn strided_split4(store: &TripleStore) -> Split4 {
    let (mut train, mut valid, mut transfer, mut test) = (vec![], vec![], vec![], vec![]);
    for (i, &t) in store.triples().iter().enumerate() {
        match i % 10 {
            7 => valid.push(t),
            8 => transfer.push(t),
            9 => test.push(t),
            _ => train.push(t),
        }
    }
    let split = Split4 {
        train,
        valid,
        transfer,
        test,
    };
    split.assert_disjoint().unwrap();
    split
}

/// Deliberately crippled incumbent: 2 dims and a tiny learning rate barely
/// move off random init.
fn poor_baseline() -> Knobs {
    Knobs {
        dims: 2,
        lr: 0.01,
        optimizer: Optimizer::Adagrad,
        loss: Loss::CrossEntropy,
        neg_count: 100,
        temperature: 1.0,
        n3_lambda: 0.0,
        epochs: 20,
        scorer: ScorerKind::Hole,
    }
}

fn one_good_config_grid() -> HpoGrid {
    // The HPO grid varies only dims/lr/loss/n3 over the baseline, so this yields
    // one proposal: 24 dims at a healthy lr — clearly better than the baseline.
    HpoGrid {
        dims: vec![24],
        lrs: vec![0.5],
        losses: vec![Loss::CrossEntropy],
        n3_lambdas: vec![0.0],
    }
}

#[test]
fn well_tuned_arm_promoted_over_crippled_baseline() {
    let store = synthetic_kg(3, 16); // 48 entities, 3 relations
    let split = strided_split4(&store);
    assert!(!split.valid.is_empty() && !split.test.is_empty() && !split.transfer.is_empty());

    let mut ev = TrainerEvaluator::new(store, split, 7);
    let spec = CampaignSpec {
        baseline: poor_baseline(),
        grid: one_good_config_grid(),
        transfer_tolerance: 0.05,
        ..Default::default()
    };
    let report = Campaign::run(&spec, &mut ev);

    // The 24-dim HPO arm is the champion (the 2-dim baseline is crippled).
    assert_eq!(report.champion.dims, 24, "well-tuned arm must win");
    assert!(!report.paused);
    assert!(
        report.champion_val_mrr > report.baseline_val_mrr,
        "champion val MRR {} must beat baseline {}",
        report.champion_val_mrr,
        report.baseline_val_mrr
    );

    // At least one Promote was recorded.
    assert!(report
        .proposals
        .iter()
        .any(|d| matches!(d.decision, GateDecision::Promote)));

    // Test split scored exactly twice (baseline + champion) — ADR-004 gate 1.
    assert_eq!(ev.test_scorings(), 2);

    // The champion's exact trained tables are retained for install.
    assert!(ev.tables(report.champion_id).is_some());

    // The dual receipt chain verifies; one row per proposal plus the champion
    // confirmation receipt.
    let log = KgeReceiptLog::from_jsonl(&report.receipts_jsonl).unwrap();
    assert!(log.verify_chain().is_ok());
    assert_eq!(log.len(), report.proposals.len() + 1);
}

#[test]
fn campaign_is_deterministic() {
    let run = || {
        let store = synthetic_kg(3, 12);
        let split = strided_split4(&store);
        let mut ev = TrainerEvaluator::new(store, split, 11);
        let spec = CampaignSpec {
            baseline: poor_baseline(),
            grid: one_good_config_grid(),
            transfer_tolerance: 0.05,
            ..Default::default()
        };
        let r = Campaign::run(&spec, &mut ev);
        (r.champion_id, r.champion_val_mrr, r.receipts_jsonl)
    };
    let a = run();
    let b = run();
    assert_eq!(a.0, b.0);
    assert_eq!(a.1, b.1);
    assert_eq!(
        a.2, b.2,
        "same inputs must produce a byte-identical receipt log"
    );
}
