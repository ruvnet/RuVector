//! Campaign tests driven by a stub `Evaluator` (no trainer present).

use super::*;
use crate::optimize::proposals::{HpoGrid, Loss};
use crate::optimize::KgeReceiptLog;
use crate::optimize::{ArmOutcome, Evaluator, Proposal};
use crate::ScorerKind;
use ruvector_typesafe_core::loop_gate::GateDecision;
use std::cell::Cell;
use std::rc::Rc;

/// How the stub should shape one arm's synthetic outcome.
struct ArmPlan {
    wins: usize,
    losses: usize,
    concordant: usize,
    transfer_baseline: f32,
    transfer_candidate: f32,
    val_mrr: f32,
    cost: f32,
    test_acc: f32,
}

impl ArmPlan {
    /// A genuinely better arm: `wins/losses` discordant at ~p, transfer up.
    fn better() -> Self {
        Self {
            wins: 140,
            losses: 60,
            concordant: 0,
            transfer_baseline: 0.78,
            transfer_candidate: 0.80,
            val_mrr: 0.50,
            cost: 1.0,
            test_acc: 0.55,
        }
    }
    /// A no-op arm: all concordant, transfer flat.
    fn noop() -> Self {
        Self {
            wins: 0,
            losses: 0,
            concordant: 200,
            transfer_baseline: 0.78,
            transfer_candidate: 0.78,
            val_mrr: 0.20,
            cost: 1.0,
            test_acc: 0.40,
        }
    }
}

struct Counters {
    /// Number of test-split scorings (each `score_test` call).
    test_scorings: Rc<Cell<u32>>,
    /// Number of `fit_and_eval` calls.
    fits: Rc<Cell<u32>>,
}

struct Stub {
    counters: Counters,
    plan: Box<dyn FnMut(&Proposal) -> ArmPlan>,
}

impl Stub {
    fn new(plan: impl FnMut(&Proposal) -> ArmPlan + 'static) -> (Self, Counters) {
        let test_scorings = Rc::new(Cell::new(0));
        let fits = Rc::new(Cell::new(0));
        (
            Self {
                counters: Counters {
                    test_scorings: test_scorings.clone(),
                    fits: fits.clone(),
                },
                plan: Box::new(plan),
            },
            Counters {
                test_scorings,
                fits,
            },
        )
    }
}

impl Evaluator for Stub {
    fn fit_and_eval(&mut self, proposal: &Proposal) -> ArmOutcome {
        self.counters.fits.set(self.counters.fits.get() + 1);
        let plan = (self.plan)(proposal);
        let mut paired = Vec::new();
        paired.extend(std::iter::repeat_n((false, true), plan.wins));
        paired.extend(std::iter::repeat_n((true, false), plan.losses));
        paired.extend(std::iter::repeat_n((true, true), plan.concordant));
        let counter = self.counters.test_scorings.clone();
        let test_acc = plan.test_acc;
        ArmOutcome {
            val_paired: paired,
            val_mrr: plan.val_mrr,
            transfer_baseline: plan.transfer_baseline,
            transfer_candidate: plan.transfer_candidate,
            transfer_n: 50,
            cost: plan.cost,
            score_test: Box::new(move || {
                counter.set(counter.get() + 1);
                (test_acc, 100)
            }),
        }
    }
}

fn two_config_grid() -> HpoGrid {
    HpoGrid {
        dims: vec![64, 256],
        lrs: vec![0.01],
        losses: vec![Loss::CrossEntropy],
        n3_lambdas: vec![0.0],
    }
}

#[test]
fn better_arm_promoted_noop_rejected_test_scored_twice_chain_verifies() {
    // dims 256 is the genuinely better config; dims 64 is a no-op.
    let (mut stub, counters) = Stub::new(|p| {
        if p.knobs.dims == 256 {
            ArmPlan::better()
        } else {
            ArmPlan::noop()
        }
    });
    let spec = CampaignSpec {
        grid: two_config_grid(),
        ..Default::default()
    };
    let report = Campaign::run(&spec, &mut stub);

    // The better config (dims 256) is the champion.
    assert_eq!(report.champion.dims, 256);
    assert!(!report.paused);

    // At least one promotion and at least one no-op rejection were recorded.
    assert!(report
        .proposals
        .iter()
        .any(|d| matches!(d.decision, GateDecision::Promote)));
    assert!(report
        .proposals
        .iter()
        .any(|d| matches!(d.decision, GateDecision::Reject(_))));

    // Test split scored exactly twice (baseline + champion).
    assert_eq!(counters.test_scorings.get(), 2);
    assert!(report.champion_test.champion_accuracy > report.baseline_test.baseline_accuracy);

    // The receipt chain is intact; it holds one row per proposal plus the
    // champion-confirmation receipt.
    let log = KgeReceiptLog::from_jsonl(&report.receipts_jsonl).unwrap();
    assert!(log.verify_chain().is_ok());
    assert_eq!(log.len(), report.proposals.len() + 1);
}

#[test]
fn transfer_regression_rejected_despite_validation_win() {
    // Big validation win, but the transfer split regresses hard.
    let (mut stub, _counters) = Stub::new(|_p| ArmPlan {
        transfer_candidate: 0.50, // << baseline 0.78, beyond tolerance
        ..ArmPlan::better()
    });
    let spec = CampaignSpec {
        grid: HpoGrid {
            dims: vec![200],
            lrs: vec![0.01],
            losses: vec![Loss::CrossEntropy],
            n3_lambdas: vec![0.0],
        },
        ..Default::default()
    };
    let report = Campaign::run(&spec, &mut stub);

    // Nothing promoted → champion stays the baseline incumbent.
    let baseline_id = Proposal::baseline(spec.baseline.clone()).id;
    assert_eq!(report.champion_id, baseline_id);
    assert!(report
        .proposals
        .iter()
        .all(|d| matches!(d.decision, GateDecision::Reject(_))));
    assert!(report
        .proposals
        .iter()
        .any(|d| matches!(&d.decision, GateDecision::Reject(r) if r.contains("transfer"))));
    // The audit log must not claim the (unpromoted) baseline was promoted.
    let log = KgeReceiptLog::from_jsonl(&report.receipts_jsonl).unwrap();
    assert!(!matches!(
        log.iter().last().unwrap().receipt.decision,
        GateDecision::Promote
    ));
}

#[test]
fn budget_exhaustion_pauses() {
    let (mut stub, counters) = Stub::new(|_p| ArmPlan::better());
    let spec = CampaignSpec {
        grid: two_config_grid(),
        per_day_evals: 1, // only the first proposal may be evaluated
        ..Default::default()
    };
    let report = Campaign::run(&spec, &mut stub);

    assert!(report.paused);
    assert_eq!(report.budget_consumed, 1);
    assert!(report
        .proposals
        .iter()
        .any(|d| matches!(d.decision, GateDecision::Paused(_))));
    // Fitting stopped at the pause: baseline + the one gated proposal = 2 fits.
    // The budget pre-check records the pause WITHOUT fitting the third proposal.
    assert_eq!(counters.fits.get(), 2);
    // Chain still verifies through a paused campaign.
    let log = KgeReceiptLog::from_jsonl(&report.receipts_jsonl).unwrap();
    assert!(log.verify_chain().is_ok());
}

#[test]
fn model_arms_run_after_hpo() {
    // Every arm promotes. The HPO winner is Hole-scored, so exactly one model
    // arm (Rotate) is gated after HPO — the Hole arm is skipped as redundant.
    let (mut stub, _counters) = Stub::new(|_p| ArmPlan::better());
    let spec = CampaignSpec {
        grid: two_config_grid(),
        ..Default::default()
    };
    let report = Campaign::run(&spec, &mut stub);
    let arms: Vec<_> = report
        .proposals
        .iter()
        .filter(|d| d.arm == crate::optimize::KgeArm::ModelArm)
        .collect();
    assert_eq!(arms.len(), 1);
    // Champion is the first promoted proposal at max reward (a Hole HPO config).
    assert_eq!(report.champion.scorer, ScorerKind::Hole);
}
