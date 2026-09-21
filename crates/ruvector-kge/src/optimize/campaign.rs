//! `Campaign::run` — one self-optimization campaign over typesafe's promotion
//! gate (ADR-004). The gate, sequential test, transfer holdout and budget are
//! reused unchanged from `ruvector-typesafe-core`; each decision's receipt is
//! wrapped in a [`KgeReceipt`] so the KGE-specific fields ride beside the
//! typesafe receipt instead of overloading it. `fit → gate → receipt` is
//! interleaved so an exhausted budget stops further fitting.

use super::proposals::{model_arms, successive_halving, KgeArm, Knobs, Proposal};
use super::receipt::{KgeReceiptExtra, KgeReceiptLog};
use super::{ArmOutcome, Evaluator};
use ruvector_typesafe_core::loop_gate::{
    Budget, Campaign as TestTokens, Evidence, Gate, GateDecision, GateOutcome,
    PairedSequentialTest, Proposal as TsProposal, ProposalKind, TransferHoldout,
};
use ruvector_typesafe_core::receipt::{Metrics, Receipt, TestStatistic};
use ruvector_typesafe_core::Head;
use serde::{Deserialize, Serialize};

/// Sentinel head on every KGE receipt: typesafe's `Head` enum has no KGE
/// variant, so the loop pins one value; the scorer identity lives in the
/// [`KgeReceiptExtra`], not in an overloaded typesafe field.
const SENTINEL_HEAD: Head = Head::NearestPrototype;

/// Everything a campaign needs. Defaults match a small, fast sweep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignSpec {
    pub baseline: Knobs,
    pub grid: super::proposals::HpoGrid,
    /// Successive-halving reduction factor (≥ 2).
    pub eta: usize,
    /// Anytime paired-test type-I bound.
    pub alpha: f32,
    /// Anytime paired-test betting fraction.
    pub lambda: f32,
    /// Cost weight in the bandit reward `MRR − λ·cost`.
    pub lambda_cost: f32,
    /// Transfer-split non-regression tolerance.
    pub transfer_tolerance: f32,
    /// Per-day evaluation budget (gate 5).
    pub per_day_evals: u32,
    /// Caller-supplied day key (no `SystemTime` in core).
    pub day_key: String,
}

impl Default for CampaignSpec {
    fn default() -> Self {
        Self {
            baseline: Knobs::default(),
            grid: super::proposals::HpoGrid::default(),
            eta: 2,
            alpha: 0.05,
            lambda: 0.5,
            lambda_cost: 0.1,
            transfer_tolerance: 0.01,
            per_day_evals: 64,
            day_key: "day-0".to_string(),
        }
    }
}

impl CampaignSpec {
    fn gate(&self) -> Gate {
        let mut g = Gate::new(Budget::new(self.per_day_evals, self.day_key.clone()));
        g.alpha = self.alpha;
        g.lambda = self.lambda;
        g.transfer = TransferHoldout {
            tolerance: self.transfer_tolerance,
        };
        g
    }
}

/// One row of the campaign report: the decision and the numbers behind it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProposalDecision {
    /// Content-hash id; serialized as a JSON **string** (full `u64` range would
    /// lose precision as a JS number) so a `championId` matches exactly.
    #[serde(with = "super::serde_ids::id_str")]
    pub id: u64,
    #[serde(with = "super::serde_ids::opt_id_str")]
    pub parent: Option<u64>,
    pub arm: KgeArm,
    pub decision: GateDecision,
    pub val: Metrics,
    pub statistic: TestStatistic,
}

/// The campaign outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignReport {
    pub proposals: Vec<ProposalDecision>,
    pub champion: Knobs,
    /// Serialized as a JSON string (see [`ProposalDecision::id`]).
    #[serde(with = "super::serde_ids::id_str")]
    pub champion_id: u64,
    pub budget_consumed: u32,
    pub paused: bool,
    /// Test-split metrics, scored exactly twice (baseline, champion).
    pub baseline_test: Metrics,
    pub champion_test: Metrics,
    /// Filtered combined **validation** MRR of the baseline and the champion —
    /// the reward signal behind the gate, reported so a caller can show the
    /// tuning win without re-deriving it from the paired win-fractions.
    pub baseline_val_mrr: f32,
    pub champion_val_mrr: f32,
    /// Filtered combined **transfer** MRR of the baseline and the champion (the
    /// non-regression check's two numbers).
    pub baseline_transfer_mrr: f32,
    pub champion_transfer_mrr: f32,
    /// The dual hash-chained receipt log as JSONL (verifiable with
    /// [`KgeReceiptLog`]).
    pub receipts_jsonl: String,
}

/// Map a KGE arm onto typesafe's `ProposalKind`: `ModelArm` is the one semantic
/// match; HPO and Continual have no typesafe counterpart, so they fall back to
/// `ModelArm` (never `BankGrowth`). The true loop identity is carried by
/// [`ProposalDecision::arm`] and the [`KgeReceiptExtra`].
fn to_ts_kind(arm: KgeArm) -> ProposalKind {
    match arm {
        KgeArm::ModelArm => ProposalKind::ModelArm,
        KgeArm::Hpo | KgeArm::Continual => ProposalKind::ModelArm,
    }
}

fn scorer_model_id(p: &Proposal) -> String {
    let s = match p.knobs.scorer {
        crate::ScorerKind::Hole => "hole",
        crate::ScorerKind::Rotate => "rotate",
    };
    format!("{s}-d{}", p.knobs.dims)
}

fn ts_proposal(p: &Proposal) -> TsProposal {
    TsProposal {
        id: p.id,
        parent: p.parent,
        kind: to_ts_kind(p.arm),
        description_hash: p.id,
    }
}

/// The KGE-specific receipt fields (ADR-004). For campaign proposals the EWC
/// and Fisher slots are empty (those belong to continual-update receipts); the
/// manifest hash and RVF lineage are filled by the coordinator when present.
fn kge_extra(p: &Proposal) -> KgeReceiptExtra {
    KgeReceiptExtra {
        scorer: p.knobs.scorer,
        dims: p.knobs.dims,
        knobs_hash: p.id,
        ewc_lambda: None,
        fisher_id: None,
        manifest_hash: None,
        lineage: p.parent.map(|x| format!("{x:016x}")),
        sampling_temperature: Some(p.knobs.temperature),
    }
}

/// Live state threaded through every proposal step so `fit → gate → receipt`
/// stays interleaved.
struct CampaignState<'a> {
    evaluator: &'a mut dyn Evaluator,
    gate: Gate,
    log: KgeReceiptLog,
    decisions: Vec<ProposalDecision>,
    lambda_cost: f32,
    day_key: String,
    seq: u64,
    paused: bool,
    /// Best promoted proposal so far, with its outcome and reward (bandit pick).
    champion: Option<(Proposal, ArmOutcome, f32)>,
}

impl CampaignState<'_> {
    fn evidence(&self, p: &Proposal, out: &ArmOutcome) -> Evidence {
        Evidence {
            paired: out.val_paired.clone(),
            baseline_transfer_acc: out.transfer_baseline,
            champion_transfer_acc: out.transfer_candidate,
            transfer_n: out.transfer_n,
            model_id: scorer_model_id(p),
            head: SENTINEL_HEAD,
            temperature: p.knobs.temperature,
            created_seq: self.seq,
            created: None,
        }
    }

    /// Zeroed evidence for a proposal that is paused before it is ever fitted.
    fn empty_evidence(&self, p: &Proposal) -> Evidence {
        Evidence {
            paired: Vec::new(),
            baseline_transfer_acc: 0.0,
            champion_transfer_acc: 0.0,
            transfer_n: 0,
            model_id: scorer_model_id(p),
            head: SENTINEL_HEAD,
            temperature: p.knobs.temperature,
            created_seq: self.seq,
            created: None,
        }
    }

    fn record(&mut self, p: &Proposal, outcome: GateOutcome) {
        self.decisions.push(ProposalDecision {
            id: p.id,
            parent: p.parent,
            arm: p.arm,
            decision: outcome.decision.clone(),
            val: outcome.receipt.val.clone(),
            statistic: outcome.receipt.statistic.clone(),
        });
        self.log.push(outcome.receipt, kge_extra(p));
        self.seq += 1;
    }

    /// Fit, gate and record one proposal. Returns the bandit reward, or
    /// `NEG_INFINITY` once the campaign has paused (so no further fitting
    /// happens — ADR-004 "exhaustion pauses").
    fn step(&mut self, p: &Proposal) -> f32 {
        if self.paused {
            return f32::NEG_INFINITY;
        }
        // Pre-check: if the day's budget is already spent, record the pause
        // WITHOUT fitting — the paused proposal is never trained.
        if self.gate.budget.remaining() == 0 {
            let ev = self.empty_evidence(p);
            let outcome = self.gate.evaluate(ts_proposal(p), &ev, &self.day_key);
            self.record(p, outcome);
            self.paused = true;
            return f32::NEG_INFINITY;
        }

        let out = self.evaluator.fit_and_eval(p);
        let ev = self.evidence(p, &out);
        let outcome = self.gate.evaluate(ts_proposal(p), &ev, &self.day_key);
        let reward = out.reward(self.lambda_cost);
        let decision = outcome.decision.clone();
        self.record(p, outcome);
        match decision {
            GateDecision::Paused(_) => {
                self.paused = true;
                return f32::NEG_INFINITY;
            }
            GateDecision::Promote => {
                // Chain the incumbent: later proposals pair against this champion.
                self.evaluator.set_incumbent(p);
                if self.champion.as_ref().is_none_or(|(_, _, r)| reward > *r) {
                    self.champion = Some((p.clone(), out, reward));
                }
            }
            GateDecision::Reject(_) => {}
        }
        reward
    }
}

/// A campaign: successive-halving HPO, then the two model arms, all gated.
pub struct Campaign;

impl Campaign {
    /// Run one campaign against `evaluator`. The baseline is the incumbent;
    /// each proposal is paired against it. The test split is scored exactly
    /// twice — baseline and final champion — via minted `TestToken`s.
    pub fn run(spec: &CampaignSpec, evaluator: &mut dyn Evaluator) -> CampaignReport {
        let baseline = Proposal::baseline(spec.baseline.clone());
        let baseline_out = evaluator.fit_and_eval(&baseline);
        evaluator.set_incumbent(&baseline); // the initial incumbent

        let mut state = CampaignState {
            evaluator,
            gate: spec.gate(),
            log: KgeReceiptLog::new(),
            decisions: Vec::new(),
            lambda_cost: spec.lambda_cost,
            day_key: spec.day_key.clone(),
            seq: 0,
            paused: false,
            champion: None,
        };

        // Loop 1: HPO via successive halving (ranked first — ADR-004).
        let grid = spec.grid.expand(&spec.baseline);
        let hpo = successive_halving(grid, spec.eta, baseline.id, |p| state.step(p));

        // Loop 2: model arms, seeded by AND parented on the best HPO proposal
        // (or the baseline when the grid was empty) — consistent lineage.
        let (best_knobs, best_parent) = hpo
            .best
            .map_or_else(|| (spec.baseline.clone(), baseline.id), |p| (p.knobs, p.id));
        for arm in model_arms(&best_knobs, best_parent) {
            state.step(&arm);
        }

        Self::finish(spec, baseline, baseline_out, state)
    }

    fn finish(
        spec: &CampaignSpec,
        baseline: Proposal,
        baseline_out: ArmOutcome,
        state: CampaignState<'_>,
    ) -> CampaignReport {
        let CampaignState {
            gate,
            mut log,
            decisions,
            paused,
            champion,
            seq,
            ..
        } = state;

        let promoted = champion.is_some();
        // Champion is the best promoted proposal, else the baseline incumbent.
        let champ_proposal = champion
            .as_ref()
            .map_or_else(|| baseline.clone(), |(p, _, _)| p.clone());
        // Borrow the champion's outcome, or the baseline's when nothing promoted
        // (the baseline `Fn` is simply called twice — it is not `FnOnce`).
        let champ_out: &ArmOutcome = champion.as_ref().map_or(&baseline_out, |(_, o, _)| o);

        // Test scored exactly twice: baseline, then champion (ADR-004 gate 1).
        let mut tokens = TestTokens::new();
        let tb = tokens.test_token().expect("baseline test token");
        let (base_acc, n) = (baseline_out.score_test)();
        let tc = tokens.test_token().expect("champion test token");
        let (champ_acc, _) = (champ_out.score_test)();
        debug_assert!(tokens.test_token().is_none(), "only two test tokens exist");

        let baseline_test = gate.score_test(tb, base_acc, base_acc, n);
        let champion_test = gate.score_test(tc, base_acc, champ_acc, n);

        // A final champion receipt carrying the test metrics, chained on.
        let champ_val = paired_metrics(&champ_out.val_paired);
        let champ_transfer = Metrics {
            baseline_accuracy: champ_out.transfer_baseline,
            champion_accuracy: champ_out.transfer_candidate,
            n: champ_out.transfer_n,
            ece: None,
            brier: None,
        };
        let mut test = PairedSequentialTest::new(spec.alpha, spec.lambda);
        test.update_all(&champ_out.val_paired);
        // The final receipt records what actually happened to the incumbent —
        // not a hard-coded Promote (ADR-004/005: the log is the audit artifact).
        let final_decision = if promoted {
            GateDecision::Promote
        } else if paused {
            GateDecision::Paused("campaign paused; incumbent retained".into())
        } else {
            GateDecision::Reject("no proposal promoted; incumbent retained".into())
        };
        let final_receipt = Receipt {
            seq: 0,
            proposal: ts_proposal(&champ_proposal),
            parent: champ_proposal.parent,
            kind: to_ts_kind(champ_proposal.arm),
            val: champ_val,
            transfer: champ_transfer,
            test: Some(champion_test.clone()),
            statistic: test.statistic(),
            decision: final_decision,
            model_id: scorer_model_id(&champ_proposal),
            head: SENTINEL_HEAD,
            temperature: champ_proposal.knobs.temperature,
            budget_consumed: gate.budget.consumed,
            created_seq: seq,
            created: None,
            prev_hash: String::new(),
            hash: String::new(),
            signature: None,
        };
        log.push(final_receipt, kge_extra(&champ_proposal));

        CampaignReport {
            proposals: decisions,
            champion: champ_proposal.knobs,
            champion_id: champ_proposal.id,
            budget_consumed: gate.budget.consumed,
            paused,
            baseline_test,
            champion_test,
            baseline_val_mrr: baseline_out.val_mrr,
            champion_val_mrr: champ_out.val_mrr,
            baseline_transfer_mrr: baseline_out.transfer_candidate,
            champion_transfer_mrr: champ_out.transfer_candidate,
            receipts_jsonl: log.to_jsonl(),
        }
    }
}

fn paired_metrics(paired: &[(bool, bool)]) -> Metrics {
    let n = paired.len() as u32;
    let base_hits = paired.iter().filter(|(b, _)| *b).count() as u32;
    let champ_hits = paired.iter().filter(|(_, c)| *c).count() as u32;
    Metrics {
        baseline_accuracy: Metrics::ratio(base_hits, n),
        champion_accuracy: Metrics::ratio(champ_hits, n),
        n,
        ece: None,
        brier: None,
    }
}

#[cfg(test)]
mod tests;
