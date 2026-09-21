//! The self-optimization loop (ADR-004). HPO arms are weighted at least as
//! heavily as model arms (LibKGE/KGTuner), every promotion passes the one gate
//! defined in typesafe ADR-004 (reused verbatim from `ruvector-typesafe-core`,
//! not restated), and every decision is recorded in a hash-chained receipt.
//!
//! The trainer/evaluator is not in this crate yet — two sibling agents are
//! writing `data`/`train`/`eval`. This module codes against the pinned
//! [`Scorer`](crate::Scorer) trait plus its own [`Evaluator`] seam, so the loop
//! is testable in isolation; the coordinator wires the real trainer to it.

pub mod campaign;
pub mod continual;
pub mod proposals;
pub mod receipt;
pub(crate) mod serde_ids;
pub mod trainer_eval;

pub use campaign::{Campaign, CampaignReport, CampaignSpec, ProposalDecision};
pub use continual::ContinualUpdate;
pub use proposals::{HalvingResult, HpoGrid, KgeArm, Knobs, Loss, Optimizer, Proposal};
pub use receipt::{KgeReceipt, KgeReceiptExtra, KgeReceiptLog};
pub use trainer_eval::TrainerEvaluator;

/// Scores one model on the frozen **test** split → `(accuracy_or_mrr, n)`.
/// Called at most twice per campaign (baseline, champion) and only after a
/// [`TestToken`](ruvector_typesafe_core::loop_gate) is minted (ADR-004 gate 1).
pub type TestScoreFn = Box<dyn Fn() -> (f32, u32)>;

/// What an [`Evaluator`] returns for one proposal: everything the promotion
/// gate and the cost-aware bandit reward need, and nothing that could leak
/// triple text (ADR-005). Not `Clone`/`Debug`: it owns the test-scoring
/// closure.
pub struct ArmOutcome {
    /// Per validation item, paired with the incumbent:
    /// `(incumbent_correct, candidate_correct)`, fed straight into typesafe's
    /// `PairedSequentialTest`. A stub may read "correct" as a Hits@k hit; the
    /// real [`TrainerEvaluator`] instead encodes a per-query *rank improvement*
    /// (candidate ranks the true entity above the incumbent), so its receipt
    /// val accuracies are McNemar win-fractions, not Hits@k — see
    /// [`trainer_eval`](crate::optimize::trainer_eval) for the exact mapping.
    pub val_paired: Vec<(bool, bool)>,
    /// Candidate filtered MRR on validation — the reward signal (ADR-004's
    /// cost-aware `MRR − λ·cost`).
    pub val_mrr: f32,
    /// Incumbent filtered MRR/accuracy on the **transfer** split (gate 3).
    pub transfer_baseline: f32,
    /// Candidate filtered MRR/accuracy on the transfer split.
    pub transfer_candidate: f32,
    pub transfer_n: u32,
    /// `train_time + score_ms` proxy, normalised; the `cost` in `MRR − λ·cost`.
    pub cost: f32,
    /// Scores this arm on the frozen test split (see [`TestScoreFn`]).
    pub score_test: TestScoreFn,
}

impl ArmOutcome {
    /// Candidate accuracy on validation, derived from `val_paired`.
    #[must_use]
    pub fn champion_val_acc(&self) -> f32 {
        if self.val_paired.is_empty() {
            return 0.0;
        }
        let hits = self.val_paired.iter().filter(|(_, c)| *c).count();
        hits as f32 / self.val_paired.len() as f32
    }

    /// Cost-aware reward `val_mrr − λ·cost` (ADR-004 loop 1).
    #[must_use]
    pub fn reward(&self, lambda_cost: f32) -> f32 {
        self.val_mrr - lambda_cost * self.cost
    }
}

/// The seam the campaign drives. A real implementation fits `proposal.knobs`
/// on the frozen train split and evaluates it against the current incumbent; a
/// stub (see the tests) returns synthetic outcomes. `fit_and_eval`'s signature
/// is pinned — the coordinator wires the trainer to exactly it.
///
/// The campaign starts with the baseline incumbent and calls
/// [`Evaluator::set_incumbent`] after every promotion, so later proposals pair
/// against the current champion (a chain of promotions), not the campaign
/// baseline. Choosing *which* promoted proposal to report as the champion is a
/// separate reward-based bandit selection, not a second gate.
pub trait Evaluator {
    fn fit_and_eval(&mut self, proposal: &Proposal) -> ArmOutcome;

    /// Adopt `proposal` as the incumbent that subsequent `fit_and_eval` calls
    /// pair against. Default no-op for evaluators that pair against a fixed
    /// reference.
    fn set_incumbent(&mut self, proposal: &Proposal) {
        let _ = proposal;
    }
}
