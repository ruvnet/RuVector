//! The promotion gate (ADR-004 "The promotion gate"). A [`Proposal`] is
//! promoted only when an anytime-valid paired test on the validation split
//! rejects "no improvement" AND the transfer split does not regress beyond a
//! tolerance AND the day's evaluation budget is not exhausted. Exhaustion
//! *pauses* — it never lowers the bar.
//!
//! Every module here is a pure struct so it can be unit-tested with no engine,
//! embedder or bank present; the gate depends only on outcomes the caller feeds
//! it, never on `heads`/`calibration`/`engine`.

use crate::receipt::{Metrics, Receipt, TestStatistic};
use crate::Head;
use serde::{Deserialize, Serialize};

/// What a proposal changes (ADR-004 loops 1–4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProposalKind {
    /// Loop 1: grow the example bank (active learning).
    BankGrowth,
    /// Loop 2: model-arm / head selection (bandit).
    ModelArm,
    /// Loop 3: speed self-tuning (quantisation, adaptive `ef`).
    SpeedTuning,
    /// Loop 4: criteria mutation (highest risk, strictest budget).
    CriteriaMutation,
}

/// A candidate change awaiting the gate. Carries only a `description_hash`, not
/// the description text (ADR-005: receipts store hashes, never text).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Proposal {
    pub id: u64,
    pub parent: Option<u64>,
    pub kind: ProposalKind,
    pub description_hash: u64,
}

/// The gate's verdict. `Reject`/`Paused` carry a machine-stable reason.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "decision", content = "reason", rename_all = "kebab-case")]
pub enum GateDecision {
    Promote,
    Reject(String),
    Paused(String),
}

/// Anytime-valid paired test by betting (ADR-004 gate 2; e-processes,
/// arXiv:2606.00878 and arXiv:2501.03982). For each *discordant* validation
/// pair — one model right, the other wrong — wealth updates
/// `W ← W · (1 + λ·(X − ½))`, where `X = 1` iff the champion won the pair.
/// Concordant pairs carry no paired information and are skipped (McNemar's
/// structure). Under the null "no improvement", `X ~ Bernoulli(½)` so `E[W]`
/// stays 1: `W` is a non-negative martingale and Ville's inequality gives
/// `P(sup_t W_t ≥ 1/α) ≤ α`. The rejection therefore latches on the first
/// crossing of `1/α` and never un-latches.
#[derive(Debug, Clone)]
pub struct PairedSequentialTest {
    alpha: f32,
    lambda: f32,
    wealth: f64,
    max_wealth: f64,
    n_champion_wins: u32,
    n_baseline_wins: u32,
    rejected: bool,
    n_discordant_at_rejection: Option<u32>,
}

impl PairedSequentialTest {
    /// `alpha` is the type-I bound (default 0.05); `lambda` is the betting
    /// fraction. `lambda` must lie in `(0, 2)` — outside it wealth can go
    /// negative and the martingale argument breaks — so it is clamped into a
    /// safe interior band.
    #[must_use]
    pub fn new(alpha: f32, lambda: f32) -> Self {
        let alpha = alpha.clamp(1e-6, 0.5);
        let lambda = lambda.clamp(0.01, 1.99);
        Self {
            alpha,
            lambda,
            wealth: 1.0,
            max_wealth: 1.0,
            n_champion_wins: 0,
            n_baseline_wins: 0,
            rejected: false,
            n_discordant_at_rejection: None,
        }
    }

    /// α = 0.05, λ = 0.5 (wealth factor in `{0.75, 1.25}` — never negative).
    #[must_use]
    pub fn standard() -> Self {
        Self::new(0.05, 0.5)
    }

    fn threshold(&self) -> f64 {
        1.0 / self.alpha as f64
    }

    /// Feed one paired outcome. Returns `true` once the rejection has latched.
    pub fn update(&mut self, baseline_correct: bool, champion_correct: bool) -> bool {
        if baseline_correct == champion_correct {
            return self.rejected; // concordant: no information
        }
        let champion_won = champion_correct && !baseline_correct;
        let x = if champion_won { 1.0 } else { 0.0 };
        if champion_won {
            self.n_champion_wins += 1;
        } else {
            self.n_baseline_wins += 1;
        }
        self.wealth *= 1.0 + self.lambda as f64 * (x - 0.5);
        if self.wealth > self.max_wealth {
            self.max_wealth = self.wealth;
        }
        if !self.rejected && self.max_wealth >= self.threshold() {
            self.rejected = true;
            self.n_discordant_at_rejection = Some(self.n_champion_wins + self.n_baseline_wins);
        }
        self.rejected
    }

    /// Feed a whole stream of `(baseline_correct, champion_correct)` pairs.
    pub fn update_all(&mut self, pairs: &[(bool, bool)]) -> bool {
        for &(b, c) in pairs {
            self.update(b, c);
        }
        self.rejected
    }

    /// `true` once wealth has ever reached `1/α` (latched, anytime-valid).
    #[must_use]
    pub fn rejected(&self) -> bool {
        self.rejected
    }

    /// Plain paired counts `(champion_wins, baseline_wins)` — the McNemar cells.
    #[must_use]
    pub fn discordant_counts(&self) -> (u32, u32) {
        (self.n_champion_wins, self.n_baseline_wins)
    }

    #[must_use]
    pub fn statistic(&self) -> TestStatistic {
        TestStatistic {
            alpha: self.alpha,
            lambda: self.lambda,
            wealth: self.wealth,
            max_wealth: self.max_wealth,
            threshold: self.threshold(),
            n_champion_wins: self.n_champion_wins,
            n_baseline_wins: self.n_baseline_wins,
            rejected: self.rejected,
            n_discordant_at_rejection: self.n_discordant_at_rejection,
        }
    }
}

/// ADR-004 gate 3: the champion must not regress on the transfer split beyond
/// `tolerance` (default 0.01), so the loop cannot overfit validation phrasing.
#[derive(Debug, Clone, Copy)]
pub struct TransferHoldout {
    pub tolerance: f32,
}

impl Default for TransferHoldout {
    fn default() -> Self {
        Self { tolerance: 0.01 }
    }
}

impl TransferHoldout {
    /// `true` if the champion is acceptable (did not regress beyond tolerance).
    #[must_use]
    pub fn passes(&self, baseline_acc: f32, champion_acc: f32) -> bool {
        champion_acc >= baseline_acc - self.tolerance
    }
}

/// ADR-004 gate 4: a permanent control arm running the pre-mutation config.
/// Drift of the champion below the control by more than `threshold`, once at
/// least `min_samples` paired observations are in, raises the regression alarm.
#[derive(Debug, Clone)]
pub struct ControlArm {
    threshold: f32,
    min_samples: u32,
    control_correct: u32,
    champion_correct: u32,
    n: u32,
}

impl ControlArm {
    #[must_use]
    pub fn new(threshold: f32, min_samples: u32) -> Self {
        Self {
            threshold,
            min_samples,
            control_correct: 0,
            champion_correct: 0,
            n: 0,
        }
    }

    pub fn observe(&mut self, control_correct: bool, champion_correct: bool) {
        self.n += 1;
        self.control_correct += u32::from(control_correct);
        self.champion_correct += u32::from(champion_correct);
    }

    /// `champion_rate − control_rate`.
    #[must_use]
    pub fn drift(&self) -> f32 {
        if self.n == 0 {
            return 0.0;
        }
        (self.champion_correct as f32 - self.control_correct as f32) / self.n as f32
    }

    /// `true` when enough samples are in and the champion trails the control by
    /// more than `threshold`.
    #[must_use]
    pub fn alarm(&self) -> bool {
        self.n >= self.min_samples && self.drift() < -self.threshold
    }

    #[must_use]
    pub fn samples(&self) -> u32 {
        self.n
    }
}

/// Per-day evaluation budget (ADR-004 gate 5). The `day_key` is caller-supplied
/// (no `SystemTime` in core); a new key rolls the counter over.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Budget {
    pub per_day_evals: u32,
    pub consumed: u32,
    pub day_key: String,
}

impl Budget {
    #[must_use]
    pub fn new(per_day_evals: u32, day_key: impl Into<String>) -> Self {
        Self {
            per_day_evals,
            consumed: 0,
            day_key: day_key.into(),
        }
    }

    /// Try to spend one evaluation for `day_key`. Rolls over on a new day.
    /// Returns `false` when the day's budget is exhausted.
    pub fn try_consume(&mut self, day_key: &str) -> bool {
        if day_key != self.day_key {
            self.day_key = day_key.to_string();
            self.consumed = 0;
        }
        if self.consumed < self.per_day_evals {
            self.consumed += 1;
            true
        } else {
            false
        }
    }

    #[must_use]
    pub fn remaining(&self) -> u32 {
        self.per_day_evals.saturating_sub(self.consumed)
    }
}

/// Everything the gate needs to judge one proposal. `paired` is the validation
/// split's `(baseline_correct, champion_correct)` outcomes; val accuracies are
/// derived from it, never passed redundantly. `model_id`/`head`/`temperature`
/// identify the scorer for the receipt.
#[derive(Debug, Clone)]
pub struct Evidence {
    pub paired: Vec<(bool, bool)>,
    pub baseline_transfer_acc: f32,
    pub champion_transfer_acc: f32,
    pub transfer_n: u32,
    pub model_id: String,
    pub head: Head,
    pub temperature: f32,
    pub created_seq: u64,
    pub created: Option<String>,
}

impl Evidence {
    fn val_metrics(&self) -> Metrics {
        let n = self.paired.len() as u32;
        let base_hits = self.paired.iter().filter(|(b, _)| *b).count() as u32;
        let champ_hits = self.paired.iter().filter(|(_, c)| *c).count() as u32;
        Metrics {
            baseline_accuracy: Metrics::ratio(base_hits, n),
            champion_accuracy: Metrics::ratio(champ_hits, n),
            n,
            ece: None,
            brier: None,
        }
    }

    fn transfer_metrics(&self) -> Metrics {
        Metrics {
            baseline_accuracy: self.baseline_transfer_acc,
            champion_accuracy: self.champion_transfer_acc,
            n: self.transfer_n,
            ece: None,
            brier: None,
        }
    }
}

/// The gate's return: a decision and the receipt that records it. The receipt
/// is not yet chained — hand it to a [`crate::receipt::ReceiptLog`] to append.
#[derive(Debug, Clone)]
pub struct GateOutcome {
    pub decision: GateDecision,
    pub receipt: Receipt,
}

/// Single-use token authorising ONE test-split scoring. The loop cannot obtain
/// one, so it can never score test inside a campaign (ADR-004 gate 1: test is
/// scored only for the baseline and the final champion).
#[derive(Debug)]
pub struct TestToken {
    _private: (),
}

/// A campaign hands out exactly two [`TestToken`]s — one for the baseline, one
/// for the final champion — and no more.
#[derive(Debug)]
pub struct Campaign {
    tokens_left: u8,
}

impl Default for Campaign {
    fn default() -> Self {
        Self { tokens_left: 2 }
    }
}

impl Campaign {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Mint a test-scoring token; `None` once both (baseline, champion) are out.
    pub fn test_token(&mut self) -> Option<TestToken> {
        if self.tokens_left > 0 {
            self.tokens_left -= 1;
            Some(TestToken { _private: () })
        } else {
            None
        }
    }
}

/// The promotion gate. Holds the sequential-test parameters, the transfer
/// tolerance and the (stateful) day budget.
#[derive(Debug, Clone)]
pub struct Gate {
    pub alpha: f32,
    pub lambda: f32,
    pub transfer: TransferHoldout,
    pub budget: Budget,
}

impl Gate {
    #[must_use]
    pub fn new(budget: Budget) -> Self {
        Self {
            alpha: 0.05,
            lambda: 0.5,
            transfer: TransferHoldout::default(),
            budget,
        }
    }

    /// Judge one proposal. Takes `&mut self` because the budget is stateful
    /// (the brief's `&self` cannot spend a budget without interior mutability,
    /// which would be worse); `day_key` charges the right day.
    pub fn evaluate(
        &mut self,
        proposal: Proposal,
        evidence: &Evidence,
        day_key: &str,
    ) -> GateOutcome {
        let val = evidence.val_metrics();
        let transfer = evidence.transfer_metrics();

        // Budget first: exhaustion pauses, never lowers the bar.
        if !self.budget.try_consume(day_key) {
            let stat = PairedSequentialTest::new(self.alpha, self.lambda).statistic();
            return self.outcome(
                proposal,
                evidence,
                val,
                transfer,
                stat,
                GateDecision::Paused("daily evaluation budget exhausted".into()),
            );
        }

        let mut test = PairedSequentialTest::new(self.alpha, self.lambda);
        test.update_all(&evidence.paired);
        let stat = test.statistic();

        let transfer_ok = self.transfer.passes(
            evidence.baseline_transfer_acc,
            evidence.champion_transfer_acc,
        );

        let decision = if !transfer_ok {
            GateDecision::Reject("transfer split regressed beyond tolerance".into())
        } else if test.rejected() {
            GateDecision::Promote
        } else {
            GateDecision::Reject("sequential test did not reject no-improvement".into())
        };

        self.outcome(proposal, evidence, val, transfer, stat, decision)
    }

    #[allow(clippy::too_many_arguments)]
    fn outcome(
        &self,
        proposal: Proposal,
        evidence: &Evidence,
        val: Metrics,
        transfer: Metrics,
        statistic: TestStatistic,
        decision: GateDecision,
    ) -> GateOutcome {
        let receipt = Receipt {
            seq: 0,
            proposal,
            parent: proposal.parent,
            kind: proposal.kind,
            val,
            transfer,
            test: None,
            statistic,
            decision: decision.clone(),
            model_id: evidence.model_id.clone(),
            head: evidence.head,
            temperature: evidence.temperature,
            budget_consumed: self.budget.consumed,
            created_seq: evidence.created_seq,
            created: evidence.created.clone(),
            prev_hash: String::new(),
            hash: String::new(),
            signature: None,
        };
        GateOutcome { decision, receipt }
    }

    /// Score the frozen test split. Requires a [`TestToken`], which only a
    /// [`Campaign`] can mint (twice) — so this is unreachable from inside the
    /// loop. Consumes the token by value.
    #[must_use]
    pub fn score_test(
        &self,
        _token: TestToken,
        baseline_acc: f32,
        champion_acc: f32,
        n: u32,
    ) -> Metrics {
        Metrics {
            baseline_accuracy: baseline_acc,
            champion_accuracy: champion_acc,
            n,
            ece: None,
            brier: None,
        }
    }
}

#[cfg(test)]
mod tests;
