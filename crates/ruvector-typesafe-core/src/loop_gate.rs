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

/// Which criterion carried a promotion (ADR-004 gate 2 / 2b). `Accuracy`: the
/// paired accuracy test rejected "no improvement". `Calibration`: accuracy was
/// non-inferior within tolerance and the paired NLL test rejected — so a
/// calibration-only change (e.g. a logit-scale / temperature-floor tweak that
/// never moves the argmax) can be promoted, which the accuracy test alone can
/// never do (identical argmax → all concordant pairs → zero paired information).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PromotionCriterion {
    Accuracy,
    Calibration,
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
    /// Per-item validation `(baseline_nll, champion_nll)` for the calibration
    /// criterion (ADR-004 gate 2b). `None` → calibration test is not run (the
    /// original accuracy-only behaviour).
    pub paired_nll: Option<Vec<(f32, f32)>>,
    pub baseline_transfer_acc: f32,
    pub champion_transfer_acc: f32,
    pub transfer_n: u32,
    pub model_id: String,
    pub head: Head,
    pub temperature: f32,
    pub created_seq: u64,
    pub created: Option<String>,
}

/// Ties in a per-item NLL comparison (within this) carry no paired information.
const NLL_TIE_EPS: f32 = 1e-6;

impl Evidence {
    /// Paired NLL outcomes as `(baseline_better, champion_better)` for a
    /// [`PairedSequentialTest`]: the champion "wins" a discordant pair iff its
    /// per-item NLL is lower (better calibrated) by more than `NLL_TIE_EPS`.
    fn calibration_pairs(&self) -> Vec<(bool, bool)> {
        match &self.paired_nll {
            None => Vec::new(),
            Some(rows) => rows
                .iter()
                .map(|(b, c)| {
                    let champion_better = *c + NLL_TIE_EPS < *b;
                    let baseline_better = *b + NLL_TIE_EPS < *c;
                    (baseline_better, champion_better)
                })
                .collect(),
        }
    }

    /// Champion validation accuracy minus baseline (from the paired outcomes).
    fn val_accuracy_delta(&self) -> f32 {
        let n = self.paired.len().max(1) as f32;
        let base = self.paired.iter().filter(|(b, _)| *b).count() as f32;
        let champ = self.paired.iter().filter(|(_, c)| *c).count() as f32;
        (champ - base) / n
    }

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

mod gate;
mod sequential;

pub use gate::Gate;
pub use sequential::PairedSequentialTest;

#[cfg(test)]
mod tests;
