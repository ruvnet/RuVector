//! The optimize campaign (ADR-004). One embedder is fixed; a campaign proposes
//! a deterministic grid over [`EngineOptions`] and promotes a proposal only
//! through the [`crate::loop_gate`] — a paired anytime-valid test on the
//! validation split, a transfer-holdout, and a per-day budget. The model arm
//! (bge vs int8 vs MiniLM) is driven at the binding level by constructing one
//! engine per model; this routine stays over options for a single embedder.
//!
//! It is *pure* with respect to the engine's decision-path caches: it embeds
//! the campaign rows once, fits each proposal with the shared [`fit`] fitters,
//! scores validation/transfer directly, and touches neither the artifact cache
//! nor the bank. Test is scored exactly twice — baseline and final champion —
//! each gated by a single-use [`TestToken`] a [`Campaign`] mints.

use serde::{Deserialize, Serialize};

use crate::bank::{Split, TrustTier};
use crate::engine::fit::{self, Artifact};
use crate::engine::options::EngineOptions;
use crate::heads::{build_compiled, question_texts, ClassProtos, Compiled};
use crate::loop_gate::{
    Budget, Campaign, Evidence, Gate, GateDecision, PairedSequentialTest, PromotionCriterion,
    Proposal, ProposalKind, TestToken,
};
use crate::receipt::{Metrics, Receipt, ReceiptLog, TestStatistic};
use crate::{Answer, Embedder, Question, Result, TypesafeError};
use std::collections::BTreeMap;

fn default_tier() -> TrustTier {
    TrustTier::A
}
fn default_alpha() -> f32 {
    0.05
}
fn default_lambda() -> f32 {
    0.5
}
fn default_tolerance() -> f32 {
    0.01
}
fn default_accuracy_tolerance() -> f32 {
    0.02
}
fn default_budget() -> u32 {
    64
}
fn default_day() -> String {
    "campaign".to_string()
}

/// One labeled campaign row with an explicit frozen split (the caller's
/// fixture, not a re-hash — so the champion's "test" is the fixture's test ids).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CampaignRow {
    pub text: String,
    pub label: String,
    pub split: Split,
    #[serde(default = "default_tier")]
    pub tier: TrustTier,
}

/// A campaign over one question for the engine's fixed embedder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignSpec {
    pub question: String,
    /// The compiled scorer: the criteria (`choice`) / legend (`score`) /
    /// predicate (`noul`) whose prototypes every arm shares.
    pub question_def: Question,
    pub rows: Vec<CampaignRow>,
    /// The incumbent options the campaign starts from (the baseline).
    #[serde(default)]
    pub base_options: EngineOptions,
    /// Explicit proposals; when empty a deterministic default grid over the
    /// base options is used ([`default_grid`]).
    #[serde(default)]
    pub proposals: Vec<EngineOptions>,
    #[serde(default = "default_alpha")]
    pub alpha: f32,
    #[serde(default = "default_lambda")]
    pub lambda: f32,
    #[serde(default = "default_tolerance")]
    pub transfer_tolerance: f32,
    /// Accuracy non-inferiority margin for a calibration-only promotion (gate 2b).
    #[serde(default = "default_accuracy_tolerance")]
    pub accuracy_tolerance: f32,
    #[serde(default = "default_budget")]
    pub budget_per_day: u32,
    #[serde(default = "default_day")]
    pub day_key: String,
    #[serde(default)]
    pub created_seq_base: u64,
    #[serde(default)]
    pub created: Option<String>,
}

/// One arm's outcome inside a campaign.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmResult {
    pub proposal_id: u64,
    pub options: EngineOptions,
    pub decision: GateDecision,
    pub val: Metrics,
    pub transfer: Metrics,
    pub statistic: TestStatistic,
    /// The calibration (paired NLL) test statistic (ADR-004 gate 2b).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_statistic: Option<TestStatistic>,
    /// Which criterion promoted this arm, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub promoted_by: Option<PromotionCriterion>,
    pub promoted: bool,
}

/// The campaign's result: baseline → champion, per-split metrics, the chained
/// receipt log, and the test-split scores (each from one [`TestToken`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignReport {
    pub question: String,
    pub embedder_id: String,
    pub baseline_options: EngineOptions,
    pub champion_options: EngineOptions,
    pub baseline_val: Metrics,
    pub champion_val: Metrics,
    pub baseline_transfer: Metrics,
    pub champion_transfer: Metrics,
    pub baseline_test: Metrics,
    pub champion_test: Metrics,
    pub arms: Vec<ArmResult>,
    pub promotions: usize,
    pub budget_consumed: u32,
    /// Number of `score_test` calls — must be exactly 2 (baseline + champion).
    pub test_scorings: u8,
    pub receipts: ReceiptLog,
}

/// A deterministic default grid over the base options: the campaign's four
/// levers (logit sharpening, probe regularisation/iterations, class balancing).
/// Order is fixed, so a campaign is reproducible.
#[must_use]
pub fn default_grid(base: &EngineOptions) -> Vec<EngineOptions> {
    let mut out = Vec::new();
    // Sharpen flat logits before calibration (the 8-shot underconfidence fix).
    for scale in [8.0f32, 20.0] {
        out.push(EngineOptions {
            logit_scale: scale,
            ..base.clone()
        });
    }
    // Lower the calibration-slice floor so temperature fits at low shot counts
    // (the decisive 8-shot ECE fix): a calibration-only change the accuracy test
    // cannot promote but the NLL criterion (gate 2b) can.
    out.push(EngineOptions {
        min_calibration: 10,
        ..base.clone()
    });
    out.push(EngineOptions {
        min_calibration: 10,
        logit_scale: 8.0,
        ..base.clone()
    });
    // Class-balanced probe (minority-class recall under an imbalanced sample).
    out.push(EngineOptions {
        probe_class_balanced: true,
        ..base.clone()
    });
    out.push(EngineOptions {
        logit_scale: 20.0,
        probe_class_balanced: true,
        ..base.clone()
    });
    // Stronger / weaker probe regularisation.
    out.push(EngineOptions {
        probe_l2: 1e-2,
        logit_scale: 20.0,
        ..base.clone()
    });
    out.push(EngineOptions {
        probe_iterations: 800,
        logit_scale: 20.0,
        ..base.clone()
    });
    out
}

/// One row resolved to a state embedding, its true class index (`choice`/
/// `score`) or noul label, and its split.
struct Resolved {
    emb: Vec<f32>,
    class: Option<usize>,
    y: Option<f32>,
    split: Split,
}

mod campaign;
mod evaluate;

fn noul_correct(art: &Artifact, predicate: &[f32], emb: &[f32], model: &str, y: f32) -> bool {
    if let Answer::Noul { noul, .. } = fit::noul_answer(art, predicate, emb, model) {
        (noul >= 0.5) == (y >= 0.5)
    } else {
        false
    }
}

/// `(correct, NLL of the true label)` for a noul item.
fn noul_correct_nll(
    art: &Artifact,
    predicate: &[f32],
    emb: &[f32],
    model: &str,
    y: f32,
) -> (bool, f32) {
    if let Answer::Noul { noul, .. } = fit::noul_answer(art, predicate, emb, model) {
        let p_true = if y >= 0.5 { noul } else { 1.0 - noul };
        ((noul >= 0.5) == (y >= 0.5), nll(p_true))
    } else {
        (false, nll(0.0))
    }
}

/// Negative log-likelihood of the true class, floored so a zero probability
/// gives a large-but-finite loss instead of infinity.
fn nll(p_true: f32) -> f32 {
    -p_true.max(1e-9).ln()
}

fn class_index(compiled: &Compiled) -> Option<BTreeMap<String, usize>> {
    match compiled {
        Compiled::Class(cp) => Some(
            cp.keys
                .iter()
                .enumerate()
                .map(|(i, k)| (k.clone(), i))
                .collect(),
        ),
        Compiled::Noul { .. } => None,
    }
}

fn class_pairs(resolved: &[Resolved], split: Split) -> Vec<(Vec<f32>, usize)> {
    resolved
        .iter()
        .filter(|r| r.split == split)
        .filter_map(|r| r.class.map(|c| (r.emb.clone(), c)))
        .collect()
}

fn noul_pairs(resolved: &[Resolved], split: Split) -> Vec<(Vec<f32>, f32)> {
    resolved
        .iter()
        .filter(|r| r.split == split)
        .filter_map(|r| r.y.map(|y| (r.emb.clone(), y)))
        .collect()
}

fn empty_metrics() -> Metrics {
    Metrics {
        baseline_accuracy: 0.0,
        champion_accuracy: 0.0,
        n: 0,
        ece: None,
        brier: None,
    }
}

/// Expected calibration error, 10 equal-width bins over `[0, 1]` confidence.
fn ece(conf: &[f32], correct: &[bool]) -> f32 {
    if conf.is_empty() || conf.len() != correct.len() {
        return 0.0;
    }
    const BINS: usize = 10;
    let mut bin_conf = [0.0f32; BINS];
    let mut bin_acc = [0.0f32; BINS];
    let mut bin_n = [0u32; BINS];
    for (c, ok) in conf.iter().zip(correct) {
        let b = ((c * BINS as f32) as usize).min(BINS - 1);
        bin_conf[b] += *c;
        bin_acc[b] += if *ok { 1.0 } else { 0.0 };
        bin_n[b] += 1;
    }
    let total = conf.len() as f32;
    let mut e = 0.0f32;
    for b in 0..BINS {
        if bin_n[b] == 0 {
            continue;
        }
        let n = bin_n[b] as f32;
        let acc = bin_acc[b] / n;
        let cf = bin_conf[b] / n;
        e += (n / total) * (acc - cf).abs();
    }
    e
}

/// Stable FNV-1a over the canonical options JSON — a receipt's description hash.
fn options_hash(opts: &EngineOptions) -> u64 {
    let s = serde_json::to_string(opts).unwrap_or_default();
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(all(test, feature = "hash-embedder"))]
mod tests;
