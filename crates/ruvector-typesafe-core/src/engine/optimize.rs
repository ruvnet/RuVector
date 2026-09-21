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

impl<E: Embedder> super::Engine<E> {
    /// Run a campaign over [`EngineOptions`] for this engine's embedder.
    /// Returns the baseline → champion comparison, the chained receipts, and
    /// the (twice-scored) test metrics. Does not mutate the engine's own bank or
    /// caches — pass the champion options to [`with_options`](Self::with_options)
    /// to adopt them.
    pub fn optimize(&self, spec: &CampaignSpec) -> Result<CampaignReport> {
        let dims = self.embedder().dims();
        let model_id = self.embedder().id().to_string();
        let allow_cal = !model_id.ends_with("@test-double");

        // Compile the scorer once (embed criteria / legend / predicate).
        let qtexts = question_texts(&spec.question_def);
        let qrefs: Vec<&str> = qtexts.iter().map(String::as_str).collect();
        let qembs = if qrefs.is_empty() {
            Vec::new()
        } else {
            self.embedder().embed(&qrefs)?
        };
        let compiled = build_compiled(&spec.question_def, &qembs);

        // Embed every row's state once.
        let row_refs: Vec<&str> = spec.rows.iter().map(|r| r.text.as_str()).collect();
        let row_embs = if row_refs.is_empty() {
            Vec::new()
        } else {
            self.embedder().embed(&row_refs)?
        };
        if row_embs.len() != spec.rows.len() {
            return Err(TypesafeError::Embedder(
                "row embedding count mismatch".into(),
            ));
        }

        let index = class_index(&compiled);
        let resolved: Vec<Resolved> = spec
            .rows
            .iter()
            .zip(row_embs)
            .map(|(row, emb)| Resolved {
                emb,
                class: index
                    .as_ref()
                    .and_then(|m| m.get(row.label.as_str()).copied()),
                y: super::support::parse_noul_label(&row.label),
                split: row.split,
            })
            .collect();

        // Training / calibration splits (shared by every arm).
        let class_train = class_pairs(&resolved, Split::Train);
        let class_calib = class_pairs(&resolved, Split::Calibration);
        let noul_train = noul_pairs(&resolved, Split::Train);
        let noul_calib = noul_pairs(&resolved, Split::Calibration);

        let fit_opts = |opts: &EngineOptions| -> Artifact {
            match &compiled {
                Compiled::Class(cp) => {
                    fit::fit_class_artifact(opts, cp, &class_train, &class_calib, dims, allow_cal)
                }
                Compiled::Noul { .. } => {
                    fit::fit_noul_artifact(opts, &noul_train, &noul_calib, dims, allow_cal)
                }
            }
        };

        // Incumbent (baseline) — the champion of the campaign starts here and
        // is replaced only on a promotion (successive).
        let mut champ_opts = spec.base_options.clone();
        let mut champ_art = fit_opts(&champ_opts);
        let baseline_art = fit_opts(&spec.base_options);

        let proposals = if spec.proposals.is_empty() {
            default_grid(&spec.base_options)
        } else {
            spec.proposals.clone()
        };

        let mut gate = Gate {
            alpha: spec.alpha,
            lambda: spec.lambda,
            transfer: crate::loop_gate::TransferHoldout {
                tolerance: spec.transfer_tolerance,
            },
            budget: Budget::new(spec.budget_per_day, spec.day_key.clone()),
            accuracy_tolerance: spec.accuracy_tolerance,
        };
        let mut log = ReceiptLog::new();
        let mut arms = Vec::new();
        let mut promotions = 0usize;

        for (i, opts) in proposals.iter().enumerate() {
            let cand_art = fit_opts(opts);
            // Paired validation outcomes (accuracy + NLL): incumbent vs candidate.
            let (paired, paired_nll) = self.paired_validation(
                &compiled,
                &index,
                &resolved,
                &champ_opts,
                &champ_art,
                opts,
                &cand_art,
                &model_id,
            );
            let base_xfer = self.split_accuracy(
                &compiled,
                &index,
                &resolved,
                &champ_opts,
                &champ_art,
                Split::Transfer,
                &model_id,
            );
            let cand_xfer = self.split_accuracy(
                &compiled,
                &index,
                &resolved,
                opts,
                &cand_art,
                Split::Transfer,
                &model_id,
            );
            let evidence = Evidence {
                paired,
                paired_nll: Some(paired_nll),
                baseline_transfer_acc: base_xfer.0,
                champion_transfer_acc: cand_xfer.0,
                transfer_n: cand_xfer.1,
                model_id: model_id.clone(),
                head: cand_art.head(),
                temperature: cand_art.temperature(),
                created_seq: spec.created_seq_base + i as u64,
                created: spec.created.clone(),
            };
            let proposal = Proposal {
                id: spec.created_seq_base + i as u64 + 1,
                parent: Some(spec.created_seq_base),
                kind: ProposalKind::ModelArm,
                description_hash: options_hash(opts),
            };
            let outcome = gate.evaluate(proposal, &evidence, &spec.day_key);
            let promoted = matches!(outcome.decision, GateDecision::Promote);
            log.push(outcome.receipt.clone());
            arms.push(ArmResult {
                proposal_id: proposal.id,
                options: opts.clone(),
                decision: outcome.decision.clone(),
                val: outcome.receipt.val.clone(),
                transfer: outcome.receipt.transfer.clone(),
                statistic: outcome.receipt.statistic.clone(),
                calibration_statistic: outcome.receipt.calibration_statistic.clone(),
                promoted_by: outcome.receipt.promoted_by,
                promoted,
            });
            if promoted {
                champ_opts = opts.clone();
                champ_art = cand_art;
                promotions += 1;
            }
        }

        // Final per-split metrics for baseline and champion.
        let (baseline_val, champion_val) = self.pair_metrics(
            &compiled,
            &index,
            &resolved,
            &spec.base_options,
            &baseline_art,
            &champ_opts,
            &champ_art,
            Split::Validation,
            &model_id,
        );
        let (baseline_transfer, champion_transfer) = self.pair_metrics(
            &compiled,
            &index,
            &resolved,
            &spec.base_options,
            &baseline_art,
            &champ_opts,
            &champ_art,
            Split::Transfer,
            &model_id,
        );

        // Test — reachable ONLY through `score_test_split`, which consumes a
        // single-use `TestToken`. A `Campaign` mints exactly two (baseline,
        // champion), so the test split is physically unscoreable a third time
        // (ADR-004 gate 1). `split_scored` refuses `Split::Test` outright.
        let mut campaign = Campaign::new();
        let mut test_scorings = 0u8;
        let baseline_test = match campaign.test_token() {
            Some(token) => {
                test_scorings += 1;
                let (acc, n, e) = self.score_test_split(
                    token,
                    &compiled,
                    &index,
                    &resolved,
                    &spec.base_options,
                    &baseline_art,
                    &model_id,
                );
                Metrics {
                    baseline_accuracy: acc,
                    champion_accuracy: acc,
                    n,
                    ece: Some(e),
                    brier: None,
                }
            }
            None => empty_metrics(),
        };
        let champion_test = match campaign.test_token() {
            Some(token) => {
                test_scorings += 1;
                let (acc, n, e) = self.score_test_split(
                    token,
                    &compiled,
                    &index,
                    &resolved,
                    &champ_opts,
                    &champ_art,
                    &model_id,
                );
                Metrics {
                    baseline_accuracy: baseline_test.baseline_accuracy,
                    champion_accuracy: acc,
                    n,
                    ece: Some(e),
                    brier: None,
                }
            }
            None => empty_metrics(),
        };
        // A third scoring is impossible: the campaign is out of tokens.
        debug_assert!(campaign.test_token().is_none());

        // Final champion receipt — carries the test metrics into the hash chain
        // (ADR-004 "every promotion writes a receipt … with the test statistic").
        let champion_receipt = Receipt {
            seq: 0,
            proposal: Proposal {
                id: spec.created_seq_base,
                parent: None,
                kind: ProposalKind::ModelArm,
                description_hash: options_hash(&champ_opts),
            },
            parent: None,
            kind: ProposalKind::ModelArm,
            val: champion_val.clone(),
            transfer: champion_transfer.clone(),
            test: Some(champion_test.clone()),
            statistic: PairedSequentialTest::new(spec.alpha, spec.lambda).statistic(),
            calibration_statistic: None,
            promoted_by: None,
            decision: if promotions > 0 {
                GateDecision::Promote
            } else {
                GateDecision::Reject("no proposal reached significance".into())
            },
            model_id: model_id.clone(),
            head: champ_art.head(),
            temperature: champ_art.temperature(),
            budget_consumed: gate.budget.consumed,
            created_seq: spec.created_seq_base,
            created: spec.created.clone(),
            prev_hash: String::new(),
            hash: String::new(),
            signature: None,
        };
        log.push(champion_receipt);

        let budget_consumed = gate.budget.consumed;
        Ok(CampaignReport {
            question: spec.question.clone(),
            embedder_id: model_id,
            baseline_options: spec.base_options.clone(),
            champion_options: champ_opts,
            baseline_val,
            champion_val,
            baseline_transfer,
            champion_transfer,
            baseline_test,
            champion_test,
            arms,
            promotions,
            budget_consumed,
            test_scorings,
            receipts: log,
        })
    }

    /// Predict the class index and confidence for one state under `opts`.
    fn class_predict(
        &self,
        cp: &ClassProtos,
        art: &Artifact,
        opts: &EngineOptions,
        emb: &[f32],
        model: &str,
        index: &BTreeMap<String, usize>,
    ) -> (usize, f32) {
        match fit::class_answer(opts, cp, art, emb, model) {
            Answer::Choice { choice, meta, .. } => {
                (*index.get(&choice).unwrap_or(&usize::MAX), meta.confidence)
            }
            Answer::Score { score, meta, .. } => (score, meta.confidence),
            Answer::Noul { .. } => (usize::MAX, 0.0),
        }
    }

    /// `(correct, NLL of the true class)` for one class item.
    #[allow(clippy::too_many_arguments)]
    fn class_predict_nll(
        &self,
        cp: &ClassProtos,
        art: &Artifact,
        opts: &EngineOptions,
        emb: &[f32],
        model: &str,
        index: &BTreeMap<String, usize>,
        truth: usize,
    ) -> (bool, f32) {
        let (pred, _) = self.class_predict(cp, art, opts, emb, model, index);
        let p_true = match fit::class_answer(opts, cp, art, emb, model) {
            Answer::Choice { probabilities, .. } => cp
                .keys
                .get(truth)
                .and_then(|k| probabilities.get(k).copied())
                .unwrap_or(0.0),
            Answer::Score { probabilities, .. } => probabilities.get(truth).copied().unwrap_or(0.0),
            Answer::Noul { .. } => 0.0,
        };
        (pred == truth, nll(p_true))
    }

    /// Paired validation outcomes: `(accuracy pairs, NLL pairs)`. The NLL pairs
    /// (baseline, champion) drive the calibration criterion (ADR-004 gate 2b).
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn paired_validation(
        &self,
        compiled: &Compiled,
        index: &Option<BTreeMap<String, usize>>,
        resolved: &[Resolved],
        base_opts: &EngineOptions,
        base_art: &Artifact,
        cand_opts: &EngineOptions,
        cand_art: &Artifact,
        model: &str,
    ) -> (Vec<(bool, bool)>, Vec<(f32, f32)>) {
        let mut acc = Vec::new();
        let mut nlls = Vec::new();
        for r in resolved.iter().filter(|r| r.split == Split::Validation) {
            match (compiled, index) {
                (Compiled::Class(cp), Some(idx)) => {
                    let Some(truth) = r.class else { continue };
                    let (bc, bn) =
                        self.class_predict_nll(cp, base_art, base_opts, &r.emb, model, idx, truth);
                    let (cc, cn) =
                        self.class_predict_nll(cp, cand_art, cand_opts, &r.emb, model, idx, truth);
                    acc.push((bc, cc));
                    nlls.push((bn, cn));
                }
                (Compiled::Noul { predicate }, _) => {
                    let Some(y) = r.y else { continue };
                    let (bc, bn) = noul_correct_nll(base_art, predicate, &r.emb, model, y);
                    let (cc, cn) = noul_correct_nll(cand_art, predicate, &r.emb, model, y);
                    acc.push((bc, cc));
                    nlls.push((bn, cn));
                }
                _ => {}
            }
        }
        (acc, nlls)
    }

    /// `(accuracy, n)` on one split under `opts`.
    #[allow(clippy::too_many_arguments)]
    fn split_accuracy(
        &self,
        compiled: &Compiled,
        index: &Option<BTreeMap<String, usize>>,
        resolved: &[Resolved],
        opts: &EngineOptions,
        art: &Artifact,
        split: Split,
        model: &str,
    ) -> (f32, u32) {
        let (acc, n, _) = self.split_scored(compiled, index, resolved, opts, art, split, model);
        (acc, n)
    }

    /// Score the frozen test split — reachable ONLY with a [`TestToken`], which
    /// only a [`Campaign`] can mint (twice). Consumes the token by value, so the
    /// number of test scorings is bounded by the tokens the campaign hands out
    /// (ADR-004 gate 1: test scored only for baseline and final champion).
    #[allow(clippy::too_many_arguments)]
    fn score_test_split(
        &self,
        _token: TestToken,
        compiled: &Compiled,
        index: &Option<BTreeMap<String, usize>>,
        resolved: &[Resolved],
        opts: &EngineOptions,
        art: &Artifact,
        model: &str,
    ) -> (f32, u32, f32) {
        self.split_scored_any(compiled, index, resolved, opts, art, Split::Test, model)
    }

    /// `(accuracy, n, ece)` on a NON-test split under `opts`. Refuses
    /// `Split::Test` so the only path to test scoring is [`score_test_split`].
    #[allow(clippy::too_many_arguments)]
    fn split_scored(
        &self,
        compiled: &Compiled,
        index: &Option<BTreeMap<String, usize>>,
        resolved: &[Resolved],
        opts: &EngineOptions,
        art: &Artifact,
        split: Split,
        model: &str,
    ) -> (f32, u32, f32) {
        assert!(
            split != Split::Test,
            "the test split is scored only through a TestToken (score_test_split)",
        );
        self.split_scored_any(compiled, index, resolved, opts, art, split, model)
    }

    /// The shared scorer for any split (token-agnostic). Private, called only by
    /// the two guarded wrappers above.
    #[allow(clippy::too_many_arguments)]
    fn split_scored_any(
        &self,
        compiled: &Compiled,
        index: &Option<BTreeMap<String, usize>>,
        resolved: &[Resolved],
        opts: &EngineOptions,
        art: &Artifact,
        split: Split,
        model: &str,
    ) -> (f32, u32, f32) {
        let mut correct = Vec::new();
        let mut conf = Vec::new();
        for r in resolved.iter().filter(|r| r.split == split) {
            match (compiled, index) {
                (Compiled::Class(cp), Some(idx)) => {
                    let Some(truth) = r.class else { continue };
                    let (pred, c) = self.class_predict(cp, art, opts, &r.emb, model, idx);
                    correct.push(pred == truth);
                    conf.push(c);
                }
                (Compiled::Noul { predicate }, _) => {
                    let Some(y) = r.y else { continue };
                    let ok = noul_correct(art, predicate, &r.emb, model, y);
                    correct.push(ok);
                    // Confidence is max(p, 1-p) from the noul answer.
                    if let Answer::Noul { noul, .. } =
                        fit::noul_answer(art, predicate, &r.emb, model)
                    {
                        conf.push(noul.max(1.0 - noul));
                    }
                }
                _ => {}
            }
        }
        let n = correct.len() as u32;
        let hits = correct.iter().filter(|&&c| c).count() as u32;
        (Metrics::ratio(hits, n), n, ece(&conf, &correct))
    }

    /// Baseline and champion metrics on one split, packaged as receipt Metrics.
    #[allow(clippy::too_many_arguments)]
    fn pair_metrics(
        &self,
        compiled: &Compiled,
        index: &Option<BTreeMap<String, usize>>,
        resolved: &[Resolved],
        base_opts: &EngineOptions,
        base_art: &Artifact,
        champ_opts: &EngineOptions,
        champ_art: &Artifact,
        split: Split,
        model: &str,
    ) -> (Metrics, Metrics) {
        let (ba, bn, be) =
            self.split_scored(compiled, index, resolved, base_opts, base_art, split, model);
        let (ca, cn, ce) = self.split_scored(
            compiled, index, resolved, champ_opts, champ_art, split, model,
        );
        (
            Metrics {
                baseline_accuracy: ba,
                champion_accuracy: ba,
                n: bn,
                ece: Some(be),
                brier: None,
            },
            Metrics {
                baseline_accuracy: ba,
                champion_accuracy: ca,
                n: cn.max(bn),
                ece: Some(ce),
                brier: None,
            },
        )
    }
}

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
