//! Decision heads (ADR-003). This module owns the option/bucket geometry:
//! how a question's criteria are compiled into prototypes, how a `state`
//! scores against them, how the abstain mass is formed, and how a scored
//! question becomes a Jev-shaped [`Answer`]. The trainable heads (the linear
//! probe and the binary logistic head) live in the `probe` / `logistic`
//! submodules; the engine wires embeddings, training data and calibration in.
//!
//! `abstain` is always computed from the **prototype** geometry — the (K+1)
//! softmax over the option cosine scores plus the abstain logit — even when a
//! trained probe supplies the option shares. Probe logits are unbounded and
//! would otherwise drown a cosine-space abstain term; taking the mass from the
//! prototype layer keeps abstain meaningful under either head, and is
//! bit-identical to the ADR's formulation when the prototype head is active.

pub(crate) mod logistic;
pub(crate) mod probe;

pub(crate) use probe::softmax;

use crate::calibration::apply_temperature;
use crate::embedder::{dot, l2_normalize};
use crate::engine::AbstainMode;
use crate::{Answer, AnswerMeta, Head, Question};
use std::collections::BTreeMap;

// Penalty weight on the `not_for` hard-negative similarity, the abstain
// similarity threshold and its scale are now [`EngineOptions`] knobs
// (`not_for_lambda`, `abstain_tau`, `abstain_scale`); their defaults reproduce
// the original constants 0.5 / 0.35 / 0.5 (ADR-003).

/// Whether a compiled class question is reported as `choice` or `score`.
#[derive(Clone, Copy)]
pub(crate) enum ClassKind {
    Choice,
    Score,
}

/// Compiled prototypes for a `choice` or `score` question. `keys` are the
/// option keys (choice) or legend labels (score) in reporting order.
pub(crate) struct ClassProtos {
    pub keys: Vec<String>,
    pub protos: Vec<Vec<f32>>,
    pub not_for: Vec<Option<Vec<f32>>>,
    pub kind: ClassKind,
}

/// A compiled question: prototypes plus, for `noul`, the predicate embedding.
pub(crate) enum Compiled {
    Class(ClassProtos),
    Noul { predicate: Vec<f32> },
}

/// The texts a question needs embedded, in a fixed order that
/// [`build_compiled`] consumes identically. Keeping the two in lock-step lets
/// the engine batch every question's texts into one `embed` call.
///
/// `choice_instructions` (off by default, see
/// [`EngineOptions::choice_instructions`](crate::EngineOptions)) prefixes a
/// `choice` question's `instructions` to each option text, exactly as `score`
/// already does for its legend. Off, `choice` embeds only the criteria, so its
/// instructions do not change the answer.
pub(crate) fn question_texts(q: &Question, choice_instructions: bool) -> Vec<String> {
    match q {
        Question::Choice {
            instructions,
            criteria,
        } => {
            let prefix = choice_instructions && !instructions.trim().is_empty();
            let text = |t: &str| {
                if prefix {
                    bucket_text(instructions, t)
                } else {
                    t.to_string()
                }
            };
            let mut out = Vec::new();
            for c in criteria.values() {
                out.push(text(c.what()));
                out.extend(c.examples().iter().map(|e| text(e)));
                if let Some(nf) = c.not_for() {
                    out.push(text(nf));
                }
            }
            out
        }
        Question::Score {
            instructions,
            legend,
        } => legend
            .iter()
            .map(|l| bucket_text(instructions, l))
            .collect(),
        Question::Noul { instructions } => vec![instructions.clone()],
    }
}

fn bucket_text(instructions: &str, label: &str) -> String {
    if instructions.trim().is_empty() {
        label.to_string()
    } else {
        format!("{instructions}. {label}")
    }
}

/// Build the compiled prototypes from the embeddings of [`question_texts`]
/// (same order). `embs` are L2-normalised unit vectors from the embedder.
pub(crate) fn build_compiled(q: &Question, embs: &[Vec<f32>]) -> Compiled {
    match q {
        Question::Choice { criteria, .. } => {
            let mut keys = Vec::with_capacity(criteria.len());
            let mut protos = Vec::with_capacity(criteria.len());
            let mut not_for = Vec::with_capacity(criteria.len());
            let mut cursor = 0usize;
            for (key, c) in criteria {
                let n_pos = 1 + c.examples().len();
                let proto = mean_normalize(&embs[cursor..cursor + n_pos]);
                cursor += n_pos;
                let nf = if c.not_for().is_some() {
                    let v = embs[cursor].clone();
                    cursor += 1;
                    Some(v)
                } else {
                    None
                };
                keys.push(key.clone());
                protos.push(proto);
                not_for.push(nf);
            }
            Compiled::Class(ClassProtos {
                keys,
                protos,
                not_for,
                kind: ClassKind::Choice,
            })
        }
        Question::Score { legend, .. } => {
            let keys = legend.clone();
            let protos = embs.to_vec();
            let not_for = vec![None; legend.len()];
            Compiled::Class(ClassProtos {
                keys,
                protos,
                not_for,
                kind: ClassKind::Score,
            })
        }
        Question::Noul { .. } => Compiled::Noul {
            predicate: embs[0].clone(),
        },
    }
}

fn mean_normalize(vs: &[Vec<f32>]) -> Vec<f32> {
    let dims = vs.first().map(|v| v.len()).unwrap_or(0);
    let mut mean = vec![0.0f32; dims];
    for v in vs {
        for (m, x) in mean.iter_mut().zip(v) {
            *m += x;
        }
    }
    let n = vs.len().max(1) as f32;
    for m in &mut mean {
        *m /= n;
    }
    l2_normalize(&mut mean);
    mean
}

/// Cosine-space geometry of a `state` against a class question's prototypes.
pub(crate) struct Geometry {
    /// Per-option score `sim(state, proto) − λ·sim(state, not_for)`.
    pub proto_scores: Vec<f32>,
    max_sim: f32,
    best_not_for: Option<f32>,
}

pub(crate) fn geometry(state: &[f32], cp: &ClassProtos, not_for_lambda: f32) -> Geometry {
    let mut proto_scores = Vec::with_capacity(cp.protos.len());
    let mut max_sim = f32::NEG_INFINITY;
    let mut best_not_for: Option<f32> = None;
    for (proto, nf) in cp.protos.iter().zip(&cp.not_for) {
        let sim = dot(state, proto);
        max_sim = max_sim.max(sim);
        let penalty = match nf {
            Some(v) => {
                let nfs = dot(state, v);
                best_not_for = Some(best_not_for.map_or(nfs, |b: f32| b.max(nfs)));
                not_for_lambda * nfs
            }
            None => 0.0,
        };
        proto_scores.push(sim - penalty);
    }
    Geometry {
        proto_scores,
        max_sim,
        best_not_for,
    }
}

/// Out-of-scope logit over every option except `skip` (the catch-all option,
/// whose own text would otherwise be the nearest prototype for an off-topic
/// state): same formula as [`Geometry::abstain_logit`].
pub(crate) fn oos_logit_excluding(
    state: &[f32],
    cp: &ClassProtos,
    skip: usize,
    tau: f32,
    scale: f32,
) -> f32 {
    let scale = if scale.abs() < f32::EPSILON {
        0.5
    } else {
        scale
    };
    let mut max_sim = f32::NEG_INFINITY;
    let mut best_not_for: Option<f32> = None;
    for (i, (proto, nf)) in cp.protos.iter().zip(&cp.not_for).enumerate() {
        if i == skip {
            continue;
        }
        max_sim = max_sim.max(dot(state, proto));
        if let Some(v) = nf {
            let nfs = dot(state, v);
            best_not_for = Some(best_not_for.map_or(nfs, |b: f32| b.max(nfs)));
        }
    }
    let dist = (tau - max_sim) / scale;
    match best_not_for {
        Some(nf) => nf.max(dist),
        None => dist,
    }
}

/// Re-weight a `choice` answer around a catch-all option `k`: `k` gets the
/// out-of-scope probability `p_oos`, every other option `(1 − p_oos)` times
/// its share among the real options (the head never scores `k` itself), and
/// the choice is `k` when `p_oos ≥ threshold`, else the best real option.
pub(crate) fn apply_catch_all(
    answer: Answer,
    k: usize,
    keys: &[String],
    p_oos: f32,
    threshold: f32,
) -> Answer {
    let Answer::Choice {
        probabilities,
        mut meta,
        ..
    } = answer
    else {
        return answer;
    };
    let mut probs = BTreeMap::new();
    let mut best: Option<(&String, f32)> = None;
    for (i, key) in keys.iter().enumerate() {
        if i == k {
            continue;
        }
        let p = (1.0 - p_oos) * probabilities.get(key).copied().unwrap_or(0.0);
        let better = match best {
            None => true,
            Some((_, b)) => p > b,
        };
        if better {
            best = Some((key, p));
        }
        probs.insert(key.clone(), p);
    }
    probs.insert(keys[k].clone(), p_oos);
    let (choice, confidence) = match best {
        Some((key, p)) if p_oos < threshold => (key.clone(), p),
        _ => (keys[k].clone(), p_oos),
    };
    meta.confidence = confidence;
    Answer::Choice {
        choice,
        probabilities: probs,
        meta,
    }
}

impl Geometry {
    /// Abstain logit: the larger of the best `not_for` match and a
    /// distance-to-nearest-prototype term. When no option carries a `not_for`,
    /// only the distance term contributes (never floored at zero, so the
    /// in-scope/out-of-scope contrast survives — ADR-003).
    pub(crate) fn abstain_logit(&self, tau: f32, scale: f32) -> f32 {
        let scale = if scale.abs() < f32::EPSILON {
            0.5
        } else {
            scale
        };
        let dist = (tau - self.max_sim) / scale;
        match self.best_not_for {
            Some(nf) => nf.max(dist),
            None => dist,
        }
    }
}

/// A scored class question ready to be turned into an [`Answer`]. `head_logits`
/// come from the active head (prototype scores, or probe logits) in `keys`
/// order; `proto_scores` and `abstain_logit` come from the prototype geometry.
pub(crate) struct Classified<'a> {
    pub keys: &'a [String],
    pub kind: ClassKind,
    pub head_logits: Vec<f32>,
    pub proto_scores: &'a [f32],
    pub abstain_logit: f32,
    pub head: Head,
    pub temperature: f32,
    /// Inverse-temperature prior applied to every logit before temperature
    /// scaling and the abstain softmax (ADR-004 `logitScale`). `1.0` is a no-op.
    pub logit_scale: f32,
    pub calibrated: bool,
    pub model: &'a str,
    /// How the reported `abstain` is formed (see [`AbstainMode`]).
    pub abstain_mode: AbstainMode,
}

impl Classified<'_> {
    /// Returns `(shares, abstain_for_confidence, reported_abstain)`.
    /// `confidence` always uses the (K+1)-softmax mass, so it is identical in
    /// every mode; only the reported `abstain` changes under `Sigmoid`.
    fn shares_and_abstain(&self) -> (Vec<f32>, f32, f32) {
        let s = self.logit_scale;
        let mut proto_full: Vec<f32> = self.proto_scores.iter().map(|l| l * s).collect();
        proto_full.push(self.abstain_logit * s);
        let proto_masses = softmax(&apply_temperature(&proto_full, self.temperature));
        let abstain = *proto_masses.last().unwrap_or(&0.0);
        let scaled: Vec<f32> = self.head_logits.iter().map(|l| l * s).collect();
        let shares = softmax(&apply_temperature(&scaled, self.temperature));
        let reported = match self.abstain_mode {
            AbstainMode::Softmax => abstain,
            AbstainMode::Sigmoid => sigmoid(self.abstain_logit),
        };
        (shares, abstain, reported)
    }

    fn meta(&self, confidence: f32, abstain: f32) -> AnswerMeta {
        AnswerMeta {
            confidence,
            abstain,
            calibrated: self.calibrated,
            head: self.head,
            model: self.model.to_string(),
            temperature: self.temperature,
        }
    }

    pub(crate) fn into_answer(self) -> Answer {
        match self.kind {
            ClassKind::Choice => self.make_choice(),
            ClassKind::Score => self.make_score(),
        }
    }

    fn make_choice(&self) -> Answer {
        let (shares, abstain, reported) = self.shares_and_abstain();
        let best = argmax(&shares);
        let confidence = shares[best] * (1.0 - abstain);
        let probabilities: BTreeMap<String, f32> = self
            .keys
            .iter()
            .cloned()
            .zip(shares.iter().copied())
            .collect();
        Answer::Choice {
            choice: self.keys[best].clone(),
            probabilities,
            meta: self.meta(confidence, reported),
        }
    }

    fn make_score(&self) -> Answer {
        let (shares, abstain, reported) = self.shares_and_abstain();
        let expected: f32 = shares.iter().enumerate().map(|(i, p)| i as f32 * p).sum();
        let score = (expected.round() as usize).min(self.keys.len() - 1);
        let confidence = shares[score] * (1.0 - abstain);
        Answer::Score {
            score,
            legend: self.keys[score].clone(),
            probabilities: shares,
            meta: self.meta(confidence, reported),
        }
    }
}

/// Assemble a `noul` answer. `noul` is already a probability (a trained
/// logistic/Platt output, or the uncalibrated similarity fallback); there is
/// no abstain bucket for a binary predicate.
pub(crate) fn assemble_noul(noul: f32, head: Head, calibrated: bool, model: &str) -> Answer {
    let confidence = noul.max(1.0 - noul);
    Answer::Noul {
        noul,
        meta: AnswerMeta {
            confidence,
            abstain: 0.0,
            calibrated,
            head,
            model: model.to_string(),
            temperature: 1.0,
        },
    }
}

/// Map a raw cosine similarity in `[-1, 1]` to a `[0, 1]` predicate score.
pub(crate) fn similarity_to_unit(sim: f32) -> f32 {
    ((sim + 1.0) * 0.5).clamp(0.0, 1.0)
}

pub(crate) fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold(0, |best, (i, &x)| if x > v[best] { i } else { best })
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_confidence_describes_the_emitted_bucket() {
        let keys = vec!["low".into(), "medium".into(), "high".into()];
        let protos = vec![0.0; 3];
        let answer = Classified {
            keys: &keys,
            kind: ClassKind::Score,
            head_logits: vec![3.0, 0.0, 3.0],
            proto_scores: &protos,
            abstain_logit: 0.0,
            head: Head::NearestPrototype,
            temperature: 1.0,
            logit_scale: 1.0,
            calibrated: false,
            model: "test",
            abstain_mode: AbstainMode::Softmax,
        }
        .into_answer();
        let Answer::Score {
            score,
            probabilities,
            meta,
            ..
        } = answer
        else {
            panic!("expected score answer")
        };
        assert_eq!(score, 1, "rounded expected score differs from top class");
        assert!((meta.confidence - probabilities[score] * (1.0 - meta.abstain)).abs() < 1e-6);
    }

    fn classified_with<'a>(
        keys: &'a [String],
        protos: &'a [f32],
        abstain_logit: f32,
        temperature: f32,
        mode: AbstainMode,
    ) -> Answer {
        Classified {
            keys,
            kind: ClassKind::Choice,
            head_logits: protos.to_vec(),
            proto_scores: protos,
            abstain_logit,
            head: Head::LinearProbe,
            temperature,
            logit_scale: 1.0,
            calibrated: true,
            model: "test",
            abstain_mode: mode,
        }
        .into_answer()
    }

    fn meta_of(a: &Answer) -> &AnswerMeta {
        match a {
            Answer::Choice { meta, .. }
            | Answer::Score { meta, .. }
            | Answer::Noul { meta, .. } => meta,
        }
    }

    #[test]
    fn sigmoid_abstain_ignores_option_count_and_temperature() {
        // Same nearest-prototype distance, 8 vs 150 options, T = 1 vs a sharp
        // fitted T = 0.1: the sigmoid score is identical; the softmax mass is not.
        let k8: Vec<String> = (0..8).map(|i| format!("o{i}")).collect();
        let k150: Vec<String> = (0..150).map(|i| format!("o{i}")).collect();
        let p8 = vec![0.6_f32; 8];
        let p150 = vec![0.6_f32; 150];
        let logit = (0.35 - 0.6) / 0.5;
        let sig = |k: &[String], p: &[f32], t| {
            meta_of(&classified_with(k, p, logit, t, AbstainMode::Sigmoid)).abstain
        };
        let soft = |k: &[String], p: &[f32], t| {
            meta_of(&classified_with(k, p, logit, t, AbstainMode::Softmax)).abstain
        };
        let expected = 1.0 / (1.0 + (-logit).exp());
        for v in [
            sig(&k8, &p8, 1.0),
            sig(&k150, &p150, 1.0),
            sig(&k150, &p150, 0.1),
        ] {
            assert!((v - expected).abs() < 1e-6);
        }
        assert!(soft(&k150, &p150, 1.0) < soft(&k8, &p8, 1.0));
        assert!(soft(&k150, &p150, 0.1) < 1e-3);
    }

    #[test]
    fn sigmoid_mode_leaves_choice_and_confidence_unchanged() {
        let keys: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let protos = vec![0.7_f32, 0.4, 0.2];
        let a = classified_with(&keys, &protos, -0.7, 0.5, AbstainMode::Softmax);
        let b = classified_with(&keys, &protos, -0.7, 0.5, AbstainMode::Sigmoid);
        match (&a, &b) {
            (
                Answer::Choice {
                    choice: ca,
                    probabilities: pa,
                    meta: ma,
                },
                Answer::Choice {
                    choice: cb,
                    probabilities: pb,
                    meta: mb,
                },
            ) => {
                assert_eq!(ca, cb);
                assert_eq!(pa, pb);
                assert_eq!(ma.confidence, mb.confidence);
                assert_ne!(ma.abstain, mb.abstain);
            }
            _ => panic!("expected choice answers"),
        }
    }
}
