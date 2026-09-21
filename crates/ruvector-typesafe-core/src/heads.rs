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
use crate::{Answer, AnswerMeta, Head, Question};
use std::collections::BTreeMap;

/// Penalty weight on the `not_for` hard-negative similarity (ADR-003).
const LAMBDA_NOT_FOR: f32 = 0.5;
/// Similarity below which a `state` starts to look out-of-scope. Heuristic; it
/// sets the absolute level of the abstain logit, not the in/out-of-scope
/// ordering (which holds for any positive `tau`/`scale`).
const ABSTAIN_TAU: f32 = 0.35;
const ABSTAIN_SCALE: f32 = 0.5;

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
pub(crate) fn question_texts(q: &Question) -> Vec<String> {
    match q {
        Question::Choice { criteria, .. } => {
            let mut out = Vec::new();
            for c in criteria.values() {
                out.push(c.what().to_string());
                out.extend(c.examples().iter().cloned());
                if let Some(nf) = c.not_for() {
                    out.push(nf.to_string());
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

pub(crate) fn geometry(state: &[f32], cp: &ClassProtos) -> Geometry {
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
                LAMBDA_NOT_FOR * nfs
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

impl Geometry {
    /// Abstain logit: the larger of the best `not_for` match and a
    /// distance-to-nearest-prototype term. When no option carries a `not_for`,
    /// only the distance term contributes (never floored at zero, so the
    /// in-scope/out-of-scope contrast survives — ADR-003).
    pub(crate) fn abstain_logit(&self) -> f32 {
        let dist = (ABSTAIN_TAU - self.max_sim) / ABSTAIN_SCALE;
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
    pub calibrated: bool,
    pub model: &'a str,
}

impl Classified<'_> {
    fn shares_and_abstain(&self) -> (Vec<f32>, f32) {
        let mut proto_full = self.proto_scores.to_vec();
        proto_full.push(self.abstain_logit);
        let proto_masses = softmax(&apply_temperature(&proto_full, self.temperature));
        let abstain = *proto_masses.last().unwrap_or(&0.0);
        let shares = softmax(&apply_temperature(&self.head_logits, self.temperature));
        (shares, abstain)
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
        let (shares, abstain) = self.shares_and_abstain();
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
            meta: self.meta(confidence, abstain),
        }
    }

    fn make_score(&self) -> Answer {
        let (shares, abstain) = self.shares_and_abstain();
        let best = argmax(&shares);
        let confidence = shares[best] * (1.0 - abstain);
        let expected: f32 = shares.iter().enumerate().map(|(i, p)| i as f32 * p).sum();
        let score = (expected.round() as usize).min(self.keys.len() - 1);
        Answer::Score {
            score,
            legend: self.keys[score].clone(),
            probabilities: shares,
            meta: self.meta(confidence, abstain),
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
