//! The engine: validates a request, embeds `state` and the question texts,
//! runs the per-question head (ADR-003), calibrates, and returns a
//! Jev-shaped response with the additive fields. The public signature below
//! is the contract the bindings compile against; two other agents compile
//! against it, so `new`, `embedder`, `decide`, `train`, `LabeledExample` and
//! `TrainReport` keep their exact shapes. Everything else is additive.

use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use crate::calibration::{fit_temperature, Platt};
use crate::heads::logistic::{BinaryLogistic, LogisticConfig};
use crate::heads::probe::{MultiProbe, ProbeConfig};
use crate::heads::{
    assemble_noul, build_compiled, geometry, question_texts, similarity_to_unit, ClassProtos,
    Classified, Compiled,
};
use crate::{
    Answer, DecisionRequest, DecisionResponse, Embedder, Head, Question, Result, TypesafeError,
    Usage,
};

/// Fewest examples of a class (in the training slice) before the linear probe
/// takes over from the nearest-prototype head (ADR-003).
const MIN_EXAMPLES_PER_CLASS: usize = 4;
/// Calibration slice size below which `confidence` stays uncalibrated (ADR-003;
/// ADR-006 sets the floor). Temperature/Platt need a held-out slice this large.
const MIN_CALIBRATION: usize = 20;
/// Every Nth admitted example is reserved for the calibration slice, disjoint
/// from the head's training examples.
const CALIB_EVERY: usize = 5;

/// Labeled example used by `train` (text, option key / legend bucket / "yes"|"no").
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LabeledExample {
    pub text: String,
    pub label: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TrainReport {
    pub question: String,
    pub accepted: usize,
    pub rejected: usize,
    pub head: crate::Head,
    pub calibrated: bool,
}

/// Admitted examples for one question id, in insertion order (deterministic).
#[derive(Default)]
struct QuestionTraining {
    examples: Vec<(Vec<f32>, String)>,
}

/// A per-question trained artifact, cached by `(question content hash, train
/// generation)` so a repeated request with unchanged training reuses it.
enum Artifact {
    Class {
        probe: Option<MultiProbe>,
        temperature: f32,
        calibrated: bool,
        head: Head,
    },
    Noul {
        model: Option<BinaryLogistic>,
        platt: Option<Platt>,
        calibrated: bool,
        head: Head,
    },
}

pub struct Engine<E: Embedder> {
    embedder: E,
    compiled_cache: RwLock<BTreeMap<u64, Arc<Compiled>>>,
    artifact_cache: RwLock<BTreeMap<u64, Arc<Artifact>>>,
    training: RwLock<BTreeMap<String, QuestionTraining>>,
    train_gen: RwLock<BTreeMap<String, u64>>,
}

enum Slot {
    Cached(Arc<Compiled>),
    Pending { start: usize, len: usize },
}

impl<E: Embedder> Engine<E> {
    pub fn new(embedder: E) -> Self {
        Self {
            embedder,
            compiled_cache: RwLock::new(BTreeMap::new()),
            artifact_cache: RwLock::new(BTreeMap::new()),
            training: RwLock::new(BTreeMap::new()),
            train_gen: RwLock::new(BTreeMap::new()),
        }
    }

    pub fn embedder(&self) -> &E {
        &self.embedder
    }

    /// Answer every question in `req` against `req.state`. `state` is embedded
    /// once, and every criterion text that is not already compiled is embedded
    /// in the *same* batched `embed` call (ADR-003 §4).
    pub fn decide(&self, req: &DecisionRequest) -> Result<DecisionResponse> {
        crate::limits::validate(req)?;
        let model = self.embedder.id();

        // Pass A: content-hash each question, reuse compiled prototypes where
        // possible, and gather the texts that still need embedding.
        let mut texts: Vec<String> = vec![req.state.clone()];
        let mut ids: Vec<&str> = Vec::new();
        let mut qs: Vec<&Question> = Vec::new();
        let mut hashes: Vec<u64> = Vec::new();
        let mut slots: Vec<Slot> = Vec::new();
        {
            let cache = self.compiled_cache.read().unwrap();
            for (id, q) in &req.questions {
                let hash = stable_hash(id, q)?;
                let slot = if let Some(c) = cache.get(&hash) {
                    Slot::Cached(c.clone())
                } else {
                    let start = texts.len();
                    let t = question_texts(q);
                    let len = t.len();
                    texts.extend(t);
                    Slot::Pending { start, len }
                };
                ids.push(id);
                qs.push(q);
                hashes.push(hash);
                slots.push(slot);
            }
        }

        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let embs = self.embedder.embed(&refs)?;
        if embs.len() != texts.len() {
            return Err(TypesafeError::Embedder("embedding count mismatch".into()));
        }
        let state_emb = &embs[0];

        // Pass B: build and cache any freshly compiled questions.
        {
            let mut cache = self.compiled_cache.write().unwrap();
            for i in 0..slots.len() {
                let pend = match &slots[i] {
                    Slot::Pending { start, len } => Some((*start, *len)),
                    Slot::Cached(_) => None,
                };
                if let Some((start, len)) = pend {
                    let compiled = build_compiled(qs[i], &embs[start..start + len]);
                    let arc = Arc::new(compiled);
                    cache.insert(hashes[i], arc.clone());
                    slots[i] = Slot::Cached(arc);
                }
            }
        }

        // Assemble answers.
        let mut answers = BTreeMap::new();
        for i in 0..ids.len() {
            let compiled = match &slots[i] {
                Slot::Cached(a) => a.clone(),
                Slot::Pending { .. } => unreachable!("all slots resolved in pass B"),
            };
            let art = self.artifact_for(ids[i], hashes[i], &compiled);
            let answer = self.assemble(&compiled, &art, state_emb, model);
            answers.insert(ids[i].to_string(), answer);
        }

        Ok(DecisionResponse {
            answers,
            usage: Usage {
                embed_calls: 1,
                texts_embedded: texts.len() as u32,
                state_bytes: req.state.len() as u32,
            },
        })
    }

    /// Admit labeled examples for one question (ADR-004 loop 1). Examples are
    /// stored keyed by question id; criteria arrive only at `decide` time, so a
    /// label that does not match any criterion is validated *lazily* there (it
    /// is simply ignored by the head). Empty text/label pairs are rejected.
    pub fn train(&mut self, question: &str, examples: &[LabeledExample]) -> Result<TrainReport> {
        let valid = |e: &&LabeledExample| !e.text.trim().is_empty() && !e.label.trim().is_empty();
        let filtered: Vec<&LabeledExample> = examples.iter().filter(valid).collect();
        let rejected = examples.len() - filtered.len();

        let refs: Vec<&str> = filtered.iter().map(|e| e.text.as_str()).collect();
        let embs = if refs.is_empty() {
            Vec::new()
        } else {
            self.embedder.embed(&refs)?
        };
        if embs.len() != filtered.len() {
            return Err(TypesafeError::Embedder("embedding count mismatch".into()));
        }

        let accepted = filtered.len();
        {
            let mut training = self.training.write().unwrap();
            let entry = training.entry(question.to_string()).or_default();
            for (e, emb) in filtered.iter().zip(embs) {
                entry.examples.push((emb, e.label.clone()));
            }
        }
        {
            let mut gens = self.train_gen.write().unwrap();
            *gens.entry(question.to_string()).or_insert(0) += 1;
        }

        let training = self.training.read().unwrap();
        let stored = &training.get(question).unwrap().examples;
        Ok(TrainReport {
            question: question.to_string(),
            accepted,
            rejected,
            head: provisional_head(stored),
            calibrated: provisional_calibrated(stored, self.embedder.id()),
        })
    }

    fn artifact_for(&self, id: &str, hash: u64, compiled: &Compiled) -> Arc<Artifact> {
        let gen = *self.train_gen.read().unwrap().get(id).unwrap_or(&0);
        let akey = mix(hash, gen);
        if let Some(a) = self.artifact_cache.read().unwrap().get(&akey) {
            return a.clone();
        }
        let art = Arc::new(self.build_artifact(id, compiled));
        self.artifact_cache
            .write()
            .unwrap()
            .insert(akey, art.clone());
        art
    }

    fn build_artifact(&self, id: &str, compiled: &Compiled) -> Artifact {
        let training = self.training.read().unwrap();
        let stored = training
            .get(id)
            .map(|t| t.examples.as_slice())
            .unwrap_or(&[]);
        let test_double = is_test_double(self.embedder.id());
        match compiled {
            Compiled::Class(cp) => self.build_class_artifact(cp, stored, test_double),
            Compiled::Noul { .. } => self.build_noul_artifact(stored, test_double),
        }
    }

    fn build_class_artifact(
        &self,
        cp: &ClassProtos,
        stored: &[(Vec<f32>, String)],
        test_double: bool,
    ) -> Artifact {
        let index: BTreeMap<&str, usize> = cp
            .keys
            .iter()
            .enumerate()
            .map(|(i, k)| (k.as_str(), i))
            .collect();
        let relevant: Vec<(&Vec<f32>, usize)> = stored
            .iter()
            .filter_map(|(emb, lab)| index.get(lab.as_str()).map(|&i| (emb, i)))
            .collect();
        let (train_ex, calib) = split_class(&relevant);

        let mut counts = vec![0usize; cp.keys.len()];
        for (_, ci) in &train_ex {
            counts[*ci] += 1;
        }
        let use_probe = cp.keys.len() >= 2 && counts.iter().all(|&c| c >= MIN_EXAMPLES_PER_CLASS);

        let probe = if use_probe {
            Some(MultiProbe::train(
                &train_ex,
                cp.keys.len(),
                self.embedder.dims(),
                &ProbeConfig::default(),
            ))
        } else {
            None
        };
        let head = if probe.is_some() {
            Head::LinearProbe
        } else {
            Head::NearestPrototype
        };

        let (temperature, calibrated) = if calib.len() >= MIN_CALIBRATION && !test_double {
            let mut logits = Vec::with_capacity(calib.len());
            let mut labels = Vec::with_capacity(calib.len());
            for (emb, ci) in &calib {
                let row = match &probe {
                    Some(p) => p.logits(emb),
                    None => geometry(emb, cp).proto_scores,
                };
                logits.push(row);
                labels.push(*ci);
            }
            (fit_temperature(&logits, &labels), true)
        } else {
            (1.0, false)
        };

        Artifact::Class {
            probe,
            temperature,
            calibrated,
            head,
        }
    }

    fn build_noul_artifact(&self, stored: &[(Vec<f32>, String)], test_double: bool) -> Artifact {
        let relevant: Vec<(Vec<f32>, f32)> = stored
            .iter()
            .filter_map(|(emb, lab)| parse_noul_label(lab).map(|y| (emb.clone(), y)))
            .collect();
        let (train_ex, calib) = split_noul(&relevant);
        let has_pos = train_ex.iter().any(|(_, y)| *y >= 0.5);
        let has_neg = train_ex.iter().any(|(_, y)| *y < 0.5);

        if train_ex.is_empty() || !has_pos || !has_neg {
            return Artifact::Noul {
                model: None,
                platt: None,
                calibrated: false,
                head: Head::SimilarityUncalibrated,
            };
        }

        let model =
            BinaryLogistic::train(&train_ex, self.embedder.dims(), &LogisticConfig::default());
        let (platt, calibrated) = if calib.len() >= MIN_CALIBRATION && !test_double {
            let scores: Vec<f32> = calib.iter().map(|(x, _)| model.raw(x)).collect();
            let labels: Vec<f32> = calib.iter().map(|(_, y)| *y).collect();
            (Some(Platt::fit(&scores, &labels)), true)
        } else {
            (None, false)
        };

        Artifact::Noul {
            model: Some(model),
            platt,
            calibrated,
            head: Head::Logistic,
        }
    }

    fn assemble(
        &self,
        compiled: &Compiled,
        art: &Artifact,
        state_emb: &[f32],
        model: &str,
    ) -> Answer {
        match (compiled, art) {
            (
                Compiled::Class(cp),
                Artifact::Class {
                    probe,
                    temperature,
                    calibrated,
                    head,
                },
            ) => {
                let g = geometry(state_emb, cp);
                let head_logits = match probe {
                    Some(p) => p.logits(state_emb),
                    None => g.proto_scores.clone(),
                };
                Classified {
                    keys: &cp.keys,
                    kind: cp.kind,
                    head_logits,
                    proto_scores: &g.proto_scores,
                    abstain_logit: g.abstain_logit(),
                    head: *head,
                    temperature: *temperature,
                    calibrated: *calibrated,
                    model,
                }
                .into_answer()
            }
            (
                Compiled::Noul { predicate },
                Artifact::Noul {
                    model: probe,
                    platt,
                    calibrated,
                    head,
                },
            ) => {
                let noul = match probe {
                    Some(m) => match platt {
                        Some(p) => p.apply(m.raw(state_emb)),
                        None => m.prob(state_emb),
                    },
                    None => similarity_to_unit(crate::embedder::dot(state_emb, predicate)),
                };
                assemble_noul(noul, *head, *calibrated, model)
            }
            // Compiled/Artifact kinds are built together, so a mismatch is a bug.
            _ => unreachable!("compiled and artifact kinds always agree"),
        }
    }
}

mod support;
use support::{
    is_test_double, mix, parse_noul_label, provisional_calibrated, provisional_head, split_class,
    split_noul, stable_hash,
};

#[cfg(all(test, feature = "hash-embedder"))]
mod tests;
