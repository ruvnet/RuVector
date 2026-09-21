//! The engine: validates a request, embeds `state` and the question texts,
//! runs the per-question head (ADR-003), calibrates, and returns a
//! Jev-shaped response with the additive fields. The public signature below
//! is the contract the bindings compile against; two other agents compile
//! against it, so `new`, `embedder`, `decide`, `train`, `LabeledExample` and
//! `TrainReport` keep their exact shapes. Everything else is additive.
//!
//! Training now flows through the append-only [`Bank`](crate::bank) (ADR-004
//! loop 1): `train` admits examples into the bank's frozen Train/Calibration
//! splits, embeddings are cached by content id, and the head/calibration read
//! those splits. `export_bank`/`import_bank` persist it. `with_options` and
//! `optimize` add the campaign surface without changing the pinned methods.

use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use crate::bank::{Admission, Bank, Split, TrustTier};
use crate::heads::{build_compiled, question_texts, ClassProtos, Compiled};
use crate::{
    Answer, DecisionRequest, DecisionResponse, Embedder, Head, Question, Result, TypesafeError,
    Usage,
};

pub mod fit;
pub mod optimize;
pub mod options;

pub use options::{EngineOptions, HeadChoice};

use fit::{Artifact, MIN_EXAMPLES_PER_CLASS};

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

pub struct Engine<E: Embedder> {
    embedder: E,
    options: EngineOptions,
    compiled_cache: RwLock<BTreeMap<u64, Arc<Compiled>>>,
    artifact_cache: RwLock<BTreeMap<u64, Arc<Artifact>>>,
    /// Append-only example bank: raw text + frozen splits (ADR-004 loop 1).
    bank: RwLock<Bank>,
    /// Embedding cache keyed by `ExampleId.0`, so a bank example is embedded once.
    embeds: RwLock<BTreeMap<u64, Vec<f32>>>,
    train_gen: RwLock<BTreeMap<String, u64>>,
}

enum Slot {
    Cached(Arc<Compiled>),
    Pending { start: usize, len: usize },
}

impl<E: Embedder> Engine<E> {
    pub fn new(embedder: E) -> Self {
        Self::with_options(embedder, EngineOptions::default())
    }

    /// Construct with tunable [`EngineOptions`]. The bank's Train/Calibration
    /// ratio follows `options.calibration_fraction`; the plain `train` path
    /// carves only those two splits (validation/transfer/test come from a
    /// campaign's explicit rows, never from `train` input).
    pub fn with_options(embedder: E, options: EngineOptions) -> Self {
        let ratios = options.train_ratios();
        Self {
            embedder,
            options,
            compiled_cache: RwLock::new(BTreeMap::new()),
            artifact_cache: RwLock::new(BTreeMap::new()),
            bank: RwLock::new(Bank::new(ratios)),
            embeds: RwLock::new(BTreeMap::new()),
            train_gen: RwLock::new(BTreeMap::new()),
        }
    }

    pub fn embedder(&self) -> &E {
        &self.embedder
    }

    /// The engine's current tunable options.
    pub fn options(&self) -> &EngineOptions {
        &self.options
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

    /// Admit labeled examples for one question into the bank (ADR-004 loop 1).
    /// Examples are stored by content id in frozen Train/Calibration splits;
    /// duplicates are deduplicated (append-only), empty pairs are rejected, and
    /// their embeddings are cached so the head can be re-fit without re-embedding.
    /// Criteria arrive only at `decide` time, so a label that matches no
    /// criterion is simply ignored by the head then.
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

        let mut accepted = 0usize;
        {
            let mut bank = self.bank.write().unwrap();
            let mut embeds = self.embeds.write().unwrap();
            for (e, emb) in filtered.iter().zip(embs) {
                match bank.admit(question, &e.text, &e.label, TrustTier::A) {
                    Admission::Accepted(id) => {
                        embeds.insert(id.0, emb);
                        accepted += 1;
                    }
                    // A duplicate is already stored (and embedded); a quarantine
                    // is held out. Neither is a fresh accept.
                    Admission::Duplicate(_) | Admission::Quarantined(_) => {}
                }
            }
        }
        {
            let mut gens = self.train_gen.write().unwrap();
            *gens.entry(question.to_string()).or_insert(0) += 1;
        }

        Ok(TrainReport {
            question: question.to_string(),
            accepted,
            rejected,
            head: self.provisional_head(question),
            calibrated: self.provisional_calibrated(question),
        })
    }

    /// Full-fidelity bank JSON (the user's own examples) for `typesafe train
    /// --bank bank.json`. Text included — never a receipt.
    pub fn export_bank(&self) -> Result<String> {
        self.bank.read().unwrap().to_json()
    }

    /// Replace the bank from JSON, re-embedding every stored text so the head
    /// can be re-fit, and invalidating the artifact cache. Split assignments in
    /// the JSON are authoritative (frozen).
    pub fn import_bank(&mut self, json: &str) -> Result<()> {
        let bank = Bank::from_json(json)?;
        let items: Vec<(u64, String)> = bank
            .iter()
            .filter_map(|e| bank.text_of(e.id).map(|t| (e.id.0, t.to_string())))
            .collect();
        let texts: Vec<&str> = items.iter().map(|(_, t)| t.as_str()).collect();
        let embs = if texts.is_empty() {
            Vec::new()
        } else {
            self.embedder.embed(&texts)?
        };
        if embs.len() != items.len() {
            return Err(TypesafeError::Embedder("embedding count mismatch".into()));
        }
        let mut embeds = BTreeMap::new();
        for ((id, _), emb) in items.iter().zip(embs) {
            embeds.insert(*id, emb);
        }
        *self.embeds.write().unwrap() = embeds;
        *self.bank.write().unwrap() = bank;
        self.artifact_cache.write().unwrap().clear();
        Ok(())
    }

    /// Receipt-safe rollup of the bank (counts and hashes, never text).
    pub fn bank_summary(&self) -> crate::bank::RedactedSummary {
        self.bank.read().unwrap().redacted_summary()
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
        let allow_calibration = !is_test_double(self.embedder.id());
        let dims = self.embedder.dims();
        match compiled {
            Compiled::Class(cp) => {
                let (train_ex, calib) = self.class_split_for(id, cp);
                fit::fit_class_artifact(
                    &self.options,
                    cp,
                    &train_ex,
                    &calib,
                    dims,
                    allow_calibration,
                )
            }
            Compiled::Noul { .. } => {
                let (train_ex, calib) = self.noul_split_for(id);
                fit::fit_noul_artifact(&self.options, &train_ex, &calib, dims, allow_calibration)
            }
        }
    }

    /// The (train, calibration) class-example split for `question`: the bank's
    /// `Train` examples in insertion order, class-mapped, with every
    /// `calib_stride`-th one carved out for calibration (labels that match no
    /// criterion are dropped, as before). The positional carve keeps the slice
    /// class-stratified for small few-shot samples (bit-identical to the
    /// original every-`N`th rule).
    #[allow(clippy::type_complexity)]
    fn class_split_for(
        &self,
        question: &str,
        cp: &ClassProtos,
    ) -> (Vec<(Vec<f32>, usize)>, Vec<(Vec<f32>, usize)>) {
        let index: BTreeMap<&str, usize> = cp
            .keys
            .iter()
            .enumerate()
            .map(|(i, k)| (k.as_str(), i))
            .collect();
        let bank = self.bank.read().unwrap();
        let embeds = self.embeds.read().unwrap();
        let relevant: Vec<(Vec<f32>, usize)> = bank
            .iter_split(question, Split::Train)
            .filter_map(|e| {
                let ci = *index.get(e.label.as_str())?;
                let emb = embeds.get(&e.id.0)?.clone();
                Some((emb, ci))
            })
            .collect();
        carve_calibration(relevant, self.options.calib_stride())
    }

    #[allow(clippy::type_complexity)]
    fn noul_split_for(&self, question: &str) -> (Vec<(Vec<f32>, f32)>, Vec<(Vec<f32>, f32)>) {
        let bank = self.bank.read().unwrap();
        let embeds = self.embeds.read().unwrap();
        let relevant: Vec<(Vec<f32>, f32)> = bank
            .iter_split(question, Split::Train)
            .filter_map(|e| {
                let y = parse_noul_label(&e.label)?;
                let emb = embeds.get(&e.id.0)?.clone();
                Some((emb, y))
            })
            .collect();
        carve_calibration(relevant, self.options.calib_stride())
    }

    /// The head a `TrainReport` announces, from the bank's Train split for this
    /// question with the calibration positions excluded (matching the head
    /// `decide` will pick under the current options).
    fn provisional_head(&self, question: &str) -> Head {
        let bank = self.bank.read().unwrap();
        let stride = self.options.calib_stride();
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for (i, e) in bank.iter_split(question, Split::Train).enumerate() {
            if is_calib_pos(i, stride) {
                continue;
            }
            *counts.entry(e.label.clone()).or_insert(0) += 1;
        }
        if !counts.is_empty() && counts.keys().all(|k| parse_noul_label(k).is_some()) {
            return Head::Logistic;
        }
        let counts_vec: Vec<usize> = counts.values().copied().collect();
        let probe = match self.options.head {
            HeadChoice::Prototype => false,
            HeadChoice::Probe => counts.len() >= 2 && counts_vec.iter().all(|&c| c >= 1),
            HeadChoice::Auto => {
                counts.len() >= 2 && counts_vec.iter().all(|&c| c >= MIN_EXAMPLES_PER_CLASS)
            }
        };
        if probe {
            Head::LinearProbe
        } else {
            Head::NearestPrototype
        }
    }

    fn provisional_calibrated(&self, question: &str) -> bool {
        if is_test_double(self.embedder.id()) {
            return false;
        }
        let bank = self.bank.read().unwrap();
        let stride = self.options.calib_stride();
        let calib = (0..bank.iter_split(question, Split::Train).count())
            .filter(|&i| is_calib_pos(i, stride))
            .count();
        calib >= self.options.min_calibration
    }

    fn assemble(
        &self,
        compiled: &Compiled,
        art: &Artifact,
        state_emb: &[f32],
        model: &str,
    ) -> Answer {
        match compiled {
            Compiled::Class(cp) => fit::class_answer(&self.options, cp, art, state_emb, model),
            Compiled::Noul { predicate } => fit::noul_answer(art, predicate, state_emb, model),
        }
    }
}

mod support;
use support::{
    carve_calibration, is_calib_pos, is_test_double, mix, parse_noul_label, stable_hash,
};

#[cfg(all(test, feature = "hash-embedder"))]
mod tests;
