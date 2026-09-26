//! Frozen decision-head export for the ESP-IDF C runtime. Embedding/feature
//! extraction stays outside this snapshot and must match `model_id` exactly.
//! This is a new versioned format, not RVF or the training bank format.
use super::*;
use crate::heads::ClassKind;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EmbeddedSnapshot {
    pub version: u32,
    pub dims: usize,
    pub model_id: String,
    pub question_id: String,
    pub kind: String,
    pub head: Head,
    pub labels: Vec<String>,
    pub prototypes: Vec<Vec<f32>>,
    pub negatives: Vec<Option<Vec<f32>>>,
    pub weights: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
    pub temperature: f32,
    pub platt_a: f32,
    pub platt_b: f32,
    pub source_calibrated: bool,
    pub not_for_lambda: f32,
    pub abstain_tau: f32,
    pub abstain_scale: f32,
    pub logit_scale: f32,
}

impl<E: Embedder> Engine<E> {
    /// Export one fitted head with the same parameters used by `decide`.
    /// The MCU envelope is 1..=768 features and at most 16 class options.
    /// Quantize this snapshot with examples/esp32-decision/tools/quantize.py.
    pub fn export_embedded(&self, id: &str, question: &Question) -> Result<EmbeddedSnapshot> {
        crate::limits::validate(&DecisionRequest {
            state: "export".into(),
            questions: BTreeMap::from([(id.to_string(), question.clone())]),
        })?;
        let dims = self.embedder.dims();
        if !(1..=768).contains(&dims) {
            return Err(TypesafeError::Limit("embedded dimensions must be 1..=768"));
        }
        let texts = question_texts(question);
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let embs = self.embedder.embed(&refs)?;
        if embs.len() != texts.len()
            || embs.iter().any(|v| {
                v.len() != dims
                    || v.iter().any(|x| !x.is_finite())
                    || (v.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() > 0.01
            })
        {
            return Err(TypesafeError::Embedder(
                "embedded export requires finite unit vectors of declared width".into(),
            ));
        }
        let compiled = build_compiled(question, &embs);
        let art = self.artifact_for(id, stable_hash(id, question)?, &compiled);
        let mut snapshot = EmbeddedSnapshot {
            version: 1,
            dims,
            model_id: self.embedder.id().into(),
            question_id: id.into(),
            kind: String::new(),
            head: art.head(),
            labels: vec![],
            prototypes: vec![],
            negatives: vec![],
            weights: vec![],
            bias: vec![],
            temperature: art.temperature(),
            platt_a: 1.0,
            platt_b: 0.0,
            source_calibrated: false,
            not_for_lambda: self.options.not_for_lambda,
            abstain_tau: self.options.abstain_tau,
            abstain_scale: if self.options.abstain_scale.abs() < f32::EPSILON {
                0.5
            } else {
                self.options.abstain_scale
            },
            logit_scale: self.options.logit_scale,
        };
        match (&compiled, art.as_ref()) {
            (
                Compiled::Class(cp),
                Artifact::Class {
                    probe, calibrated, ..
                },
            ) => {
                if cp.keys.len() > 16 {
                    return Err(TypesafeError::Limit("embedded classes must be <=16"));
                }
                snapshot.kind = match cp.kind {
                    ClassKind::Choice => "choice",
                    ClassKind::Score => "score",
                }
                .into();
                snapshot.labels = cp.keys.clone();
                snapshot.prototypes = cp.protos.clone();
                snapshot.negatives = cp.not_for.clone();
                snapshot.source_calibrated = *calibrated;
                if let Some(p) = probe {
                    let (w, b) = p.parameters();
                    snapshot.weights = w.to_vec();
                    snapshot.bias = b.to_vec();
                }
            }
            (
                Compiled::Noul { predicate },
                Artifact::Noul {
                    model,
                    platt,
                    calibrated,
                    ..
                },
            ) => {
                snapshot.kind = "noul".into();
                snapshot.labels = vec!["noul".into()];
                snapshot.prototypes = vec![predicate.clone()];
                snapshot.negatives = vec![None];
                snapshot.source_calibrated = *calibrated;
                if let Some(m) = model {
                    let (w, b) = m.parameters();
                    snapshot.weights = vec![w.to_vec()];
                    snapshot.bias = vec![b];
                }
                if let Some(p) = platt {
                    (snapshot.platt_a, snapshot.platt_b) = p.parameters();
                }
            }
            _ => return Err(TypesafeError::Invalid("head export mismatch".into())),
        }
        Ok(snapshot)
    }
}
