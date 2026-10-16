//! Validated data-only model artifacts and bounded exhaustive native ranking.
use crate::{content_hash, decode, native_error};
use ruvector_kge::{HolE, RotatE, Scorer, Tables};
use ruvector_language::{
    error::{Error, Result},
    types::{Prediction, PredictionBatch, PredictionRequest, Predictor},
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const MAX_WEIGHTS: usize = 1_000_000;
pub const MAX_ENTITIES: usize = 10_000;
pub const MAX_MODELS: usize = 16;
pub const MAX_PREDICTION_BYTES: usize = 16 * 1024 * 1024;

// Exact JSON string size for valid model identities, including surrounding quotes.
// Non-ASCII UTF8 bytes are emitted directly by serde_json. Control characters are
// disallowed by valid_name, but accounted for defensively here.
fn json_string_bytes(text: &str) -> usize {
    2 + text
        .bytes()
        .map(|b| match b {
            b'"' | b'\\' => 2,
            0..=31 => 6,
            _ => 1,
        })
        .sum::<usize>()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScorerKind {
    Hole,
    Rotate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Origin {
    Imported,
    Trained {
        request_hash: String,
        training_hash: String,
        evaluation_hash: String,
        epochs: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelArtifact {
    pub version: u32,
    pub name: String,
    pub revision: String,
    pub scorer: ScorerKind,
    pub dims: usize,
    /// Stable row order, not lexical order. Predictions use these exact identities.
    pub entities: Vec<String>,
    pub relations: Vec<String>,
    pub entity_weights: Vec<f32>,
    pub relation_weights: Vec<f32>,
    pub origin: Origin,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelBundle {
    pub version: u32,
    pub models: Vec<ModelArtifact>,
}

pub(crate) fn valid_name(s: &str) -> bool {
    !s.trim().is_empty() && s.len() <= 512 && !s.chars().any(char::is_control)
}

pub(crate) fn labels(values: &[String], maximum: usize) -> Result<()> {
    if values.is_empty()
        || values.len() > maximum
        || values.iter().any(|s| !valid_name(s))
        || values.iter().collect::<BTreeSet<_>>().len() != values.len()
    {
        return Err(Error::new(
            "MODEL_FORMAT",
            "invalid, duplicate or excessive vocabulary",
        ));
    }
    Ok(())
}

impl ModelArtifact {
    pub fn validate(&self) -> Result<()> {
        if self.version != 1
            || !valid_name(&self.name)
            || !valid_name(&self.revision)
            || !(2..=1024).contains(&self.dims)
            || !self.dims.is_power_of_two()
        {
            return Err(Error::new(
                "MODEL_FORMAT",
                "unsupported version, identity or dimensions",
            ));
        }
        labels(&self.entities, MAX_ENTITIES)?;
        labels(&self.relations, 256)?;
        if self.entity_weights.len() != self.entities.len() * self.dims
            || self.relation_weights.len() != self.relations.len() * self.dims
            || self.entity_weights.len() + self.relation_weights.len() > MAX_WEIGHTS
            || self
                .entity_weights
                .iter()
                .chain(&self.relation_weights)
                .any(|x| !x.is_finite() || x.abs() > 1_000_000.0)
        {
            return Err(Error::new(
                "MODEL_FORMAT",
                "invalid weight shape, magnitude or size",
            ));
        }
        if let Origin::Trained {
            request_hash,
            training_hash,
            evaluation_hash,
            epochs,
        } = &self.origin
        {
            if *epochs == 0
                || *epochs > 1000
                || [request_hash, training_hash, evaluation_hash]
                    .iter()
                    .any(|h| h.len() != 64 || !h.bytes().all(|b| b.is_ascii_hexdigit()))
            {
                return Err(Error::new("MODEL_FORMAT", "invalid training provenance"));
            }
        }
        Ok(())
    }

    pub fn tables(&self) -> Result<Tables> {
        self.validate()?;
        let mut tables = Tables::new(self.entities.len(), self.relations.len(), self.dims, 0);
        tables
            .entities_raw_mut()
            .copy_from_slice(&self.entity_weights);
        tables
            .relations_raw_mut()
            .copy_from_slice(&self.relation_weights);
        Ok(tables)
    }

    pub fn scorer(&self) -> Result<Box<dyn Scorer>> {
        self.validate()?;
        match self.scorer {
            ScorerKind::Hole => Ok(Box::new(HolE::new(self.dims).map_err(native_error)?)),
            ScorerKind::Rotate => Ok(Box::new(RotatE::new(self.dims).map_err(native_error)?)),
        }
    }
}

/// Conservative deterministic operation units, not instruction counts or time.
/// Restricting dimensions to powers of two keeps the FFT estimate bounded.
pub fn score_work(kind: ScorerKind, dims: usize) -> u64 {
    match kind {
        ScorerKind::Hole => 16 * dims as u64 * (dims.ilog2() as u64 + 1),
        ScorerKind::Rotate => 24 * dims as u64,
    }
}

struct Loaded {
    artifact: ModelArtifact,
    hash: String,
    tables: Tables,
    scorer: Box<dyn Scorer>,
    entities: BTreeMap<String, u32>,
    relations: BTreeMap<String, u32>,
    maximum_entity_bytes: usize,
}

pub struct NativePredictor {
    models: BTreeMap<String, Loaded>,
}

impl NativePredictor {
    pub fn new(bundle: ModelBundle) -> Result<Self> {
        if bundle.version != 1 || bundle.models.is_empty() || bundle.models.len() > MAX_MODELS {
            return Err(Error::new(
                "MODEL_FORMAT",
                "unsupported bundle version or model count",
            ));
        }
        let mut models = BTreeMap::new();
        let mut weights = 0usize;
        for artifact in bundle.models {
            artifact.validate()?;
            weights += artifact.entity_weights.len() + artifact.relation_weights.len();
            if weights > MAX_WEIGHTS || models.contains_key(&artifact.name) {
                return Err(Error::new(
                    "MODEL_FORMAT",
                    "excessive weights or duplicate model name",
                ));
            }
            let loaded = Loaded {
                hash: content_hash(&artifact)?,
                tables: artifact.tables()?,
                scorer: artifact.scorer()?,
                entities: artifact
                    .entities
                    .iter()
                    .enumerate()
                    .map(|(i, s)| (s.clone(), i as u32))
                    .collect(),
                relations: artifact
                    .relations
                    .iter()
                    .enumerate()
                    .map(|(i, s)| (s.clone(), i as u32))
                    .collect(),
                maximum_entity_bytes: artifact
                    .entities
                    .iter()
                    .map(|s| json_string_bytes(s))
                    .max()
                    .unwrap_or(0),
                artifact,
            };
            models.insert(loaded.artifact.name.clone(), loaded);
        }
        Ok(Self { models })
    }
}

pub fn load_models(json: &str) -> Result<Box<dyn Predictor>> {
    Ok(Box::new(NativePredictor::new(decode(json)?)?))
}

impl Predictor for NativePredictor {
    fn predict(&self, request: &PredictionRequest) -> Result<PredictionBatch> {
        let m = self
            .models
            .get(&request.model)
            .ok_or_else(|| Error::new("MODEL", "unknown model"))?;
        let relation = *m
            .relations
            .get(&request.relation)
            .ok_or_else(|| Error::new("MODEL", "unknown relation"))?;
        let n = m.artifact.entities.len();
        if request.k == 0
            || request.k > n
            || request.subjects.len() > 10_000
            || request.k.saturating_mul(request.subjects.len()) > 100_000
        {
            return Err(Error::new("LIMIT", "prediction result bound exceeded"));
        }
        let subjects: BTreeSet<&String> = request.subjects.iter().collect();
        if subjects.len() != request.subjects.len()
            || subjects.iter().any(|s| !m.entities.contains_key(*s))
        {
            return Err(Error::new("MODEL", "unknown or duplicate subject"));
        }
        // The estimate covers serialized bytes AND owned Prediction storage.
        // 128 bytes per row covers three String headers, f32, punctuation and
        // a conservative 32-character finite f32 representation. Using the
        // longest possible object rejects oversized results before scoring or
        // allocating their candidate array, even when top-k is not yet known.
        let relation_bytes = json_string_bytes(&request.relation);
        let mut projected_bytes = 256
            + json_string_bytes(&request.model)
            + json_string_bytes(&m.artifact.revision)
            + m.hash.len();
        for subject in &subjects {
            let row_bytes =
                128 + json_string_bytes(subject) + relation_bytes + m.maximum_entity_bytes;
            projected_bytes = projected_bytes.saturating_add(row_bytes.saturating_mul(request.k));
            if projected_bytes > MAX_PREDICTION_BYTES {
                return Err(Error::new(
                    "LIMIT",
                    "projected native prediction output exceeds 16 MiB",
                ));
            }
        }
        let comparison_units = (m.maximum_entity_bytes / 64 + 1) as u64;
        let units = score_work(m.artifact.scorer, m.artifact.dims)
            + 8 * (n.ilog2() as u64 + 1) * comparison_units;
        let work = (subjects.len() as u64)
            .checked_mul(n as u64)
            .and_then(|v| v.checked_mul(units))
            .and_then(|v| v.checked_add((projected_bytes / 64) as u64))
            .ok_or_else(|| Error::new("BUDGET", "prediction work overflow"))?;
        if work > request.max_work {
            return Err(Error::new("BUDGET", "native prediction budget exhausted"));
        }
        let r = m.tables.relation(relation).map_err(native_error)?;
        let mut candidates = Vec::with_capacity(subjects.len() * request.k);
        for subject in subjects {
            let s = m.tables.entity(m.entities[subject]).map_err(native_error)?;
            let mut ranked = Vec::with_capacity(n);
            for i in 0..n {
                let score = m
                    .scorer
                    .score(s, r, m.tables.entity(i as u32).map_err(native_error)?);
                if !score.is_finite() {
                    return Err(Error::new("MODEL", "nonfinite native score"));
                }
                ranked.push((score, i));
            }
            ranked.sort_unstable_by(|(a, ai), (b, bi)| {
                b.total_cmp(a)
                    .then(m.artifact.entities[*ai].cmp(&m.artifact.entities[*bi]))
            });
            for (score, index) in ranked.into_iter().take(request.k) {
                candidates.push(Prediction {
                    subject: subject.clone(),
                    relation: request.relation.clone(),
                    object: m.artifact.entities[index].clone(),
                    score,
                });
            }
        }
        Ok(PredictionBatch {
            model: request.model.clone(),
            revision: m.artifact.revision.clone(),
            model_hash: m.hash.clone(),
            exact: true,
            work,
            candidates,
        })
    }
}
