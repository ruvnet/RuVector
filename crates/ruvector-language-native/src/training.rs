//! Offline native training and held out evaluation. Promotion is explicit.
mod promotion;
use crate::{
    artifact::{score_work, valid_name, ModelArtifact, Origin, ScorerKind},
    content_hash, native_error,
};
pub use promotion::promote;
use ruvector_kge::{
    evaluate, Differentiable, EvalConfig, EvalReport, HolE, LossKind, OptimKind, RotatE, Tables,
    TrainConfig, Trainer, Triple, TripleStore,
};
use ruvector_language::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fact {
    pub subject: String,
    pub relation: String,
    pub object: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingData {
    pub train: Vec<Fact>,
    pub evaluation: Vec<Fact>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingRequest {
    pub version: u32,
    pub name: String,
    pub revision: String,
    pub scorer: ScorerKind,
    pub dims: usize,
    pub epochs: usize,
    pub seed: u64,
    pub learning_rate: f32,
    pub max_work: u64,
    pub data: TrainingData,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metrics {
    pub count: usize,
    pub mr: f32,
    pub mrr: f32,
    pub hits1: f32,
    pub hits3: f32,
    pub hits10: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricsReport {
    pub head: Metrics,
    pub tail: Metrics,
    pub combined: Metrics,
}

impl From<EvalReport> for MetricsReport {
    fn from(value: EvalReport) -> Self {
        let convert = |m: ruvector_kge::MetricSet| Metrics {
            count: m.count,
            mr: m.mr,
            mrr: m.mrr,
            hits1: m.hits1,
            hits3: m.hits3,
            hits10: m.hits10,
        };
        Self {
            head: convert(value.head),
            tail: convert(value.tail),
            combined: convert(value.combined),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingReport {
    pub version: u32,
    pub request_hash: String,
    pub model_hash: String,
    pub training_hash: String,
    pub evaluation_hash: String,
    pub train_count: usize,
    pub evaluation_count: usize,
    pub initial: MetricsReport,
    pub trained: MetricsReport,
    pub losses: Vec<f32>,
    pub estimated_work: u64,
    pub seed: u64,
    pub epochs: usize,
    pub learning_rate: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub version: u32,
    pub model: ModelArtifact,
    pub report: TrainingReport,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationRequest {
    pub version: u32,
    pub data: TrainingData,
    pub seed: u64,
    pub max_work: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationReport {
    pub version: u32,
    pub model_hash: String,
    pub data_hash: String,
    pub evaluation_count: usize,
    pub metrics: MetricsReport,
    pub estimated_work: u64,
}

struct Encoded {
    entities: Vec<String>,
    relations: Vec<String>,
    train: Vec<Triple>,
    eval: Vec<Triple>,
}

fn encode(data: &TrainingData) -> Result<Encoded> {
    if data.train.is_empty()
        || data.evaluation.is_empty()
        || data.train.len() > 10_000
        || data.evaluation.len() > 2000
    {
        return Err(Error::new(
            "TRAINING",
            "nonempty bounded training and evaluation sets required",
        ));
    }
    let train: BTreeSet<_> = data.train.iter().collect();
    let eval: BTreeSet<_> = data.evaluation.iter().collect();
    if train.len() != data.train.len()
        || eval.len() != data.evaluation.len()
        || !train.is_disjoint(&eval)
    {
        return Err(Error::new(
            "TRAINING",
            "duplicate triples or train/evaluation leakage",
        ));
    }
    if data
        .train
        .iter()
        .chain(&data.evaluation)
        .any(|f| !valid_name(&f.subject) || !valid_name(&f.relation) || !valid_name(&f.object))
    {
        return Err(Error::new("TRAINING", "invalid fact identity"));
    }
    let entities: Vec<_> = data
        .train
        .iter()
        .flat_map(|f| [&f.subject, &f.object])
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let relations: Vec<_> = data
        .train
        .iter()
        .map(|f| f.relation.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if entities.len() > 2000 || relations.len() > 128 {
        return Err(Error::new("TRAINING", "training vocabulary exceeds limits"));
    }
    let ei: BTreeMap<_, _> = entities
        .iter()
        .enumerate()
        .map(|(i, s)| (s, i as u32))
        .collect();
    let ri: BTreeMap<_, _> = relations
        .iter()
        .enumerate()
        .map(|(i, s)| (s, i as u32))
        .collect();
    let convert = |set: &Vec<Fact>| -> Result<Vec<Triple>> {
        set.iter()
            .map(|f| {
                Ok(Triple::new(
                    *ei.get(&f.subject).ok_or_else(|| {
                        Error::new(
                            "TRAINING",
                            "evaluation subject absent from training vocabulary",
                        )
                    })?,
                    *ri.get(&f.relation).ok_or_else(|| {
                        Error::new(
                            "TRAINING",
                            "evaluation relation absent from training vocabulary",
                        )
                    })?,
                    *ei.get(&f.object).ok_or_else(|| {
                        Error::new(
                            "TRAINING",
                            "evaluation object absent from training vocabulary",
                        )
                    })?,
                ))
            })
            .collect()
    };
    let train = convert(&data.train)?;
    let eval = convert(&data.evaluation)?;
    Ok(Encoded {
        entities,
        relations,
        train,
        eval,
    })
}

fn budget(work: u64, limit: u64) -> Result<()> {
    if limit == 0 || limit > 1_000_000_000 || work > limit {
        Err(Error::new(
            "BUDGET",
            "offline model work budget exceeded (maximum 1 billion units)",
        ))
    } else {
        Ok(())
    }
}

fn store(encoded: &Encoded, all: bool) -> Result<TripleStore> {
    let mut triples = encoded.train.clone();
    if all {
        triples.extend(&encoded.eval);
    }
    TripleStore::with_counts(
        triples,
        Some(encoded.entities.len()),
        Some(encoded.relations.len()),
    )
    .map_err(native_error)
}

/// Each explicit evaluation triple is absent from training. Evaluation uses native
/// filtered ranking on BOTH head and tail with deterministic RANDOM tie breaking.
/// These are transductive results: unseen entity/relation generalization is not tested.
pub fn train_candidate(request: &TrainingRequest) -> Result<Candidate> {
    if request.version != 1
        || !valid_name(&request.name)
        || !valid_name(&request.revision)
        || !(2..=128).contains(&request.dims)
        || !request.dims.is_power_of_two()
        || request.epochs == 0
        || request.epochs > 1000
        || !request.learning_rate.is_finite()
        || request.learning_rate <= 0.0
        || request.learning_rate > 1.0
    {
        return Err(Error::new(
            "TRAINING",
            "invalid version, identity or training configuration",
        ));
    }
    let encoded = encode(&request.data)?;
    let unit = score_work(request.scorer, request.dims);
    let work = encoded.entities.len() as u64
        * unit
        * (request.epochs as u64 * encoded.train.len() as u64 * 12 + encoded.eval.len() as u64 * 4);
    budget(work, request.max_work)?;
    let scorer: Box<dyn Differentiable> = match request.scorer {
        ScorerKind::Hole => Box::new(HolE::new(request.dims).map_err(native_error)?),
        ScorerKind::Rotate => Box::new(RotatE::new(request.dims).map_err(native_error)?),
    };
    let mut tables = Tables::new(
        encoded.entities.len(),
        encoded.relations.len(),
        request.dims,
        request.seed,
    );
    let training = store(&encoded, false)?;
    let filter = store(&encoded, true)?;
    let eval_config = EvalConfig::random(request.seed);
    let initial =
        evaluate(&tables, &*scorer, &filter, &encoded.eval, &eval_config).map_err(native_error)?;
    let config = TrainConfig {
        dims: request.dims,
        epochs: request.epochs,
        batch_size: 64,
        lr: request.learning_rate,
        seed: request.seed,
        optimizer: OptimKind::Adagrad { epsilon: 1e-8 },
        loss: LossKind::OneVsAll,
        n3_lambda: 0.0001,
    };
    let mut losses = Vec::with_capacity(request.epochs);
    Trainer::fit(&mut tables, &*scorer, &training, &config, |p| {
        losses.push(p.loss)
    })
    .map_err(native_error)?;
    if losses.len() != request.epochs || losses.iter().any(|v| !v.is_finite()) {
        return Err(Error::new("TRAINING", "training produced invalid progress"));
    }
    let trained =
        evaluate(&tables, &*scorer, &filter, &encoded.eval, &eval_config).map_err(native_error)?;
    let request_hash = content_hash(request)?;
    let training_hash = content_hash(&request.data.train)?;
    let evaluation_hash = content_hash(&request.data.evaluation)?;
    let model = ModelArtifact {
        version: 1,
        name: request.name.clone(),
        revision: request.revision.clone(),
        scorer: request.scorer,
        dims: request.dims,
        entities: encoded.entities,
        relations: encoded.relations,
        entity_weights: tables.entities_raw().to_vec(),
        relation_weights: tables.relations_raw().to_vec(),
        origin: Origin::Trained {
            request_hash: request_hash.clone(),
            training_hash: training_hash.clone(),
            evaluation_hash: evaluation_hash.clone(),
            epochs: request.epochs,
        },
    };
    model.validate()?;
    let report = TrainingReport {
        version: 1,
        request_hash,
        model_hash: content_hash(&model)?,
        training_hash,
        evaluation_hash,
        train_count: encoded.train.len(),
        evaluation_count: encoded.eval.len(),
        initial: initial.into(),
        trained: trained.into(),
        losses,
        estimated_work: work,
        seed: request.seed,
        epochs: request.epochs,
        learning_rate: request.learning_rate,
    };
    Ok(Candidate {
        version: 1,
        model,
        report,
    })
}

pub fn evaluate_model(
    model: &ModelArtifact,
    request: &EvaluationRequest,
) -> Result<EvaluationReport> {
    model.validate()?;
    if request.version != 1 {
        return Err(Error::new("TRAINING", "unsupported evaluation version"));
    }
    let mut encoded = encode(&request.data)?;
    // Input rows may have a different order from an imported model's vocabulary.
    let ei: BTreeMap<_, _> = model
        .entities
        .iter()
        .enumerate()
        .map(|(i, s)| (s, i as u32))
        .collect();
    let ri: BTreeMap<_, _> = model
        .relations
        .iter()
        .enumerate()
        .map(|(i, s)| (s, i as u32))
        .collect();
    for triple in encoded.train.iter_mut().chain(&mut encoded.eval) {
        triple.s = *ei
            .get(&encoded.entities[triple.s as usize])
            .ok_or_else(|| Error::new("TRAINING", "unknown evaluation entity"))?;
        triple.o = *ei
            .get(&encoded.entities[triple.o as usize])
            .ok_or_else(|| Error::new("TRAINING", "unknown evaluation entity"))?;
        triple.r = *ri
            .get(&encoded.relations[triple.r as usize])
            .ok_or_else(|| Error::new("TRAINING", "unknown evaluation relation"))?;
    }
    encoded.entities = model.entities.clone();
    encoded.relations = model.relations.clone();
    let work = encoded.eval.len() as u64
        * 2
        * encoded.entities.len() as u64
        * score_work(model.scorer, model.dims);
    budget(work, request.max_work)?;
    let metrics = evaluate(
        &model.tables()?,
        &*model.scorer()?,
        &store(&encoded, true)?,
        &encoded.eval,
        &EvalConfig::random(request.seed),
    )
    .map_err(native_error)?;
    Ok(EvaluationReport {
        version: 1,
        model_hash: content_hash(model)?,
        data_hash: content_hash(&request.data)?,
        evaluation_count: encoded.eval.len(),
        metrics: metrics.into(),
        estimated_work: work,
    })
}
