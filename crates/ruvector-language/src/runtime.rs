//! Pure bounded execution against immutable snapshots and an explicit clock.
mod evidence;
mod graph;
mod receipt;
mod validation;

use crate::{
    compiler,
    error::{Error, Result},
    types::*,
};
use evidence::EvidenceSet;
use graph::Graph;
pub use receipt::{hash_receipt, verify_receipt};
use std::collections::BTreeMap;

pub(super) const MAX_RECORDS: usize = 100_000;
pub(super) const MAX_BYTES: usize = 16 * 1024 * 1024;
pub(super) const MAX_VECTOR: usize = 4096;
pub(super) const MAX_BUDGET: u64 = 1_000_000_000;

pub(super) struct Meter {
    pub used: u64,
    limit: u64,
}
impl Meter {
    fn new(limit: u64) -> Self {
        Self { used: 0, limit }
    }
    pub fn charge(&mut self, amount: usize) -> Result<()> {
        let next = self
            .used
            .checked_add(amount as u64)
            .ok_or_else(|| Error::new("BUDGET_EXCEEDED", "work counter overflow"))?;
        if next > self.limit {
            return Err(Error::new(
                "BUDGET_EXCEEDED",
                "deterministic work budget exhausted",
            ));
        }
        self.used = next;
        Ok(())
    }
    pub fn remaining(&self) -> u64 {
        self.limit - self.used
    }
}

pub(super) fn bounded(size: usize, limit: usize) -> Result<()> {
    if size > limit {
        Err(Error::new(
            "RESULT_LIMIT",
            "result or provenance limit exceeded",
        ))
    } else {
        Ok(())
    }
}
fn invalid(message: impl Into<String>) -> Error {
    Error::new("INVALID_INPUT", message)
}

enum Value<'a> {
    Nodes(Vec<&'a str>),
    Evidence(EvidenceSet),
    Decisions(Vec<Decision>),
    Predictions(PredictionBatch),
}
impl Value<'_> {
    fn retained_bytes(&self) -> Result<usize> {
        match self {
            Self::Nodes(nodes) => Ok(nodes.iter().map(|id| id.len() + 32).sum()),
            Self::Evidence(evidence) => Ok(evidence.retained_bytes()),
            Self::Decisions(decisions) => validation::serialized_size(decisions),
            Self::Predictions(predictions) => validation::serialized_size(predictions),
        }
    }
}
fn nodes<'v, 's>(values: &'v BTreeMap<String, Value<'s>>, key: &str) -> Result<&'v [&'s str]> {
    match values.get(key) {
        Some(Value::Nodes(value)) => Ok(value),
        _ => Err(Error::new(
            "TYPE_ERROR",
            format!("{key} is not a node collection"),
        )),
    }
}
fn evidence<'a>(values: &'a BTreeMap<String, Value<'_>>, key: &str) -> Result<&'a EvidenceSet> {
    match values.get(key) {
        Some(Value::Evidence(value)) => Ok(value),
        _ => Err(Error::new("TYPE_ERROR", format!("{key} is not evidence"))),
    }
}

/// Validate a host-created snapshot with the same hard limits as execution.
pub fn validate_snapshot(snapshot: &Snapshot) -> Result<()> {
    validation::snapshot(snapshot, &mut Meter::new(MAX_BUDGET))
}

/// Immutable, borrowed snapshot preparation for repeated Rust host queries.
/// The borrow prevents mutation while indexes and canonical hash are retained.
/// Warm executions charge the same logical work as ordinary executions.
pub struct PreparedSnapshot<'a> {
    snapshot: &'a Snapshot,
    graph: Graph<'a>,
    index_work: u64,
    serialized_bytes: usize,
    snapshot_hash: String,
}
impl<'a> PreparedSnapshot<'a> {
    pub fn new(snapshot: &'a Snapshot) -> Result<Self> {
        let mut meter = Meter::new(MAX_BUDGET);
        validation::snapshot(snapshot, &mut meter)?;
        let graph = Graph::new(snapshot, &mut meter)?;
        let serialized_bytes = validation::serialized_size(snapshot)?;
        let snapshot_hash = receipt::snapshot_hash(snapshot)?;
        Ok(Self {
            snapshot,
            graph,
            index_work: meter.used,
            serialized_bytes,
            snapshot_hash,
        })
    }
    /// Context, program, provider and capabilities are checked on EVERY query.
    /// This is index/hash reuse, never cached decisions or cached authority.
    pub fn execute(
        &self,
        program: &Program,
        context: &Context,
        predictor: Option<&dyn Predictor>,
    ) -> Result<Receipt> {
        execute_internal::<true>(program, self.snapshot, context, predictor, Some(self))
    }
}

/// Execute without I/O, mutation, ambient time, or implicit model promotion.
/// All errors discard intermediate results; budget exhaustion never returns a partial receipt.
pub fn execute(
    program: &Program,
    snapshot: &Snapshot,
    context: &Context,
    predictor: Option<&dyn Predictor>,
) -> Result<Receipt> {
    execute_internal::<false>(program, snapshot, context, predictor, None)
}

fn execute_internal<const PREPARED: bool>(
    program: &Program,
    snapshot: &Snapshot,
    context: &Context,
    predictor: Option<&dyn Predictor>,
    prepared: Option<&PreparedSnapshot<'_>>,
) -> Result<Receipt> {
    compiler::check(program)?;
    validation::context(context)?;
    if !program.effects.is_subset(&context.capabilities) {
        return Err(Error::new(
            "UNAUTHORIZED",
            "program effects exceed granted capabilities",
        ));
    }
    let mut meter = Meter::new(program.budget.min(context.max_steps));
    let program_bytes = validation::serialized_size(program)?;
    meter.charge(program_bytes / 64 + 1)?;
    let owned_graph;
    let graph = if PREPARED {
        let prepared = prepared.expect("prepared execution requires preparation");
        meter.charge(prepared.index_work as usize)?;
        &prepared.graph
    } else {
        validation::snapshot(snapshot, &mut meter)?;
        owned_graph = Graph::new(snapshot, &mut meter)?;
        &owned_graph
    };
    let mut values = BTreeMap::new();
    let mut retained_bytes = 0_usize;
    for step in &program.steps {
        meter.charge(1)?;
        let value = match &step.op {
            Op::Nodes { label } => {
                Value::Nodes(graph.nodes(label, context.max_results, &mut meter)?)
            }
            Op::Select { ids } => {
                Value::Nodes(graph.select(ids, context.max_results, &mut meter)?)
            }
            Op::Traverse {
                input,
                relation,
                depth,
            } => Value::Nodes(graph.traverse(
                nodes(&values, input)?,
                relation,
                *depth,
                context.max_results,
                &mut meter,
            )?),
            Op::Similar { input, vector, k } => Value::Nodes(graph.similar(
                nodes(&values, input)?,
                vector,
                *k,
                context.max_results,
                &mut meter,
            )?),
            Op::Observe {
                input,
                predicate,
                within,
            } => Value::Evidence(graph.observe(
                nodes(&values, input)?,
                predicate,
                *within,
                context,
                &mut meter,
            )?),
            Op::Infer { inputs, rules } => {
                let inputs = inputs
                    .iter()
                    .map(|key| evidence(&values, key))
                    .collect::<Result<Vec<_>>>()?;
                Value::Evidence(evidence::infer(&inputs, rules, graph, context, &mut meter)?)
            }
            Op::Predict {
                input,
                relation,
                model,
                k,
            } => {
                let subjects = nodes(&values, input)?;
                bounded(*k, context.max_results)?;
                bounded(subjects.len().saturating_mul(*k), context.max_results)?;
                let adapter = predictor.ok_or_else(|| {
                    Error::new("PROVIDER_UNAVAILABLE", "no model provider supplied")
                })?;
                meter.charge(subjects.len())?;
                let request = PredictionRequest {
                    model: model.clone(),
                    relation: relation.clone(),
                    subjects: subjects.iter().map(|id| (*id).to_string()).collect(),
                    k: *k,
                    max_work: meter.remaining(),
                };
                let mut batch = adapter.predict(&request)?;
                validation::prediction(&batch, &request, context.max_results)?;
                meter.charge(
                    usize::try_from(batch.work)
                        .map_err(|_| invalid("provider work exceeds platform range"))?,
                )?;
                meter.charge(
                    batch
                        .candidates
                        .len()
                        .saturating_mul(graph::log_work(batch.candidates.len())),
                )?;
                batch.candidates.sort_by(|a, b| {
                    a.subject
                        .cmp(&b.subject)
                        .then(b.score.total_cmp(&a.score))
                        .then(a.object.cmp(&b.object))
                });
                Value::Predictions(batch)
            }
            Op::Assess { input } => {
                Value::Decisions(evidence(&values, input)?.assess(context.max_results, &mut meter)?)
            }
        };
        retained_bytes = retained_bytes.saturating_add(value.retained_bytes()?);
        bounded(retained_bytes, MAX_BYTES)?;
        values.insert(step.bind.clone(), value);
    }
    let output = match values.remove(&program.emit) {
        Some(Value::Decisions(value)) => Output::Decisions(value),
        Some(Value::Predictions(value)) => Output::Predictions(value),
        _ => {
            return Err(Error::new(
                "TYPE_ERROR",
                "emit requires decisions or predictions",
            ))
        }
    };
    // Serialization size is bounded before hashing and the input hash cost is metered.
    meter.charge(program_bytes / 64 + 1)?;
    let snapshot_bytes = if PREPARED {
        prepared
            .expect("prepared execution requires preparation")
            .serialized_bytes
    } else {
        validation::serialized_size(snapshot)?
    };
    meter.charge(snapshot_bytes / 64 + 1)?;
    meter.charge(validation::serialized_size(&output)? / 64 + 1)?;
    meter.charge(validation::serialized_size(context)? / 64 + 1)?;
    let records = snapshot.nodes.len()
        + snapshot.edges.len()
        + snapshot.observations.len()
        + snapshot.rules.len();
    meter.charge(records.saturating_mul(graph::log_work(records)))?;
    let program_hash = receipt::hash(program)?;
    let snapshot_hash = if PREPARED {
        prepared
            .expect("prepared execution requires preparation")
            .snapshot_hash
            .clone()
    } else {
        receipt::snapshot_hash(snapshot)?
    };
    let context_hash = receipt::hash(context)?;
    let mut receipt = Receipt {
        version: 1,
        program_hash,
        snapshot_hash,
        context_hash,
        snapshot_revision: snapshot.revision.clone(),
        now: context.now,
        work: meter.used,
        output,
        receipt_hash: String::new(),
    };
    receipt.receipt_hash = hash_receipt(&receipt)?;
    Ok(receipt)
}
