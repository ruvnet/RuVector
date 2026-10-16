//! Bounded counterfactual sensitivity, not causal attribution or source authentication.
use crate::{runtime, Error, Op, Output, Result, RunRequest, Status};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const MAX_TRIALS: usize = 64;
const MAX_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Intervention {
    Observation { id: String },
    Source { id: String },
    Edge { id: String },
}
impl Intervention {
    fn id(&self) -> &str {
        match self {
            Self::Observation { id } | Self::Source { id } | Self::Edge { id } => id,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuditOptions {
    pub interventions: Vec<Intervention>,
    /// Upper bound on the SUM of reserved runtime budgets, including baseline.
    pub max_execution_work: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Change {
    pub subject: String,
    pub predicate: String,
    /// None means the decision left or entered the retrieval domain, not Unknown.
    pub before: Option<Status>,
    pub after: Option<Status>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Trial {
    pub intervention: Intervention,
    pub removed_records: usize,
    pub receipt_hash: String,
    pub snapshot_hash: String,
    pub work: u64,
    pub changes: Vec<Change>,
    /// Includes changed witness lineage even when the status is unchanged.
    pub changed_decisions: usize,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuditReport {
    pub version: u32,
    pub baseline_receipt_hash: String,
    pub program_hash: String,
    pub snapshot_hash: String,
    pub context_hash: String,
    pub reserved_execution_work: u64,
    pub execution_work: u64,
    pub baseline_decisions: usize,
    pub trials: Vec<Trial>,
}

fn invalid(message: &str) -> Error {
    Error::new("INVALID_AUDIT", message)
}

/// Run independent single interventions against the original snapshot.
/// Predictions are excluded, including unused prediction steps, so audit never
/// invokes an external provider. No mutable state or ambient clock is used.
/// At most 65 full executions; failure returns no partial report. Input cloning
/// and report comparison are bounded by record/byte limits, not runtime work.
pub fn run(request: &RunRequest, options: &AuditOptions) -> Result<AuditReport> {
    if options.interventions.is_empty() || options.interventions.len() > MAX_TRIALS {
        return Err(invalid("audit requires 1..=64 interventions"));
    }
    if options.max_execution_work == 0 || options.max_execution_work > 1_000_000_000 {
        return Err(invalid("max_execution_work must be in 1..=1000000000"));
    }
    let mut seen = BTreeSet::new();
    for intervention in &options.interventions {
        if intervention.id().trim().is_empty() || intervention.id().len() > 4096 {
            return Err(invalid("intervention id must contain 1..=4096 UTF8 bytes"));
        }
        if !seen.insert(intervention) {
            return Err(invalid("duplicate intervention"));
        }
    }
    if request
        .program
        .steps
        .iter()
        .any(|s| matches!(s.op, Op::Predict { .. }))
    {
        return Err(invalid("audit supports evidence programs only"));
    }
    let reserved = request
        .program
        .budget
        .min(request.context.max_steps)
        .checked_mul(options.interventions.len() as u64 + 1)
        .ok_or_else(|| invalid("execution reservation overflow"))?;
    if reserved > options.max_execution_work {
        return Err(Error::new(
            "AUDIT_BUDGET",
            "aggregate execution reservation exceeds audit limit",
        ));
    }
    // Validates the entire input before any potentially large clone or trial.
    let baseline = runtime::execute(&request.program, &request.snapshot, &request.context, None)?;
    let Output::Decisions(ref decisions) = baseline.output else {
        return Err(invalid("audit requires decision output"));
    };
    // Check every target before trials, so a typo cannot look like robustness.
    for target in &options.interventions {
        let exists = match target {
            Intervention::Observation { id } => {
                request.snapshot.observations.iter().any(|o| &o.id == id)
            }
            Intervention::Source { id } => request
                .snapshot
                .observations
                .iter()
                .any(|o| &o.source == id),
            Intervention::Edge { id } => request.snapshot.edges.iter().any(|e| &e.id == id),
        };
        if !exists {
            return Err(invalid("intervention target does not exist"));
        }
    }
    let before: BTreeMap<_, _> = decisions
        .iter()
        .map(|d| ((d.subject.as_str(), d.predicate.as_str()), d))
        .collect();
    let mut report = AuditReport {
        version: 1,
        baseline_receipt_hash: baseline.receipt_hash,
        program_hash: baseline.program_hash,
        snapshot_hash: baseline.snapshot_hash,
        context_hash: baseline.context_hash,
        reserved_execution_work: reserved,
        execution_work: baseline.work,
        baseline_decisions: decisions.len(),
        trials: vec![],
    };
    let mut report_bytes = serde_json::to_vec(&report)
        .map_err(|_| invalid("serialization failed"))?
        .len();
    for target in &options.interventions {
        let mut snapshot = request.snapshot.clone();
        let old_count = snapshot.observations.len() + snapshot.edges.len();
        match target {
            Intervention::Observation { id } => snapshot.observations.retain(|o| &o.id != id),
            Intervention::Source { id } => snapshot.observations.retain(|o| &o.source != id),
            Intervention::Edge { id } => snapshot.edges.retain(|e| &e.id != id),
        }
        let removed_records = old_count - snapshot.observations.len() - snapshot.edges.len();
        let receipt = runtime::execute(&request.program, &snapshot, &request.context, None)?;
        let Output::Decisions(ref output) = receipt.output else {
            unreachable!()
        };
        let after: BTreeMap<_, _> = output
            .iter()
            .map(|d| ((d.subject.as_str(), d.predicate.as_str()), d))
            .collect();
        let keys: BTreeSet<_> = before.keys().chain(after.keys()).copied().collect();
        let mut trial = Trial {
            intervention: target.clone(),
            removed_records,
            receipt_hash: receipt.receipt_hash,
            snapshot_hash: receipt.snapshot_hash,
            work: receipt.work,
            changes: vec![],
            changed_decisions: 0,
        };
        // Count serialized metadata and separators before retaining changes.
        report_bytes += serde_json::to_vec(&trial)
            .map_err(|_| invalid("serialization failed"))?
            .len()
            + 64;
        for key in keys {
            let left = before.get(&key);
            let right = after.get(&key);
            if left != right {
                trial.changed_decisions += 1;
            }
            let old = left.map(|d| d.status.clone());
            let new = right.map(|d| d.status.clone());
            if old != new {
                let change = Change {
                    subject: key.0.into(),
                    predicate: key.1.into(),
                    before: old,
                    after: new,
                };
                report_bytes += serde_json::to_vec(&change)
                    .map_err(|_| invalid("serialization failed"))?
                    .len()
                    + 1;
                if report_bytes > MAX_BYTES {
                    return Err(Error::new("AUDIT_LIMIT", "audit report exceeds 16 MiB"));
                }
                trial.changes.push(change);
            }
        }
        if report_bytes > MAX_BYTES {
            return Err(Error::new("AUDIT_LIMIT", "audit report exceeds 16 MiB"));
        }
        report.execution_work += trial.work;
        report.trials.push(trial);
    }
    Ok(report)
}
