//! Versioned wire contracts. Unknown fields fail closed at every boundary.
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Effect {
    #[serde(rename = "graph.read")]
    GraphRead,
    #[serde(rename = "model.predict")]
    ModelPredict,
    #[serde(rename = "decision.emit")]
    DecisionEmit,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Program {
    pub version: u32,
    pub name: String,
    pub effects: BTreeSet<Effect>,
    pub budget: u64,
    pub steps: Vec<Step>,
    pub emit: String,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub bind: String,
    #[serde(flatten)]
    pub op: Op,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Op {
    Nodes {
        label: String,
    },
    Select {
        ids: Vec<String>,
    },
    Traverse {
        input: String,
        relation: String,
        depth: usize,
    },
    Similar {
        input: String,
        vector: Vec<f32>,
        k: usize,
    },
    Observe {
        input: String,
        predicate: String,
        within: u64,
    },
    Infer {
        inputs: Vec<String>,
        rules: Vec<String>,
    },
    Predict {
        input: String,
        relation: String,
        model: String,
        k: usize,
    },
    Assess {
        input: String,
    },
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Node {
    pub id: String,
    pub label: String,
    #[serde(default)]
    pub properties: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub vector: Option<Vec<f32>>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Edge {
    pub id: String,
    pub source: String,
    pub target: String,
    pub relation: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub id: String,
    pub subject: String,
    pub predicate: String,
    pub value: bool,
    pub source: String,
    pub observed_at: u64,
    pub expires_at: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Condition {
    pub predicate: String,
    pub value: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rule {
    pub id: String,
    pub requires: Vec<Condition>,
    pub conclusion: Condition,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Snapshot {
    pub version: u32,
    pub revision: String,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub observations: Vec<Observation>,
    #[serde(default)]
    pub rules: Vec<Rule>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub now: u64,
    pub capabilities: BTreeSet<Effect>,
    pub max_steps: u64,
    pub max_results: usize,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Supported,
    Refuted,
    Unknown,
    Conflicting,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    Observed,
    Derived,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Evidence {
    pub subject: String,
    pub predicate: String,
    pub value: bool,
    pub kind: EvidenceKind,
    pub observation_ids: BTreeSet<String>,
    pub sources: BTreeSet<String>,
    pub rule_ids: BTreeSet<String>,
    pub expires_at: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Decision {
    pub subject: String,
    pub predicate: String,
    pub status: Status,
    pub evidence: Vec<Evidence>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prediction {
    pub subject: String,
    pub relation: String,
    pub object: String,
    pub score: f32,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictionBatch {
    pub model: String,
    pub revision: String,
    pub model_hash: String,
    pub exact: bool,
    pub work: u64,
    pub candidates: Vec<Prediction>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "items",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum Output {
    Decisions(Vec<Decision>),
    Predictions(PredictionBatch),
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    pub version: u32,
    pub program_hash: String,
    pub snapshot_hash: String,
    pub snapshot_revision: String,
    pub context_hash: String,
    pub now: u64,
    pub work: u64,
    pub output: Output,
    pub receipt_hash: String,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunRequest {
    pub program: Program,
    pub snapshot: Snapshot,
    pub context: Context,
}
#[derive(Debug, Clone)]
pub struct PredictionRequest {
    pub model: String,
    pub relation: String,
    pub subjects: Vec<String>,
    pub k: usize,
    pub max_work: u64,
}
pub trait Predictor {
    /// Trusted host adapter. Must enforce max_work before expensive computation.
    fn predict(&self, request: &PredictionRequest) -> crate::error::Result<PredictionBatch>;
}
