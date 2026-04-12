//! Shared data model for the RuVector field subsystem.
//!
//! Mirrors section 6 of `docs/research/ruvector-field/SPEC.md`. Kept in a single
//! file so this example stays easy to read end to end.

use std::time::{SystemTime, UNIX_EPOCH};

/// Logical abstraction depth. Distinct from physical memory tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Shell {
    Event,
    Pattern,
    Concept,
    Principle,
}

impl Shell {
    /// Ordinal depth, 0 = Event, 3 = Principle.
    pub fn depth(self) -> u8 {
        match self {
            Shell::Event => 0,
            Shell::Pattern => 1,
            Shell::Concept => 2,
            Shell::Principle => 3,
        }
    }

    /// Phi-scaled compression budget for the shell, starting from `base`.
    /// Spec section 9.3.
    pub fn budget(self, base: f32) -> f32 {
        let phi: f32 = 1.618_034;
        base / phi.powi(self.depth() as i32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    Interaction,
    Summary,
    Policy,
    Agent,
    Partition,
    Region,
    Witness,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    Supports,
    Contrasts,
    Refines,
    RoutesTo,
    DerivedFrom,
    SharesRegion,
    BindsWitness,
}

/// Normalized axis scores. Each field is expected in `[0, 1]`.
#[derive(Debug, Clone, Copy)]
pub struct AxisScores {
    pub limit: f32,
    pub care: f32,
    pub bridge: f32,
    pub clarity: f32,
}

impl AxisScores {
    pub fn new(limit: f32, care: f32, bridge: f32, clarity: f32) -> Self {
        Self {
            limit: clamp01(limit),
            care: clamp01(care),
            bridge: clamp01(bridge),
            clarity: clamp01(clarity),
        }
    }

    pub fn product(&self) -> f32 {
        self.limit * self.care * self.bridge * self.clarity
    }
}

/// Dense embedding vector. Real systems use ids into an embedding store; this
/// example owns the vector directly for clarity.
#[derive(Debug, Clone)]
pub struct Embedding {
    pub values: Vec<f32>,
}

impl Embedding {
    pub fn new(values: Vec<f32>) -> Self {
        Self {
            values: l2_normalize(values),
        }
    }

    /// Geometric antipode used by the search geometry layer. This is the
    /// normalized negative, nothing more. Semantic opposition lives on a
    /// separate explicit link per spec section 5.2.
    pub fn geometric_antipode(&self) -> Embedding {
        Embedding {
            values: self.values.iter().map(|v| -v).collect(),
        }
    }

    pub fn cosine(&self, other: &Embedding) -> f32 {
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}

#[derive(Debug, Clone)]
pub struct FieldNode {
    pub id: u64,
    pub kind: NodeKind,
    pub semantic_embedding: Embedding,
    pub geometric_antipode: Embedding,
    pub semantic_antipode: Option<u64>,
    pub shell: Shell,
    pub axes: AxisScores,
    pub coherence: f32,
    pub continuity: f32,
    pub resonance: f32,
    pub policy_mask: u64,
    pub witness_ref: Option<u64>,
    pub ts_ns: u64,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct FieldEdge {
    pub src: u64,
    pub dst: u64,
    pub kind: EdgeKind,
    pub weight: f32,
    pub ts_ns: u64,
}

#[derive(Debug, Clone, Default)]
pub struct DriftSignal {
    pub semantic: f32,
    pub structural: f32,
    pub policy: f32,
    pub identity: f32,
    pub total: f32,
}

#[derive(Debug, Clone)]
pub struct RoutingHint {
    pub target_partition: Option<u64>,
    pub target_agent: Option<u64>,
    pub gain_estimate: f32,
    pub cost_estimate: f32,
    pub ttl_epochs: u16,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct RetrievalResult {
    pub selected: Vec<u64>,
    pub rejected: Vec<u64>,
    pub contradiction_frontier: Vec<u64>,
    pub explanation: Vec<String>,
}

// ---- helpers ----

pub fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

pub fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

pub fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}
