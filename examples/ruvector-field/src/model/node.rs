//! Field node — a single semantic unit carrying shell, axes, and signals.

use core::fmt;

use super::{clamp01, EmbeddingId, NodeId, Shell};

/// Kinds of first-class field nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    /// Raw user or agent interaction.
    Interaction,
    /// Summary, pattern, or concept node.
    Summary,
    /// Policy or principle node.
    Policy,
    /// Agent or role node.
    Agent,
    /// Partition node (logical region).
    Partition,
    /// Physical region node.
    Region,
    /// Witness binding node.
    Witness,
}

/// Four-axis score vector. Every field is expected in `[0, 1]`.
///
/// # Example
///
/// ```
/// use ruvector_field::model::AxisScores;
/// let a = AxisScores::new(0.7, 0.6, 0.5, 0.8);
/// assert!((a.product() - 0.168).abs() < 1e-3);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AxisScores {
    /// Limit — constraint / bound preservation.
    pub limit: f32,
    /// Care — impact awareness.
    pub care: f32,
    /// Bridge — synthesis across contexts.
    pub bridge: f32,
    /// Clarity — explainability.
    pub clarity: f32,
}

impl AxisScores {
    /// Construct axes, clamping each into `[0, 1]`.
    pub fn new(limit: f32, care: f32, bridge: f32, clarity: f32) -> Self {
        Self {
            limit: clamp01(limit),
            care: clamp01(care),
            bridge: clamp01(bridge),
            clarity: clamp01(clarity),
        }
    }

    /// Product of all four axes — spec 8.1 component.
    pub fn product(&self) -> f32 {
        self.limit * self.care * self.bridge * self.clarity
    }

    /// Reinforce one axis by `delta`, clamped.
    pub fn reinforce(&mut self, which: AxisKind, delta: f32) {
        let f = match which {
            AxisKind::Limit => &mut self.limit,
            AxisKind::Care => &mut self.care,
            AxisKind::Bridge => &mut self.bridge,
            AxisKind::Clarity => &mut self.clarity,
        };
        *f = clamp01(*f + delta);
    }

    /// Decay one axis by `delta`, clamped to `[0, 1]`.
    pub fn decay(&mut self, which: AxisKind, delta: f32) {
        self.reinforce(which, -delta);
    }
}

impl Default for AxisScores {
    fn default() -> Self {
        Self {
            limit: 0.5,
            care: 0.5,
            bridge: 0.5,
            clarity: 0.5,
        }
    }
}

/// Single axis tag used by `AxisScores::reinforce` / `decay`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisKind {
    /// Limit axis.
    Limit,
    /// Care axis.
    Care,
    /// Bridge axis.
    Bridge,
    /// Clarity axis.
    Clarity,
}

/// A single field node.
///
/// Embedding vectors are interned in the engine's [`super::EmbeddingStore`];
/// the node itself only carries ids so clones stay cheap.
#[derive(Debug, Clone)]
pub struct FieldNode {
    /// Stable id.
    pub id: NodeId,
    /// Kind of node.
    pub kind: NodeKind,
    /// Semantic embedding id.
    pub semantic_embedding: EmbeddingId,
    /// Geometric antipode embedding id (distinct from the semantic one).
    pub geometric_antipode: EmbeddingId,
    /// Explicit semantic antipode node id, if one is bound.
    pub semantic_antipode: Option<NodeId>,
    /// Current shell assignment.
    pub shell: Shell,
    /// Axis scores.
    pub axes: AxisScores,
    /// Coherence signal in `[0, 1]`.
    pub coherence: f32,
    /// Continuity signal in `[0, 1]`.
    pub continuity: f32,
    /// Resonance signal in `[0, 1]`.
    pub resonance: f32,
    /// Policy mask bitset.
    pub policy_mask: u64,
    /// Witness binding if applicable.
    pub witness_ref: Option<u64>,
    /// Creation timestamp, nanoseconds.
    pub ts_ns: u64,
    /// Hour-bucket for temporal queries.
    pub temporal_bucket: u64,
    /// Raw text payload.
    pub text: String,
    /// Timestamp of entry into the current shell.
    pub shell_entered_ts: u64,
    /// Consecutive passes above promotion thresholds.
    pub promotion_streak: u32,
    /// Last N shell transitions (bounded to `HYSTERESIS_WINDOW`).
    pub promotion_history: Vec<Shell>,
    /// Number of times this node was selected in retrieval (for axis tick).
    pub selection_count: u32,
    /// Number of contradictions observed against this node.
    pub contradiction_hits: u32,
    /// Edges incident at the last tick — for continuity churn.
    pub edges_at_last_tick: u32,
}

impl fmt::Display for FieldNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {:?} shell={} coh={:.3} cont={:.3} res={:.3} text={:?}",
            self.id,
            self.kind,
            self.shell,
            self.coherence,
            self.continuity,
            self.resonance,
            truncate_chars(&self.text, 48),
        )
    }
}

fn truncate_chars(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(n).collect();
        out.push_str("...");
        out
    }
}
