//! Multiplicative resonance — spec section 8.1.
//!
//! `resonance = limit * care * bridge * clarity * coherence * continuity`
//!
//! All factors are normalized to `[0, 1]`, and the product collapses when any
//! factor is zero. That is the whole point: a single missing component
//! zeroes the signal so callers cannot trade off care for coherence.

use crate::model::FieldNode;

/// Spec 8.1: multiplicative resonance bounded to `[0, 1]`.
///
/// # Example
///
/// ```
/// use ruvector_field::model::{AxisScores, EmbeddingId, FieldNode, NodeId, NodeKind, Shell};
/// use ruvector_field::scoring::resonance_score;
/// let node = FieldNode {
///     id: NodeId(1),
///     kind: NodeKind::Interaction,
///     semantic_embedding: EmbeddingId(1),
///     geometric_antipode: EmbeddingId(2),
///     semantic_antipode: None,
///     shell: Shell::Event,
///     axes: AxisScores::new(1.0, 1.0, 1.0, 1.0),
///     coherence: 1.0,
///     continuity: 1.0,
///     resonance: 0.0,
///     policy_mask: 0,
///     witness_ref: None,
///     ts_ns: 0,
///     temporal_bucket: 0,
///     text: String::new(),
///     shell_entered_ts: 0,
///     promotion_streak: 0,
///     promotion_history: vec![],
///     selection_count: 0,
///     contradiction_hits: 0,
///     edges_at_last_tick: 0,
/// };
/// let r = resonance_score(&node);
/// assert!((r - 1.0).abs() < 1e-6);
/// ```
pub fn resonance_score(node: &FieldNode) -> f32 {
    (node.axes.product() * node.coherence.clamp(0.0, 1.0) * node.continuity.clamp(0.0, 1.0))
        .clamp(0.0, 1.0)
}
