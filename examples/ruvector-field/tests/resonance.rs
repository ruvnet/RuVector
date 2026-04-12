//! Resonance monotonicity and product bounds.

use ruvector_field::prelude::*;
use ruvector_field::scoring::resonance_score;

fn mock_node(axes: AxisScores, coh: f32, cont: f32) -> FieldNode {
    FieldNode {
        id: NodeId(1),
        kind: NodeKind::Interaction,
        semantic_embedding: EmbeddingId(1),
        geometric_antipode: EmbeddingId(2),
        semantic_antipode: None,
        shell: Shell::Event,
        axes,
        coherence: coh,
        continuity: cont,
        resonance: 0.0,
        policy_mask: 0,
        witness_ref: None,
        ts_ns: 0,
        temporal_bucket: 0,
        text: String::new(),
        shell_entered_ts: 0,
        promotion_streak: 0,
        promotion_history: vec![],
        selection_count: 0,
        contradiction_hits: 0,
        edges_at_last_tick: 0,
    }
}

#[test]
fn zero_factor_collapses_product() {
    let a = mock_node(AxisScores::new(0.0, 1.0, 1.0, 1.0), 1.0, 1.0);
    assert_eq!(resonance_score(&a), 0.0);
    let b = mock_node(AxisScores::new(0.9, 0.9, 0.9, 0.9), 0.0, 1.0);
    assert_eq!(resonance_score(&b), 0.0);
    let c = mock_node(AxisScores::new(0.9, 0.9, 0.9, 0.9), 1.0, 0.0);
    assert_eq!(resonance_score(&c), 0.0);
}

#[test]
fn resonance_bounded_in_unit_interval() {
    let n = mock_node(AxisScores::new(1.0, 1.0, 1.0, 1.0), 1.0, 1.0);
    let r = resonance_score(&n);
    assert!((0.999..=1.001).contains(&r));

    let n2 = mock_node(AxisScores::new(0.5, 0.5, 0.5, 0.5), 0.5, 0.5);
    let r2 = resonance_score(&n2);
    assert!((0.0..=1.0).contains(&r2));
    assert!(r2 > 0.0);
}

#[test]
fn monotonic_in_single_factor() {
    let a = mock_node(AxisScores::new(0.4, 0.6, 0.6, 0.6), 0.6, 0.6);
    let b = mock_node(AxisScores::new(0.5, 0.6, 0.6, 0.6), 0.6, 0.6);
    assert!(resonance_score(&b) > resonance_score(&a));
}
