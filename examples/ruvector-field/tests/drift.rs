//! Four-channel drift detection.

use ruvector_field::prelude::*;
use ruvector_field::scoring::DriftSignal;

#[test]
fn identical_centroid_produces_near_zero_semantic_drift() {
    let mut engine = FieldEngine::new();
    let emb = Embedding::new(vec![1.0, 0.0, 0.0, 0.0]);
    engine
        .ingest(
            NodeKind::Interaction,
            "a",
            emb.clone(),
            AxisScores::new(0.5, 0.5, 0.5, 0.5),
            0,
        )
        .unwrap();
    let drift = engine.drift(&emb);
    assert!(drift.semantic < 0.1, "drift.semantic was {}", drift.semantic);
}

#[test]
fn symmetric_axes_produce_equal_semantic_drift_signs() {
    let mut engine = FieldEngine::new();
    let a = Embedding::new(vec![1.0, 0.0, 0.0]);
    let b = Embedding::new(vec![0.0, 1.0, 0.0]);
    engine
        .ingest(
            NodeKind::Interaction,
            "a",
            a.clone(),
            AxisScores::new(0.5, 0.5, 0.5, 0.5),
            0,
        )
        .unwrap();
    let d = engine.drift(&b);
    // Orthogonal -> cos 0 -> semantic drift 0.5.
    assert!((d.semantic - 0.5).abs() < 0.05);
}

#[test]
fn agreement_rule_requires_two_channels() {
    let one = DriftSignal {
        semantic: 0.5,
        structural: 0.0,
        policy: 0.0,
        identity: 0.0,
        total: 0.5,
    };
    assert!(!one.agreement_fires(0.4, 0.1));

    let two = DriftSignal {
        semantic: 0.3,
        structural: 0.2,
        policy: 0.0,
        identity: 0.0,
        total: 0.5,
    };
    assert!(two.agreement_fires(0.4, 0.1));
}
