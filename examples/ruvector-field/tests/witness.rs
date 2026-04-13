//! Witness log — mutations emit exactly one event, reads zero.

use ruvector_field::prelude::*;

#[test]
fn ingest_emits_one_event() {
    let mut engine = FieldEngine::new();
    let before = engine.witness.len();
    let p = HashEmbeddingProvider::new(16);
    engine
        .ingest(
            NodeKind::Interaction,
            "a",
            p.embed("a"),
            AxisScores::new(0.5, 0.5, 0.5, 0.5),
            0,
        )
        .unwrap();
    assert_eq!(engine.witness.len() - before, 1);
    assert_eq!(
        engine.witness.events().last().unwrap().tag(),
        "field_node_created"
    );
}

#[test]
fn retrieval_does_not_emit_for_empty_frontier() {
    let mut engine = FieldEngine::new();
    let p = HashEmbeddingProvider::new(16);
    let _ = engine
        .ingest(
            NodeKind::Interaction,
            "a",
            p.embed("a"),
            AxisScores::new(0.5, 0.5, 0.5, 0.5),
            0,
        )
        .unwrap();
    let before = engine.witness.len();
    let q = p.embed("a");
    let _ = engine.retrieve(&q, &[Shell::Event], 1, None);
    assert_eq!(engine.witness.len(), before, "retrieval without frontier should not witness");
}

#[test]
fn bind_antipode_and_edge_each_emit_one_event() {
    let mut engine = FieldEngine::new();
    let p = HashEmbeddingProvider::new(16);
    let a = engine
        .ingest(
            NodeKind::Summary,
            "a",
            p.embed("a"),
            AxisScores::new(0.5, 0.5, 0.5, 0.5),
            0,
        )
        .unwrap();
    let b = engine
        .ingest(
            NodeKind::Summary,
            "b",
            p.embed("b"),
            AxisScores::new(0.5, 0.5, 0.5, 0.5),
            0,
        )
        .unwrap();
    let before = engine.witness.len();
    engine.add_edge(a, b, EdgeKind::Supports, 0.8).unwrap();
    assert_eq!(engine.witness.len() - before, 1);
    let before2 = engine.witness.len();
    engine.bind_semantic_antipode(a, b, 0.9).unwrap();
    assert_eq!(engine.witness.len() - before2, 1);
}

#[test]
fn flush_drains_events() {
    let mut engine = FieldEngine::new();
    let p = HashEmbeddingProvider::new(16);
    let _ = engine
        .ingest(
            NodeKind::Summary,
            "a",
            p.embed("a"),
            AxisScores::new(0.5, 0.5, 0.5, 0.5),
            0,
        )
        .unwrap();
    assert!(engine.witness.len() > 0);
    let drained = engine.witness.flush();
    assert!(!drained.is_empty());
    assert_eq!(engine.witness.len(), 0);
}
