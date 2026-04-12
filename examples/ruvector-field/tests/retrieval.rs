//! Retrieval basics: shell filters, contradiction frontier, explanation trace.

use ruvector_field::prelude::*;

fn seed(engine: &mut FieldEngine) -> Vec<NodeId> {
    let p = HashEmbeddingProvider::new(16);
    let mut ids = Vec::new();
    for (i, text) in ["alpha one", "alpha two", "beta one", "gamma one"].iter().enumerate() {
        let id = engine
            .ingest(
                NodeKind::Summary,
                *text,
                p.embed(text),
                AxisScores::new(0.8, 0.7, 0.6, 0.8),
                0b0001,
            )
            .unwrap();
        ids.push(id);
        let _ = i;
    }
    ids
}

#[test]
fn excludes_disallowed_shells() {
    let mut engine = FieldEngine::new();
    let _ids = seed(&mut engine);
    engine.tick();
    let q = HashEmbeddingProvider::new(16).embed("alpha");
    // Only Concept allowed, but all nodes are Event -> empty result.
    let r = engine.retrieve(&q, &[Shell::Concept], 5, None);
    assert!(r.selected.is_empty());
}

#[test]
fn retrieval_produces_explanation_trace() {
    let mut engine = FieldEngine::new();
    let _ids = seed(&mut engine);
    engine.tick();
    let q = HashEmbeddingProvider::new(16).embed("alpha");
    let r = engine.retrieve(&q, &[Shell::Event], 3, None);
    assert!(!r.selected.is_empty());
    assert!(!r.explanation.is_empty());
}

#[test]
fn contradiction_frontier_is_populated_for_linked_pair() {
    let mut engine = FieldEngine::new();
    let ids = seed(&mut engine);
    engine
        .bind_semantic_antipode(ids[0], ids[3], 0.95)
        .unwrap();
    engine.tick();
    let q = HashEmbeddingProvider::new(16).embed("alpha");
    let r = engine.retrieve(&q, &[Shell::Event], 3, None);
    assert!(
        !r.contradiction_frontier.is_empty(),
        "expected frontier to be populated after antipode bind"
    );
}
