//! Antipode binding symmetry and geometric vs semantic separation.

use ruvector_field::prelude::*;

#[test]
fn semantic_antipode_is_symmetric() {
    let mut engine = FieldEngine::new();
    let p = HashEmbeddingProvider::new(16);
    let a = engine
        .ingest(
            NodeKind::Summary,
            "a",
            p.embed("a"),
            AxisScores::new(0.7, 0.7, 0.7, 0.7),
            0,
        )
        .unwrap();
    let b = engine
        .ingest(
            NodeKind::Summary,
            "b",
            p.embed("b"),
            AxisScores::new(0.7, 0.7, 0.7, 0.7),
            0,
        )
        .unwrap();
    engine.bind_semantic_antipode(a, b, 0.9).unwrap();
    assert_eq!(engine.node(a).unwrap().semantic_antipode, Some(b));
    assert_eq!(engine.node(b).unwrap().semantic_antipode, Some(a));
}

#[test]
fn geometric_antipode_is_distinct_from_semantic() {
    let mut engine = FieldEngine::new();
    let emb = Embedding::new(vec![0.7, 0.2, 0.1]);
    let a = engine
        .ingest(
            NodeKind::Summary,
            "a",
            emb.clone(),
            AxisScores::new(0.7, 0.7, 0.7, 0.7),
            0,
        )
        .unwrap();
    let node = engine.node(a).unwrap();
    assert_ne!(node.semantic_embedding, node.geometric_antipode);
    let sem = engine.store.get(node.semantic_embedding).unwrap();
    let geo = engine.store.get(node.geometric_antipode).unwrap();
    // Cosine of a vector and its negation is -1.
    let cos = sem.cosine(geo);
    assert!(cos < -0.999);
    // Semantic antipode remains unset — geometric flip does not imply opposition.
    assert!(engine.node(a).unwrap().semantic_antipode.is_none());
}
