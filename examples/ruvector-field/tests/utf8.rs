//! UTF-8 safe truncation used by node Display.

use ruvector_field::prelude::*;

#[test]
fn display_node_with_multibyte_text_does_not_panic() {
    let mut engine = FieldEngine::new();
    let p = HashEmbeddingProvider::new(16);
    let id = engine
        .ingest(
            NodeKind::Interaction,
            "こんにちは世界 — this is multibyte content that must truncate safely",
            p.embed("hello"),
            AxisScores::new(0.5, 0.5, 0.5, 0.5),
            0,
        )
        .unwrap();
    let n = engine.node(id).unwrap();
    let s = format!("{}", n);
    assert!(s.contains("node#"));
}
