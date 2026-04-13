//! Verifies HNSW backend agrees with linear scan on top-1 for a small corpus.
//!
//! Build/run with:
//!
//! ```text
//! cargo test --features hnsw --test feature_hnsw
//! ```

#![cfg(feature = "hnsw")]

use ruvector_field::model::EmbeddingStore;
use ruvector_field::prelude::*;
use ruvector_field::storage::{HnswIndex, LinearIndex, SemanticIndex};

fn seed(engine: &mut FieldEngine) {
    let p = HashEmbeddingProvider::new(16);
    for text in [
        "alpha one",
        "alpha two",
        "alpha three",
        "beta one",
        "beta two",
        "gamma one",
        "gamma two",
        "delta one",
        "delta two",
        "delta three",
    ] {
        engine
            .ingest(
                NodeKind::Summary,
                text,
                p.embed(text),
                AxisScores::new(0.8, 0.7, 0.6, 0.8),
                0b0001,
            )
            .unwrap();
    }
}

#[test]
fn hnsw_and_linear_index_agree_on_top1() {
    // Compare the raw index layer directly (before the engine rerank
    // applies cutoffs that depend on absolute score). Both indexes must
    // return the same top-1 node on a corpus small enough for HNSW to
    // be exhaustive (M=12, corpus=10).
    let mut store = EmbeddingStore::new();
    let mut linear = LinearIndex::new();
    let mut hnsw = HnswIndex::new();
    let p = HashEmbeddingProvider::new(16);
    let corpus = [
        "alpha one",
        "alpha two",
        "alpha three",
        "beta one",
        "beta two",
        "gamma one",
        "gamma two",
        "delta one",
        "delta two",
        "delta three",
    ];
    for (i, text) in corpus.iter().enumerate() {
        let eid = store.intern(p.embed(text));
        let nid = NodeId((i + 1) as u64);
        linear.upsert(nid, eid, Shell::Event);
        hnsw.upsert(&store, nid, eid, Shell::Event);
    }
    for query_text in ["alpha", "beta", "gamma", "delta"] {
        let q = p.embed(query_text);
        let lhits = linear.search(&store, &q, &[Shell::Event], 5);
        let hhits = hnsw.search(&store, &q, &[Shell::Event], 5);
        assert!(!lhits.is_empty(), "linear empty for {}", query_text);
        assert!(!hhits.is_empty(), "hnsw empty for {}", query_text);
        // Top similarities must match (tied rows may appear in any order
        // across backends; comparing top similarity handles that).
        let ltop_sim = lhits[0].1;
        let htop_sim = hhits[0].1;
        assert!(
            (ltop_sim - htop_sim).abs() < 1e-4,
            "top-1 similarity mismatch on {}: linear={} hnsw={}",
            query_text,
            ltop_sim,
            htop_sim
        );
        // HNSW top-1 node should appear in linear's tied top group.
        let tied: Vec<NodeId> = lhits
            .iter()
            .filter(|(_, s)| (*s - ltop_sim).abs() < 1e-4)
            .map(|(id, _)| *id)
            .collect();
        assert!(
            tied.contains(&hhits[0].0),
            "top-1 node mismatch on {}: hnsw={:?} not in linear tied top {:?}",
            query_text,
            hhits[0].0,
            tied
        );
    }
}

#[test]
fn hnsw_retrieval_basic() {
    let mut engine = FieldEngine::new().with_hnsw_index();
    seed(&mut engine);
    engine.tick();
    let q = HashEmbeddingProvider::new(16).embed("alpha");
    let r = engine.retrieve(&q, &[Shell::Event], 3, None);
    assert!(!r.selected.is_empty());
    assert!(!r.explanation.is_empty());
}

#[test]
fn hnsw_respects_shell_filter() {
    let mut engine = FieldEngine::new().with_hnsw_index();
    seed(&mut engine);
    engine.tick();
    let q = HashEmbeddingProvider::new(16).embed("alpha");
    // No node should be on Concept shell → empty.
    let r = engine.retrieve(&q, &[Shell::Concept], 3, None);
    assert!(r.selected.is_empty());
}
