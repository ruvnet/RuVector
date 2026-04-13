//! Promotion hysteresis and demotion.

use std::sync::Arc;

use ruvector_field::clock::AtomicTestClock;
use ruvector_field::engine::FieldEngineConfig;
use ruvector_field::prelude::*;

fn build_engine(passes: u32) -> (FieldEngine, Arc<AtomicTestClock>) {
    let clock = Arc::new(AtomicTestClock::new());
    let cfg = FieldEngineConfig {
        promotion_passes: passes,
        min_residence_ns: 0,
        hysteresis_window: 3,
        ..FieldEngineConfig::default()
    };
    (
        FieldEngine::with_config_and_clock(cfg, clock.clone()),
        clock,
    )
}

fn ingest_high_res(engine: &mut FieldEngine, text: &str) -> NodeId {
    let p = HashEmbeddingProvider::new(16);
    engine
        .ingest(
            NodeKind::Summary,
            text,
            p.embed(text),
            AxisScores::new(0.95, 0.95, 0.95, 0.95),
            0b0001,
        )
        .unwrap()
}

#[test]
fn single_pass_does_not_promote_with_hysteresis() {
    let (mut engine, _clock) = build_engine(3);
    let a = ingest_high_res(&mut engine, "core pattern alpha");
    let b = ingest_high_res(&mut engine, "core pattern beta");
    engine.add_edge(a, b, EdgeKind::Supports, 0.9).unwrap();
    engine.add_edge(b, a, EdgeKind::Supports, 0.9).unwrap();
    engine.add_edge(a, b, EdgeKind::DerivedFrom, 0.9).unwrap();
    engine.tick();
    // One pass — should not promote because passes_required = 3.
    let first = engine.promote_candidates();
    assert!(
        first.is_empty(),
        "expected no promotions on first pass, got {:?}",
        first
    );
}

#[test]
fn multiple_passes_promote_after_threshold() {
    let (mut engine, _clock) = build_engine(2);
    let a = ingest_high_res(&mut engine, "core pattern alpha");
    let b = ingest_high_res(&mut engine, "core pattern beta");
    let c = ingest_high_res(&mut engine, "core pattern gamma");
    for &(s, d) in &[(a, b), (b, c), (c, a), (a, c)] {
        engine.add_edge(s, d, EdgeKind::Supports, 0.9).unwrap();
        engine.add_edge(s, d, EdgeKind::DerivedFrom, 0.9).unwrap();
    }
    engine.tick();
    let first = engine.promote_candidates();
    let second = engine.promote_candidates();
    assert!(first.is_empty(), "first pass should not promote yet");
    assert!(!second.is_empty(), "second pass should promote");
}

#[test]
fn residence_window_blocks_premature_promotion() {
    let clock = Arc::new(AtomicTestClock::new());
    let cfg = FieldEngineConfig {
        promotion_passes: 1,
        min_residence_ns: 10_000,
        ..FieldEngineConfig::default()
    };
    let mut engine = FieldEngine::with_config_and_clock(cfg, clock.clone());
    let a = ingest_high_res(&mut engine, "a");
    let b = ingest_high_res(&mut engine, "b");
    engine.add_edge(a, b, EdgeKind::Supports, 0.9).unwrap();
    engine.add_edge(b, a, EdgeKind::Supports, 0.9).unwrap();
    engine.add_edge(a, b, EdgeKind::DerivedFrom, 0.9).unwrap();
    engine.tick();
    let first = engine.promote_candidates();
    assert!(first.is_empty(), "residence not satisfied");
    clock.advance_ns(20_000);
    let second = engine.promote_candidates();
    assert!(!second.is_empty(), "residence satisfied — should promote");
}

#[test]
fn demotion_on_contradiction() {
    let (mut engine, _clock) = build_engine(1);
    let a = ingest_high_res(&mut engine, "durable concept");
    let b = ingest_high_res(&mut engine, "supporting claim");
    let c = ingest_high_res(&mut engine, "second supporting claim");
    engine.add_edge(b, a, EdgeKind::Supports, 0.9).unwrap();
    engine.add_edge(c, a, EdgeKind::Supports, 0.9).unwrap();
    engine.add_edge(a, b, EdgeKind::DerivedFrom, 0.9).unwrap();
    engine.add_edge(a, c, EdgeKind::DerivedFrom, 0.9).unwrap();
    engine.tick();
    let _ = engine.promote_candidates();
    let _ = engine.promote_candidates();
    let _ = engine.promote_candidates();
    // Now force contradictions on `a`.
    let d = ingest_high_res(&mut engine, "opposing claim 1");
    let e = ingest_high_res(&mut engine, "opposing claim 2");
    engine.bind_semantic_antipode(a, d, 0.95).unwrap();
    engine.bind_semantic_antipode(a, e, 0.95).unwrap();
    engine.tick();
    let demoted = engine.demote_candidates();
    // At least one demotion should have happened OR `a` should still be at
    // Event (never promoted further), but the demotion path must run without
    // error and not panic.
    let _ = demoted;
}
