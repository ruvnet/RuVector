//! Integration tests for content-correlation-augmented arbitration (ADR-350,
//! nightly 2026-11-01). Deliberately mirrors `tests/arbitration.rs`'s fixture
//! helpers so these tests are an apples-to-apples extension of the locked
//! baseline numbers in
//! `content_copying_yields_no_correlation_downgrade_under_default_reliability`.

use std::collections::BTreeMap;

use rand::rngs::OsRng;
use ruvector_agent_memory::{
    arbitrate, arbitrate_with_content_correlation, ArbitrationConfig, ArbitrationOutcome,
    AtomicObservation, CausalEpisodicGraph, ContentCorrelationConfig, ContentCorrelationError,
    CorrelationMode, NodeRef, ObservationSource, ReliabilityModel, SourceKind, Tenant,
};
use rvf_types::Ed25519Keypair;

fn kp() -> Ed25519Keypair {
    Ed25519Keypair::generate(&mut OsRng)
}

fn source(id: &str) -> ObservationSource {
    ObservationSource {
        kind: SourceKind::AgentObservation,
        id: id.to_string(),
        public_key: [0u8; 32],
    }
}

fn signed(
    keypair: &Ed25519Keypair,
    src: &str,
    tenant: &str,
    confidence: f32,
    time_ns: u64,
    parents: Vec<ruvector_agent_memory::ObservationId>,
    payload: &[u8],
) -> AtomicObservation {
    AtomicObservation::new_signed(
        source(src),
        time_ns,
        confidence,
        Tenant::new(tenant),
        parents,
        payload.to_vec(),
        keypair,
    )
    .unwrap()
}

fn config() -> ArbitrationConfig {
    ArbitrationConfig {
        now_ns: 1,
        half_life_ns: u64::MAX,
        sufficiency_threshold: 3,
        reliability: ReliabilityModel::default(),
    }
}

/// Deterministic "near-identical content" embedding: a fixed base vector
/// plus tiny per-copy noise, so every pair's cosine similarity is well
/// above any reasonable near-duplicate threshold.
fn near_duplicate_embedding(seed: u8) -> Vec<f32> {
    let mut v = vec![1.0f32, 0.3, -0.4, 0.2, 0.1, -0.1, 0.05, -0.05];
    // Perturb deterministically by `seed`, kept tiny (<=0.002 per component)
    // so pairwise cosine similarity stays > 0.999.
    for (i, x) in v.iter_mut().enumerate() {
        let jitter = ((seed as i32 * 7 + i as i32 * 13) % 5) as f32 - 2.0;
        *x += jitter * 0.001;
    }
    v
}

/// A genuinely distinct embedding per index: one-hot-ish basis vectors in an
/// 8-dim space, pairwise cosine similarity 0.0 (orthogonal).
fn independent_embedding(index: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; 8];
    v[index % 8] = 1.0;
    v
}

fn hard_merge(threshold: f32) -> ContentCorrelationConfig {
    ContentCorrelationConfig {
        similarity_threshold: threshold,
        mode: CorrelationMode::HardMerge,
    }
}

fn soft_discount(threshold: f32, strength: f32) -> ContentCorrelationConfig {
    ContentCorrelationConfig {
        similarity_threshold: threshold,
        mode: CorrelationMode::SoftDiscount { strength },
    }
}

/// The exact astroturf fixture from
/// `content_copying_yields_no_correlation_downgrade_under_default_reliability`
/// (origin + 20 parentless copies of its content, no declared derivation):
/// declared-ancestry arbitration mints 20 lineages (naive 18.0, arbitrated
/// 16.2). With content embeddings marking the 20 copies as near-duplicates,
/// hard-merge must collapse them to a single lineage.
#[test]
fn astroturf_without_declared_parents_is_detected_by_hard_merge() {
    let keypair = kp();
    let mut graph = CausalEpisodicGraph::new(Tenant::new("acme"));

    graph
        .ingest(signed(&keypair, "origin", "acme", 0.9, 0, vec![], b"rumor"))
        .unwrap();
    let copies: Vec<NodeRef> = (0..20u8)
        .map(|i| {
            NodeRef::Observation(
                graph
                    .ingest(signed(&keypair, "copier", "acme", 0.9, 1, vec![], &[i]))
                    .unwrap(),
            )
        })
        .collect();

    // Baseline: the documented blind spot, re-verified here.
    let baseline = arbitrate(&graph, &copies, &config()).unwrap();
    assert_eq!(baseline.verdict().independent_lineages(), 20);
    assert!((baseline.verdict().naive_confidence - 18.0).abs() < 1e-6);
    assert!((baseline.verdict().arbitrated_confidence - 16.2).abs() < 1e-6);

    let embeddings: BTreeMap<NodeRef, Vec<f32>> = copies
        .iter()
        .enumerate()
        .map(|(i, n)| (*n, near_duplicate_embedding(i as u8)))
        .collect();

    let outcome = arbitrate_with_content_correlation(
        &graph,
        &copies,
        &embeddings,
        &config(),
        &hard_merge(0.95),
    )
    .unwrap();
    let verdict = outcome.verdict();
    assert_eq!(
        verdict.independent_lineages(),
        1,
        "near-identical content with no declared parent must collapse to one lineage"
    );
    assert!(
        verdict.arbitrated_confidence < baseline.verdict().arbitrated_confidence,
        "content-correlated arbitration ({}) must score strictly below the \
         declared-ancestry-only baseline ({})",
        verdict.arbitrated_confidence,
        baseline.verdict().arbitrated_confidence
    );
    // Single lineage's weight is one member's confidence * reliability *
    // freshness = 0.9 * 0.9 * 1.0 = 0.81.
    assert!((verdict.arbitrated_confidence - 0.81).abs() < 1e-6);
}

/// Genuinely independent, orthogonal content must NOT be falsely merged —
/// the false-merge-rate side of the acceptance bar.
#[test]
fn independent_content_is_not_falsely_merged() {
    let keypair = kp();
    let mut graph = CausalEpisodicGraph::new(Tenant::new("acme"));

    let independent: Vec<NodeRef> = (0..8u8)
        .map(|i| {
            NodeRef::Observation(
                graph
                    .ingest(signed(
                        &keypair,
                        "witness",
                        "acme",
                        0.9,
                        10,
                        vec![],
                        &[i, 0xff],
                    ))
                    .unwrap(),
            )
        })
        .collect();

    let embeddings: BTreeMap<NodeRef, Vec<f32>> = independent
        .iter()
        .enumerate()
        .map(|(i, n)| (*n, independent_embedding(i)))
        .collect();

    let baseline = arbitrate(&graph, &independent, &config()).unwrap();
    assert_eq!(baseline.verdict().independent_lineages(), 8);

    let outcome = arbitrate_with_content_correlation(
        &graph,
        &independent,
        &embeddings,
        &config(),
        &hard_merge(0.95),
    )
    .unwrap();
    assert_eq!(
        outcome.verdict().independent_lineages(),
        8,
        "orthogonal content must not be merged by the near-duplicate pass"
    );
    assert!(
        (outcome.verdict().arbitrated_confidence - baseline.verdict().arbitrated_confidence).abs()
            < 1e-9,
        "no detected near-duplicates must mean no change to the score"
    );
}

/// `SoftDiscount` preserves the lineage count (the documented real
/// diagnostic) but still reduces the arbitrated score below the baseline.
#[test]
fn soft_discount_reduces_score_without_merging_lineages() {
    let keypair = kp();
    let mut graph = CausalEpisodicGraph::new(Tenant::new("acme"));
    graph
        .ingest(signed(&keypair, "origin", "acme", 0.9, 0, vec![], b"rumor"))
        .unwrap();
    let copies: Vec<NodeRef> = (0..20u8)
        .map(|i| {
            NodeRef::Observation(
                graph
                    .ingest(signed(&keypair, "copier", "acme", 0.9, 1, vec![], &[i]))
                    .unwrap(),
            )
        })
        .collect();
    let embeddings: BTreeMap<NodeRef, Vec<f32>> = copies
        .iter()
        .enumerate()
        .map(|(i, n)| (*n, near_duplicate_embedding(i as u8)))
        .collect();

    let baseline = arbitrate(&graph, &copies, &config()).unwrap();
    let outcome = arbitrate_with_content_correlation(
        &graph,
        &copies,
        &embeddings,
        &config(),
        &soft_discount(0.95, 0.5),
    )
    .unwrap();

    assert_eq!(
        outcome.verdict().independent_lineages(),
        20,
        "soft discount must not change the lineage count"
    );
    assert!(
        outcome.verdict().arbitrated_confidence < baseline.verdict().arbitrated_confidence,
        "soft discount ({}) must still score below baseline ({})",
        outcome.verdict().arbitrated_confidence,
        baseline.verdict().arbitrated_confidence
    );
}

/// Downgrade-only invariant: content correlation can only ever reduce (never
/// increase) the arbitrated score relative to the declared-ancestry-only
/// baseline, across both modes.
#[test]
fn downgrade_only_holds_for_both_modes() {
    let keypair = kp();
    let mut graph = CausalEpisodicGraph::new(Tenant::new("acme"));
    let memories: Vec<NodeRef> = (0..12u8)
        .map(|i| {
            NodeRef::Observation(
                graph
                    .ingest(signed(&keypair, "a", "acme", 0.7, i as u64, vec![], &[i]))
                    .unwrap(),
            )
        })
        .collect();
    // Half near-duplicate, half independent — a mixed, realistic case.
    let embeddings: BTreeMap<NodeRef, Vec<f32>> = memories
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let v = if i < 6 {
                near_duplicate_embedding(i as u8)
            } else {
                independent_embedding(i)
            };
            (*n, v)
        })
        .collect();

    let baseline = arbitrate(&graph, &memories, &config()).unwrap();
    for cfg in [hard_merge(0.95), soft_discount(0.95, 0.5)] {
        let outcome =
            arbitrate_with_content_correlation(&graph, &memories, &embeddings, &config(), &cfg)
                .unwrap();
        assert!(
            outcome.verdict().arbitrated_confidence
                <= baseline.verdict().arbitrated_confidence + 1e-9,
            "mode {:?}: {} exceeded baseline {}",
            cfg.mode,
            outcome.verdict().arbitrated_confidence,
            baseline.verdict().arbitrated_confidence
        );
        assert!(
            outcome.verdict().arbitrated_confidence <= outcome.verdict().naive_confidence + 1e-9
        );
    }
}

/// A memory with no supplied embedding is a hard error, never silently
/// treated as independent.
#[test]
fn missing_embedding_is_hard_error() {
    let keypair = kp();
    let mut graph = CausalEpisodicGraph::new(Tenant::new("acme"));
    let memories: Vec<NodeRef> = (0..3u8)
        .map(|i| {
            NodeRef::Observation(
                graph
                    .ingest(signed(&keypair, "a", "acme", 0.9, 0, vec![], &[i]))
                    .unwrap(),
            )
        })
        .collect();
    // Omit the embedding for the last memory.
    let embeddings: BTreeMap<NodeRef, Vec<f32>> = memories[..2]
        .iter()
        .enumerate()
        .map(|(i, n)| (*n, independent_embedding(i)))
        .collect();

    let err = arbitrate_with_content_correlation(
        &graph,
        &memories,
        &embeddings,
        &config(),
        &hard_merge(0.95),
    )
    .unwrap_err();
    assert!(matches!(err, ContentCorrelationError::MissingEmbedding(_)));
}

/// Config choke point: an out-of-range threshold or discount strength is
/// refused before any scoring happens.
#[test]
fn invalid_config_rejected() {
    let keypair = kp();
    let mut graph = CausalEpisodicGraph::new(Tenant::new("acme"));
    let id = graph
        .ingest(signed(&keypair, "a", "acme", 0.9, 0, vec![], b"x"))
        .unwrap();
    let memories = [NodeRef::Observation(id)];
    let embeddings: BTreeMap<NodeRef, Vec<f32>> = memories
        .iter()
        .map(|n| (*n, independent_embedding(0)))
        .collect();

    assert!(matches!(
        arbitrate_with_content_correlation(
            &graph,
            &memories,
            &embeddings,
            &config(),
            &hard_merge(f32::NAN),
        ),
        Err(ContentCorrelationError::InvalidConfig(_))
    ));
    assert!(matches!(
        arbitrate_with_content_correlation(
            &graph,
            &memories,
            &embeddings,
            &config(),
            &hard_merge(1.5),
        ),
        Err(ContentCorrelationError::InvalidConfig(_))
    ));
    assert!(matches!(
        arbitrate_with_content_correlation(
            &graph,
            &memories,
            &embeddings,
            &config(),
            &soft_discount(0.9, 1.5),
        ),
        Err(ContentCorrelationError::InvalidConfig(_))
    ));
}

/// Sufficiency re-derivation: merging the astroturf set below the
/// sufficiency threshold must report `InsufficientIndependentEvidence`
/// naming the single merged lineage's roots, exactly like the underlying
/// `arbitrate` tail-end logic would for one over-represented lineage.
#[test]
fn sufficiency_outcome_recomputed_after_merge() {
    let keypair = kp();
    let mut graph = CausalEpisodicGraph::new(Tenant::new("acme"));
    graph
        .ingest(signed(&keypair, "origin", "acme", 0.9, 0, vec![], b"rumor"))
        .unwrap();
    let copies: Vec<NodeRef> = (0..5u8)
        .map(|i| {
            NodeRef::Observation(
                graph
                    .ingest(signed(&keypair, "copier", "acme", 0.9, 1, vec![], &[i]))
                    .unwrap(),
            )
        })
        .collect();
    let embeddings: BTreeMap<NodeRef, Vec<f32>> = copies
        .iter()
        .enumerate()
        .map(|(i, n)| (*n, near_duplicate_embedding(i as u8)))
        .collect();

    let mut cfg = config();
    cfg.sufficiency_threshold = 3; // baseline (5 lineages) would be Sufficient
    let baseline = arbitrate(&graph, &copies, &cfg).unwrap();
    assert!(matches!(baseline, ArbitrationOutcome::Sufficient(_)));

    let outcome =
        arbitrate_with_content_correlation(&graph, &copies, &embeddings, &cfg, &hard_merge(0.95))
            .unwrap();
    match outcome {
        ArbitrationOutcome::InsufficientIndependentEvidence {
            verdict, required, ..
        } => {
            assert_eq!(required, 3);
            assert_eq!(verdict.independent_lineages(), 1);
        }
        other => panic!("expected InsufficientIndependentEvidence, got {other:?}"),
    }
}
