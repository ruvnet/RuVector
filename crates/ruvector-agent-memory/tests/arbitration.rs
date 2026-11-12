//! Integration tests for CAMA-pattern correlation-aware arbitration
//! (ADR-330, PIR WP27), exercised through the crate's public API only.

use std::collections::BTreeMap;

use rand::rngs::OsRng;
use ruvector_agent_memory::{
    arbitrate, ArbitrationConfig, ArbitrationError, ArbitrationOutcome, AtomicObservation,
    CausalEpisodicGraph, FusionError, NodeRef, ObservationId, ObservationSource, ReliabilityModel,
    SourceKind, Tenant,
};
use rvf_types::{ed25519_sign, Ed25519Keypair};

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
    parents: Vec<ObservationId>,
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
        now_ns: 1_000,
        half_life_ns: u64::MAX, // effectively no decay unless a test wants it
        sufficiency_threshold: 1,
        reliability: ReliabilityModel {
            default: 1.0,
            overrides: BTreeMap::new(),
        },
    }
}

/// The Wave-4 acceptance test, literal: ten memories all derived from ONE
/// atomic source arbitrate to exactly one effective evidence lineage, and
/// score strictly below ten genuinely independent memories.
#[test]
fn ten_memories_one_source_is_one_lineage() {
    let keypair = kp();
    let mut graph = CausalEpisodicGraph::new(Tenant::new("acme"));

    // One original source observation.
    let root = graph
        .ingest(signed(&keypair, "origin", "acme", 0.9, 0, vec![], b"rumor"))
        .unwrap();

    // Ten memories derived from it (some via chains, some directly).
    let mut derived: Vec<NodeRef> = Vec::new();
    let mut prev = root;
    for i in 0..10u8 {
        let parents = if i % 2 == 0 { vec![root] } else { vec![prev] };
        let id = graph
            .ingest(signed(
                &keypair,
                "agent",
                "acme",
                0.9,
                10 + i as u64,
                parents,
                &[i],
            ))
            .unwrap();
        derived.push(NodeRef::Observation(id));
        prev = id;
    }

    let outcome = arbitrate(&graph, &derived, &config()).unwrap();
    let verdict = outcome.verdict();
    assert_eq!(
        verdict.independent_lineages(),
        1,
        "ten memories from one origin must be ONE evidence lineage"
    );
    assert_eq!(verdict.lineages[0].roots, vec![root]);
    assert_eq!(verdict.lineages[0].members.len(), 10);
    // Naive vote says ~9.0; arbitrated must collapse to a single lineage ≤ 1.
    assert!((verdict.naive_confidence - 9.0).abs() < 1e-6);
    assert!(verdict.arbitrated_confidence <= 1.0 + 1e-9);

    // Contrast: ten genuinely independent memories (no shared ancestry).
    let mut graph2 = CausalEpisodicGraph::new(Tenant::new("acme"));
    let independent: Vec<NodeRef> = (0..10u8)
        .map(|i| {
            NodeRef::Observation(
                graph2
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
    let outcome2 = arbitrate(&graph2, &independent, &config()).unwrap();
    let verdict2 = outcome2.verdict();
    assert_eq!(verdict2.independent_lineages(), 10);
    assert!(
        verdict.arbitrated_confidence < verdict2.arbitrated_confidence,
        "one correlated lineage ({}) must score strictly below ten independent ones ({})",
        verdict.arbitrated_confidence,
        verdict2.arbitrated_confidence
    );
}

/// Mixed case: 10 memories tracing to 3 disjoint origins form 3 lineages,
/// and a bridging memory (ancestry in two origins) merges lineages.
#[test]
fn mixed_ancestry_clusters_and_bridges() {
    let keypair = kp();
    let mut graph = CausalEpisodicGraph::new(Tenant::new("acme"));

    let roots: Vec<ObservationId> = (0..3u8)
        .map(|i| {
            graph
                .ingest(signed(&keypair, "origin", "acme", 0.8, 0, vec![], &[i]))
                .unwrap()
        })
        .collect();

    // 4 memories from root 0, 3 from root 1, 3 from root 2.
    let mut memories: Vec<NodeRef> = Vec::new();
    for (r, count) in roots.iter().zip([4u8, 3, 3]) {
        for i in 0..count {
            let id = graph
                .ingest(signed(
                    &keypair,
                    "agent",
                    "acme",
                    0.8,
                    5,
                    vec![*r],
                    &[i, r.0[0]],
                ))
                .unwrap();
            memories.push(NodeRef::Observation(id));
        }
    }
    let outcome = arbitrate(&graph, &memories, &config()).unwrap();
    assert_eq!(outcome.verdict().independent_lineages(), 3);

    // A bridging memory citing roots 0 and 1 merges them into one lineage.
    let bridge = graph
        .ingest(signed(
            &keypair,
            "agent",
            "acme",
            0.8,
            6,
            vec![roots[0], roots[1]],
            b"bridge",
        ))
        .unwrap();
    let mut with_bridge = memories.clone();
    with_bridge.push(NodeRef::Observation(bridge));
    let outcome = arbitrate(&graph, &with_bridge, &config()).unwrap();
    assert_eq!(
        outcome.verdict().independent_lineages(),
        2,
        "shared ancestry anywhere must collapse independence"
    );
}

/// Downgrade-only: over randomized graphs, arbitrated ≤ naive, always.
#[test]
fn downgrade_only_over_random_graphs() {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xCA4A);
    let keypair = kp();

    for round in 0..20 {
        let mut graph = CausalEpisodicGraph::new(Tenant::new("t"));
        let n_roots = rng.gen_range(1..=5usize);
        let roots: Vec<ObservationId> = (0..n_roots)
            .map(|i| {
                graph
                    .ingest(signed(
                        &keypair,
                        "o",
                        "t",
                        rng.gen_range(0.0..=1.0),
                        rng.gen_range(0..500),
                        vec![],
                        &[round, i as u8],
                    ))
                    .unwrap()
            })
            .collect();
        let mut all: Vec<ObservationId> = roots.clone();
        let mut memories: Vec<NodeRef> = Vec::new();
        for i in 0..rng.gen_range(1..=20usize) {
            let n_parents = rng.gen_range(1..=2usize).min(all.len());
            let parents: Vec<ObservationId> = (0..n_parents)
                .map(|_| all[rng.gen_range(0..all.len())])
                .collect();
            let id = graph
                .ingest(signed(
                    &keypair,
                    "a",
                    "t",
                    rng.gen_range(0.0..=1.0),
                    rng.gen_range(0..1000),
                    parents,
                    &[round, 0x80, i as u8],
                ))
                .unwrap();
            all.push(id);
            memories.push(NodeRef::Observation(id));
        }
        let mut cfg = config();
        cfg.half_life_ns = rng.gen_range(1..=2000u64);
        cfg.reliability.default = rng.gen_range(0.0..=1.0);
        let verdict = arbitrate(&graph, &memories, &cfg).unwrap();
        let v = verdict.verdict();
        assert!(
            v.arbitrated_confidence <= v.naive_confidence + 1e-9,
            "round {round}: arbitrated {} exceeded naive {}",
            v.arbitrated_confidence,
            v.naive_confidence
        );
        assert!(
            v.arbitrated_confidence <= v.independent_lineages() as f64 + 1e-9,
            "arbitrated must also be bounded by the lineage count"
        );
    }
}

/// The non-finite choke point: a NaN confidence that *passes ingest* (the
/// signature covers the NaN bit pattern, and ingest checks only signatures,
/// tenancy, duplicates, and parents) must be rejected here, not compared.
#[test]
fn nan_confidence_rejected_at_choke_point() {
    let keypair = kp();
    let mut graph = CausalEpisodicGraph::new(Tenant::new("acme"));

    let mut obs = signed(&keypair, "evil", "acme", 0.5, 0, vec![], b"nan");
    obs.confidence = f32::NAN;
    // Re-sign the mutated content so it verifies and ingests cleanly.
    obs.signature = ed25519_sign(&keypair.secret_key(), &obs.id().0);
    let id = graph
        .ingest(obs)
        .expect("ingest does not range-check confidence");

    let err = arbitrate(&graph, &[NodeRef::Observation(id)], &config()).unwrap_err();
    match err {
        ArbitrationError::ConfidenceInvalid(_) => {}
        other => panic!("expected ConfidenceInvalid, got {other:?}"),
    }
}

/// Tenant isolation is inherited from the graph binding: a node from another
/// tenant's graph is unknown here — a hard error, never a silent skip.
#[test]
fn cross_tenant_node_is_unknown() {
    let keypair = kp();
    let mut acme = CausalEpisodicGraph::new(Tenant::new("acme"));
    let mut globex = CausalEpisodicGraph::new(Tenant::new("globex"));
    let a = acme
        .ingest(signed(&keypair, "a", "acme", 0.9, 0, vec![], b"a"))
        .unwrap();
    let g = globex
        .ingest(signed(&keypair, "g", "globex", 0.9, 0, vec![], b"g"))
        .unwrap();

    let err = arbitrate(
        &acme,
        &[NodeRef::Observation(a), NodeRef::Observation(g)],
        &config(),
    )
    .unwrap_err();
    match err {
        ArbitrationError::UnknownObservation(id) => assert_eq!(id, g),
        other => panic!("expected UnknownObservation, got {other:?}"),
    }
}

/// Below the sufficiency threshold the outcome is the typed insufficient
/// variant naming the over-represented roots — the CAMA active-search hook.
#[test]
fn insufficient_independent_evidence_names_over_represented_roots() {
    let keypair = kp();
    let mut graph = CausalEpisodicGraph::new(Tenant::new("acme"));
    let root = graph
        .ingest(signed(&keypair, "o", "acme", 0.9, 0, vec![], b"r"))
        .unwrap();
    let memories: Vec<NodeRef> = (0..5u8)
        .map(|i| {
            NodeRef::Observation(
                graph
                    .ingest(signed(&keypair, "a", "acme", 0.9, 1, vec![root], &[i]))
                    .unwrap(),
            )
        })
        .collect();

    let mut cfg = config();
    cfg.sufficiency_threshold = 3;
    match arbitrate(&graph, &memories, &cfg).unwrap() {
        ArbitrationOutcome::InsufficientIndependentEvidence {
            verdict,
            required,
            over_represented_roots,
        } => {
            assert_eq!(required, 3);
            assert_eq!(verdict.independent_lineages(), 1);
            assert_eq!(over_represented_roots, vec![root]);
        }
        other => panic!("expected InsufficientIndependentEvidence, got {other:?}"),
    }
}

/// Config choke point: non-finite reliability and a zero half-life are
/// refused before any scoring happens; an empty memory set is refused too.
#[test]
fn invalid_config_and_empty_set_rejected() {
    let keypair = kp();
    let mut graph = CausalEpisodicGraph::new(Tenant::new("acme"));
    let id = graph
        .ingest(signed(&keypair, "a", "acme", 0.9, 0, vec![], b"x"))
        .unwrap();
    let node = [NodeRef::Observation(id)];

    let mut cfg = config();
    cfg.reliability.default = f64::NAN;
    assert!(matches!(
        arbitrate(&graph, &node, &cfg),
        Err(ArbitrationError::InvalidConfig(_))
    ));

    let mut cfg = config();
    cfg.half_life_ns = 0;
    assert!(matches!(
        arbitrate(&graph, &node, &cfg),
        Err(ArbitrationError::InvalidConfig(_))
    ));

    assert!(matches!(
        arbitrate(&graph, &[], &config()),
        Err(ArbitrationError::EmptyMemorySet)
    ));
}

/// Locks the Known-limitation numbers in the module rustdoc so the honesty
/// text cannot silently drift from behavior: twenty agents re-asserting one
/// claim's content **without declaring a parent** mint twenty lineages, and
/// under the shipped `ReliabilityModel::default()` (0.9) score naive 18.0 vs
/// arbitrated 16.2 — a reduction that is entirely the reliability multiplier
/// and reflects zero detected correlation. The diagnostic is the lineage
/// count equalling the memory count, not the score delta.
#[test]
fn content_copying_yields_no_correlation_downgrade_under_default_reliability() {
    let keypair = kp();
    let mut graph = CausalEpisodicGraph::new(Tenant::new("acme"));

    // One genuine origin, then 20 agents asserting the same claim with no
    // declared derivation — the astroturfing shape the module cannot detect.
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

    let cfg = ArbitrationConfig {
        now_ns: 1,
        half_life_ns: u64::MAX, // freshness 1.0, isolating the reliability factor
        sufficiency_threshold: 3,
        reliability: ReliabilityModel::default(), // the SHIPPED default: 0.9
    };
    let outcome = arbitrate(&graph, &copies, &cfg).unwrap();

    // Nothing was clustered: lineages == memories is the real diagnostic.
    let verdict = outcome.verdict();
    assert_eq!(verdict.independent_lineages(), copies.len());
    assert!((verdict.naive_confidence - 18.0).abs() < 1e-6);
    assert!((verdict.arbitrated_confidence - 16.2).abs() < 1e-6);
    // And the sufficiency gate passes, which is exactly why the docs warn it
    // is not a trust gate against repeated-claim attacks.
    assert!(matches!(outcome, ArbitrationOutcome::Sufficient(_)));
}

/// `fuse()` must reject a non-finite member confidence at the source, rather
/// than folding it away. `f32::min` returns the non-NaN operand, so the old
/// `INFINITY`-seeded fold gave a lone-NaN cluster `confidence = +inf` and let
/// a NaN in `[NaN, 0.9]` vanish into `0.9` — both violating the documented
/// weakest-link `[0, 1]` contract on a public field.
#[test]
fn fuse_rejects_non_finite_member_confidence() {
    let keypair = kp();
    let mut graph = CausalEpisodicGraph::new(Tenant::new("acme"));

    let mut poisoned = signed(&keypair, "evil", "acme", 0.5, 0, vec![], b"nan");
    poisoned.confidence = f32::NAN;
    poisoned.signature = ed25519_sign(&keypair.secret_key(), &poisoned.id().0);
    let bad = graph
        .ingest(poisoned)
        .expect("ingest does not range-check confidence");
    let good = graph
        .ingest(signed(&keypair, "ok", "acme", 0.9, 0, vec![], b"ok"))
        .unwrap();

    // Lone NaN member: must not become +inf.
    assert!(matches!(
        graph.fuse(&[NodeRef::Observation(bad)], "lone"),
        Err(FusionError::MemberConfidenceInvalid(_))
    ));
    // NaN alongside a valid member: must not silently vanish into 0.9.
    assert!(matches!(
        graph.fuse(
            &[NodeRef::Observation(bad), NodeRef::Observation(good)],
            "mixed"
        ),
        Err(FusionError::MemberConfidenceInvalid(_))
    ));
    // A well-formed fusion still succeeds and keeps the weakest link.
    let ok = graph.fuse(&[NodeRef::Observation(good)], "fine").unwrap();
    assert!((graph.cluster(ok).unwrap().confidence - 0.9).abs() < 1e-6);
}

/// Clusters participate as memories: a fused cluster resolves to its atomic
/// constituents and clusters with anything sharing those origins.
#[test]
fn fused_cluster_shares_lineage_with_its_origins() {
    let keypair = kp();
    let mut graph = CausalEpisodicGraph::new(Tenant::new("acme"));
    let root = graph
        .ingest(signed(&keypair, "o", "acme", 0.9, 0, vec![], b"r"))
        .unwrap();
    let child = graph
        .ingest(signed(&keypair, "a", "acme", 0.8, 1, vec![root], b"c"))
        .unwrap();
    let cluster = graph.fuse(&[NodeRef::Observation(child)], "topic").unwrap();

    let outcome = arbitrate(
        &graph,
        &[NodeRef::Cluster(cluster), NodeRef::Observation(child)],
        &config(),
    )
    .unwrap();
    assert_eq!(outcome.verdict().independent_lineages(), 1);
}
