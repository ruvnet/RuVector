//! Integration tests for the ADR-320 (PIR WP18) cross-source causal fusion
//! layer: AtomicObservation construction + signature round-trip, multi-source
//! fusion into the causal episodic graph, provenance resolution back to atomic
//! sources, confidence/tenant carry-through, tampered-observation rejection,
//! and composition with the WP4 (ADR-307) transactional ledger.
//!
//! Pattern informed by MemFuse (arXiv:2608.18704, `Darwin-Agent/Mi-Memory`);
//! explicitly distinct from the unrelated `memfuse/memfuse` OSS project.

use rand::rngs::OsRng;
use ruvector_agent_memory::ops::{LedgerState, MemoryWitnessLog};
use ruvector_agent_memory::{
    AlwaysAdmitGate, AtomicObservation, CausalEpisodicGraph, FusionError, NodeRef, ObservationId,
    ObservationSource, SourceKind, Tenant, TransactionalLedger,
};
use rvf_types::Ed25519Keypair;

/// Sign an observation from a fresh per-source keypair (each source authenticates
/// under its own key, as heterogeneous real sources would).
fn observe(
    kind: SourceKind,
    src_id: &str,
    tenant: &str,
    confidence: f32,
    parents: Vec<ObservationId>,
    payload: &[u8],
) -> AtomicObservation {
    let keypair = Ed25519Keypair::generate(&mut OsRng);
    AtomicObservation::new_signed(
        ObservationSource {
            kind,
            id: src_id.to_string(),
            public_key: [0u8; 32], // overwritten from the keypair on signing
        },
        1_700_000_000_000,
        confidence,
        Tenant::new(tenant),
        parents,
        payload.to_vec(),
        &keypair,
    )
    .expect("valid observation")
}

#[test]
fn atomic_observation_signature_round_trip() {
    let obs = observe(
        SourceKind::AgentObservation,
        "agent-alpha",
        "acme",
        0.92,
        vec![],
        b"door opened at 14:03",
    );
    assert!(obs.verify(), "a freshly signed observation must verify");

    // Round-tripping through the content address is stable.
    let id_a = obs.id();
    let id_b = obs.id();
    assert_eq!(id_a, id_b);
}

#[test]
fn multi_source_fusion_into_graph() {
    // Three heterogeneous source kinds converging into one tenant's graph.
    let mut graph = CausalEpisodicGraph::new(Tenant::new("acme"));
    let rf = observe(
        SourceKind::RuViewRf,
        "rf-antenna-2",
        "acme",
        0.80,
        vec![],
        b"rf-burst",
    );
    let net = observe(
        SourceKind::NetworkTelemetry,
        "netprobe-7",
        "acme",
        0.95,
        vec![],
        b"tcp-syn-spike",
    );
    let user = observe(
        SourceKind::UserActivity,
        "session-42",
        "acme",
        0.70,
        vec![],
        b"login-attempt",
    );

    let rf_id = graph.ingest(rf).unwrap();
    let net_id = graph.ingest(net).unwrap();
    let user_id = graph.ingest(user).unwrap();
    assert_eq!(graph.observation_count(), 3);

    let cluster = graph
        .fuse(
            &[
                NodeRef::Observation(rf_id),
                NodeRef::Observation(net_id),
                NodeRef::Observation(user_id),
            ],
            "correlated-access-event",
        )
        .unwrap();
    assert_eq!(graph.cluster_count(), 1);
    assert_eq!(graph.cluster(cluster).unwrap().members.len(), 3);
}

/// THE provenance-resolution proof (ADR-320's load-bearing guarantee): a
/// derived, fused node resolves back to exactly the atomic source observations
/// it was built from — including transitively through a cluster-of-clusters —
/// and each resolved id looks up the original source evidence.
#[test]
fn provenance_resolves_derived_node_to_atomic_sources() {
    let mut graph = CausalEpisodicGraph::new(Tenant::new("acme"));

    let rf = observe(
        SourceKind::RuViewRf,
        "rf-1",
        "acme",
        0.6,
        vec![],
        b"rf-evidence",
    );
    let net = observe(
        SourceKind::NetworkTelemetry,
        "net-1",
        "acme",
        0.9,
        vec![],
        b"net-evidence",
    );
    let user = observe(
        SourceKind::UserActivity,
        "user-1",
        "acme",
        0.8,
        vec![],
        b"user-evidence",
    );

    let rf_id = graph.ingest(rf).unwrap();
    let net_id = graph.ingest(net).unwrap();
    let user_id = graph.ingest(user).unwrap();

    // Cluster A fuses two event-layer sources; cluster B fuses cluster A with a
    // third source — so B is a *derived* node two hops from some atomic sources.
    let cluster_a = graph
        .fuse(
            &[NodeRef::Observation(rf_id), NodeRef::Observation(net_id)],
            "rf+net",
        )
        .unwrap();
    let cluster_b = graph
        .fuse(
            &[NodeRef::Cluster(cluster_a), NodeRef::Observation(user_id)],
            "everything",
        )
        .unwrap();

    // Provenance of the derived node B is exactly the three atomic sources.
    let provenance = graph
        .resolve_provenance(NodeRef::Cluster(cluster_b))
        .unwrap();
    let expected: std::collections::BTreeSet<ObservationId> =
        [rf_id, net_id, user_id].into_iter().collect();
    assert_eq!(
        provenance, expected,
        "derived node must resolve to exactly its atomic sources"
    );

    // And every resolved id traces to the original source evidence.
    for id in &provenance {
        let atomic = graph.observation(*id).expect("resolved id must be present");
        assert!(atomic.verify(), "resolved atomic source stays authentic");
    }
    // Spot-check one payload round-trips to the original evidence bytes.
    assert_eq!(graph.observation(rf_id).unwrap().payload, b"rf-evidence");

    // Provenance of an intermediate cluster is its two atomic sources only.
    let prov_a = graph
        .resolve_provenance(NodeRef::Cluster(cluster_a))
        .unwrap();
    assert_eq!(prov_a, [rf_id, net_id].into_iter().collect());
}

#[test]
fn confidence_and_tenant_carried_through_fusion() {
    let mut graph = CausalEpisodicGraph::new(Tenant::new("acme"));
    let a = observe(SourceKind::RuViewRf, "rf", "acme", 0.9, vec![], b"a");
    let b = observe(
        SourceKind::NetworkTelemetry,
        "net",
        "acme",
        0.4,
        vec![],
        b"b",
    );
    let c = observe(SourceKind::UserActivity, "user", "acme", 0.7, vec![], b"c");

    let a_id = graph.ingest(a).unwrap();
    let b_id = graph.ingest(b).unwrap();
    let c_id = graph.ingest(c).unwrap();

    let cluster = graph
        .fuse(
            &[
                NodeRef::Observation(a_id),
                NodeRef::Observation(b_id),
                NodeRef::Observation(c_id),
            ],
            "mixed",
        )
        .unwrap();
    let fused = graph.cluster(cluster).unwrap();

    // Weakest-link confidence carries through fusion.
    assert!((fused.confidence - 0.4).abs() < 1e-6);
    // Tenant is preserved as the graph's isolation boundary.
    assert_eq!(fused.tenant, Tenant::new("acme"));
}

#[test]
fn tampered_observation_is_rejected_at_ingest() {
    let mut graph = CausalEpisodicGraph::new(Tenant::new("acme"));
    let mut obs = observe(
        SourceKind::AgentObservation,
        "a",
        "acme",
        0.9,
        vec![],
        b"authentic",
    );
    assert!(obs.verify());

    // A compromised or malfunctioning source flips the evidence after signing.
    obs.payload = b"forged".to_vec();

    match graph.ingest(obs) {
        Err(FusionError::SignatureInvalid(_)) => {}
        other => panic!("tampered observation must be rejected, got {other:?}"),
    }
    assert_eq!(
        graph.observation_count(),
        0,
        "nothing tampered enters the graph"
    );
}

/// ADR-320 §3 transactional integrity: an observation entering memory is a
/// governed transition through the WP4 (ADR-307 TARL) ledger, emitting a
/// witness record, with causal parents mapped to ledger dependency edges.
#[test]
fn governed_ingest_composes_with_wp4_ledger() {
    let mut graph = CausalEpisodicGraph::new(Tenant::new("acme"));
    let mut ledger =
        TransactionalLedger::new(MemoryWitnessLog::default(), AlwaysAdmitGate::default());

    let parent = observe(
        SourceKind::RuViewRf,
        "rf",
        "acme",
        0.9,
        vec![],
        b"parent-evidence",
    );
    let (parent_id, parent_entry) = graph
        .ingest_governed(parent, &mut ledger, "fusion-layer")
        .unwrap();

    // A child observation citing the parent as causal lineage.
    let child = observe(
        SourceKind::AgentObservation,
        "agent",
        "acme",
        0.85,
        vec![parent_id],
        b"child-evidence",
    );
    let (child_id, child_entry) = graph
        .ingest_governed(child, &mut ledger, "fusion-layer")
        .unwrap();

    // Both admissions are governed ledger entries in Pending (each was
    // witnessed before it mutated memory — "no witness, no mutation").
    assert_eq!(ledger.state_of(parent_entry), Some(LedgerState::Pending));
    assert_eq!(ledger.state_of(child_entry), Some(LedgerState::Pending));
    assert_eq!(graph.ledger_id(child_id), Some(child_entry));

    // The causal parent became a ledger dependency edge.
    assert_eq!(
        ledger.entry(child_entry).unwrap().depends_on,
        vec![parent_entry]
    );

    // Witness chain over the governed transitions is intact and non-empty.
    assert!(ledger.witness_sink().verify_chain());
    assert!(!ledger.witness_sink().records.is_empty());

    // Provenance still resolves through the governed graph.
    let cluster = graph
        .fuse(
            &[
                NodeRef::Observation(parent_id),
                NodeRef::Observation(child_id),
            ],
            "lineage",
        )
        .unwrap();
    let prov = graph.resolve_provenance(NodeRef::Cluster(cluster)).unwrap();
    assert_eq!(prov, [parent_id, child_id].into_iter().collect());
}

/// Depth-safety (security audit #874, MEDIUM): a deeply nested linear fuse
/// chain (`C1=fuse([obs])`, `C2=fuse([C1])`, … `CN`) must resolve without
/// overflowing the stack. The old recursive `collect_provenance` aborted the
/// process (SIGABRT) at this depth; the iterative worklist bounds depth by heap.
#[test]
fn deep_fuse_chain_provenance_is_depth_safe() {
    let mut graph = CausalEpisodicGraph::new(Tenant::new("acme"));
    let obs = observe(
        SourceKind::RuViewRf,
        "rf",
        "acme",
        0.5,
        vec![],
        b"root-evidence",
    );
    let obs_id = graph.ingest(obs).unwrap();

    // 200_000 levels deep — far past what the recursive version could survive,
    // but linear and fast iteratively.
    const DEPTH: usize = 200_000;
    let mut current = graph
        .fuse(&[NodeRef::Observation(obs_id)], "level-0")
        .unwrap();
    for level in 1..DEPTH {
        current = graph
            .fuse(&[NodeRef::Cluster(current)], format!("level-{level}"))
            .unwrap();
    }

    // The load-bearing provenance query resolves to exactly the one atomic
    // source, without aborting.
    let provenance = graph.resolve_provenance(NodeRef::Cluster(current)).unwrap();
    assert_eq!(provenance, [obs_id].into_iter().collect());
    // Weakest-link confidence propagated unchanged through the whole chain.
    assert!((graph.cluster(current).unwrap().confidence - 0.5).abs() < 1e-6);
}

#[test]
fn governed_ingest_rejects_ungoverned_parent() {
    let mut graph = CausalEpisodicGraph::new(Tenant::new("acme"));
    let mut ledger =
        TransactionalLedger::new(MemoryWitnessLog::default(), AlwaysAdmitGate::default());

    // Parent admitted on the pure path (no ledger entry)...
    let parent = observe(SourceKind::RuViewRf, "rf", "acme", 0.9, vec![], b"p");
    let parent_id = graph.ingest(parent).unwrap();

    // ...so a governed child citing it has no ledger dependency to record.
    let child = observe(
        SourceKind::AgentObservation,
        "agent",
        "acme",
        0.8,
        vec![parent_id],
        b"c",
    );
    match graph.ingest_governed(child, &mut ledger, "fusion-layer") {
        Err(FusionError::ParentNotGoverned(id)) => assert_eq!(id, parent_id),
        other => panic!("expected ParentNotGoverned, got {other:?}"),
    }
}
