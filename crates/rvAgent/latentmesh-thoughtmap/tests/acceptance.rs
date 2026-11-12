//! WP23 acceptance tests (ADR-326 / RuVector#882).
//!
//! One test per acceptance criterion, exercising the crate end-to-end through
//! the public API — the same surface a real fabric would use.

use latentmesh_thoughtmap::{
    integrity_tag, CausalTag, ControlledReplacementVerifier, DeadEndPolicy, Incorporation,
    LatentMessage, LocalView, PeerId, PeerProfile, Quarantine, Query, ReasoningEngine,
    RejectReason, RejectionKind, ThoughtGraphBackend, ThoughtMap, ROOT_STEP,
};

/// Build a well-formed (correctly stamped) message.
fn signed_msg(
    from: u64,
    to: u64,
    latent: Vec<f32>,
    antecedent: latentmesh_thoughtmap::StepId,
) -> LatentMessage {
    LatentMessage {
        from: PeerId(from),
        to: PeerId(to),
        integrity: integrity_tag(PeerId(from), PeerId(to), &latent, antecedent),
        causal: CausalTag { antecedent },
        latent,
        summary: String::new(),
    }
}

fn engine_with_root() -> ReasoningEngine<ControlledReplacementVerifier> {
    let mut map = ThoughtMap::new();
    // Seed a root reasoning step (topic-A direction) plus an unrelated step so the
    // causal control population is meaningful.
    map.add_step(PeerId(0), vec![1.0, 0.0, 0.0], "root/topic-A");
    map.add_step(PeerId(0), vec![0.0, 1.0, 0.0], "topic-B");
    ReasoningEngine::new(
        map,
        ControlledReplacementVerifier::default(),
        Quarantine::default(),
        DeadEndPolicy::default(),
    )
}

// 1 — Query-dependent peer specialization with NO central role-assigner.
#[test]
fn local_peer_selection_is_decentralized_and_query_dependent() {
    // Two agents, each with only their OWN local view, select peers for the same
    // candidate set but different queries — and reach different, query-appropriate
    // choices with no shared/central router involved.
    let candidates = vec![
        PeerProfile {
            id: PeerId(10),
            skill: vec![1.0, 0.0, 0.0], // specialist A
            trust: 0.9,
            cost: 1.0,
            last_active_step: 5,
        },
        PeerProfile {
            id: PeerId(11),
            skill: vec![0.0, 1.0, 0.0], // specialist B
            trust: 0.9,
            cost: 1.0,
            last_active_step: 5,
        },
    ];

    let view = LocalView::new(PeerId(1), vec![1.0, 1.0, 1.0], 5);

    let q_a = Query::new(1, vec![1.0, 0.0, 0.0]);
    let q_b = Query::new(2, vec![0.0, 1.0, 0.0]);

    let pick_a = view.select_peer(&candidates, &q_a, &[]).unwrap();
    let pick_b = view.select_peer(&candidates, &q_b, &[]).unwrap();

    assert_eq!(
        pick_a.peer,
        PeerId(10),
        "query A → specialist A, chosen locally"
    );
    assert_eq!(
        pick_b.peer,
        PeerId(11),
        "query B → specialist B, chosen locally"
    );
    assert_ne!(
        pick_a.peer, pick_b.peer,
        "same peers, different query → different specialization (no static role)"
    );
}

// 2 — Shared thought map is directly navigable; success reinforces / failure weakens.
#[test]
fn thought_map_edges_reinforce_on_success_and_weaken_on_failure() {
    let mut map = ThoughtMap::new();
    let a = map.add_step(PeerId(1), vec![1.0, 0.0], "a");
    let b = map.add_step(PeerId(2), vec![0.9, 0.1], "b");
    let c = map.add_step(PeerId(3), vec![0.8, 0.2], "c");
    map.connect(a, b, PeerId(2), 0.5);
    map.connect(b, c, PeerId(3), 0.5);

    // A confirmed-good path a→b→c reinforces both edges.
    map.reinforce_path(&[a, b, c], 0.3);
    assert!((map.edge_weight(a, b).unwrap() - 0.8).abs() < 1e-6);
    assert!((map.edge_weight(b, c).unwrap() - 0.8).abs() < 1e-6);

    // A confirmed-bad path weakens.
    map.weaken_path(&[a, b], 0.7);
    assert!((map.edge_weight(a, b).unwrap() - 0.1).abs() < 1e-6);

    // Navigation: strongest neighbor of a is now b→c's chain start via weight.
    assert_eq!(map.neighbors(a).len(), 1);
}

// 3 — SECURITY PROOF: a latent message failing causal verification is REJECTED,
//     never incorporated.
#[test]
fn latent_message_failing_causal_verification_is_rejected_not_incorporated() {
    let mut engine = engine_with_root();
    let nodes_before = engine.map.node_count();
    let root = ROOT_STEP; // topic-A direction [1,0,0]

    // A forged message: latent orthogonal to its claimed antecedent (root/topic-A).
    // Correctly integrity-stamped, so ONLY the causal gate can catch it.
    let forged = signed_msg(99, 0, vec![0.0, 0.0, 1.0], root);

    let outcome = engine.incorporate(&forged);

    assert_eq!(
        outcome,
        Incorporation::Rejected(RejectionKind::Causal(RejectReason::Unattributable)),
        "message not attributable to its claimed cause must be rejected"
    );
    assert_eq!(
        engine.map.node_count(),
        nodes_before,
        "REJECTED message must NOT be written to the thought graph"
    );

    // Contrast: a genuine message derived from the same antecedent IS incorporated.
    let genuine = signed_msg(5, 0, vec![0.97, 0.03, 0.0], root);
    match engine.incorporate(&genuine) {
        Incorporation::Accepted { .. } => {}
        other => panic!("genuine causally-attributable message should be accepted, got {other:?}"),
    }
    assert_eq!(engine.map.node_count(), nodes_before + 1);
}

// 3n — SECURITY PROOF (NaN bypass): a NON-FINITE latent must NOT slip through
//      the causal gate. A NaN would make the gate's `<`-comparisons false and
//      fall through to Accept; it is instead rejected at the choke point, never
//      written, and the sender is struck (not left at strikes=0 to retry).
#[test]
fn nan_latent_is_rejected_not_accepted_and_strikes_sender() {
    let mut engine = engine_with_root();
    let before = engine.map.node_count();
    let attacker = PeerId(77);

    // NaN latent that claims a REAL antecedent (root). Correctly integrity-stamped
    // so only the finiteness guard can catch it.
    let nan = signed_msg(attacker.0, 0, vec![f32::NAN, 0.1, 0.0], ROOT_STEP);
    assert_eq!(
        engine.incorporate(&nan),
        Incorporation::Rejected(RejectionKind::NonFinite),
        "a NaN latent must be REJECTED, never fall through to Accept"
    );
    assert_eq!(
        engine.map.node_count(),
        before,
        "rejected NaN message must NOT be written to the thought graph"
    );
    assert!(
        engine.quarantine.strikes(attacker) > 0,
        "malformed NaN message must strike the sender, not leave it at strikes=0"
    );

    // A repeat offender quarantines instead of retrying forever.
    let nan2 = signed_msg(attacker.0, 0, vec![f32::NAN, 0.0, 0.0], ROOT_STEP);
    let _ = engine.incorporate(&nan2);
    assert!(
        engine.quarantine.is_quarantined(attacker),
        "a repeat non-finite offender must eventually quarantine"
    );
}

// 3i — Same for +inf / -inf latents.
#[test]
fn infinite_latent_is_rejected() {
    let mut engine = engine_with_root();
    let before = engine.map.node_count();
    for bad in [f32::INFINITY, f32::NEG_INFINITY] {
        let msg = signed_msg(78, 0, vec![bad, 0.0, 0.0], ROOT_STEP);
        assert_eq!(
            engine.incorporate(&msg),
            Incorporation::Rejected(RejectionKind::NonFinite)
        );
    }
    assert_eq!(
        engine.map.node_count(),
        before,
        "no inf latent may be stored"
    );
}

// 3c — CASCADE PROOF: a rejected non-finite injection must NOT poison the
//      map-wide control. A genuinely-spurious message stays Rejected(Spurious)
//      afterwards (the specificity check still fires — control is not NaN).
#[test]
fn nonfinite_injection_does_not_poison_mapwide_specificity_control() {
    // A map whose nodes share one direction, so a message aligned with it is
    // genuinely spurious (explained equally by any node).
    let mut map = ThoughtMap::new();
    let a = map.add_step(PeerId(0), vec![1.0, 1.0, 1.0], "a");
    map.add_step(PeerId(0), vec![1.0, 1.0, 1.0], "b");
    map.add_step(PeerId(0), vec![1.0, 1.0, 1.0], "c");
    let mut engine = ReasoningEngine::new(
        map,
        ControlledReplacementVerifier::default(),
        Quarantine::default(),
        DeadEndPolicy::default(),
    );

    // Baseline: a spurious message is correctly Rejected(Spurious).
    let spurious_before = signed_msg(6, 0, vec![1.0, 1.0, 1.0], a);
    assert_eq!(
        engine.incorporate(&spurious_before),
        Incorporation::Rejected(RejectionKind::Causal(RejectReason::Spurious))
    );

    // Attempt a NaN injection — rejected, nothing written.
    let nan = signed_msg(50, 0, vec![f32::NAN, 0.0, 0.0], a);
    assert_eq!(
        engine.incorporate(&nan),
        Incorporation::Rejected(RejectionKind::NonFinite)
    );
    assert_eq!(engine.map.node_count(), 3, "NaN injection wrote nothing");

    // CASCADE CHECK: the spurious message is STILL Rejected(Spurious). If the NaN
    // had poisoned the control (mean → NaN → control_attribution NaN), the
    // specificity check would silently pass and this would wrongly Accept.
    let spurious_after = signed_msg(7, 0, vec![1.0, 1.0, 1.0], a);
    assert_eq!(
        engine.incorporate(&spurious_after),
        Incorporation::Rejected(RejectionKind::Causal(RejectReason::Spurious)),
        "map-wide specificity control must survive a rejected non-finite injection"
    );
}

// 3b — A tampered (bad-integrity) message is rejected before the causal gate.
#[test]
fn tampered_message_is_rejected_on_integrity() {
    let mut engine = engine_with_root();
    let before = engine.map.node_count();
    let mut msg = signed_msg(99, 0, vec![0.97, 0.03, 0.0], ROOT_STEP);
    msg.latent[0] = 0.5; // mutate payload after stamping → integrity mismatch
    assert_eq!(
        engine.incorporate(&msg),
        Incorporation::Rejected(RejectionKind::Integrity)
    );
    assert_eq!(engine.map.node_count(), before);
}

// 4 — Dead-end → topology change CONTINUES from current state (NOT restart),
//     and the choice is swappable. (UNVERIFIED design choice, ADR-326 §6.)
#[test]
fn dead_end_continues_from_state_and_is_swappable() {
    let mut engine = engine_with_root();
    // Add a step d that has no productive outgoing edge → dead end.
    let d = engine.map.add_step(PeerId(3), vec![0.5, 0.5, 0.0], "stuck");
    assert!(engine.map.is_dead_end(d));

    let view = LocalView::new(PeerId(1), vec![1.0, 1.0, 1.0], 10);
    let query = Query::new(1, vec![1.0, 0.0, 0.0]);
    let candidates = vec![PeerProfile {
        id: PeerId(20),
        skill: vec![1.0, 0.0, 0.0],
        trust: 0.9,
        cost: 1.0,
        last_active_step: 10,
    }];

    // Default policy = ContinueWithTopologyChange (the unverified design choice).
    let change = engine
        .handle_dead_end(d, &view, &candidates, &query, &[], PeerId(1))
        .expect("non-quarantined trigger yields a change");
    assert!(change.continued, "continue, NOT restart");
    assert_eq!(
        change.resume_from, d,
        "resume from current state, not the root"
    );
    assert_eq!(change.new_peer, Some(PeerId(20)), "re-routed to a new peer");

    // Swap the policy → Restart resumes from ROOT and discards progress.
    engine.policy = DeadEndPolicy::Restart;
    let restart = engine
        .handle_dead_end(d, &view, &candidates, &query, &[], PeerId(1))
        .unwrap();
    assert!(!restart.continued);
    assert_eq!(restart.resume_from, ROOT_STEP);
}

// 5 — A misbehaving peer is dampened/quarantined and CANNOT churn the thought map.
#[test]
fn misbehaving_peer_is_quarantined_and_cannot_churn_topology() {
    let mut engine = engine_with_root();
    let bad = PeerId(66);

    // The bad peer sends three forged (unattributable) messages. Each is rejected
    // and accrues a strike; after threshold it is quarantined.
    for _ in 0..3 {
        let forged = signed_msg(bad.0, 0, vec![0.0, 0.0, 1.0], ROOT_STEP);
        assert!(matches!(
            engine.incorporate(&forged),
            Incorporation::Rejected(RejectionKind::Causal(_))
        ));
    }
    assert!(
        engine.quarantine.is_quarantined(bad),
        "sustained anomalies quarantine the peer"
    );

    let nodes_after_rejects = engine.map.node_count();

    // Now the quarantined peer sends a *well-formed, causally valid* message.
    // It must be DAMPENED (not incorporated) purely because it is quarantined —
    // a bad actor cannot buy its way back into the map with one good message.
    let good_but_quarantined = signed_msg(bad.0, 0, vec![0.97, 0.03, 0.0], ROOT_STEP);
    assert_eq!(
        engine.incorporate(&good_but_quarantined),
        Incorporation::Dampened
    );
    assert_eq!(
        engine.map.node_count(),
        nodes_after_rejects,
        "quarantined peer cannot add nodes"
    );

    // And it cannot trigger a topology change (no spurious churn).
    let view = LocalView::new(PeerId(1), vec![1.0, 1.0, 1.0], 10);
    let query = Query::new(1, vec![1.0, 0.0, 0.0]);
    let churn = engine.handle_dead_end(ROOT_STEP, &view, &[], &query, &[], bad);
    assert!(
        churn.is_none(),
        "quarantined peer cannot trigger topology churn"
    );
}

// 6 — Topology changes remain attributable (every change names its cause + trigger).
#[test]
fn topology_changes_are_attributable() {
    let mut engine = engine_with_root();
    let d = engine.map.add_step(PeerId(3), vec![0.5, 0.5, 0.0], "stuck");
    let view = LocalView::new(PeerId(1), vec![1.0, 1.0, 1.0], 10);
    let query = Query::new(1, vec![1.0, 0.0, 0.0]);

    let change = engine
        .handle_dead_end(d, &view, &[], &query, &[], PeerId(1))
        .unwrap();
    // Attribution: cause is machine-readable and the trigger is named.
    assert_eq!(change.cause, latentmesh_thoughtmap::TopologyCause::DeadEnd);
    assert_eq!(change.triggered_by, PeerId(1));
}

// Extra — the peer-selection choke ties into quarantine: re-routing skips
// quarantined peers, so churn can't be redirected onto a bad actor either.
#[test]
fn rerouting_skips_quarantined_candidates() {
    let mut engine = engine_with_root();
    let bad = PeerId(66);
    // Quarantine `bad` via an integrity failure (weight 3).
    let mut tampered = signed_msg(bad.0, 0, vec![0.97, 0.03, 0.0], ROOT_STEP);
    tampered.latent[0] = 0.1;
    let _ = engine.incorporate(&tampered);
    assert!(engine.quarantine.is_quarantined(bad));

    let d = engine.map.add_step(PeerId(3), vec![0.5, 0.5, 0.0], "stuck");
    let view = LocalView::new(PeerId(1), vec![1.0, 1.0, 1.0], 10);
    let query = Query::new(1, vec![1.0, 0.0, 0.0]);
    let candidates = vec![
        PeerProfile {
            id: bad,
            skill: vec![1.0, 0.0, 0.0],
            trust: 0.9,
            cost: 1.0,
            last_active_step: 10,
        },
        PeerProfile {
            id: PeerId(21),
            skill: vec![1.0, 0.0, 0.0],
            trust: 0.9,
            cost: 1.0,
            last_active_step: 10,
        },
    ];
    let change = engine
        .handle_dead_end(d, &view, &candidates, &query, &[], PeerId(1))
        .unwrap();
    assert_eq!(
        change.new_peer,
        Some(PeerId(21)),
        "re-route skips the quarantined peer"
    );
}
