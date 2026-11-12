//! ADR-318 invariant tests: hash binding round-trips, mutation invalidates
//! bound views, stale-view operations fail closed, current-view operations
//! succeed, and anchor rejection aborts commits.
//!
//! Benchmark note (preprint-reproduction rule): StagedWorkspace
//! (arXiv:2608.18050) reports +8.3-12.1pp OfficeQA Pass@1 for its own
//! models/implementation. That figure is NOT asserted by any test here; the
//! program's own delta on an internal OfficeQA-equivalent task set is a
//! future `research-gate` benchmark deliverable (ADR-318 §Decision 5).

use std::cell::Cell;

use ruvector_staged_workspace::{
    ArtifactRef, ContentHash, EvidenceGrade, RejectingAnchor, StagedView, ViewKind, WorkspaceError,
    WorkspaceState, TRANSITION_DOMAIN,
};

const DOC: &str = "reports/q3.md";

// ── 1. Hash binding round-trips ─────────────────────────────────────────────

#[test]
fn commit_binds_content_hash_and_revision() {
    let mut ws = WorkspaceState::with_noop_anchor();
    let head = ws.commit(DOC, b"v1 bytes", "agent-a").unwrap();
    // The recorded hash is exactly SHA-256 of the committed bytes,
    // independently recomputable.
    assert_eq!(head.content_hash, ContentHash::of(b"v1 bytes"));
    assert_eq!(head.revision_id, 1);
    assert_eq!(ws.head(DOC), Some(&head));
}

#[test]
fn staged_view_serde_round_trip_preserves_binding() {
    let mut ws = WorkspaceState::with_noop_anchor();
    ws.commit(DOC, b"v1", "agent-a").unwrap();
    let view = ws
        .stage_view(ViewKind::ParserResult, DOC, b"parsed")
        .unwrap();
    let json = serde_json::to_string(&view).unwrap();
    let back: StagedView = serde_json::from_str(&json).unwrap();
    assert_eq!(back, view);
    // The deserialized view still validates against the live workspace.
    ws.validate(&back).unwrap();
}

#[test]
fn identical_content_commit_is_idempotent() {
    let mut ws = WorkspaceState::with_noop_anchor();
    let first = ws.commit(DOC, b"same", "agent-a").unwrap();
    let view = ws.stage_view(ViewKind::Output, DOC, b"out").unwrap();
    let second = ws.commit(DOC, b"same", "agent-a").unwrap();
    // No version change: same ref, no new transition, view stays current.
    assert_eq!(first, second);
    assert_eq!(ws.transitions().len(), 1);
    ws.validate(&view).unwrap();
}

#[test]
fn revisions_are_monotonic_and_history_is_kept() {
    let mut ws = WorkspaceState::with_noop_anchor();
    for (i, content) in [b"a".as_ref(), b"b", b"c"].into_iter().enumerate() {
        let head = ws.commit(DOC, content, "agent-a").unwrap();
        assert_eq!(head.revision_id, i as u64 + 1);
    }
    let history = ws.history(DOC).unwrap();
    assert_eq!(history.len(), 3);
    assert_eq!(history[2], ContentHash::of(b"c"));
}

// ── 2. Mutation invalidates every view bound to the old hash ────────────────

#[test]
fn mutation_invalidates_all_five_view_kinds() {
    let mut ws = WorkspaceState::with_noop_anchor();
    ws.commit(DOC, b"v1", "agent-a").unwrap();
    // Bind one view of every kind ADR-318 enumerates to revision 1.
    let views: Vec<StagedView> = ViewKind::ALL
        .iter()
        .map(|&kind| ws.stage_view(kind, DOC, b"derived").unwrap())
        .collect();
    for v in &views {
        ws.validate(v).unwrap(); // current before the mutation
    }
    // The workspace changes...
    ws.commit(DOC, b"v2", "agent-b").unwrap();
    // ...and every view bound to the old hash is now structurally stale.
    for v in &views {
        match ws.validate(v) {
            Err(WorkspaceError::StaleView(conflict)) => {
                assert_eq!(conflict.bound.content_hash, ContentHash::of(b"v1"));
                assert_eq!(conflict.head.content_hash, ContentHash::of(b"v2"));
                assert_eq!(conflict.head.revision_id, 2);
            }
            other => panic!("expected StaleView for {:?}, got {other:?}", v.kind),
        }
    }
}

// ── 3. Fail-closed proof: a stale-view operation never runs ─────────────────

#[test]
fn stale_view_operation_fails_closed_op_never_runs() {
    let mut ws = WorkspaceState::with_noop_anchor();
    ws.commit(DOC, b"v1", "agent-a").unwrap();
    let view = ws
        .stage_view(ViewKind::Approval, DOC, b"approve v1")
        .unwrap();
    ws.commit(DOC, b"v2", "agent-b").unwrap();

    let ran = Cell::new(false);
    let result = ws.execute(&view, |_| ran.set(true));
    assert!(matches!(result, Err(WorkspaceError::StaleView(_))));
    // The operation body was never invoked: stale state is rejected before
    // execution, not after.
    assert!(!ran.get(), "operation ran against a stale view");
}

#[test]
fn current_view_operation_succeeds() {
    let mut ws = WorkspaceState::with_noop_anchor();
    ws.commit(DOC, b"v1", "agent-a").unwrap();
    let view = ws
        .stage_view(ViewKind::ToolCall, DOC, b"tool input")
        .unwrap();
    let seen_rev = ws.execute(&view, |head| head.revision_id).unwrap();
    assert_eq!(seen_rev, 1);
}

#[test]
fn unknown_artifact_view_fails_closed() {
    let ws = WorkspaceState::with_noop_anchor();
    let forged = StagedView {
        view_id: 0,
        kind: ViewKind::Diff,
        artifact: ArtifactRef {
            artifact_id: "no/such/artifact".into(),
            content_hash: ContentHash::of(b"x"),
            revision_id: 1,
        },
        payload_hash: ContentHash::of(b"diff"),
    };
    assert!(matches!(
        ws.validate(&forged),
        Err(WorkspaceError::UnknownArtifact(_))
    ));
    let ran = Cell::new(false);
    assert!(ws.execute(&forged, |_| ran.set(true)).is_err());
    assert!(!ran.get());
}

#[test]
fn matching_hash_with_wrong_revision_is_rejected_as_binding_mismatch() {
    let mut ws = WorkspaceState::with_noop_anchor();
    ws.commit(DOC, b"v1", "agent-a").unwrap();
    let mut view = ws.stage_view(ViewKind::Output, DOC, b"out").unwrap();
    view.artifact.revision_id = 99; // forged revision, correct hash
    assert!(matches!(
        ws.validate(&view),
        Err(WorkspaceError::BindingMismatch(_))
    ));
}

// ── 4. ADR-312 anchoring seam ───────────────────────────────────────────────

#[test]
fn anchor_rejection_aborts_commit_no_anchor_no_transition() {
    let mut ws = WorkspaceState::new(RejectingAnchor {
        reason: "ledger offline".into(),
    });
    let err = ws.commit(DOC, b"v1", "agent-a").unwrap_err();
    assert!(matches!(err, WorkspaceError::AnchorRejected(ref m) if m == "ledger offline"));
    // The mutation was NOT applied: no head, no history, no transitions.
    assert!(ws.head(DOC).is_none());
    assert!(ws.history(DOC).is_none());
    assert!(ws.transitions().is_empty());
}

#[test]
fn transitions_carry_receipts_with_record_ids_and_evidence_grades() {
    let mut ws = WorkspaceState::with_noop_anchor();
    ws.commit(DOC, b"v1", "agent-a").unwrap();
    ws.commit(DOC, b"v2", "agent-b").unwrap();
    let transitions = ws.transitions();
    assert_eq!(transitions.len(), 2);

    let (create, create_receipt) = &transitions[0];
    assert_eq!(create.prior_hash, None);
    assert_eq!(create.new_revision, 1);
    // The workspace recomputed the hash from primary bytes.
    assert_eq!(create.evidence_grade, EvidenceGrade::Recomputed);
    // Receipt identity is the 322C-style SHA-256 of the canonical record.
    assert_eq!(create_receipt.record_id, create.record_id());
    // NoopAnchor cannot claim a signature.
    assert_eq!(create_receipt.grade, EvidenceGrade::TrustedAssertion);

    let (update, _) = &transitions[1];
    assert_eq!(update.prior_hash, Some(ContentHash::of(b"v1")));
    assert_eq!(update.prior_revision, Some(1));
    assert_eq!(update.new_hash, ContentHash::of(b"v2"));
    assert_eq!(update.new_revision, 2);
}

#[test]
fn record_id_is_deterministic_and_field_sensitive() {
    let mut ws = WorkspaceState::with_noop_anchor();
    ws.commit(DOC, b"v1", "agent-a").unwrap();
    ws.commit("other.md", b"v1", "agent-a").unwrap();
    let (a, _) = &ws.transitions()[0];
    let (b, _) = &ws.transitions()[1];
    assert_eq!(
        a.record_id(),
        a.record_id(),
        "record_id must be deterministic"
    );
    assert_ne!(
        a.record_id(),
        b.record_id(),
        "records differing in any field must differ in identity"
    );
}

#[test]
fn signing_preimage_is_domain_separated() {
    let mut ws = WorkspaceState::with_noop_anchor();
    ws.commit(DOC, b"v1", "agent-a").unwrap();
    let (record, _) = &ws.transitions()[0];
    let preimage = record.signing_preimage();
    let domain = TRANSITION_DOMAIN.as_bytes();
    // ADR-322C scheme: domain || 0x00 || canonicalBytes.
    assert_eq!(&preimage[..domain.len()], domain);
    assert_eq!(preimage[domain.len()], 0x00);
    assert_eq!(&preimage[domain.len() + 1..], &record.canonical_bytes()[..]);
}
