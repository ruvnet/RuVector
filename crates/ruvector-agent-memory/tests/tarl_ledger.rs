//! Integration tests for the TARL (Transaction-Aware Reliable Ledgers)
//! transactional memory ledger (ADR-307, PIR WP4, arXiv:2608.03699).

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use ruvector_agent_memory::{
    replay_history, AcceptanceReceipt, AlwaysAdmitGate, LedgerError, LedgerState,
    LedgerWitnessRecord, MemoryOp, MemoryWitnessLog, ProofGate, TransactionalLedger,
    TransitionKind, WitnessSink,
};

/// Gate that denies everything — models the proof gate refusing a poisoned
/// acceptance.
struct DenyAllGate;

impl ProofGate for DenyAllGate {
    fn admit(&mut self, _entry_id: u64, _content: &[u8]) -> Result<AcceptanceReceipt, LedgerError> {
        Err(LedgerError::GateDenied("deny-all test gate".into()))
    }
}

fn witnessed_ledger() -> TransactionalLedger<MemoryWitnessLog, AlwaysAdmitGate> {
    TransactionalLedger::new(MemoryWitnessLog::default(), AlwaysAdmitGate::default())
}

// ── Five-operation state machine ─────────────────────────────────────────────

#[test]
fn five_op_state_machine_transitions() {
    let mut ledger = witnessed_ledger();

    // add -> Pending
    let a = ledger
        .add("the sky is blue", &[], "agent-1", "observed")
        .unwrap();
    assert_eq!(ledger.state_of(a), Some(LedgerState::Pending));

    // accept -> Accepted (through the gate)
    ledger.accept(a, "agent-1", "verified").unwrap();
    assert_eq!(ledger.state_of(a), Some(LedgerState::Accepted));

    // ignore -> no entry retained, but the decision is in history
    ledger
        .ignore("spam statement", "agent-1", "irrelevant")
        .unwrap();
    assert_eq!(ledger.len(), 1);
    let last = ledger.history().last().unwrap();
    assert_eq!(last.kind, TransitionKind::Op(MemoryOp::Ignore));
    assert_eq!(last.entry_id, None);

    // defer-for-verification: Accepted -> Pending
    ledger
        .defer_for_verification(a, "agent-2", "source challenged")
        .unwrap();
    assert_eq!(ledger.state_of(a), Some(LedgerState::Pending));

    // reject-unreliable: Pending -> Rejected
    ledger
        .reject_unreliable(a, "agent-2", "source retracted")
        .unwrap();
    assert_eq!(ledger.state_of(a), Some(LedgerState::Rejected));

    // revise-outdated-belief on an accepted entry supersedes it
    let b = ledger
        .add("water boils at 100C", &[], "agent-1", "observed")
        .unwrap();
    ledger.accept(b, "agent-1", "verified").unwrap();
    let b2 = ledger
        .revise_outdated_belief(
            b,
            "water boils at 100C at sea level",
            "agent-1",
            "more precise",
        )
        .unwrap();
    assert_eq!(ledger.state_of(b), Some(LedgerState::Rejected));
    assert_eq!(ledger.state_of(b2), Some(LedgerState::Pending));
    assert_eq!(
        ledger.entry(b2).unwrap().content,
        "water boils at 100C at sea level"
    );

    // Terminal-state guards: no op may resurrect or re-kill a Rejected entry.
    assert!(matches!(
        ledger.accept(b, "agent-1", "no"),
        Err(LedgerError::InvalidTransition { .. })
    ));
    assert!(matches!(
        ledger.reject_unreliable(b, "agent-1", "no"),
        Err(LedgerError::InvalidTransition { .. })
    ));
    assert!(matches!(
        ledger.defer_for_verification(b, "agent-1", "no"),
        Err(LedgerError::InvalidTransition { .. })
    ));
    assert!(matches!(
        ledger.revise_outdated_belief(b, "x", "agent-1", "no"),
        Err(LedgerError::InvalidTransition { .. })
    ));

    // The three ledgers are distinctly queryable.
    assert_eq!(ledger.entries_in(LedgerState::Pending).len(), 1); // b2
    assert_eq!(ledger.entries_in(LedgerState::Rejected).len(), 2); // a, b
    assert_eq!(ledger.entries_in(LedgerState::Accepted).len(), 0);
}

// ── Poisoning containment ────────────────────────────────────────────────────

#[test]
fn poisoning_containment_reject_demotes_dependents() {
    let mut ledger = witnessed_ledger();

    // base <- d1 <- d3, base <- d2 ; all accepted.
    let base = ledger.add("base fact", &[], "a", "obs").unwrap();
    ledger.accept(base, "a", "ok").unwrap();
    let d1 = ledger.add("inference 1", &[base], "a", "derived").unwrap();
    let d2 = ledger.add("inference 2", &[base], "a", "derived").unwrap();
    ledger.accept(d1, "a", "ok").unwrap();
    ledger.accept(d2, "a", "ok").unwrap();
    let d3 = ledger.add("inference 3", &[d1], "a", "derived").unwrap();
    ledger.accept(d3, "a", "ok").unwrap();

    // Unrelated accepted entry must be untouched.
    let other = ledger.add("unrelated", &[], "a", "obs").unwrap();
    ledger.accept(other, "a", "ok").unwrap();

    // Reject the base: every transitive dependent leaves Accepted.
    ledger
        .reject_unreliable(base, "auditor", "poisoned source")
        .unwrap();

    assert_eq!(ledger.state_of(base), Some(LedgerState::Rejected));
    assert_eq!(ledger.state_of(d1), Some(LedgerState::Pending));
    assert_eq!(ledger.state_of(d2), Some(LedgerState::Pending));
    assert_eq!(
        ledger.state_of(d3),
        Some(LedgerState::Pending),
        "transitive dependent demoted"
    );
    assert_eq!(ledger.state_of(other), Some(LedgerState::Accepted));

    // The cascade is provenance-tracked as DependentDemotion records.
    let demotions: Vec<_> = ledger
        .history()
        .iter()
        .filter(|r| r.kind == TransitionKind::DependentDemotion)
        .collect();
    assert_eq!(demotions.len(), 3);

    // A demoted dependent cannot be re-accepted while its premise is out.
    assert!(matches!(
        ledger.accept(d1, "a", "retry"),
        Err(LedgerError::DependencyNotAccepted { .. })
    ));
}

#[test]
fn poisoning_containment_revise_demotes_dependents() {
    let mut ledger = witnessed_ledger();
    let base = ledger.add("v1 of belief", &[], "a", "obs").unwrap();
    ledger.accept(base, "a", "ok").unwrap();
    let dep = ledger.add("built on v1", &[base], "a", "derived").unwrap();
    ledger.accept(dep, "a", "ok").unwrap();

    let base2 = ledger
        .revise_outdated_belief(base, "v2 of belief", "a", "contradicting but credible")
        .unwrap();

    assert_eq!(ledger.state_of(base), Some(LedgerState::Rejected));
    assert_eq!(
        ledger.state_of(dep),
        Some(LedgerState::Pending),
        "dependent demoted by revise"
    );
    assert_eq!(ledger.state_of(base2), Some(LedgerState::Pending));
}

/// Poisoning-style adversarial check: rejected and deferred statements never
/// reach the accepted ledger, and a denied gate keeps entries out.
#[test]
fn rejected_and_deferred_never_reach_accepted() {
    let mut ledger = TransactionalLedger::new(MemoryWitnessLog::default(), DenyAllGate);
    let e = ledger
        .add("attacker statement", &[], "attacker", "injected")
        .unwrap();

    // Gate denies: the acceptance fails and the entry stays Pending.
    assert!(matches!(
        ledger.accept(e, "attacker", "force in"),
        Err(LedgerError::GateDenied(_))
    ));
    assert_eq!(ledger.state_of(e), Some(LedgerState::Pending));
    assert!(ledger.entries_in(LedgerState::Accepted).is_empty());

    ledger
        .defer_for_verification(e, "auditor", "suspicious")
        .unwrap();
    assert_eq!(ledger.state_of(e), Some(LedgerState::Pending));

    ledger
        .reject_unreliable(e, "auditor", "confirmed poisoning")
        .unwrap();
    assert_eq!(ledger.state_of(e), Some(LedgerState::Rejected));
    assert!(ledger.entries_in(LedgerState::Accepted).is_empty());

    // A denied acceptance is not a committed transition: no Accept record
    // exists anywhere in history.
    assert!(ledger
        .history()
        .iter()
        .all(|r| r.kind != TransitionKind::Accept));
}

// ── Defer-for-verification round-trip ────────────────────────────────────────

#[test]
fn defer_for_verification_round_trip() {
    let mut ledger = witnessed_ledger();
    let e = ledger.add("needs checking", &[], "a", "obs").unwrap();

    // Pending -> (defer) -> Pending -> (accept) -> Accepted
    ledger
        .defer_for_verification(e, "a", "await source")
        .unwrap();
    assert_eq!(ledger.state_of(e), Some(LedgerState::Pending));
    let r1 = ledger.accept(e, "a", "source arrived").unwrap();
    assert_eq!(ledger.state_of(e), Some(LedgerState::Accepted));

    // Accepted -> (defer) -> Pending -> (accept) -> Accepted again
    ledger.defer_for_verification(e, "b", "re-check").unwrap();
    assert_eq!(ledger.state_of(e), Some(LedgerState::Pending));
    let r2 = ledger.accept(e, "b", "re-verified").unwrap();
    assert_eq!(ledger.state_of(e), Some(LedgerState::Accepted));

    // Each acceptance passed the gate independently.
    assert!(r2.sequence > r1.sequence);
    // Both receipts are retained in provenance.
    let accepts: Vec<_> = ledger
        .history()
        .iter()
        .filter(|r| r.kind == TransitionKind::Accept)
        .collect();
    assert_eq!(accepts.len(), 2);
    assert!(accepts.iter().all(|r| r.receipt.is_some()));
}

// ── Witness emission ─────────────────────────────────────────────────────────

#[test]
fn witness_record_emitted_per_transition() {
    let mut ledger = witnessed_ledger();
    let a = ledger.add("fact", &[], "a", "obs").unwrap();
    ledger.accept(a, "a", "ok").unwrap();
    let b = ledger.add("derived", &[a], "a", "derived").unwrap();
    ledger.accept(b, "a", "ok").unwrap();
    ledger.ignore("noise", "a", "irrelevant").unwrap();
    ledger
        .reject_unreliable(a, "auditor", "bad source")
        .unwrap(); // + 1 demotion

    let log = ledger.witness_sink();
    // Exactly one witness record per committed transition.
    assert_eq!(log.records.len(), ledger.history().len());
    assert_eq!(log.records.len(), 7);

    // The persisted chain verifies independently of the ledger...
    assert!(log.verify_chain());
    // ...sequences are dense and monotonic...
    for (i, r) in log.records.iter().enumerate() {
        assert_eq!(r.sequence, i as u64);
    }
    // ...and an injected tamper is detected.
    let mut tampered = log.clone();
    tampered.records[3].payload ^= 1;
    assert!(
        !tampered.verify_chain(),
        "tampered witness chain must fail verification"
    );
    let mut truncated = log.clone();
    truncated.records.remove(2);
    assert!(
        !truncated.verify_chain(),
        "deleted witness record must break the chain"
    );
}

#[test]
fn tail_truncation_detected_by_head_commitment() {
    let mut ledger = witnessed_ledger();
    let a = ledger.add("fact", &[], "a", "obs").unwrap();
    ledger.accept(a, "a", "ok").unwrap();
    let b = ledger.add("derived", &[a], "a", "derived").unwrap();
    ledger.accept(b, "a", "ok").unwrap();
    ledger
        .reject_unreliable(a, "auditor", "bad source")
        .unwrap(); // rejection + 1 demotion — the records a roll-back would target

    let log = ledger.witness_sink();
    assert!(log.verify_chain());
    let (count, head) = log.head_commitment();
    assert_eq!(count, log.records.len() as u64);
    assert_eq!(head, log.records.last().unwrap().chain_hash());

    // Rolling back the newest record (erasing the demotion) must be
    // detected, even though the remaining prefix is a perfectly linked
    // chain on its own.
    let mut rolled_back = log.clone();
    rolled_back.records.pop();
    assert!(
        !rolled_back.verify_chain(),
        "tail truncation must break verification via the head commitment"
    );

    // Rolling back the whole reject operation (both records) too.
    let mut rolled_back2 = log.clone();
    rolled_back2.records.pop();
    rolled_back2.records.pop();
    assert!(!rolled_back2.verify_chain());
}

#[test]
fn aux_tamper_on_newest_record_detected() {
    let mut ledger = witnessed_ledger();
    let a = ledger.add("fact", &[], "a", "obs").unwrap();
    // The newest record is an Accept, whose `aux` carries the proof-gate
    // commitment prefix — the field excluded from the 48-byte record_hash
    // preimage.
    ledger.accept(a, "a", "ok").unwrap();

    let log = ledger.witness_sink();
    assert!(log.verify_chain());
    let mut tampered = log.clone();
    tampered.records.last_mut().unwrap().aux ^= 0xDEAD_BEEF;
    assert!(
        !tampered.verify_chain(),
        "aux tamper on the newest record must be caught by the head commitment"
    );
}

#[test]
fn evidence_grade_flags_divergence_detected() {
    let mut ledger = witnessed_ledger();
    ledger.add("fact", &[], "a", "obs").unwrap();

    let log = ledger.witness_sink();
    assert!(log.verify_chain());
    // Upgrade the serde-visible grade without touching the hashed flags
    // bits: a consumer reading the named enum would see a forged stronger
    // grade. verify_chain must cross-check the two.
    let mut forged = log.clone();
    forged.records[0].evidence_grade = ruvector_agent_memory::EvidenceGrade::SignatureVerified;
    assert!(
        !forged.verify_chain(),
        "evidence_grade diverging from flags bits 12-15 must fail verification"
    );
}

#[test]
fn witness_record_serializes_with_stable_shape() {
    let mut ledger = witnessed_ledger();
    let a = ledger.add("fact", &[], "agent-1", "obs").unwrap();
    ledger.accept(a, "agent-1", "ok").unwrap();

    let records = &ledger.witness_sink().records;
    let add_json = serde_json::to_value(records[0]).unwrap();
    let accept_json = serde_json::to_value(records[1]).unwrap();

    // ADR-322C alignment: integer-only fields plus the three-value
    // evidence-grade vocabulary, serde-serializable.
    assert_eq!(add_json["evidence_grade"], "trusted-assertion");
    assert_eq!(accept_json["evidence_grade"], "recomputed");
    assert_eq!(add_json["action_kind"], 0xA0);
    assert_eq!(accept_json["action_kind"], 0xA5);
    // Round-trips losslessly.
    let back: ruvector_agent_memory::LedgerWitnessRecord =
        serde_json::from_value(accept_json).unwrap();
    assert_eq!(back, records[1]);
}

// ── Witness-sink failure: no witness, no mutation ────────────────────────────

/// Sink with a record budget that refuses any batch it cannot durably
/// retain in full, honoring the `emit_batch` atomicity contract (on `Err`,
/// nothing from the batch is retained). Shared handles let the test adjust
/// the budget and inspect the persisted log while the ledger owns the sink.
#[derive(Clone, Default)]
struct FailingSink {
    log: Rc<RefCell<MemoryWitnessLog>>,
    budget: Rc<Cell<usize>>,
}

impl WitnessSink for FailingSink {
    fn emit_batch(&mut self, records: &[LedgerWitnessRecord]) -> Result<(), LedgerError> {
        if records.len() > self.budget.get() {
            return Err(LedgerError::WitnessRejected(format!(
                "sink out of budget: batch of {} exceeds remaining {}",
                records.len(),
                self.budget.get()
            )));
        }
        self.budget.set(self.budget.get() - records.len());
        self.log.borrow_mut().emit_batch(records)
    }
}

#[test]
fn failing_sink_single_op_no_witness_no_mutation() {
    let sink = FailingSink::default(); // budget 0: refuse everything
    let log = sink.log.clone();
    let mut ledger = TransactionalLedger::new(sink, AlwaysAdmitGate::default());

    let err = ledger.add("fact", &[], "a", "obs").unwrap_err();
    assert!(matches!(err, LedgerError::WitnessRejected(_)));

    // (a) no state mutation, (b) no history append, (c) no orphan record.
    assert_eq!(ledger.len(), 0);
    assert!(ledger.history().is_empty());
    assert!(log.borrow().records.is_empty());
    assert!(log.borrow().verify_chain());
}

#[test]
fn failing_sink_mid_cascade_leaves_ledger_and_chain_consistent() {
    let sink = FailingSink::default();
    let log = sink.log.clone();
    let budget = sink.budget.clone();
    let mut ledger = TransactionalLedger::new(sink, AlwaysAdmitGate::default());

    // base <- d1, base <- d2; all accepted. 6 single-record batches.
    budget.set(6);
    let base = ledger.add("base fact", &[], "a", "obs").unwrap();
    ledger.accept(base, "a", "ok").unwrap();
    let d1 = ledger.add("inference 1", &[base], "a", "derived").unwrap();
    ledger.accept(d1, "a", "ok").unwrap();
    let d2 = ledger.add("inference 2", &[base], "a", "derived").unwrap();
    ledger.accept(d2, "a", "ok").unwrap();
    assert_eq!(budget.get(), 0);
    let history_before = ledger.history().len();
    let states_before = ledger.states();
    let persisted_before = log.borrow().records.len();
    let head_before = log.borrow().head_commitment();

    // The rejection stages a 3-record batch (rejection + 2 dependent
    // demotions). Budget 2 models the sink's medium failing at record 3 of
    // the cascade; the atomic contract means the sink refuses the whole
    // batch rather than leaking records 1-2.
    budget.set(2);
    let err = ledger
        .reject_unreliable(base, "auditor", "poisoned source")
        .unwrap_err();
    assert!(matches!(err, LedgerError::WitnessRejected(_)));

    // Mutation NOT applied — the whole cascade, not just the failing tail.
    assert_eq!(ledger.states(), states_before);
    assert_eq!(ledger.state_of(base), Some(LedgerState::Accepted));
    assert_eq!(ledger.state_of(d1), Some(LedgerState::Accepted));
    assert_eq!(ledger.state_of(d2), Some(LedgerState::Accepted));
    // No history append.
    assert_eq!(ledger.history().len(), history_before);
    // No orphan record visible; chain still verifies at the same head.
    assert_eq!(log.borrow().records.len(), persisted_before);
    assert_eq!(log.borrow().head_commitment(), head_before);
    assert!(log.borrow().verify_chain());

    // Recovery: with the sink healthy again the same operation commits
    // cleanly — no sequence reuse, no chain fork.
    budget.set(usize::MAX);
    ledger
        .reject_unreliable(base, "auditor", "poisoned source")
        .unwrap();
    assert_eq!(ledger.state_of(base), Some(LedgerState::Rejected));
    assert_eq!(ledger.state_of(d1), Some(LedgerState::Pending));
    assert_eq!(ledger.state_of(d2), Some(LedgerState::Pending));
    let log_ref = log.borrow();
    assert_eq!(log_ref.records.len(), persisted_before + 3);
    for (i, r) in log_ref.records.iter().enumerate() {
        assert_eq!(r.sequence, i as u64, "sequences stay dense, none reused");
    }
    assert!(log_ref.verify_chain(), "persisted chain never forks");
    drop(log_ref);

    // And the retained history still replays to the exact final state.
    assert_eq!(replay_history(ledger.history()), ledger.states());
}

// ── History replay ───────────────────────────────────────────────────────────

#[test]
fn history_replay_reconstructs_final_state() {
    let mut ledger = witnessed_ledger();

    // A property-ish workout: drive the ledger through a deterministic but
    // varied schedule of all five ops plus acceptances and cascades.
    let mut ids = Vec::new();
    for i in 0..25u64 {
        let deps: Vec<u64> = if i % 3 == 0 || ids.is_empty() {
            vec![]
        } else {
            let accepted: Vec<u64> = ids
                .iter()
                .copied()
                .filter(|&id| ledger.state_of(id) == Some(LedgerState::Accepted))
                .collect();
            accepted.into_iter().take((i as usize % 2) + 1).collect()
        };
        let id = ledger
            .add(format!("statement {i}"), &deps, "a", "obs")
            .unwrap();
        ids.push(id);
        if i % 2 == 0 {
            ledger.accept(id, "a", "ok").unwrap();
        }
        match i % 7 {
            3 => ledger
                .ignore(&format!("noise {i}"), "a", "irrelevant")
                .unwrap(),
            4 => ledger
                .defer_for_verification(ids[(i as usize) / 2], "a", "re-check")
                .unwrap_or(()),
            5 => ledger
                .reject_unreliable(ids[(i as usize) / 3], "a", "unreliable")
                .unwrap_or(()),
            6 => {
                let _ = ledger
                    .revise_outdated_belief(
                        ids[(i as usize) / 4],
                        format!("rev {i}"),
                        "a",
                        "new evidence",
                    )
                    .map(|new_id| ids.push(new_id));
            }
            _ => {}
        }
    }

    assert!(
        ledger.history().len() > 40,
        "schedule should produce a rich history"
    );
    // Replaying the retained provenance history alone reconstructs the
    // ledger's exact final state map.
    assert_eq!(replay_history(ledger.history()), ledger.states());
    // And the witness chain over that same run still verifies.
    assert!(ledger.witness_sink().verify_chain());
}

// ── Real proof-gate integration (feature-gated) ──────────────────────────────

#[cfg(feature = "proof-gate")]
mod proof_gate_integration {
    use super::*;
    use ruvector_agent_memory::WriteGateAdapter;
    use ruvector_proof_gate::{HashChainGate, WriteGate};

    #[test]
    fn acceptance_routes_through_real_hash_chain_gate() {
        let adapter = WriteGateAdapter::new(HashChainGate::new());
        let mut ledger = TransactionalLedger::new(MemoryWitnessLog::default(), adapter);

        let a = ledger.add("fact", &[], "a", "obs").unwrap();
        let receipt = ledger.accept(a, "a", "ok").unwrap();

        // Non-trivial commitment from the real ADR-194/047 gate.
        assert_ne!(receipt.commitment, [0u8; 32]);
        assert_eq!(receipt.sequence, 0);

        // The underlying gate saw exactly the acceptance writes and its
        // chain re-derives from genesis.
        let b = ledger.add("fact 2", &[], "a", "obs").unwrap();
        ledger.accept(b, "a", "ok").unwrap();
        // add() does not touch the gate; only acceptances do.
        assert_eq!(ledger.witness_sink().records.len(), 4);
        assert_eq!(ledger.proof_gate().gate().len(), 2);
        // Two acceptances admitted.
        assert!(ledger
            .history()
            .iter()
            .filter(|r| r.kind == TransitionKind::Accept)
            .all(|r| r.receipt.is_some()));
    }

    #[test]
    fn gate_chain_integrity_survives_ledger_run() {
        let adapter = WriteGateAdapter::new(HashChainGate::new());
        let mut ledger = TransactionalLedger::new(MemoryWitnessLog::default(), adapter);
        for i in 0..10 {
            let id = ledger.add(format!("s{i}"), &[], "a", "obs").unwrap();
            ledger.accept(id, "a", "ok").unwrap();
        }
        assert_eq!(ledger.entries_in(LedgerState::Accepted).len(), 10);
        // Full cryptographic re-derivation of the gate's hash chain from
        // genesis (ADR-194/047 machinery), plus the ledger's own witness
        // chain over the same run.
        assert!(ledger.proof_gate().gate().verify_integrity());
        assert!(ledger.witness_sink().verify_chain());
    }
}
