//! Integration tests for `TileZeroGate` (nightly research 2026-11-10,
//! `docs/research/nightly/2026-11-10-tilezero-gated-agent-memory-admission/`):
//! Permit/Deny/Defer classification on deterministic coherent/incoherent
//! fixtures, the ledger's "no witness, no mutation" invariant with
//! `TileZeroGate` installed as the `ProofGate`, and feature-flag
//! compilation sanity. Mirrors the house style of
//! `crates/ruvector-agent-memory/tests/tarl_ledger.rs`.

#![cfg(feature = "tilezero-gate")]

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use cognitum_gate_tilezero::{GateDecision, GateThresholds};
use ruvector_agent_memory::{
    embed_content, LedgerError, LedgerState, LedgerWitnessRecord, MemoryWitnessLog, ProofGate,
    TileZeroGate, TransactionalLedger, TransitionKind, WitnessSink,
};

// ── Deterministic fixtures ───────────────────────────────────────────────────

/// A small family of templated, on-topic statements sharing heavy byte
/// n-gram overlap by construction.
fn coherent_fixture(n: u32) -> String {
    format!("memory entry alpha variant {n:04} about topic coherence and routine status")
}

/// Content with no shared template structure with [`coherent_fixture`] at
/// all — a deliberately crude stand-in for poisoning/drift/injection
/// content (see `src/tilezero_gate.rs`'s module doc for the honest scope of
/// what this fixture-level separation does and does not demonstrate).
const INCOHERENT_FIXTURES: &[&str] = &[
    "zk92jf03mq8zq01pf92hdl3kq09rzpxx77vv22nn44bb66cc88dd",
    "###@@!!??~~^^&&**(())[[]]{{}}<<>>||\\\\//++==--__",
    "completely different lorem ipsum dolor sit amet consectetur",
    "9f8e7d6c5b4a3c2d1e0fQWERTYASDFGHZXCVBNMqwertyuiop1234567890",
];

fn warm_window(gate: &mut TileZeroGate, n: u32) -> u32 {
    let mut warmed = 0;
    for i in 0..n {
        if gate.admit(i as u64, coherent_fixture(i).as_bytes()).is_ok() {
            warmed += 1;
        }
    }
    warmed
}

// ── Permit / Deny / Defer classification ─────────────────────────────────────

#[test]
fn cold_start_admission_permits() {
    let mut gate = TileZeroGate::new(GateThresholds::default());
    let receipt = gate
        .admit(0, coherent_fixture(0).as_bytes())
        .expect("cold start (empty window) must permit, matching AlwaysAdmitGate");
    assert_eq!(receipt.sequence, 0);
    assert_eq!(gate.last_outcome().unwrap().decision, GateDecision::Permit);
    assert_eq!(gate.window_len(), 1);
}

#[test]
fn coherent_candidate_permits_after_warm_window() {
    let mut gate = TileZeroGate::with_window_size(GateThresholds::default(), 64);
    let warmed = warm_window(&mut gate, 64);
    assert!(warmed >= 32, "most warm-up admissions should permit");

    let outcome_ok = gate.admit(1000, coherent_fixture(999).as_bytes());
    assert!(
        outcome_ok.is_ok(),
        "a template-matching candidate should permit after a coherent warm window: {:?}",
        gate.last_outcome()
    );
    let outcome = gate.last_outcome().unwrap();
    assert_eq!(outcome.decision, GateDecision::Permit);
    assert!(outcome.structural_score > 0.0);
    assert!(outcome.evidence_score > 0.0);
}

#[test]
fn incoherent_candidates_are_denied_or_deferred_after_warm_window() {
    let mut gate = TileZeroGate::with_window_size(GateThresholds::default(), 64);
    warm_window(&mut gate, 64);
    let window_len_before = gate.window_len();

    for fixture in INCOHERENT_FIXTURES {
        let res = gate.admit(2000, fixture.as_bytes());
        assert!(
            matches!(res, Err(LedgerError::GateDenied(_))),
            "incoherent fixture {fixture:?} should be refused (Deny or Defer), got {:?}",
            gate.last_outcome()
        );
        let outcome = gate.last_outcome().unwrap();
        assert!(
            matches!(outcome.decision, GateDecision::Deny | GateDecision::Defer),
            "expected Deny or Defer for {fixture:?}, got {:?}",
            outcome.decision
        );
        assert!(outcome.rejected_by.is_some());
        // A refused admission must not grow the sliding window — only
        // genuinely admitted content joins "recent history".
        assert_eq!(gate.window_len(), window_len_before);
    }
}

#[test]
fn last_outcome_reports_component_scores() {
    let mut gate = TileZeroGate::with_window_size(GateThresholds::default(), 8);
    warm_window(&mut gate, 8);
    let _ = gate.admit(99, INCOHERENT_FIXTURES[0].as_bytes());
    let outcome = gate.last_outcome().expect("an outcome is always recorded");
    for score in [
        outcome.structural_score,
        outcome.shift_score,
        outcome.evidence_score,
    ] {
        assert!((0.0..=1.0).contains(&score), "score {score} out of [0,1]");
    }
}

#[test]
fn denied_admission_does_not_advance_gate_sequence() {
    let mut gate = TileZeroGate::with_window_size(GateThresholds::default(), 16);
    warm_window(&mut gate, 16);
    let seq_before = gate
        .admit(500, coherent_fixture(500).as_bytes())
        .expect("coherent admit permits")
        .sequence;
    // A refused admission must not consume/advance the receipt sequence
    // space the same way a permit does (no observable side effect at all
    // on refusal, beyond `last_outcome` for introspection).
    let _ = gate.admit(501, INCOHERENT_FIXTURES[0].as_bytes());
    let seq_after = gate
        .admit(502, coherent_fixture(502).as_bytes())
        .expect("coherent admit permits")
        .sequence;
    assert_eq!(seq_after, seq_before + 1);
}

// ── TileZeroGate wired into a real TransactionalLedger ───────────────────────

#[test]
fn tilezero_gate_wired_into_ledger_end_to_end() {
    let mut ledger = TransactionalLedger::new(
        MemoryWitnessLog::default(),
        TileZeroGate::with_window_size(GateThresholds::default(), 32),
    );
    let mut accepted = 0;
    let mut refused = 0;
    for i in 0..32u32 {
        let id = ledger
            .add(coherent_fixture(i), &[], "agent", "obs")
            .unwrap();
        match ledger.accept(id, "agent", "ok") {
            Ok(_) => accepted += 1,
            Err(LedgerError::GateDenied(_)) => refused += 1,
            Err(e) => panic!("unexpected error: {e}"),
        }
    }
    assert!(
        accepted > 0,
        "a coherent warm-up run should admit most entries"
    );

    // Now feed an incoherent entry through the same ledger.
    let bad_id = ledger
        .add(INCOHERENT_FIXTURES[0], &[], "attacker", "injected")
        .unwrap();
    assert!(matches!(
        ledger.accept(bad_id, "attacker", "force in"),
        Err(LedgerError::GateDenied(_))
    ));
    assert_eq!(ledger.state_of(bad_id), Some(LedgerState::Pending));
    assert!(ledger
        .history()
        .iter()
        .all(|r| r.entry_id != Some(bad_id) || r.kind != TransitionKind::Accept));

    // The ledger's own witness chain (independent of the gate) still
    // verifies over the whole run.
    assert!(ledger.witness_sink().verify_chain());
    let _ = (accepted, refused);
}

// ── Witness-sink failure: no witness, no mutation (TileZeroGate installed) ──

/// Sink with a record budget that refuses any batch it cannot durably
/// retain in full — mirrors `tarl_ledger.rs`'s `FailingSink` exactly, so the
/// same "no witness, no mutation" invariant test methodology applies with
/// `TileZeroGate` as the `ProofGate` instead of `AlwaysAdmitGate`/
/// `WriteGateAdapter`.
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
fn tilezero_gate_failing_sink_no_witness_no_mutation() {
    // Prime the gate's sliding window directly (bypassing any ledger) so
    // the *test* admission below permits deterministically through the
    // real ThreeFilterDecision logic, not by construction.
    let mut gate = TileZeroGate::with_window_size(GateThresholds::default(), 4);
    gate.admit(0, coherent_fixture(0).as_bytes())
        .expect("cold-start admit always permits");
    assert_eq!(gate.window_len(), 1);

    let sink = FailingSink::default();
    let log = sink.log.clone();
    let budget = sink.budget.clone();
    budget.set(usize::MAX); // allow the `add` below to succeed
    let mut ledger = TransactionalLedger::new(sink, gate);

    let id = ledger
        .add(coherent_fixture(0), &[], "a", "obs")
        .expect("add never consults the gate");
    // `add` already committed its own one-record witness batch (it never
    // consults the gate) — capture that baseline, since the invariant under
    // test is specifically about the *next* (accept) batch, mirroring
    // `tarl_ledger.rs`'s `failing_sink_mid_cascade_leaves_ledger_and_chain_consistent`.
    let persisted_before = log.borrow().records.len();
    let history_before = ledger.history().len();

    budget.set(0); // the witness sink now refuses the accept's batch
    let err = ledger
        .accept(id, "a", "ok")
        .expect_err("sink refusal must surface as an error");
    assert!(matches!(err, LedgerError::WitnessRejected(_)));

    // (a) no state mutation, (b) no history append, (c) no orphan record
    // from the refused Accept batch — identical invariant to
    // `tarl_ledger.rs`'s failing-sink tests, now with a content-inspecting
    // gate installed.
    assert_eq!(ledger.state_of(id), Some(LedgerState::Pending));
    assert_eq!(ledger.history().len(), history_before);
    assert!(ledger
        .history()
        .iter()
        .all(|r| r.kind != TransitionKind::Accept));
    assert_eq!(log.borrow().records.len(), persisted_before);
    assert!(log.borrow().verify_chain());

    // Known, documented asymmetry (see `ledger.rs`'s `accept()` comment,
    // and `tilezero_gate.rs`'s module doc): the gate is consulted BEFORE
    // the witness sink, so a Permit already advanced the gate's own
    // internal state (its sliding window gained the newly admitted
    // embedding, and its receipt sequence counter advanced) even though
    // the LEDGER applied nothing — the same asymmetry already accepted for
    // `WriteGateAdapter`'s `HashChainGate` in `tarl_ledger.rs`'s
    // `proof_gate_integration` tests. This is not a bypass: no `Accept`
    // record ever reached the ledger's history or witness log.
    assert_eq!(ledger.proof_gate().window_len(), 2); // primer + this admit

    // Recovery: with the sink healthy again, the same operation commits
    // cleanly and the gate is not asked to re-admit (the entry is already
    // `Pending`; a fresh `accept` call re-consults the gate, which is the
    // correct, intended behavior — not a replay of the earlier refusal).
    budget.set(usize::MAX);
    ledger.accept(id, "a", "retry").expect("gate still permits");
    assert_eq!(ledger.state_of(id), Some(LedgerState::Accepted));
    assert!(log.borrow().verify_chain());
}

// ── Feature-flag compilation sanity ──────────────────────────────────────────

#[test]
fn tilezero_feature_exports_are_usable() {
    // This test's mere existence (and the fact this whole file only
    // compiles under `--features tilezero-gate`) is most of the sanity
    // check; it also exercises every symbol this nightly run added to the
    // crate's public surface.
    assert_eq!(ruvector_agent_memory::TILEZERO_EMBED_DIM, 64);
    assert_eq!(ruvector_agent_memory::TILEZERO_DEFAULT_WINDOW, 64);
    let v = embed_content(b"sanity check content");
    assert_eq!(v.len(), ruvector_agent_memory::TILEZERO_EMBED_DIM);
    let _: Box<dyn ProofGate> = Box::new(TileZeroGate::new(GateThresholds::default()));
}
