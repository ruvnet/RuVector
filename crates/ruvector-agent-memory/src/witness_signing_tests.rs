use super::*;
use crate::ledger::{AlwaysAdmitGate, TransactionalLedger};
use crate::ops::{EvidenceGrade, MemoryWitnessLog};

const TEST_SECRET: [u8; 32] = [7u8; 32];
const GENESIS: SignedAnchor = SignedAnchor::genesis();

fn keypair() -> Ed25519Keypair {
    Ed25519Keypair::from_secret(&TEST_SECRET)
}

fn sink_for(strategy: SigningStrategy) -> SignedWitnessSink<MemoryWitnessLog> {
    SignedWitnessSink::from_keypair(MemoryWitnessLog::default(), &keypair(), strategy).unwrap()
}

/// Drive `n` add+accept pairs through a ledger (2 witness records each).
fn run(
    sink: SignedWitnessSink<MemoryWitnessLog>,
    tag: &str,
    n: usize,
) -> SignedWitnessSink<MemoryWitnessLog> {
    let mut ledger = TransactionalLedger::new(sink, AlwaysAdmitGate::default());
    for i in 0..n {
        let id = ledger.add(format!("{tag} {i}"), &[], "actor", "r").unwrap();
        ledger.accept(id, "actor", "verified").unwrap();
    }
    ledger.into_witness_sink()
}

fn populated(strategy: SigningStrategy, n: usize) -> SignedWitnessSink<MemoryWitnessLog> {
    let mut sink = run(sink_for(strategy), "memory", n);
    sink.seal();
    sink
}

const STRATEGIES: [SigningStrategy; 3] = [
    SigningStrategy::PerRecord,
    SigningStrategy::BatchTail { batch_size: 4 },
    SigningStrategy::BatchTail { batch_size: 5 },
];

/// Diligent adversary: recompute the whole FNV chain from `at` onward and
/// fix the unsigned head commitment, so `verify_chain` passes.
fn rehash_from(log: &mut MemoryWitnessLog, at: usize) {
    let mut prev = if at == 0 {
        0
    } else {
        log.records[at - 1].chain_hash()
    };
    for r in log.records.iter_mut().skip(at) {
        r.prev_hash = prev;
        r.record_hash = r.compute_record_hash();
        prev = r.chain_hash();
    }
    log.committed_count = log.records.len() as u64;
    log.committed_head = prev;
}

fn truncate(log: &mut MemoryWitnessLog, len: usize) {
    log.records.truncate(len);
    log.committed_count = len as u64;
    log.committed_head = log.records.last().map_or(0, |r| r.chain_hash());
}

fn verify(
    sink: &SignedWitnessSink<MemoryWitnessLog>,
    log: &MemoryWitnessLog,
    spans: &[SignedSpan],
    anchor: &SignedAnchor,
) -> Result<SignedChainReport, SignedChainError> {
    verify_signed_chain(log, spans, &sink.public_key(), anchor)
}

// ---------------------------------------------------------------- positive

#[test]
fn honest_chain_verifies_under_every_strategy() {
    for s in STRATEGIES {
        let sink = populated(s, 12);
        let anchor = sink.anchor();
        let report = verify(&sink, sink.inner(), sink.spans(), &anchor).unwrap();
        assert_eq!(report.records_verified, 24);
        assert_eq!(report.spans_verified, sink.spans().len());
        assert_eq!(report.head, anchor);
        assert_eq!(anchor.record_count, 24);
        // Genesis anchor also accepts an honest chain.
        assert!(verify(&sink, sink.inner(), sink.spans(), &GENESIS).is_ok());
    }
}

#[test]
fn batch_tail_amortizes_signature_count() {
    let per_record = populated(SigningStrategy::PerRecord, 23);
    let batch = populated(SigningStrategy::BatchTail { batch_size: 5 }, 23);
    assert_eq!(per_record.spans().len(), 46);
    assert_eq!(batch.spans().len(), 10); // 9 full batches + 1 sealed partial
}

#[test]
fn old_anchor_still_verifies_a_grown_log() {
    for s in STRATEGIES {
        let mut ledger = TransactionalLedger::new(sink_for(s), AlwaysAdmitGate::default());
        let mut old = SignedAnchor::genesis();
        for i in 0..17 {
            if i == 10 {
                // Last closed span; any pending BatchTail records stay unsigned.
                old = ledger.witness_sink().anchor();
                assert!(old.record_count > 0);
            }
            let id = ledger.add(format!("m {i}"), &[], "a", "r").unwrap();
            ledger.accept(id, "a", "v").unwrap();
        }
        let mut sink = ledger.into_witness_sink();
        sink.seal();
        let report = verify(&sink, sink.inner(), sink.spans(), &old).unwrap();
        assert_eq!(report.records_verified, 34);
        assert_eq!(report.head, sink.anchor());
    }
}

#[test]
fn empty_log_verifies_only_with_genesis_anchor() {
    let sink = sink_for(SigningStrategy::PerRecord);
    assert!(verify(&sink, sink.inner(), &[], &GENESIS).is_ok());
    let bogus = SignedAnchor {
        record_count: 1,
        head_digest: [1; 32],
    };
    assert_eq!(
        verify(&sink, sink.inner(), &[], &bogus),
        Err(SignedChainError::AnchorMismatch)
    );
}

// ---------------------------------------------------------------- negative

#[test]
fn zero_batch_size_is_an_error_not_a_panic() {
    let r = SignedWitnessSink::from_keypair(
        MemoryWitnessLog::default(),
        &keypair(),
        SigningStrategy::BatchTail { batch_size: 0 },
    );
    assert_eq!(r.err(), Some(WitnessSignerError::ZeroBatchSize));
}

#[test]
fn wrong_public_key_fails() {
    let sink = populated(SigningStrategy::PerRecord, 5);
    let wrong = Ed25519Keypair::from_secret(&[9u8; 32]).public_key();
    assert_eq!(
        verify_signed_chain(sink.inner(), sink.spans(), &wrong, &sink.anchor()),
        Err(SignedChainError::BadSignature { span_index: 0 })
    );
}

#[test]
fn naive_tamper_fails_chain_walk() {
    let sink = populated(SigningStrategy::PerRecord, 10);
    let mut forged = sink.inner().clone();
    forged.records[3].payload ^= 1;
    assert_eq!(
        verify(&sink, &forged, sink.spans(), &sink.anchor()),
        Err(SignedChainError::ChainWalkFailed)
    );
}

#[test]
fn diligent_forgery_passes_chain_walk_but_not_signatures() {
    for s in STRATEGIES {
        let sink = populated(s, 16);
        let mut forged = sink.inner().clone();
        forged.records[6].payload ^= 0xDEAD_BEEF;
        rehash_from(&mut forged, 6);
        assert!(forged.verify_chain(), "documented unsigned-chain gap");
        assert!(matches!(
            verify(&sink, &forged, sink.spans(), &GENESIS),
            Err(SignedChainError::RecordsMismatch { .. })
        ));
    }
}

/// PoC 1 (review finding 1): a consistently re-hashed forged log with an
/// EMPTY span list used to verify, because coverage was never checked.
#[test]
fn poc1_empty_span_list_on_forged_log_fails() {
    for s in STRATEGIES {
        let sink = populated(s, 16);
        let mut forged = sink.inner().clone();
        forged.records[0].payload ^= 0xFFFF;
        rehash_from(&mut forged, 0);
        assert!(forged.verify_chain());
        let r = verify(&sink, &forged, &[], &GENESIS);
        assert_eq!(
            r,
            Err(SignedChainError::UnsignedTail {
                signed: 0,
                unsigned: 32
            })
        );
        // With a real anchor the empty list is also an anchor failure.
        assert_eq!(
            verify(&sink, &forged, &[], &sink.anchor()),
            Err(SignedChainError::AnchorMismatch)
        );
    }
}

/// PoC 1 variant: a partial span list (middle span dropped, forged records
/// only inside the dropped span) fails on coverage.
#[test]
fn poc1_partial_span_list_fails_on_coverage_gap() {
    let sink = populated(SigningStrategy::BatchTail { batch_size: 4 }, 16);
    let mut spans = sink.spans().to_vec();
    spans.remove(2); // records 8..=11
    let mut forged = sink.inner().clone();
    forged.records[9].payload ^= 0x55;
    rehash_from(&mut forged, 9);
    assert_eq!(
        verify(&sink, &forged, &spans, &GENESIS),
        Err(SignedChainError::CoverageGap {
            span_index: 2,
            expected_from: 8
        })
    );
}

/// PoC 2 (review finding 2): under BatchTail the unsealed tail used to be
/// silently accepted and forgeable. Now an unsealed tail never verifies as
/// authenticated — whether honest, forged, or partially truncated.
#[test]
fn poc2_unsigned_batch_tail_is_never_reported_as_verified() {
    // 11 pairs = 22 records; batch 8 => 16 signed, 6 pending.
    let sink = run(
        sink_for(SigningStrategy::BatchTail { batch_size: 8 }),
        "m",
        11,
    );
    assert_eq!(sink.unsigned_pending(), 6);
    let tail = SignedChainError::UnsignedTail {
        signed: 16,
        unsigned: 6,
    };
    assert_eq!(
        verify(&sink, sink.inner(), sink.spans(), &sink.anchor()),
        Err(tail)
    );

    let mut forged = sink.inner().clone();
    forged.records[19].payload ^= 0xABCD;
    rehash_from(&mut forged, 19);
    assert!(forged.verify_chain());
    assert_eq!(
        verify(&sink, &forged, sink.spans(), &sink.anchor()),
        Err(tail)
    );

    let mut truncated = sink.inner().clone();
    truncate(&mut truncated, 20);
    assert_eq!(
        verify(&sink, &truncated, sink.spans(), &sink.anchor()),
        Err(SignedChainError::UnsignedTail {
            signed: 16,
            unsigned: 4
        })
    );
}

/// PoC 2, sealed: once `seal()` signs the partial batch, forging or
/// truncating the former tail fails outright.
#[test]
fn poc2_sealed_tail_resists_forgery_and_truncation() {
    let mut sink = run(
        sink_for(SigningStrategy::BatchTail { batch_size: 8 }),
        "m",
        11,
    );
    sink.seal();
    assert_eq!(sink.unsigned_pending(), 0);
    let anchor = sink.anchor();
    assert!(verify(&sink, sink.inner(), sink.spans(), &anchor).is_ok());

    let mut forged = sink.inner().clone();
    forged.records[21].aux ^= 1;
    rehash_from(&mut forged, 21);
    assert!(matches!(
        verify(&sink, &forged, sink.spans(), &anchor),
        Err(SignedChainError::RecordsMismatch { span_index: 2 })
    ));

    let mut truncated = sink.inner().clone();
    truncate(&mut truncated, 20);
    assert!(matches!(
        verify(&sink, &truncated, sink.spans(), &anchor),
        Err(SignedChainError::SpanBeyondLog { span_index: 2, .. })
    ));
}

/// PoC 3 (review finding 3): dropping trailing spans AND truncating the log
/// to match is internally consistent. It is rejected against a persisted
/// anchor; with the genesis anchor it verifies (documented limit).
#[test]
fn poc3_rollback_of_log_and_spans_is_caught_by_anchor() {
    for s in STRATEGIES {
        let sink = populated(s, 16);
        let anchor = sink.anchor();
        let mut spans = sink.spans().to_vec();
        spans.truncate(spans.len() / 2);
        let keep = spans.last().unwrap().covers_to_seq as usize + 1;
        let mut rolled = sink.inner().clone();
        truncate(&mut rolled, keep);
        assert!(rolled.verify_chain());
        assert!(
            verify(&sink, &rolled, &spans, &GENESIS).is_ok(),
            "without an anchor, rollback is undetectable (documented)"
        );
        assert_eq!(
            verify(&sink, &rolled, &spans, &anchor),
            Err(SignedChainError::AnchorMismatch)
        );
    }
}

#[test]
fn splicing_two_logs_signed_by_the_same_key_fails() {
    for s in STRATEGIES {
        let a = populated(s, 8);
        let b = run(sink_for(s), "other", 8);
        let mut b = b;
        b.seal();
        let cut = a.spans()[1].covers_to_seq as usize + 1;
        let mut spliced = a.inner().clone();
        spliced.records.truncate(cut);
        spliced.records.extend_from_slice(&b.inner().records[cut..]);
        rehash_from(&mut spliced, cut);
        let mut spans = a.spans()[..2].to_vec();
        spans.extend(
            b.spans()
                .iter()
                .skip_while(|sp| sp.covers_from_seq < cut as u64),
        );
        assert!(verify(&a, &spliced, &spans, &GENESIS).is_err());
        // Even without re-hashing (records byte-identical to what B signed),
        // B's spans are chained to B's history, not A's.
        let mut raw = a.inner().clone();
        raw.records.truncate(cut);
        raw.records.extend_from_slice(&b.inner().records[cut..]);
        assert!(verify(&a, &raw, &spans, &GENESIS).is_err());
    }
}

#[test]
fn reordered_spans_fail() {
    let sink = populated(SigningStrategy::BatchTail { batch_size: 4 }, 8);
    let mut spans = sink.spans().to_vec();
    spans.swap(0, 1);
    assert!(matches!(
        verify(&sink, sink.inner(), &spans, &GENESIS),
        Err(SignedChainError::CoverageGap { span_index: 0, .. })
    ));
}

#[test]
fn non_contiguous_sequence_is_refused_before_inner_commit() {
    let mut sink = populated(SigningStrategy::PerRecord, 2);
    let mut bogus = *sink.inner().records.last().unwrap();
    bogus.sequence += 5;
    assert!(sink.emit_batch(&[bogus]).is_err());
    assert_eq!(sink.inner().records.len(), 4);
    assert!(verify(&sink, sink.inner(), sink.spans(), &sink.anchor()).is_ok());
}

// ----------------------------------------------------- BoundedBatchTail

/// Emit `n` records directly (no ledger), with `timestamp_ns` spaced
/// `gap_ns` apart starting at `start_ns`, returning the sink.
fn emit_timed(
    strategy: SigningStrategy,
    n: u64,
    start_ns: u64,
    gap_ns: u64,
) -> SignedWitnessSink<MemoryWitnessLog> {
    let mut sink = sink_for(strategy);
    let mut prev = 0u64;
    let records: Vec<LedgerWitnessRecord> = (0..n)
        .map(|i| {
            let mut r = LedgerWitnessRecord {
                sequence: i,
                timestamp_ns: start_ns + i * gap_ns,
                action_kind: 0,
                proof_tier: 0,
                flags: crate::ops::pack_flags(None, None, EvidenceGrade::Recomputed),
                actor_partition_id: 0,
                target_object_id: 0,
                capability_hash: 0,
                payload: i,
                prev_hash: prev,
                record_hash: 0,
                aux: 0,
                evidence_grade: EvidenceGrade::Recomputed,
            };
            r.record_hash = r.compute_record_hash();
            prev = r.chain_hash();
            r
        })
        .collect();
    sink.emit_batch(&records).unwrap();
    sink
}

#[test]
fn zero_batch_size_is_an_error_for_bounded_variant_too() {
    let r = SignedWitnessSink::from_keypair(
        MemoryWitnessLog::default(),
        &keypair(),
        SigningStrategy::BoundedBatchTail {
            batch_size: 0,
            max_wait_ns: 1_000,
        },
    );
    assert_eq!(r.err(), Some(WitnessSignerError::ZeroBatchSize));
}

#[test]
fn bounded_batch_tail_still_closes_on_size_when_arrivals_are_fast() {
    // 10 records, 100ns apart, max_wait_ns huge: size trigger fires first.
    let sink = emit_timed(
        SigningStrategy::BoundedBatchTail {
            batch_size: 4,
            max_wait_ns: 1_000_000_000,
        },
        10,
        0,
        100,
    );
    // 2 full batches of 4 closed by size; 2 records still pending.
    assert_eq!(sink.spans().len(), 2);
    assert_eq!(sink.unsigned_pending(), 2);
}

#[test]
fn bounded_batch_tail_closes_inline_once_a_later_arrival_crosses_the_deadline() {
    // batch_size huge (never reached); max_wait_ns = 1000. First record at
    // t=0 opens the span; a later record at t=1000 is itself included in
    // the same span and crosses the deadline, so the arrival that crosses
    // it closes the span immediately (no separate timer needed here).
    let sink = emit_timed(
        SigningStrategy::BoundedBatchTail {
            batch_size: 1_000_000,
            max_wait_ns: 1_000,
        },
        3,
        0,
        1_000,
    );
    // record 0 @ t=0 opens; record 1 @ t=1000 hits the deadline exactly and
    // closes the span (covers 0..=1); record 2 @ t=2000 opens a new span,
    // still pending (nothing closes it: no record 3 arrives).
    assert_eq!(sink.spans().len(), 1);
    assert_eq!(sink.spans()[0].covers_from_seq, 0);
    assert_eq!(sink.spans()[0].covers_to_seq, 1);
    assert_eq!(sink.unsigned_pending(), 1);
}

#[test]
fn seal_if_due_bounds_latency_when_no_further_records_ever_arrive() {
    // A single record opens a span at t=0 and nothing else ever arrives —
    // exactly the scenario plain BatchTail cannot recover from without an
    // explicit seal(). A timer-driven seal_if_due must close it by the
    // deadline.
    let mut sink = emit_timed(
        SigningStrategy::BoundedBatchTail {
            batch_size: 64,
            max_wait_ns: 5_000,
        },
        1,
        0,
        0,
    );
    assert_eq!(sink.pending_opened_at_ns(), Some(0));
    assert!(!sink.seal_if_due(4_999), "deadline not yet reached");
    assert_eq!(
        sink.unsigned_pending(),
        1,
        "still open just before deadline"
    );
    assert!(sink.seal_if_due(5_000), "deadline reached, must close");
    assert_eq!(sink.unsigned_pending(), 0);
    assert_eq!(sink.spans().len(), 1);
}

#[test]
fn seal_if_due_is_a_no_op_for_plain_batch_tail_and_per_record() {
    let mut per_record = emit_timed(SigningStrategy::PerRecord, 3, 0, 10);
    assert!(!per_record.seal_if_due(u64::MAX));
    let mut batch_tail = emit_timed(SigningStrategy::BatchTail { batch_size: 64 }, 3, 0, 10);
    assert_eq!(batch_tail.unsigned_pending(), 3);
    assert!(
        !batch_tail.seal_if_due(u64::MAX),
        "plain BatchTail has no deadline concept, however large now_ns is"
    );
    assert_eq!(batch_tail.unsigned_pending(), 3, "still open, undetected");
}

#[test]
fn seal_if_due_is_a_no_op_with_nothing_pending() {
    let mut sink = SignedWitnessSink::from_keypair(
        MemoryWitnessLog::default(),
        &keypair(),
        SigningStrategy::BoundedBatchTail {
            batch_size: 4,
            max_wait_ns: 10,
        },
    )
    .unwrap();
    assert!(!sink.seal_if_due(u64::MAX));
}

#[test]
fn bounded_batch_tail_honest_chain_verifies_after_seal_if_due() {
    let mut sink = emit_timed(
        SigningStrategy::BoundedBatchTail {
            batch_size: 64,
            max_wait_ns: 1_000,
        },
        5,
        0,
        10, // well within the deadline; nothing closes on arrival alone
    );
    assert_eq!(sink.unsigned_pending(), 5);
    assert!(sink.seal_if_due(10_000));
    let anchor = sink.anchor();
    let report = verify(&sink, sink.inner(), sink.spans(), &anchor).unwrap();
    assert_eq!(report.records_verified, 5);
    assert_eq!(anchor.record_count, 5);
}

#[test]
fn second_ledger_restarting_at_sequence_zero_is_refused() {
    let sink = populated(SigningStrategy::PerRecord, 3);
    let mut ledger = TransactionalLedger::new(sink, AlwaysAdmitGate::default());
    assert!(ledger.add("restart".to_string(), &[], "a", "r").is_err());
    let sink = ledger.into_witness_sink();
    assert_eq!(sink.inner().records.len(), 6);
    assert!(verify(&sink, sink.inner(), sink.spans(), &sink.anchor()).is_ok());
}
