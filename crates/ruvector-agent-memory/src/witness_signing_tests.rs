use super::*;
use crate::ledger::{AlwaysAdmitGate, TransactionalLedger};
use crate::ops::MemoryWitnessLog;

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

#[test]
fn second_ledger_restarting_at_sequence_zero_is_refused() {
    let sink = populated(SigningStrategy::PerRecord, 3);
    let mut ledger = TransactionalLedger::new(sink, AlwaysAdmitGate::default());
    assert!(ledger.add("restart".to_string(), &[], "a", "r").is_err());
    let sink = ledger.into_witness_sink();
    assert_eq!(sink.inner().records.len(), 6);
    assert!(verify(&sink, sink.inner(), sink.spans(), &sink.anchor()).is_ok());
}

// -------------------------------------------------------- BatchTailTimeout
//
// A dedicated section (2026-10-11 nightly, ADR-348): bounds the worst-case
// signature-availability latency of a pending `BatchTail` span under
// sparse arrivals. `mk_records` manufactures a self-consistent hash chain
// directly (bypassing `TransactionalLedger`) so these tests can drive
// `emit_batch_at`/`check_timeout_at` with explicit `now_ns` values instead
// of the real wall clock — fully deterministic, no sleeping.

fn mk_records(start_seq: u64, n: usize, mut prev_hash: u64) -> Vec<LedgerWitnessRecord> {
    let grade = crate::ops::EvidenceGrade::Recomputed;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut r = LedgerWitnessRecord {
            sequence: start_seq + i as u64,
            timestamp_ns: 0,
            action_kind: 0,
            proof_tier: 0,
            flags: crate::ops::pack_flags(None, None, grade),
            actor_partition_id: 0,
            target_object_id: 0,
            capability_hash: 0,
            payload: 0xA11CE_u64.wrapping_add(i as u64),
            prev_hash,
            record_hash: 0,
            aux: 0,
            evidence_grade: grade,
        };
        r.record_hash = r.compute_record_hash();
        prev_hash = r.chain_hash();
        out.push(r);
    }
    out
}

#[test]
fn zero_max_wait_is_an_error_not_a_panic() {
    let r = SignedWitnessSink::from_keypair(
        MemoryWitnessLog::default(),
        &keypair(),
        SigningStrategy::BatchTailTimeout {
            batch_size: 4,
            max_wait_ns: 0,
        },
    );
    assert_eq!(r.err(), Some(WitnessSignerError::ZeroMaxWait));
}

#[test]
fn zero_batch_size_is_an_error_for_timeout_variant_too() {
    let r = SignedWitnessSink::from_keypair(
        MemoryWitnessLog::default(),
        &keypair(),
        SigningStrategy::BatchTailTimeout {
            batch_size: 0,
            max_wait_ns: 10,
        },
    );
    assert_eq!(r.err(), Some(WitnessSignerError::ZeroBatchSize));
}

#[test]
fn plain_batch_tail_has_no_timeout_bound_at_any_elapsed_time() {
    // The pre-existing problem this run fixes, stated as a property: under
    // `BatchTail`, there is no deadline to check at all, however much
    // wall-clock time passes — `next_timeout_deadline_ns` is always `None`.
    let mut sink = sink_for(SigningStrategy::BatchTail { batch_size: 100 });
    sink.emit_batch_at(&mk_records(0, 1, 0), 0).unwrap();
    assert_eq!(sink.unsigned_pending(), 1);
    assert_eq!(sink.next_timeout_deadline_ns(), None);
    assert!(!sink.check_timeout_at(u64::MAX));
    assert_eq!(sink.unsigned_pending(), 1); // still stuck, at any elapsed time
}

#[test]
fn batch_tail_timeout_behaves_like_batch_tail_under_dense_arrivals() {
    // Dense arrivals (now_ns pinned at 0) never reach the deadline, so the
    // timeout variant must amortize signatures identically to plain
    // BatchTail at the same batch_size.
    let batch_size = 5;
    let max_wait_ns = 50;
    let mut plain = sink_for(SigningStrategy::BatchTail { batch_size });
    let mut timed = sink_for(SigningStrategy::BatchTailTimeout {
        batch_size,
        max_wait_ns,
    });
    let records = mk_records(0, 23, 0);
    for r in &records {
        plain.emit_batch_at(std::slice::from_ref(r), 0).unwrap();
        timed.emit_batch_at(std::slice::from_ref(r), 0).unwrap();
    }
    assert_eq!(plain.spans().len(), timed.spans().len());
    assert_eq!(plain.unsigned_pending(), timed.unsigned_pending());
    plain.seal();
    timed.seal();
    assert_eq!(plain.spans().len(), timed.spans().len());
}

#[test]
fn batch_tail_timeout_does_not_force_seal_before_the_deadline() {
    let mut sink = sink_for(SigningStrategy::BatchTailTimeout {
        batch_size: 100,
        max_wait_ns: 1_000,
    });
    sink.emit_batch_at(&mk_records(0, 1, 0), 0).unwrap();
    assert_eq!(sink.next_timeout_deadline_ns(), Some(1_000));
    assert!(!sink.check_timeout_at(999));
    assert_eq!(sink.unsigned_pending(), 1);
    assert_eq!(sink.spans().len(), 0);
}

#[test]
fn batch_tail_timeout_force_seals_exactly_at_the_deadline() {
    let mut sink = sink_for(SigningStrategy::BatchTailTimeout {
        batch_size: 100,
        max_wait_ns: 1_000,
    });
    sink.emit_batch_at(&mk_records(0, 3, 0), 0).unwrap();
    assert_eq!(sink.unsigned_pending(), 3);
    assert!(sink.check_timeout_at(1_000));
    assert_eq!(sink.unsigned_pending(), 0);
    assert_eq!(sink.spans().len(), 1);
    assert_eq!(sink.spans()[0].covers_from_seq, 0);
    assert_eq!(sink.spans()[0].covers_to_seq, 2);
    // Idempotent: a later check with nothing pending is a no-op.
    assert!(!sink.check_timeout_at(2_000));
}

#[test]
fn batch_tail_timeout_count_close_wins_when_it_happens_first() {
    let mut sink = sink_for(SigningStrategy::BatchTailTimeout {
        batch_size: 3,
        max_wait_ns: 1_000_000,
    });
    // Count-close fires at now_ns=2, long before the 1_000_000ns deadline.
    sink.emit_batch_at(&mk_records(0, 3, 0), 2).unwrap();
    assert_eq!(sink.unsigned_pending(), 0);
    assert_eq!(sink.spans().len(), 1);
}

#[test]
fn batch_tail_timeout_opportunistic_seal_on_a_late_arrival() {
    // A new arrival that itself lands at/after the deadline closes the
    // stale pending span before opening its own — exercised purely through
    // `emit_batch_at`, with no separate `check_timeout` call.
    let mut sink = sink_for(SigningStrategy::BatchTailTimeout {
        batch_size: 100,
        max_wait_ns: 1_000,
    });
    sink.emit_batch_at(&mk_records(0, 1, 0), 0).unwrap();
    let chained = mk_records(1, 1, sink.inner().records[0].chain_hash());
    sink.emit_batch_at(&chained, 5_000).unwrap();
    assert_eq!(sink.spans().len(), 1); // record 0's stale span, force-sealed
    assert_eq!(sink.spans()[0].covers_to_seq, 0);
    assert_eq!(sink.unsigned_pending(), 1); // record 1's fresh span, still open
    assert_eq!(sink.next_timeout_deadline_ns(), Some(6_000));
}

#[test]
fn batch_tail_timeout_bounds_worst_case_pending_age_under_a_poller() {
    // Property test: drive a pseudo-random mix of sparse/bursty arrivals
    // and a fixed-cadence poller (`check_timeout_at` every `tick` ns).
    // Claim under test: whenever something is pending, its age never
    // exceeds `max_wait_ns + tick` — the poller's own granularity is the
    // only slack, not an unbounded wait. `BatchTail` offers no such bound
    // at all (see `plain_batch_tail_has_no_timeout_bound_at_any_elapsed_time`).
    let max_wait_ns: u64 = 10_000;
    let tick: u64 = 1_000;
    let mut seed: u64 = 0x9E3779B97F4A7C15;
    let mut next_rand = move || {
        // xorshift64 — deterministic, no external RNG dependency.
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };

    // Trial/step counts are deliberately modest (not the much larger
    // sweep an offline simulation could afford): every force-seal and
    // every verification below signs/verifies a real Ed25519 signature,
    // which this sandboxed debug build measured at roughly 20ms/op —
    // slow enough that a large sweep turns this unit test into a
    // multi-minute run. 8 trials x 20 steps (160 arrivals, randomized
    // batch_size and gaps) still exercises the boundary repeatedly while
    // staying a fast `cargo test`; the dedicated benchmark example
    // covers the release-mode, larger-N regime with real wall-clock
    // timing instead.
    for trial in 0..8u64 {
        let batch_size = 1 + (next_rand() % 20) as usize;
        let mut sink = sink_for(SigningStrategy::BatchTailTimeout {
            batch_size,
            max_wait_ns,
        });
        let mut now_ns: u64 = 0;
        let mut prev_hash: u64 = 0;
        let mut max_observed_age: u64 = 0;
        for seq in 0..20u64 {
            // Sparse: a gap of 0..=4*max_wait_ns before the next arrival.
            now_ns += next_rand() % (4 * max_wait_ns);
            let rec = mk_records(seq, 1, prev_hash);
            prev_hash = rec[0].chain_hash();
            sink.emit_batch_at(&rec, now_ns).unwrap();
            // The poller ticks forward in `tick`-sized steps up to now_ns,
            // the only thing standing between "arrival" and "force-sealed".
            let mut poll_ns = now_ns;
            loop {
                if let Some(p) = &sink_pending_opened_at(&sink) {
                    max_observed_age = max_observed_age.max(poll_ns.saturating_sub(*p));
                }
                sink.check_timeout_at(poll_ns);
                if sink.unsigned_pending() == 0 {
                    break;
                }
                poll_ns += tick;
                if poll_ns > now_ns + max_wait_ns + tick {
                    break; // safety valve; the assertion below still fires
                }
            }
        }
        sink.seal();
        assert!(
            max_observed_age <= max_wait_ns + tick,
            "trial {trial}: observed pending age {max_observed_age}ns exceeded \
             max_wait_ns ({max_wait_ns}) + poller tick ({tick})"
        );
        let pk = sink.public_key();
        assert!(
            verify_signed_chain(sink.inner(), sink.spans(), &pk, &SignedAnchor::genesis()).is_ok()
        );
    }
}

/// Test-only accessor: the open pending span's `opened_at_ns`, if any.
/// Lives outside `SignedWitnessSink`'s public API on purpose — production
/// callers only need `next_timeout_deadline_ns`/`check_timeout(_at)`; this
/// exists solely so the property test above can assert on raw age.
fn sink_pending_opened_at<S: WitnessSink>(sink: &SignedWitnessSink<S>) -> Option<u64> {
    sink.next_timeout_deadline_ns()
        .map(|deadline| deadline - max_wait_ns_of(sink))
}

fn max_wait_ns_of<S: WitnessSink>(sink: &SignedWitnessSink<S>) -> u64 {
    match sink.strategy() {
        SigningStrategy::BatchTailTimeout { max_wait_ns, .. } => max_wait_ns,
        _ => 0,
    }
}

#[test]
fn honest_chain_verifies_under_batch_tail_timeout_with_forced_partial_seals() {
    // Mix count-closes and a deliberate timeout-forced partial seal, then
    // confirm the whole thing still verifies end to end — the new strategy
    // must not weaken any existing guarantee from `verify_signed_chain`.
    let mut sink = sink_for(SigningStrategy::BatchTailTimeout {
        batch_size: 4,
        max_wait_ns: 100,
    });
    let batch1 = mk_records(0, 4, 0); // exactly fills -> count-close
    sink.emit_batch_at(&batch1, 0).unwrap();
    assert_eq!(sink.spans().len(), 1);
    let prev = sink.inner().records.last().unwrap().chain_hash();
    let batch2 = mk_records(4, 2, prev); // partial, never fills
    sink.emit_batch_at(&batch2, 10).unwrap();
    assert_eq!(sink.unsigned_pending(), 2);
    assert!(sink.check_timeout_at(110)); // forced by timeout, not count
    assert_eq!(sink.spans().len(), 2);
    sink.seal(); // no-op, nothing pending
    let pk = sink.public_key();
    let report = verify_signed_chain(sink.inner(), sink.spans(), &pk, &SignedAnchor::genesis())
        .expect("signed chain verifies");
    assert_eq!(report.records_verified, 6);
}
