//! ADR-352: `SigningStrategy::BatchTailTimeout` (count-or-timeout batch
//! closing). Elapsed time is simulated through `emit_batch_at` /
//! `seal_expired_at` with synthetic `Instant`s; exactly one test touches the
//! real clock, sleeping 3 ms.

use super::*;
use std::time::{Duration, Instant};

const MS: Duration = Duration::from_millis(1);

/// `n` honest, contiguous witness records (2 per ledger add+accept pair).
fn honest_records(pairs: usize) -> Vec<LedgerWitnessRecord> {
    let mut ledger =
        TransactionalLedger::new(MemoryWitnessLog::default(), AlwaysAdmitGate::default());
    for i in 0..pairs {
        let id = ledger.add(format!("t {i}"), &[], "a", "r").unwrap();
        ledger.accept(id, "a", "v").unwrap();
    }
    ledger.into_witness_sink().records
}

fn timeout(batch_size: usize, max_wait: Duration) -> SignedWitnessSink<MemoryWitnessLog> {
    sink_for(SigningStrategy::BatchTailTimeout {
        batch_size,
        max_wait,
    })
}

fn covers(s: &SignedSpan) -> (u64, u64) {
    (s.covers_from_seq, s.covers_to_seq)
}

#[test]
fn stale_partial_batch_is_signed_by_the_next_write_after_max_wait() {
    let recs = honest_records(3); // 6 records
    let mut sink = timeout(64, 10 * MS);
    let t0 = Instant::now();
    sink.emit_batch_at(&recs[0..1], t0).unwrap();
    sink.emit_batch_at(&recs[1..2], t0 + MS).unwrap();
    assert_eq!((sink.spans().len(), sink.unsigned_pending()), (0, 2));
    // Just before the deadline nothing closes.
    sink.emit_batch_at(&recs[2..3], t0 + 10 * MS - Duration::from_nanos(1))
        .unwrap();
    assert_eq!((sink.spans().len(), sink.unsigned_pending()), (0, 3));
    // The first write at/after the deadline closes the stale batch and rides
    // its own records along in the same signature.
    sink.emit_batch_at(&recs[3..4], t0 + 10 * MS).unwrap();
    assert_eq!(sink.spans().len(), 1);
    assert_eq!(covers(&sink.spans()[0]), (0, 3));
    assert_eq!(sink.spans()[0].purpose, SignPurpose::BatchTail);
    assert_eq!(sink.unsigned_pending(), 0);
    // A new batch opens at the next write's time, not the old deadline.
    sink.emit_batch_at(&recs[4..6], t0 + 11 * MS).unwrap();
    assert!(!sink.seal_expired_at(t0 + 20 * MS));
    assert!(sink.seal_expired_at(t0 + 21 * MS));
    assert_eq!(covers(&sink.spans()[1]), (4, 5));
    let report = verify(&sink, sink.inner(), sink.spans(), &sink.anchor()).unwrap();
    assert_eq!((report.records_verified, report.spans_verified), (6, 2));
}

#[test]
fn seal_expired_bounds_latency_without_any_further_write() {
    let recs = honest_records(1);
    let mut sink = timeout(64, 5 * MS);
    let t0 = Instant::now();
    sink.emit_batch_at(&recs, t0).unwrap();
    assert!(!sink.seal_expired_at(t0 + 4 * MS));
    assert_eq!(
        verify(&sink, sink.inner(), sink.spans(), &GENESIS),
        Err(SignedChainError::UnsignedTail {
            signed: 0,
            unsigned: 2
        })
    );
    assert!(sink.seal_expired_at(t0 + 5 * MS));
    assert!(!sink.seal_expired_at(t0 + 50 * MS), "nothing left pending");
    assert!(verify(&sink, sink.inner(), sink.spans(), &sink.anchor()).is_ok());
}

/// Batches that fill before the deadline behave byte-for-byte like plain
/// `BatchTail` (Ed25519 is deterministic, so even signatures match).
#[test]
fn size_reached_before_timeout_is_identical_to_plain_batch_tail() {
    let recs = honest_records(23); // 46 records
    let mut plain = sink_for(SigningStrategy::BatchTail { batch_size: 5 });
    let mut timed = timeout(5, 10 * MS);
    let t0 = Instant::now();
    for (i, r) in recs.iter().enumerate() {
        plain.emit_batch(std::slice::from_ref(r)).unwrap();
        // 1 µs apart: every 5-record batch fills in 4 µs << 10 ms.
        timed
            .emit_batch_at(
                std::slice::from_ref(r),
                t0 + Duration::from_micros(i as u64),
            )
            .unwrap();
    }
    assert_eq!(plain.unsigned_pending(), timed.unsigned_pending());
    plain.seal();
    timed.seal();
    assert_eq!(plain.spans().len(), 10);
    for (a, b) in plain.spans().iter().zip(timed.spans()) {
        assert_eq!(covers(a), covers(b));
        assert_eq!(a.records_digest, b.records_digest);
        assert_eq!(a.signature, b.signature);
    }
    assert_eq!(plain.anchor(), timed.anchor());
}

#[test]
fn a_multi_record_write_can_close_on_size_then_on_timeout() {
    let recs = honest_records(4); // 8 records
    let mut sink = timeout(3, 10 * MS);
    let t0 = Instant::now();
    sink.emit_batch_at(&recs[0..1], t0).unwrap();
    // At t0+10ms: records 1,2 fill the stale batch (size close, 0..=2), then
    // 3..=7 form a size close (3..=5) and a fresh batch (6..=7) opened *now*,
    // which is not expired.
    sink.emit_batch_at(&recs[1..8], t0 + 10 * MS).unwrap();
    let got: Vec<_> = sink.spans().iter().map(covers).collect();
    assert_eq!(got, vec![(0, 2), (3, 5)]);
    assert_eq!(sink.unsigned_pending(), 2);
    assert!(sink.seal_expired_at(t0 + 20 * MS));
    assert!(verify(&sink, sink.inner(), sink.spans(), &sink.anchor()).is_ok());
}

#[test]
fn refused_write_neither_signs_nor_closes_the_stale_batch() {
    let recs = honest_records(2);
    let mut sink = timeout(64, MS);
    let t0 = Instant::now();
    sink.emit_batch_at(&recs[0..2], t0).unwrap();
    let mut bogus = recs[2];
    bogus.sequence += 7;
    assert!(sink.emit_batch_at(&[bogus], t0 + 50 * MS).is_err());
    assert_eq!((sink.spans().len(), sink.inner().records.len()), (0, 2));
    assert!(sink.seal_expired_at(t0 + 50 * MS));
    assert!(verify(&sink, sink.inner(), sink.spans(), &sink.anchor()).is_ok());
}

#[test]
fn diligent_forgery_inside_a_timeout_closed_span_is_rejected() {
    let recs = honest_records(8); // 16 records
    let mut sink = timeout(64, 10 * MS);
    let t0 = Instant::now();
    // Four stale windows of 4 records each: every span closes on timeout.
    for (w, chunk) in recs.chunks(4).enumerate() {
        let base = t0 + (w as u32) * 20 * MS;
        sink.emit_batch_at(&chunk[..2], base).unwrap();
        sink.emit_batch_at(&chunk[2..], base + 3 * MS).unwrap();
        assert!(sink.seal_expired_at(base + 10 * MS));
    }
    assert_eq!(sink.spans().len(), 4);
    let anchor = sink.anchor();
    assert!(verify(&sink, sink.inner(), sink.spans(), &anchor).is_ok());
    let mut forged = sink.inner().clone();
    forged.records[9].payload ^= 0xDEAD_BEEF;
    rehash_from(&mut forged, 9);
    assert!(forged.verify_chain(), "unsigned walk is fooled");
    assert_eq!(
        verify(&sink, &forged, sink.spans(), &anchor),
        Err(SignedChainError::RecordsMismatch { span_index: 2 })
    );
}

#[test]
fn seal_expired_is_a_noop_for_other_strategies_and_zero_batch_is_refused() {
    for s in [
        SigningStrategy::PerRecord,
        SigningStrategy::BatchTail { batch_size: 64 },
    ] {
        let mut sink = run(sink_for(s), "m", 3);
        assert!(!sink.seal_expired());
        assert!(!sink.seal_expired_at(Instant::now() + 3600 * 1000 * MS));
        assert_eq!(sink.pending_age(), None);
    }
    let r = SignedWitnessSink::from_keypair(
        MemoryWitnessLog::default(),
        &keypair(),
        SigningStrategy::BatchTailTimeout {
            batch_size: 0,
            max_wait: MS,
        },
    );
    assert_eq!(r.err(), Some(WitnessSignerError::ZeroBatchSize));
}

/// The only real-clock test: the public `emit_batch` path reads `Instant`.
#[test]
fn real_clock_timeout_fires_through_the_ledger() {
    let mut ledger = TransactionalLedger::new(timeout(64, 2 * MS), AlwaysAdmitGate::default());
    let id = ledger.add("first".to_string(), &[], "a", "r").unwrap();
    assert!(ledger.witness_sink().pending_age().is_some());
    assert!(ledger.witness_sink().spans().is_empty());
    std::thread::sleep(3 * MS);
    ledger.accept(id, "a", "v").unwrap();
    let sink = ledger.into_witness_sink();
    assert_eq!(sink.spans().len(), 1);
    assert_eq!(sink.unsigned_pending(), 0);
    assert!(verify(&sink, sink.inner(), sink.spans(), &sink.anchor()).is_ok());
}
