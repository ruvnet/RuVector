//! Bounded signature-availability latency for `BatchTail` witness signing.
//!
//! [`SigningStrategy::BatchTail`] ([`crate::witness_signing`]) closes a
//! pending span once `batch_size` records have arrived, with no upper bound
//! on how long that can take: the 2026-09-16 nightly's Next Research item 1
//! named exactly this gap — "whenever the batch happens to fill" is not a
//! deployable signature-availability guarantee under a slow or bursty
//! arrival process, only under one that reliably fills `batch_size` fast.
//!
//! This module closes that gap by porting the batch-fill *decision*
//! methodology already built and measured for signed retrieval receipts
//! (`ruvector_retrieval_receipt::batch_fill::{BatchFillPolicy, BatchScheduler}`:
//! close at `max_members`, or after `max_wait_ns` since the oldest pending
//! member, whichever comes first) onto `SignedWitnessSink`'s `BatchTail`
//! strategy — as a decorator, not a modification. [`TimedBatchTailSink`]
//! wraps an existing, unmodified [`SignedWitnessSink`] and adds exactly the
//! wall-clock seal-on-timeout decision, through an injectable [`Clock`] so a
//! benchmark/simulation can drive it with virtual time
//! ([`LogicalClock`]) while a real deployment drives it with
//! [`SystemClock`]. `witness_signing.rs` itself is untouched: this module
//! only calls its existing public API (`unsigned_pending()`, `seal()`).
//!
//! A real deployment is responsible for calling [`TimedBatchTailSink::check_timeout`]
//! periodically (e.g. from a background timer, or opportunistically on every
//! unrelated tick); [`TimedBatchTailSink::pending_deadline_ns`] tells it when
//! the next check must fire at the latest, mirroring
//! `BatchScheduler::oldest_pending_arrival_ns()`'s role in the ported
//! methodology.

use crate::ops::{LedgerError, LedgerWitnessRecord, WitnessSink};
use crate::witness_signing::{SignedAnchor, SignedSpan, SignedWitnessSink};
use std::cell::Cell;
use std::time::{SystemTime, UNIX_EPOCH};

/// A nanosecond time source. Production code uses [`SystemClock`]; tests and
/// benchmarks use [`LogicalClock`] for deterministic, reproducible timing.
pub trait Clock {
    fn now_ns(&self) -> u64;
}

/// Wall-clock time since the Unix epoch, in nanoseconds.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ns(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    }
}

/// A manually-advanced clock for deterministic simulation. Starts at
/// whatever `start_ns` the caller supplies and only moves when
/// [`LogicalClock::set_ns`] is called — never on its own.
#[derive(Debug, Default)]
pub struct LogicalClock(Cell<u64>);

impl LogicalClock {
    pub fn new(start_ns: u64) -> Self {
        Self(Cell::new(start_ns))
    }

    pub fn set_ns(&self, now_ns: u64) {
        self.0.set(now_ns);
    }
}

impl Clock for LogicalClock {
    fn now_ns(&self) -> u64 {
        self.0.get()
    }
}

/// Lets a caller retain a handle to advance a [`LogicalClock`] it has
/// already handed to a [`TimedBatchTailSink`] (which otherwise takes the
/// clock by value).
impl Clock for &LogicalClock {
    fn now_ns(&self) -> u64 {
        self.0.get()
    }
}

/// Decorates a [`SignedWitnessSink`] configured with
/// [`SigningStrategy::BatchTail`](crate::witness_signing::SigningStrategy::BatchTail)
/// with a wall-clock fill timeout: the open span is sealed early once
/// `max_wait_ns` has elapsed since it opened, even if `batch_size` records
/// have not yet arrived. Wrapping a sink using `SigningStrategy::PerRecord`
/// is harmless but pointless: nothing is ever left pending, so
/// [`check_timeout`](Self::check_timeout) is always a no-op.
pub struct TimedBatchTailSink<S: WitnessSink, C: Clock = SystemClock> {
    inner: SignedWitnessSink<S>,
    max_wait_ns: u64,
    opened_at_ns: Option<u64>,
    clock: C,
    timeout_seals: usize,
}

impl<S: WitnessSink, C: Clock> TimedBatchTailSink<S, C> {
    /// Wrap `inner` (already configured with whatever `SigningStrategy` it
    /// was built with) and bound its worst-case open-span age at
    /// `max_wait_ns`. `max_wait_ns = 0` seals on every
    /// [`check_timeout`](Self::check_timeout) call while anything is
    /// pending — a legal, if degenerate, policy.
    pub fn new(inner: SignedWitnessSink<S>, max_wait_ns: u64, clock: C) -> Self {
        Self {
            inner,
            max_wait_ns,
            opened_at_ns: None,
            clock,
            timeout_seals: 0,
        }
    }

    /// Seal the open span now if `max_wait_ns` has elapsed since it opened,
    /// per the injected clock. Returns `true` iff a seal happened. A real
    /// deployment calls this periodically; [`pending_deadline_ns`](Self::pending_deadline_ns)
    /// says the latest time it must do so by.
    pub fn check_timeout(&mut self) -> bool {
        self.check_timeout_at(self.clock.now_ns())
    }

    /// As [`check_timeout`](Self::check_timeout), but against an explicit
    /// `now_ns` instead of the injected clock — what a discrete-event
    /// simulation drives directly.
    pub fn check_timeout_at(&mut self, now_ns: u64) -> bool {
        let Some(opened) = self.opened_at_ns else {
            return false;
        };
        if self.inner.unsigned_pending() == 0 {
            // The span closed by batch_size since it was opened; nothing to
            // time out. (Kept in sync eagerly by `emit_batch`, but checked
            // again here so a stale `opened_at_ns` can never fire twice.)
            self.opened_at_ns = None;
            return false;
        }
        if now_ns.saturating_sub(opened) < self.max_wait_ns {
            return false;
        }
        self.inner.seal();
        self.opened_at_ns = None;
        self.timeout_seals += 1;
        true
    }

    /// The latest time by which [`check_timeout`](Self::check_timeout) must
    /// next be called for the timeout bound to hold, or `None` if nothing is
    /// pending. Mirrors `BatchScheduler::oldest_pending_arrival_ns()`'s role
    /// in the ported methodology.
    pub fn pending_deadline_ns(&self) -> Option<u64> {
        self.opened_at_ns.map(|opened| opened + self.max_wait_ns)
    }

    /// How many of the spans sealed so far were sealed by the timeout firing
    /// rather than `batch_size` being reached.
    pub fn timeout_seals(&self) -> usize {
        self.timeout_seals
    }

    pub const fn max_wait_ns(&self) -> u64 {
        self.max_wait_ns
    }

    pub fn public_key(&self) -> [u8; 32] {
        self.inner.public_key()
    }

    pub fn spans(&self) -> &[SignedSpan] {
        self.inner.spans()
    }

    pub fn anchor(&self) -> SignedAnchor {
        self.inner.anchor()
    }

    pub fn unsigned_pending(&self) -> usize {
        self.inner.unsigned_pending()
    }

    pub fn inner_log(&self) -> &S {
        self.inner.inner()
    }

    /// Force-seal whatever is pending, bypassing both `batch_size` and the
    /// timeout (e.g. at deployment shutdown).
    pub fn seal(&mut self) {
        self.inner.seal();
        self.opened_at_ns = None;
    }
}

impl<S: WitnessSink, C: Clock> WitnessSink for TimedBatchTailSink<S, C> {
    fn emit_batch(&mut self, records: &[LedgerWitnessRecord]) -> Result<(), LedgerError> {
        if self.inner.unsigned_pending() == 0 && !records.is_empty() {
            self.opened_at_ns = Some(self.clock.now_ns());
        }
        self.inner.emit_batch(records)?;
        if self.inner.unsigned_pending() == 0 {
            // `batch_size` closed the span inside `inner` already.
            self.opened_at_ns = None;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::{AlwaysAdmitGate, TransactionalLedger};
    use crate::ops::MemoryWitnessLog;
    use crate::witness_signing::{verify_signed_chain, SigningStrategy};
    use rvf_types::ed25519::Ed25519Keypair;

    const SECRET: [u8; 32] = [19u8; 32];

    fn timed_sink(
        batch_size: usize,
        max_wait_ns: u64,
        clock: LogicalClock,
    ) -> TimedBatchTailSink<MemoryWitnessLog, LogicalClock> {
        let inner = SignedWitnessSink::from_keypair(
            MemoryWitnessLog::default(),
            &Ed25519Keypair::from_secret(&SECRET),
            SigningStrategy::BatchTail { batch_size },
        )
        .unwrap();
        TimedBatchTailSink::new(inner, max_wait_ns, clock)
    }

    #[test]
    fn size_close_clears_pending_and_timeout_never_fires() {
        let clock = LogicalClock::new(0);
        let mut sink = timed_sink(2, 1_000, clock);
        let mut ledger = TransactionalLedger::new(sink, AlwaysAdmitGate::default());
        let id = ledger.add("a", &[], "actor", "r").unwrap();
        ledger.accept(id, "actor", "r").unwrap(); // 2 records -> closes at batch_size=2
        sink = ledger.into_witness_sink();
        assert_eq!(sink.unsigned_pending(), 0);
        assert_eq!(sink.pending_deadline_ns(), None);
        assert!(!sink.check_timeout_at(10_000));
        assert_eq!(sink.timeout_seals(), 0);
        assert_eq!(sink.spans().len(), 1);
    }

    #[test]
    fn timeout_seals_a_partial_span_that_size_alone_never_would() {
        let clock = LogicalClock::new(0);
        let mut sink = timed_sink(64, 1_000, clock);
        let mut ledger = TransactionalLedger::new(sink, AlwaysAdmitGate::default());
        let id = ledger.add("a", &[], "actor", "r").unwrap();
        ledger.accept(id, "actor", "r").unwrap(); // 2 records, far below batch_size=64
        sink = ledger.into_witness_sink();
        assert_eq!(sink.unsigned_pending(), 2);
        assert_eq!(sink.pending_deadline_ns(), Some(1_000));

        // Before the deadline: no seal.
        assert!(!sink.check_timeout_at(999));
        assert_eq!(sink.unsigned_pending(), 2);

        // At the deadline: seals exactly the 2 pending records.
        assert!(sink.check_timeout_at(1_000));
        assert_eq!(sink.unsigned_pending(), 0);
        assert_eq!(sink.timeout_seals(), 1);
        assert_eq!(sink.spans().len(), 1);
        assert_eq!(sink.spans()[0].covers_to_seq, 1);

        // A stale re-check after the seal is a no-op, not a double-seal.
        assert!(!sink.check_timeout_at(5_000));
        assert_eq!(sink.spans().len(), 1);
    }

    #[test]
    fn a_fresh_span_reopens_a_fresh_deadline_after_a_timeout_seal() {
        let clock = LogicalClock::new(0);
        let inner = SignedWitnessSink::from_keypair(
            MemoryWitnessLog::default(),
            &Ed25519Keypair::from_secret(&SECRET),
            SigningStrategy::BatchTail { batch_size: 64 },
        )
        .unwrap();
        let sink = TimedBatchTailSink::new(inner, 1_000, &clock);
        // One continuous ledger throughout: `witness_sink_mut` lets a live
        // deployment call `check_timeout` between bursts without tearing
        // the ledger (and its witness-sequence counter) down. `&clock` lets
        // the test advance the same virtual clock the sink reads from.
        let mut ledger = TransactionalLedger::new(sink, AlwaysAdmitGate::default());
        let id = ledger.add("a", &[], "actor", "r").unwrap();
        ledger.accept(id, "actor", "r").unwrap();
        clock.set_ns(1_000);
        assert!(ledger.witness_sink_mut().check_timeout());

        let id = ledger.add("b", &[], "actor", "r").unwrap();
        ledger.accept(id, "actor", "r").unwrap();
        let mut sink = ledger.into_witness_sink();
        assert_eq!(sink.pending_deadline_ns(), Some(2_000));
        clock.set_ns(1_999);
        assert!(!sink.check_timeout());
        clock.set_ns(2_000);
        assert!(sink.check_timeout());
        assert_eq!(sink.timeout_seals(), 2);
        assert_eq!(sink.spans().len(), 2);
    }

    #[test]
    fn timeout_sealed_chain_verifies_like_any_other_signed_chain() {
        let clock = LogicalClock::new(0);
        let mut sink = timed_sink(64, 1_000, clock);
        let mut ledger = TransactionalLedger::new(sink, AlwaysAdmitGate::default());
        for i in 0..5 {
            let id = ledger.add(format!("m{i}"), &[], "actor", "r").unwrap();
            ledger.accept(id, "actor", "r").unwrap();
        }
        sink = ledger.into_witness_sink();
        assert!(sink.check_timeout_at(1_000));
        sink.seal(); // flush any trailing partial span

        let pk = sink.public_key();
        let anchor = sink.anchor();
        let report = verify_signed_chain(sink.inner_log(), sink.spans(), &pk, &anchor).unwrap();
        assert_eq!(report.records_verified, 10);
        assert_eq!(report.head, anchor);
    }

    #[test]
    fn zero_max_wait_seals_on_every_check_while_pending() {
        let clock = LogicalClock::new(0);
        let mut sink = timed_sink(64, 0, clock);
        let mut ledger = TransactionalLedger::new(sink, AlwaysAdmitGate::default());
        let id = ledger.add("a", &[], "actor", "r").unwrap();
        ledger.accept(id, "actor", "r").unwrap();
        sink = ledger.into_witness_sink();
        assert!(sink.check_timeout_at(0));
        assert_eq!(sink.timeout_seals(), 1);
    }

    #[test]
    fn per_record_strategy_never_has_anything_to_time_out() {
        let clock = LogicalClock::new(0);
        let inner = SignedWitnessSink::from_keypair(
            MemoryWitnessLog::default(),
            &Ed25519Keypair::from_secret(&SECRET),
            SigningStrategy::PerRecord,
        )
        .unwrap();
        let mut sink = TimedBatchTailSink::new(inner, 1_000, clock);
        let mut ledger = TransactionalLedger::new(sink, AlwaysAdmitGate::default());
        let id = ledger.add("a", &[], "actor", "r").unwrap();
        ledger.accept(id, "actor", "r").unwrap();
        sink = ledger.into_witness_sink();
        assert_eq!(sink.pending_deadline_ns(), None);
        assert!(!sink.check_timeout_at(1_000_000));
        assert_eq!(sink.timeout_seals(), 0);
    }
}
