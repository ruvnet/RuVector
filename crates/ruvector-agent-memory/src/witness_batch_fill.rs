//! Count-or-timeout batch closing for [`SignedWitnessSink`]'s batched
//! strategies (ADR-352).
//!
//! This ports the *pattern* of `ruvector-retrieval-receipt::batch_fill`
//! (ADR-343: close a batch at `N` members or `T` since the oldest pending
//! member arrived, whichever comes first) into this crate without taking
//! that crate as a dependency. The differences are deliberate:
//!
//! - ADR-343's `BatchScheduler` is a pure, clock-free decision object driven
//!   by a discrete-event simulation that decides *when* a timeout fires.
//!   Here the decision is made inside a live sink against a real
//!   [`Instant`], and the "batch" is the sink's single open [`PendingSpan`].
//! - The timeout is **checked on write**: on every `emit_batch` (after the
//!   inner sink has committed) and on every [`SignedWitnessSink::seal_expired`]
//!   call. There is no background thread or timer. The worst-case
//!   signature-availability latency is therefore
//!   `max_wait + (gap until the next write or seal_expired call)`, not
//!   `max_wait` alone — a caller wanting a hard bound must poll
//!   `seal_expired` from its own timer.
//! - A timeout close is signed exactly like a size close
//!   ([`SignPurpose::BatchTail`](super::SignPurpose)): the signed statement is "these records,
//!   after that span", independent of why the span closed, so
//!   [`verify_signed_chain`](super::verify_signed_chain) is unchanged.

use super::{PendingSpan, RecordsHasher, SignedWitnessSink, SigningStrategy, WitnessSink};
use crate::ops::LedgerWitnessRecord;
use std::time::{Duration, Instant};

impl<S: WitnessSink> SignedWitnessSink<S> {
    /// Append already-committed `records` to the open batch, closing it at
    /// `batch_size`. With `now = Some(_)` (only `BatchTailTimeout`), a batch
    /// opened by these records is stamped `now`, and after appending, a batch
    /// whose oldest record is at least `max_wait` old is closed too — so the
    /// records of the write that discovers the expiry ride in the same
    /// signature instead of opening a fresh batch.
    pub(super) fn push_batched(
        &mut self,
        records: &[LedgerWitnessRecord],
        batch_size: usize,
        now: Option<Instant>,
    ) {
        for r in records {
            let p = self.pending.get_or_insert_with(|| PendingSpan {
                from: r.sequence,
                to: r.sequence,
                count: 0,
                hasher: RecordsHasher::new(),
                opened_at: now,
            });
            p.hasher.push(r);
            p.to = r.sequence;
            p.count += 1;
            if p.count >= batch_size {
                self.seal();
            }
        }
        if let Some(now) = now {
            self.seal_expired_at(now);
        }
    }

    /// Under [`SigningStrategy::BatchTailTimeout`], sign the open batch if
    /// its oldest record has waited at least `max_wait`; returns whether a
    /// span was closed. A no-op (returns `false`) under every other
    /// strategy, or when nothing is pending or the deadline has not passed.
    ///
    /// This is the hook for a caller-owned timer: the sink itself never
    /// wakes up, so without writes an expired batch is only signed when
    /// this (or [`SignedWitnessSink::seal`]) is called.
    pub fn seal_expired(&mut self) -> bool {
        match self.strategy {
            SigningStrategy::BatchTailTimeout { .. } => self.seal_expired_at(Instant::now()),
            _ => false,
        }
    }

    /// [`Self::seal_expired`] against a caller-supplied instant. Lets tests
    /// simulate elapsed time without sleeping.
    pub(crate) fn seal_expired_at(&mut self, now: Instant) -> bool {
        let SigningStrategy::BatchTailTimeout { max_wait, .. } = self.strategy else {
            return false;
        };
        let expired = self
            .pending
            .as_ref()
            .and_then(|p| p.opened_at)
            .is_some_and(|t| now.saturating_duration_since(t) >= max_wait);
        if expired {
            self.seal();
        }
        expired
    }

    /// How long the oldest not-yet-signed record has been waiting, measured
    /// against the real clock. `None` when nothing is pending, and always
    /// `None` under plain `BatchTail` (which never records open times).
    pub fn pending_age(&self) -> Option<Duration> {
        self.pending
            .as_ref()
            .and_then(|p| p.opened_at)
            .map(|t| t.elapsed())
    }

    /// Only `BatchTailTimeout` reads the clock: one read per `emit_batch`.
    pub(super) fn clock_if_timed(&self, clock: impl FnOnce() -> Instant) -> Option<Instant> {
        match self.strategy {
            SigningStrategy::BatchTailTimeout { .. } => Some(clock()),
            _ => None,
        }
    }

    /// `emit_batch` with a caller-supplied clock reading (tests only).
    #[cfg(test)]
    pub(crate) fn emit_batch_at(
        &mut self,
        records: &[LedgerWitnessRecord],
        now: Instant,
    ) -> Result<(), crate::ops::LedgerError> {
        let now = self.clock_if_timed(|| now);
        self.emit_batch_inner(records, now)
    }
}
