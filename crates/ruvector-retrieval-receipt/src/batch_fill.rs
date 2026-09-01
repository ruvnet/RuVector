//! Batch-fill scheduling for signed retrieval-receipt batches.
//!
//! ADR-340's `signing` benchmark measured only the CPU cost of
//! signing/verifying an already-assembled batch. It named, but explicitly
//! did not model, the wall-clock cost of *assembling* that batch from a
//! live query stream: a fixed-size-only policy can leave a query's signed
//! anchor unavailable indefinitely if queries arrive slower than the batch
//! fills (see that ADR's Limitations and Failure Modes sections).
//!
//! This module implements the batch-fill *decision* in isolation from the
//! cryptography, so a discrete-event simulation (`bin/batch_latency.rs`)
//! can combine it with real `Issuer`/`BatchAnchor` signing costs to produce
//! an end-to-end receipt-availability latency measurement.
//!
//! Time is represented as `u64` nanoseconds throughout, matching the
//! resolution `std::time::Instant` gives the real signing-cost
//! measurements this module is paired with.

/// A batch-fill policy: close a batch once it reaches `max_members`, or
/// once `max_wait_ns` has elapsed since the batch's oldest pending member
/// arrived — whichever happens first. `max_wait_ns = None` means
/// fixed-size-only: a batch never closes early, however long it waits.
#[derive(Clone, Copy, Debug)]
pub struct BatchFillPolicy {
    pub max_members: usize,
    pub max_wait_ns: Option<u64>,
}

impl BatchFillPolicy {
    /// Fixed-size-only: a batch closes only once `max_members` have
    /// arrived, with no upper bound on wait time.
    pub const fn fixed_size(max_members: usize) -> Self {
        Self {
            max_members,
            max_wait_ns: None,
        }
    }

    /// Hybrid: closes at `max_members`, or after `max_wait_ns` since the
    /// oldest pending member, whichever comes first.
    pub const fn hybrid(max_members: usize, max_wait_ns: u64) -> Self {
        Self {
            max_members,
            max_wait_ns: Some(max_wait_ns),
        }
    }
}

/// One arrived, not-yet-anchored receipt root awaiting a batch close.
#[derive(Clone, Copy, Debug)]
pub struct PendingMember {
    pub query_index: usize,
    pub arrived_at_ns: u64,
}

/// Decides when a run of arrivals closes into a batch to sign. Holds no
/// cryptographic state — only arrival bookkeeping — so it can be driven
/// deterministically in unit tests and reused by a simulation that supplies
/// real signing costs.
#[derive(Debug)]
pub struct BatchScheduler {
    policy: BatchFillPolicy,
    pending: Vec<PendingMember>,
}

impl BatchScheduler {
    pub fn new(policy: BatchFillPolicy) -> Self {
        Self {
            policy,
            pending: Vec::with_capacity(policy.max_members),
        }
    }

    pub const fn policy(&self) -> BatchFillPolicy {
        self.policy
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// The arrival time of the oldest pending (unclosed) member, if any.
    /// A caller drives timeout scheduling from this: the next timeout
    /// deadline is `oldest_pending_arrival_ns() + max_wait_ns`.
    pub fn oldest_pending_arrival_ns(&self) -> Option<u64> {
        self.pending.first().map(|m| m.arrived_at_ns)
    }

    /// Record a new arrival. Returns the closed batch if this arrival fills
    /// it to `max_members`.
    pub fn arrive(&mut self, query_index: usize, arrived_at_ns: u64) -> Option<Vec<PendingMember>> {
        self.pending.push(PendingMember {
            query_index,
            arrived_at_ns,
        });
        if self.pending.len() >= self.policy.max_members {
            return Some(std::mem::take(&mut self.pending));
        }
        None
    }

    /// Close the current pending batch because its fill-timeout elapsed at
    /// `now_ns`. The caller is responsible for only invoking this once
    /// `now_ns >= oldest_pending_arrival_ns() + max_wait_ns`; this method
    /// does not re-check the deadline so a caller driving a discrete-event
    /// simulation controls exactly when the timeout fires. Returns `None`
    /// if nothing is pending (a stale/already-closed timeout event).
    pub fn close_on_timeout(&mut self) -> Option<Vec<PendingMember>> {
        if self.pending.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.pending))
        }
    }

    /// Force-close whatever is pending, e.g. at the end of a simulation or
    /// deployment shutdown. Returns `None` if nothing is pending.
    pub fn flush(&mut self) -> Option<Vec<PendingMember>> {
        self.close_on_timeout()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_size_closes_exactly_at_max_members() {
        let mut s = BatchScheduler::new(BatchFillPolicy::fixed_size(3));
        assert!(s.arrive(0, 100).is_none());
        assert!(s.arrive(1, 200).is_none());
        let closed = s.arrive(2, 300).expect("third arrival fills the batch");
        assert_eq!(closed.len(), 3);
        assert_eq!(
            closed.iter().map(|m| m.query_index).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(s.pending_len(), 0);
    }

    #[test]
    fn fixed_size_never_closes_on_timeout() {
        let mut s = BatchScheduler::new(BatchFillPolicy::fixed_size(32));
        s.arrive(0, 0);
        // A fixed-size-only policy has no timeout to check; a caller must
        // never derive a deadline for it (max_wait_ns is None), so there is
        // nothing to assert here except that pending state is unaffected
        // by the passage of arbitrarily large amounts of virtual time.
        assert_eq!(s.policy().max_wait_ns, None);
        assert_eq!(s.pending_len(), 1);
    }

    #[test]
    fn hybrid_derives_timeout_deadline_from_oldest_pending_arrival() {
        let mut s = BatchScheduler::new(BatchFillPolicy::hybrid(32, 50_000_000));
        assert_eq!(s.oldest_pending_arrival_ns(), None);
        s.arrive(0, 1_000_000);
        assert_eq!(s.oldest_pending_arrival_ns(), Some(1_000_000));
        s.arrive(1, 2_000_000);
        // Oldest pending member does not change when a second one arrives.
        assert_eq!(s.oldest_pending_arrival_ns(), Some(1_000_000));
    }

    #[test]
    fn hybrid_close_on_timeout_flushes_a_partial_batch() {
        let mut s = BatchScheduler::new(BatchFillPolicy::hybrid(32, 50_000_000));
        s.arrive(0, 1_000_000);
        s.arrive(1, 2_000_000);
        assert_eq!(s.pending_len(), 2);
        let closed = s.close_on_timeout().expect("two pending members to flush");
        assert_eq!(closed.len(), 2);
        assert_eq!(s.pending_len(), 0);
        assert_eq!(s.oldest_pending_arrival_ns(), None);
    }

    #[test]
    fn close_on_timeout_is_none_when_nothing_pending() {
        let mut s = BatchScheduler::new(BatchFillPolicy::hybrid(4, 1_000));
        assert!(s.close_on_timeout().is_none());
    }

    #[test]
    fn a_fresh_batch_starts_after_a_close() {
        let mut s = BatchScheduler::new(BatchFillPolicy::fixed_size(2));
        s.arrive(0, 0);
        let closed = s.arrive(1, 10).expect("fills at 2");
        assert_eq!(closed.len(), 2);
        assert_eq!(s.oldest_pending_arrival_ns(), None);
        s.arrive(2, 20);
        assert_eq!(s.oldest_pending_arrival_ns(), Some(20));
        assert_eq!(s.pending_len(), 1);
    }

    #[test]
    fn flush_drains_remaining_pending_then_reports_empty() {
        let mut s = BatchScheduler::new(BatchFillPolicy::fixed_size(8));
        s.arrive(0, 0);
        s.arrive(1, 5);
        s.arrive(2, 9);
        let flushed = s.flush().expect("three pending members to flush");
        assert_eq!(flushed.len(), 3);
        assert!(s.flush().is_none());
    }
}
