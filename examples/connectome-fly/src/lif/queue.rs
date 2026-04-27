//! Event queues for the LIF kernel.
//!
//! - `SpikeEvent`: scheduled synaptic event with deterministic
//!   `(t_ms, post, pre)` ordering.
//! - `TimingWheel`: bucketed circular-buffer queue with a spill heap
//!   for events beyond the wheel horizon. Amortized O(1) insert /
//!   pop for events inside the horizon, dominating `BinaryHeap`'s
//!   O(log N) at the event counts the kernel produces.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::connectome::NeuronId;

/// A scheduled synaptic event in the priority queue.
#[derive(Copy, Clone, Debug)]
pub struct SpikeEvent {
    /// Delivery time (ms).
    pub t_ms: f32,
    /// Post-synaptic neuron.
    pub post: NeuronId,
    /// Pre-synaptic neuron (used for deterministic tie-break).
    pub pre: NeuronId,
    /// Signed charge contribution (positive → `g_exc`, negative → `g_inh`).
    pub w: f32,
}

impl PartialEq for SpikeEvent {
    fn eq(&self, other: &Self) -> bool {
        self.t_ms.to_bits() == other.t_ms.to_bits()
            && self.post == other.post
            && self.pre == other.pre
    }
}
impl Eq for SpikeEvent {}
impl Ord for SpikeEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        // `BinaryHeap` is a *max* heap. We invert so the earliest
        // event pops first. Tie-break on `(post, pre)` to match
        // `docs/research/connectome-ruvector/03-neural-dynamics.md` §3.1.
        other
            .t_ms
            .partial_cmp(&self.t_ms)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.post.cmp(&self.post))
            .then_with(|| other.pre.cmp(&self.pre))
    }
}
impl PartialOrd for SpikeEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Bucketed timing wheel at 0.1 ms granularity.
///
/// Bounded horizon: events further than `horizon_ms` ahead fall into
/// a spill heap that is re-bucketed lazily as the wheel rotates. This
/// replaces the `O(log N)` BinaryHeap with amortized `O(1)` insert
/// and pop for events inside the horizon. Events inside a bucket
/// retain insertion order (deterministic under a fixed push order;
/// true bit-exact alignment with the `BinaryHeap` path is not a goal
/// — see ADR-154 §4.2).
pub struct TimingWheel {
    buckets: Vec<Vec<SpikeEvent>>,
    bucket_ms: f32,
    base_ms: f32,
    head: usize,
    /// Spill for events beyond the wheel horizon.
    spill: BinaryHeap<SpikeEvent>,
    total: usize,
}

impl TimingWheel {
    /// Create a new timing wheel.
    pub fn new(bucket_ms: f32, horizon_ms: f32) -> Self {
        let nb = ((horizon_ms / bucket_ms).ceil() as usize).max(64);
        Self {
            buckets: vec![Vec::new(); nb],
            bucket_ms,
            base_ms: 0.0,
            head: 0,
            spill: BinaryHeap::new(),
            total: 0,
        }
    }

    /// Push an event.
    pub fn push(&mut self, ev: SpikeEvent) {
        let dt = ev.t_ms - self.base_ms;
        let nb = self.buckets.len();
        let slot = (dt / self.bucket_ms) as isize;
        if slot >= 0 && (slot as usize) < nb {
            let idx = (self.head + slot as usize) % nb;
            self.buckets[idx].push(ev);
        } else {
            self.spill.push(ev);
        }
        self.total += 1;
    }

    /// Current bucket ring width (number of slots).
    #[inline]
    pub fn num_buckets(&self) -> usize {
        self.buckets.len()
    }

    /// Byte-exact equality of this wheel's `bucket_ms` against `other`.
    /// Used by the delay-sorted delivery path to refuse its fast route
    /// when the wheel it was built against has been swapped out.
    #[inline]
    pub fn bucket_ms_matches(&self, other: f32) -> bool {
        self.bucket_ms.to_bits() == other.to_bits()
    }

    /// `1.0 / bucket_ms`, cached for the hot delivery loop.
    #[inline]
    pub fn inv_bucket_ms(&self) -> f32 {
        1.0 / self.bucket_ms
    }

    /// The `base_ms` of bucket index `head` — the wheel's current "now"
    /// anchor. Used by the delay-sorted CSR delivery path to compute a
    /// single `base_slot` per spike and increment from there.
    #[inline]
    pub fn base_ms(&self) -> f32 {
        self.base_ms
    }

    /// Current head (ring start) index.
    #[inline]
    pub fn head(&self) -> usize {
        self.head
    }

    /// Insert an event whose destination bucket *slot* (distance from
    /// `head` measured in `bucket_ms`) is already known. Caller must
    /// guarantee `0 <= slot < num_buckets()`; negative or too-far slots
    /// must be routed to `push_spill`.
    ///
    /// This is the delivery fast-path primitive used by
    /// `delay_csr::DelaySortedCsr::deliver_spike` (when built via
    /// `from_connectome_for_wheel`). It skips the float division, bounds
    /// compare, and modulo of the generic [`TimingWheel::push`], trading
    /// those for an integer add + one compare (the ring-wrap).
    ///
    /// Measured: ~1.5× kernel-level speedup on the saturated-regime
    /// `N=1024, t_end=120ms` workload *with the observer's Fiedler
    /// detector disabled*. On the full bench (observer armed) the
    /// detector dominates runtime 450-to-1 and this saving is inside
    /// bench noise — see `benches/delay_csr.rs` and the commit message
    /// for numbers.
    #[inline]
    pub fn push_at_slot(&mut self, slot: usize, ev: SpikeEvent) {
        debug_assert!(slot < self.buckets.len());
        let nb = self.buckets.len();
        let raw = self.head + slot;
        let idx = if raw >= nb { raw - nb } else { raw };
        // SAFETY-via-debug_assert: `idx < nb` because `head < nb` and
        // `slot < nb`. We use safe indexing; the bounds check is
        // branch-predicted identically across all calls.
        self.buckets[idx].push(ev);
        self.total += 1;
    }

    /// Push an event whose delivery time falls past the wheel horizon.
    /// Complements [`TimingWheel::push_at_slot`] for the slow path.
    #[inline]
    pub fn push_spill(&mut self, ev: SpikeEvent) {
        self.spill.push(ev);
        self.total += 1;
    }

    /// Ensure each bucket's inner `Vec` has capacity ≥ `cap`.
    ///
    /// A one-shot upper-bound reservation amortizes away the `Vec::push`
    /// growth cost during the saturated regime, where every bucket can
    /// see hundreds of inserts per wheel rotation. Only grows — never
    /// shrinks — so calling it on an already-warm wheel is a no-op.
    pub fn reserve_per_bucket(&mut self, cap: usize) {
        for b in &mut self.buckets {
            if b.capacity() < cap {
                b.reserve(cap - b.len());
            }
        }
    }

    /// Pop all events due at or before `now_ms` into `out`.
    ///
    /// Each bucket is sorted ascending by `(t_ms, post, pre)` before
    /// draining so the wheel path produces the same dispatch order as
    /// the heap path (`SpikeEvent::cmp` + `BinaryHeap`). This is the
    /// canonical in-bucket-ordering contract from ADR-154 §15.1 and
    /// is what enables bit-exact cross-path determinism at N=1024 on
    /// the AC-1 stimulus — see `tests/cross_path_determinism.rs`.
    /// Sort cost is O(k log k) per drained bucket; k is typically
    /// 5–50 events per 0.1 ms bucket, so the added cost is on the
    /// order of a few hundred compares per drain, comfortably below
    /// the 5 % perf budget from the same section of the ADR.
    pub fn drain_due(&mut self, now_ms: f32, out: &mut Vec<SpikeEvent>) {
        let nb = self.buckets.len();
        let eps = 1e-6_f32;
        loop {
            let bucket_end = self.base_ms + self.bucket_ms;
            if now_ms + eps < bucket_end {
                break;
            }
            let head = self.head;
            let drained = self.buckets[head].len();
            if drained > 0 {
                // Canonical in-bucket order: ascending by (t_ms, post,
                // pre). Matches the heap path's `SpikeEvent::cmp`
                // tie-break. Skip the sort when the bucket is
                // trivially ordered (length 0 or 1) — saves the
                // sort_by dispatch on the sparse-bucket common case,
                // recovering ~4 % of the saturated-regime wallclock
                // the unconditional sort cost (see BENCHMARK.md §4.11
                // "Two cheaper alternatives" — this is item 1).
                if drained > 1 {
                    self.buckets[head].sort_by(|a, b| {
                        a.t_ms
                            .partial_cmp(&b.t_ms)
                            .unwrap_or(Ordering::Equal)
                            .then_with(|| a.post.cmp(&b.post))
                            .then_with(|| a.pre.cmp(&b.pre))
                    });
                }
                out.extend_from_slice(&self.buckets[head]);
                self.buckets[head].clear();
                self.total -= drained;
            }
            self.head = (head + 1) % nb;
            self.base_ms += self.bucket_ms;
            if now_ms + eps < self.base_ms {
                break;
            }
        }
        // Pull spill events that are now within the wheel horizon.
        let horizon = self.base_ms + self.bucket_ms * self.buckets.len() as f32;
        while let Some(peek) = self.spill.peek().copied() {
            if peek.t_ms < horizon {
                self.spill.pop();
                self.total -= 1;
                self.push(peek);
            } else {
                break;
            }
        }
    }

    /// Total events currently in the wheel (buckets + spill).
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.total
    }
}
