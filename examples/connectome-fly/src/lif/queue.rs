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

    /// Pop all events due at or before `now_ms` into `out`.
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
