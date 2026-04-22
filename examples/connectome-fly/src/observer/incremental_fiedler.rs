//! Incremental co-firing accumulator for the Fiedler detector.
//!
//! ADR-154 §16 lever 3. The dense pair sweep in
//! `super::core::Observer::compute_fiedler` walks every spike pair in
//! the 50 ms window on every detect call. Under saturation
//! (~21 000 spikes in the window at N=1024) that is ~2.2·10⁸ pair
//! touches per detect, all of which are redundant work: the same pairs
//! were valid in the previous detect too, minus a thin prefix of
//! expired spikes and plus a thin suffix of newly-arrived spikes.
//!
//! This module maintains an `O(|edges|)` running count of τ-coincident
//! spike pairs inside the rolling window. The count is updated in
//! amortised `O(τ · rate · N)` per `push`/`expire` — which for τ=5 ms,
//! a 50 ms window, and the saturated N=1024 regime is ~2 000 touches
//! per spike instead of ~21 000. Each `compute_fiedler` then just
//! iterates the accumulator to build the `n × n` adjacency, with no
//! pair-sweep at all.
//!
//! ## Determinism
//!
//! AC-1 (bit-exact repeatability at N=1024) requires that iteration
//! order over the accumulator be identical across runs. We use
//! `BTreeMap` keyed by `(lo, hi)` sorted `NeuronId` pairs, so iteration
//! is in lexicographic key order — byte-for-byte reproducible.
//! `HashMap` would hash-randomise and silently break AC-1.
//!
//! ## Responsibility split
//!
//! The `Observer` owns the `VecDeque<Spike>` that is the physical
//! window. This module is stateless with respect to that window — it
//! is told "a spike arrived" (`push`) or "a spike left" (`expire`)
//! and is given a reference to the spikes it should pair against.
//! It owns only the `BTreeMap` of pair counts.

use std::collections::BTreeMap;

use crate::connectome::NeuronId;
use crate::lif::Spike;

/// Co-firing coincidence window in ms. Matches the dense path in
/// `super::core::Observer::compute_fiedler` and the sparse path in
/// `super::sparse_fiedler`.
pub const COFIRE_TAU_MS: f32 = 5.0;

/// Lexicographically-ordered neuron pair, used as the accumulator key.
/// Guarantees `(a, b)` and `(b, a)` map to the same entry.
pub type PairKey = (NeuronId, NeuronId);

/// Maintains a rolling `BTreeMap<(NeuronId, NeuronId), u32>` of
/// τ-coincident spike-pair counts inside the observer's sliding window.
///
/// The accumulator does not own the window spikes — it is updated
/// incrementally by the `Observer` whenever a spike is added
/// (`push`) or removed (`expire`). `snapshot_adjacency` consumes the
/// current map into a dense `n²` float vector suitable for the
/// existing Jacobi / shifted-power-iteration paths; `snapshot_sparse`
/// returns an iterator of `(u, v, weight)` triples for the sparse
/// path at `n > 1024`.
pub struct IncrementalCofireAccumulator {
    /// Coincidence half-window in ms.
    tau_ms: f32,
    /// Pair counts, keyed by lexicographically-ordered `(lo, hi)`.
    /// `BTreeMap` (not `HashMap`) — iteration order is deterministic,
    /// which AC-1 bit-exactness requires.
    counts: BTreeMap<PairKey, u32>,
}

impl IncrementalCofireAccumulator {
    /// New empty accumulator with τ = 5 ms (matches the dense path).
    pub fn new() -> Self {
        Self::with_tau(COFIRE_TAU_MS)
    }

    /// New empty accumulator with a custom coincidence half-window.
    pub fn with_tau(tau_ms: f32) -> Self {
        Self {
            tau_ms,
            counts: BTreeMap::new(),
        }
    }

    /// Number of distinct undirected pairs currently tracked.
    pub fn edge_count(&self) -> usize {
        self.counts.len()
    }

    /// Tau in ms.
    pub fn tau_ms(&self) -> f32 {
        self.tau_ms
    }

    /// Wipe the accumulator. Used when the window itself is reset.
    pub fn clear(&mut self) {
        self.counts.clear();
    }

    /// Record a new spike `s` entering the window. `prior_spikes` are
    /// the spikes that were already in the window *before* `s` was
    /// pushed, ordered oldest → newest (i.e. the `VecDeque` iterated
    /// via `iter()` with `s` omitted).
    ///
    /// For each prior spike `p` with `|s.t_ms − p.t_ms| ≤ tau` we
    /// increment the pair count. Since spikes arrive in monotonically-
    /// non-decreasing time order, walking from the newest prior spike
    /// backward lets us break out as soon as we exit the τ band.
    pub fn push<'a, I>(&mut self, s: Spike, prior_spikes: I)
    where
        I: IntoIterator<Item = &'a Spike>,
        I::IntoIter: DoubleEndedIterator,
    {
        for p in prior_spikes.into_iter().rev() {
            let dt = (s.t_ms - p.t_ms).abs();
            if dt > self.tau_ms {
                break;
            }
            if p.neuron == s.neuron {
                continue;
            }
            let key = ordered_pair(s.neuron, p.neuron);
            *self.counts.entry(key).or_insert(0) += 1;
        }
    }

    /// Record a spike `q` leaving the window. `remaining_spikes` are
    /// the spikes still in the window *after* `q` was popped, ordered
    /// oldest → newest.
    ///
    /// For each remaining spike `r` with `|q.t_ms − r.t_ms| ≤ tau` we
    /// decrement the pair count. Counts that reach zero are removed
    /// so `edge_count()` accurately reflects live edges and the
    /// snapshot path skips dead entries.
    ///
    /// Walking remaining spikes from oldest forward lets us break out
    /// as soon as we exit the τ band, since times are non-decreasing
    /// and `q` (just popped from front) is oldest.
    pub fn expire<'a, I>(&mut self, q: Spike, remaining_spikes: I)
    where
        I: IntoIterator<Item = &'a Spike>,
    {
        for r in remaining_spikes {
            let dt = (r.t_ms - q.t_ms).abs();
            if dt > self.tau_ms {
                break;
            }
            if r.neuron == q.neuron {
                continue;
            }
            let key = ordered_pair(q.neuron, r.neuron);
            if let Some(entry) = self.counts.get_mut(&key) {
                debug_assert!(
                    *entry > 0,
                    "expire decremented zero entry — push/expire imbalance"
                );
                *entry -= 1;
                if *entry == 0 {
                    self.counts.remove(&key);
                }
            }
        }
    }

    /// Build the dense `n × n` symmetric adjacency vector consumed by
    /// the existing Jacobi / dense shifted-power-iteration eigen-paths.
    ///
    /// `active` is the sorted, deduplicated list of `NeuronId`s whose
    /// spikes lie in the current window. The caller is responsible for
    /// maintaining this list (the `Observer` derives it on demand from
    /// its `cofire_window`). Returns a flattened `n*n` row-major vector
    /// with edge weights on both `(i, j)` and `(j, i)`.
    ///
    /// Pairs whose endpoints are not in `active` are silently skipped —
    /// this should not happen if `push` / `expire` were called
    /// correctly (every neuron with a live count is by construction
    /// in the live window) but the guard keeps the snapshot robust to
    /// future filter logic.
    pub fn snapshot_adjacency(&self, active: &[NeuronId]) -> Vec<f32> {
        let n = active.len();
        let mut a = vec![0.0_f32; n * n];
        for (&(u, v), &c) in &self.counts {
            let Ok(ai) = active.binary_search(&u) else {
                continue;
            };
            let Ok(bi) = active.binary_search(&v) else {
                continue;
            };
            if ai == bi {
                continue;
            }
            let w = c as f32;
            a[ai * n + bi] = w;
            a[bi * n + ai] = w;
        }
        a
    }

    /// Iterator of `(lo_id, hi_id, weight)` triples for the currently-
    /// tracked pair counts. Used by the sparse path at `n > 1024` to
    /// drive CSR assembly without materialising an `n × n` matrix.
    ///
    /// Iteration order is deterministic (`BTreeMap` key order).
    pub fn snapshot_sparse(&self) -> impl Iterator<Item = (NeuronId, NeuronId, f32)> + '_ {
        self.counts.iter().map(|(&(u, v), &c)| (u, v, c as f32))
    }

    /// Read-only handle to the underlying map. Used by tests and
    /// diagnostics; hot-path callers should use `snapshot_adjacency`
    /// or `snapshot_sparse` so the accumulator owns the iteration
    /// discipline.
    pub fn raw_counts(&self) -> &BTreeMap<PairKey, u32> {
        &self.counts
    }
}

impl Default for IncrementalCofireAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Sort a neuron pair into `(lo, hi)` canonical order so `(a, b)` and
/// `(b, a)` map to the same `BTreeMap` key.
#[inline]
fn ordered_pair(a: NeuronId, b: NeuronId) -> PairKey {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

// ---------------------------------------------------------------------
// Unit tests — local invariants. Equivalence against the pair-sweep
// path and AC-1 bit-exactness live in `tests/incremental_fiedler.rs`.
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    fn spike(t: f32, n: u32) -> Spike {
        Spike {
            t_ms: t,
            neuron: NeuronId(n),
        }
    }

    /// Push two τ-coincident spikes on distinct neurons — one edge,
    /// one count.
    #[test]
    fn push_single_coincident_pair() {
        let mut acc = IncrementalCofireAccumulator::new();
        let mut w = VecDeque::new();
        let a = spike(10.0, 0);
        w.push_back(a);
        acc.push(a, w.iter().take(w.len() - 1));
        let b = spike(12.0, 1);
        w.push_back(b);
        acc.push(b, w.iter().take(w.len() - 1));
        assert_eq!(acc.edge_count(), 1);
        let (&key, &c) = acc.raw_counts().iter().next().unwrap();
        assert_eq!(key, (NeuronId(0), NeuronId(1)));
        assert_eq!(c, 1);
    }

    /// Push two spikes on the same neuron — no self-edge is ever
    /// recorded.
    #[test]
    fn push_rejects_self_edges() {
        let mut acc = IncrementalCofireAccumulator::new();
        let mut w = VecDeque::new();
        for t in [10.0, 12.0] {
            let s = spike(t, 7);
            w.push_back(s);
            acc.push(s, w.iter().take(w.len() - 1));
        }
        assert_eq!(acc.edge_count(), 0);
    }

    /// Push two spikes separated by > τ — no edge.
    #[test]
    fn push_skips_beyond_tau() {
        let mut acc = IncrementalCofireAccumulator::new();
        let mut w = VecDeque::new();
        let a = spike(10.0, 0);
        w.push_back(a);
        acc.push(a, w.iter().take(w.len() - 1));
        let b = spike(20.0, 1); // 10 ms apart, τ = 5 ms
        w.push_back(b);
        acc.push(b, w.iter().take(w.len() - 1));
        assert_eq!(acc.edge_count(), 0);
    }

    /// Push then expire the same pair — count drops to zero and the
    /// entry is purged.
    #[test]
    fn push_then_expire_symmetric() {
        let mut acc = IncrementalCofireAccumulator::new();
        let mut w = VecDeque::new();
        let a = spike(10.0, 0);
        let b = spike(12.0, 1);
        w.push_back(a);
        acc.push(a, w.iter().take(w.len() - 1));
        w.push_back(b);
        acc.push(b, w.iter().take(w.len() - 1));
        assert_eq!(acc.edge_count(), 1);
        // Expire `a` (simulates window pop from front).
        let q = w.pop_front().unwrap();
        acc.expire(q, w.iter());
        assert_eq!(acc.edge_count(), 0);
    }

    /// Multi-count: same pair co-fires twice, count = 2; expire one,
    /// count = 1.
    #[test]
    fn pair_count_accumulates_and_decrements() {
        let mut acc = IncrementalCofireAccumulator::new();
        let mut w = VecDeque::new();
        let spikes = [
            spike(10.0, 0),
            spike(12.0, 1),
            spike(14.0, 0),
            spike(16.0, 1),
        ];
        for s in spikes {
            w.push_back(s);
            acc.push(s, w.iter().take(w.len() - 1));
        }
        // (0,1) pairs τ-coincident within 5 ms, skipping same-neuron:
        //   (t=10,n=0) ↔ (t=12,n=1): Δt=2  ✓
        //   (t=10,n=0) ↔ (t=16,n=1): Δt=6  ✗
        //   (t=12,n=1) ↔ (t=14,n=0): Δt=2  ✓
        //   (t=14,n=0) ↔ (t=16,n=1): Δt=2  ✓
        // → 3 counts.
        let &c = acc.raw_counts().get(&(NeuronId(0), NeuronId(1))).unwrap();
        assert_eq!(c, 3);
        // Expire t=10 (neuron 0). Pairs lost:
        //   (t=10,n=0) ↔ (t=12,n=1): Δt=2  ✓
        //   (t=10,n=0) ↔ (t=14,n=0): same neuron, skipped
        //   (t=10,n=0) ↔ (t=16,n=1): Δt=6, expire's forward scan
        //     breaks before reaching this (remaining: t=12,14,16).
        // → 1 lost, count 3 → 2.
        let q = w.pop_front().unwrap();
        acc.expire(q, w.iter());
        let &c = acc.raw_counts().get(&(NeuronId(0), NeuronId(1))).unwrap();
        assert_eq!(c, 2);
    }

    /// Snapshot to dense adjacency: symmetric, correct magnitudes.
    #[test]
    fn snapshot_adjacency_is_symmetric() {
        let mut acc = IncrementalCofireAccumulator::new();
        let mut w = VecDeque::new();
        for s in [
            spike(10.0, 0),
            spike(11.0, 1),
            spike(12.0, 2),
            spike(13.0, 0),
        ] {
            w.push_back(s);
            acc.push(s, w.iter().take(w.len() - 1));
        }
        let active = vec![NeuronId(0), NeuronId(1), NeuronId(2)];
        let a = acc.snapshot_adjacency(&active);
        assert_eq!(a.len(), 9);
        // Symmetry.
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(a[i * 3 + j], a[j * 3 + i]);
            }
        }
        // Diagonal is zero (self-edges rejected).
        for i in 0..3 {
            assert_eq!(a[i * 3 + i], 0.0);
        }
    }

    /// `BTreeMap` iteration is deterministic across fresh allocations;
    /// the sparse snapshot must produce the same ordered triples on
    /// repeated construction.
    #[test]
    fn sparse_snapshot_is_deterministic() {
        let build = || {
            let mut acc = IncrementalCofireAccumulator::new();
            let mut w = VecDeque::new();
            for s in [
                spike(10.0, 5),
                spike(11.0, 2),
                spike(12.0, 9),
                spike(13.0, 2),
                spike(14.0, 5),
            ] {
                w.push_back(s);
                acc.push(s, w.iter().take(w.len() - 1));
            }
            acc.snapshot_sparse().collect::<Vec<_>>()
        };
        let a = build();
        let b = build();
        assert_eq!(a, b);
    }
}
