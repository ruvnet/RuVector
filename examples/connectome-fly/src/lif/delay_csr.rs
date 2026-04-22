//! Delay-sorted CSR for spike delivery (Opt D from ADR-154 §3.2 step 10).
//!
//! Complements the existing `Connectome::outgoing` CSR, which is in
//! generator-insertion order and stores `Synapse { post, weight, delay,
//! sign }` as an array-of-structs with trailing enum padding (≈16 bytes
//! per synapse on x86_64). The delivery hot path at the saturated regime
//! — see `BENCHMARK.md` §4.5 for the diagnosis — is bottlenecked on
//! those loads plus the per-delivery sign branch, not on the subthreshold
//! loop that `simd.rs` already vectorizes.
//!
//! This module rebuilds the outgoing table once, at engine construction
//! time, in three packed structure-of-arrays vectors:
//!
//! - `post`           — `u32` post-synaptic neuron id
//! - `delay_ms`       — `f32` axonal + synaptic delay, ms
//! - `signed_weight`  — `f32` `weight_gain * weight` with the sign of the
//!                      synapse folded in (positive → excitatory kick,
//!                      negative → inhibitory kick). Pre-multiplying
//!                      removes the per-delivery `match Sign` branch and
//!                      the `weight_gain * weight` multiplication from
//!                      the innermost loop.
//!
//! Rows are **sorted by `delay_ms` ascending**. Wheel inserts for a
//! single spike therefore walk buckets in monotonically-nondecreasing
//! order, so the slot index is a monotone function of the synapse index
//! and (a) improves branch prediction on the bucket-bound check, and (b)
//! keeps the active bucket `Vec<SpikeEvent>` hot in L1 across several
//! consecutive inserts. The sort is also what enables the optional
//! fast path in [`DelaySortedCsr::from_connectome_for_wheel`] — see
//! that constructor for the precomputed-bucket-offset variant.
//!
//! # Measured speedup
//!
//! On `lif_throughput_n_1024` (120 ms simulated, saturated firing) the
//! delay-sorted SoA path delivers:
//!
//! - **Kernel-only** (observer's Fiedler detector disabled):
//!   ~15 ms → ~10 ms, **≈ 1.5× faster** — the win the SoA + pre-signed-
//!   weight layout targets.
//! - **Full bench** (observer armed, default config): parity with the
//!   scalar-opt path (~6.75 s both). The Fiedler detector's O(n²)-per-
//!   detect cost dominates the kernel by roughly 450-to-1 in this
//!   regime, which is the reason Opt D's kernel-level speedup does not
//!   surface at the bench level. See the commit message for the honest
//!   gap diagnosis vs the ADR-154 §3.2 ≥ 2× target.
//!
//! # Determinism
//!
//! Within-row delay sort uses a stable sort keyed on `(delay_ms.to_bits(),
//! post.0)`, so two rows with identical `(delay, post)` pairs retain
//! their insertion order. The `to_bits()` key gives byte-for-byte
//! deterministic ordering even for NaN-or-negative-zero edge cases
//! (neither can occur in practice — the generator clamps delay to
//! `[0.5, 10.0]` — but the invariant is cheap to keep).
//!
//! Cross-path bit-exactness with the insertion-order CSR is **not**
//! promised. The demonstrator already documents the cross-path spike-
//! count tolerance (README §Determinism; ADR-154 §15.1) as ~10 %, and
//! the equivalence test (`tests/delay_csr_equivalence.rs`) asserts inside
//! that envelope. AC-1 bit-exact-within-a-path at N=1024 is preserved
//! because the delay-sorted path is opt-in behind
//! `EngineConfig::use_delay_sorted_csr` (default `false`).

use crate::connectome::{Connectome, NeuronId, Sign};

use super::queue::{SpikeEvent, TimingWheel};

/// Delay-sorted packed outgoing adjacency for spike delivery.
///
/// Built once from a `Connectome` + a `weight_gain` scalar. The gain is
/// folded into `signed_weight` at build time so the delivery inner loop
/// contains no multiplications by `weight_gain` and no sign match.
pub struct DelaySortedCsr {
    /// `delay_syn[delay_ptr[i]..delay_ptr[i+1]]` is the (sorted) outgoing
    /// synapse range for pre-synaptic neuron `i`.
    delay_ptr: Vec<u32>,
    /// SoA — post-synaptic neuron id.
    post: Vec<u32>,
    /// SoA — axonal + synaptic delay, ms (sorted ascending within each row).
    delay_ms: Vec<f32>,
    /// SoA — signed weight = `weight_gain * weight * sign(±1.0)`.
    signed_weight: Vec<f32>,
    /// SoA — pre-computed bucket offset `(delay_ms / bucket_ms) as u32`
    /// using the wheel's `bucket_ms`. Lets the delivery loop avoid a
    /// per-synapse float division: `slot = base_slot + delay_buckets[k]`.
    /// Populated only when `from_connectome_for_wheel` is used; when the
    /// generic `from_connectome` constructor runs the vec is empty and
    /// `deliver_spike` falls back to the generic `queue.push()` path.
    delay_buckets: Vec<u32>,
    /// The `bucket_ms` the offsets above were computed against, or `0.0`
    /// if the fast-path offsets are not populated. Reused at delivery
    /// time as a sanity check against unexpected wheel reconfigurations.
    bucket_ms: f32,
}

impl DelaySortedCsr {
    /// Build a delay-sorted SoA view of `conn`'s outgoing edges.
    ///
    /// `weight_gain` is the engine-level scale applied to every synaptic
    /// kick; it is folded into `signed_weight` here so the delivery loop
    /// is a single fma-friendly `ev.w = signed_weight[k]` load.
    ///
    /// This constructor does **not** populate the wheel-bucket offsets;
    /// delivery via [`Self::deliver_spike`] then uses the generic
    /// `TimingWheel::push` slow path. Prefer [`Self::from_connectome_for_wheel`]
    /// when the wheel configuration is known at build time — that
    /// populates the offsets and enables the fast `push_at_slot` path.
    pub fn from_connectome(conn: &Connectome, weight_gain: f32) -> Self {
        Self::build(conn, weight_gain, None)
    }

    /// Build a delay-sorted SoA view with wheel-bucket offsets
    /// pre-computed against `bucket_ms`. Delivery then skips the
    /// per-synapse float division and goes through
    /// [`TimingWheel::push_at_slot`].
    pub fn from_connectome_for_wheel(conn: &Connectome, weight_gain: f32, bucket_ms: f32) -> Self {
        Self::build(conn, weight_gain, Some(bucket_ms))
    }

    fn build(conn: &Connectome, weight_gain: f32, wheel_bucket_ms: Option<f32>) -> Self {
        let n = conn.num_neurons();
        let total = conn.num_synapses();
        let mut delay_ptr: Vec<u32> = Vec::with_capacity(n + 1);
        let mut post: Vec<u32> = Vec::with_capacity(total);
        let mut delay_ms: Vec<f32> = Vec::with_capacity(total);
        let mut signed_weight: Vec<f32> = Vec::with_capacity(total);
        let mut delay_buckets: Vec<u32> = match wheel_bucket_ms {
            Some(_) => Vec::with_capacity(total),
            None => Vec::new(),
        };

        // Stable-sort each row by `delay_ms` ascending, tie-breaking on
        // `post` so the permutation is deterministic across rebuilds.
        let mut row_perm: Vec<u32> = Vec::new();
        delay_ptr.push(0);
        let inv_bucket = wheel_bucket_ms.map(|b| 1.0_f32 / b);
        for i in 0..n {
            let row = conn.outgoing(NeuronId(i as u32));
            row_perm.clear();
            row_perm.extend(0..row.len() as u32);
            // Stable sort by (delay_ms bits, post.0): stable so synapses
            // with identical delay+post keep generator insertion order.
            row_perm.sort_by(|&a, &b| {
                let sa = &row[a as usize];
                let sb = &row[b as usize];
                sa.delay_ms
                    .to_bits()
                    .cmp(&sb.delay_ms.to_bits())
                    .then_with(|| sa.post.0.cmp(&sb.post.0))
            });
            for &k in &row_perm {
                let s = &row[k as usize];
                let sign: f32 = match s.sign {
                    Sign::Excitatory => 1.0,
                    Sign::Inhibitory => -1.0,
                };
                post.push(s.post.0);
                delay_ms.push(s.delay_ms);
                signed_weight.push(weight_gain * s.weight * sign);
                if let Some(inv) = inv_bucket {
                    // Floor of `delay_ms / bucket_ms`. Delays are
                    // clamped to `[0.5, 10.0]` ms by the SBM generator,
                    // so the integer result always fits in `u32`.
                    delay_buckets.push((s.delay_ms * inv) as u32);
                }
            }
            delay_ptr.push(post.len() as u32);
        }

        debug_assert_eq!(post.len(), total);
        debug_assert_eq!(delay_ms.len(), total);
        debug_assert_eq!(signed_weight.len(), total);
        if wheel_bucket_ms.is_some() {
            debug_assert_eq!(delay_buckets.len(), total);
        }

        Self {
            delay_ptr,
            post,
            delay_ms,
            signed_weight,
            delay_buckets,
            bucket_ms: wheel_bucket_ms.unwrap_or(0.0),
        }
    }

    /// Number of pre-synaptic rows (== `conn.num_neurons()`).
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.delay_ptr.len().saturating_sub(1)
    }

    /// Total packed synapse count (== `conn.num_synapses()`).
    #[inline]
    pub fn num_synapses(&self) -> usize {
        self.post.len()
    }

    /// Public view on one row's `delay_ms` slice — used by the
    /// equivalence test to verify sortedness without exposing the
    /// SoA vectors directly.
    #[inline]
    pub fn row_delays(&self, pre: NeuronId) -> &[f32] {
        let s = self.delay_ptr[pre.idx()] as usize;
        let e = self.delay_ptr[pre.idx() + 1] as usize;
        &self.delay_ms[s..e]
    }

    /// Public view on one row's packed `signed_weight` slice.
    #[inline]
    pub fn row_signed_weights(&self, pre: NeuronId) -> &[f32] {
        let s = self.delay_ptr[pre.idx()] as usize;
        let e = self.delay_ptr[pre.idx() + 1] as usize;
        &self.signed_weight[s..e]
    }

    /// Deliver one spike: push all outgoing events of `pre` fired at
    /// `t_ms` into `queue`.
    ///
    /// The row is delay-sorted, so consecutive pushes drop into
    /// monotonically non-decreasing wheel buckets; that hits the hot
    /// bucket's `Vec<SpikeEvent>` backing buffer tightly in L1.
    ///
    /// When this `DelaySortedCsr` was built via
    /// [`Self::from_connectome_for_wheel`] with the wheel's `bucket_ms`,
    /// the hot path also bypasses the float division, `match Sign` /
    /// `weight_gain` multiply, and the per-event modulo of the generic
    /// [`TimingWheel::push`] — each insert is one integer add, one
    /// compare (ring-wrap), and one `Vec::push`. Otherwise delivery
    /// falls back to the generic `queue.push()`.
    ///
    /// Deterministic push order is preserved from the sort key so repeat
    /// calls on the same `(pre, t_ms)` produce identical wheel contents.
    #[inline]
    pub fn deliver_spike(&self, pre: NeuronId, t_ms: f32, queue: &mut TimingWheel) {
        let i = pre.idx();
        let start = self.delay_ptr[i] as usize;
        let end = self.delay_ptr[i + 1] as usize;
        if start == end {
            return;
        }
        if !self.delay_buckets.is_empty() && queue.bucket_ms_matches(self.bucket_ms) {
            self.deliver_spike_fast(pre, t_ms, start, end, queue);
        } else {
            self.deliver_spike_generic(pre, t_ms, start, end, queue);
        }
    }

    /// Fast path — wheel-bucket offsets are pre-computed, so each
    /// insert is `push_at_slot` / `push_spill`. No per-synapse float
    /// division, no modulo.
    #[inline]
    fn deliver_spike_fast(
        &self,
        pre: NeuronId,
        t_ms: f32,
        start: usize,
        end: usize,
        queue: &mut TimingWheel,
    ) {
        let nb = queue.num_buckets();
        let inv_bucket = queue.inv_bucket_ms();
        let base_ms = queue.base_ms();
        // One float division per SPIKE (not per synapse): compute where
        // this spike lands in the wheel relative to `base_ms`. The sim
        // only emits spikes with `t_ms >= base_ms`, so truncation
        // (`as isize`) is equivalent to floor() here.
        let base_slot = ((t_ms - base_ms) * inv_bucket) as isize;

        let post = &self.post[start..end];
        let delay = &self.delay_ms[start..end];
        let w = &self.signed_weight[start..end];
        let db = &self.delay_buckets[start..end];

        for k in 0..post.len() {
            let slot = base_slot + db[k] as isize;
            let ev = SpikeEvent {
                t_ms: t_ms + delay[k],
                post: NeuronId(post[k]),
                pre,
                w: w[k],
            };
            if slot >= 0 && (slot as usize) < nb {
                queue.push_at_slot(slot as usize, ev);
            } else {
                queue.push_spill(ev);
            }
        }
    }

    /// Generic path — falls back to `queue.push()` (one float division
    /// and one modulo per synapse). Used when the CSR was built without
    /// wheel-bucket offsets, or when the wheel's `bucket_ms` does not
    /// match what the CSR was built against.
    #[inline]
    fn deliver_spike_generic(
        &self,
        pre: NeuronId,
        t_ms: f32,
        start: usize,
        end: usize,
        queue: &mut TimingWheel,
    ) {
        let post = &self.post[start..end];
        let delay = &self.delay_ms[start..end];
        let w = &self.signed_weight[start..end];
        for k in 0..post.len() {
            let ev = SpikeEvent {
                t_ms: t_ms + delay[k],
                post: NeuronId(post[k]),
                pre,
                w: w[k],
            };
            queue.push(ev);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectome::{ConnectomeConfig, NeuronId};

    #[test]
    fn rows_are_delay_sorted() {
        let conn = crate::connectome::Connectome::generate(&ConnectomeConfig {
            num_neurons: 128,
            avg_out_degree: 16.0,
            ..ConnectomeConfig::default()
        });
        let csr = DelaySortedCsr::from_connectome(&conn, 1.0);
        assert_eq!(csr.num_synapses(), conn.num_synapses());
        assert_eq!(csr.num_rows(), conn.num_neurons());
        for i in 0..conn.num_neurons() {
            let delays = csr.row_delays(NeuronId(i as u32));
            for pair in delays.windows(2) {
                assert!(
                    pair[0].to_bits() <= pair[1].to_bits() || pair[0] <= pair[1],
                    "row {i} not delay-sorted: {} > {}",
                    pair[0],
                    pair[1]
                );
            }
        }
    }

    #[test]
    fn signed_weight_folds_gain_and_sign() {
        let conn = crate::connectome::Connectome::generate(&ConnectomeConfig {
            num_neurons: 64,
            avg_out_degree: 8.0,
            ..ConnectomeConfig::default()
        });
        // Pick a non-unit gain so a bug where we forget to multiply
        // surfaces as an order-of-magnitude divergence.
        let gain = 0.7_f32;
        let csr = DelaySortedCsr::from_connectome(&conn, gain);
        // Reconstruct the expected sum per row from the connectome's
        // canonical CSR and compare against the SoA sum (order-free).
        for i in 0..conn.num_neurons() {
            let id = NeuronId(i as u32);
            let row = conn.outgoing(id);
            let mut canon_sum = 0.0_f64;
            for s in row {
                let sign: f64 = match s.sign {
                    Sign::Excitatory => 1.0,
                    Sign::Inhibitory => -1.0,
                };
                canon_sum += (gain as f64) * (s.weight as f64) * sign;
            }
            let mut soa_sum = 0.0_f64;
            for &w in csr.row_signed_weights(id) {
                soa_sum += w as f64;
            }
            let scale = canon_sum.abs().max(1e-6);
            let rel = (canon_sum - soa_sum).abs() / scale;
            assert!(
                rel < 1e-4,
                "row {i} signed-weight sum mismatch: canon={canon_sum} soa={soa_sum} rel={rel}"
            );
        }
    }

    #[test]
    fn deliver_spike_pushes_one_event_per_synapse() {
        let conn = crate::connectome::Connectome::generate(&ConnectomeConfig {
            num_neurons: 64,
            avg_out_degree: 8.0,
            ..ConnectomeConfig::default()
        });
        let csr = DelaySortedCsr::from_connectome(&conn, 1.0);
        let mut wheel = TimingWheel::new(0.1, 32.0);
        let pre = NeuronId(7);
        let expected = conn.outgoing(pre).len();
        csr.deliver_spike(pre, 1.0, &mut wheel);
        assert_eq!(wheel.len(), expected);
    }
}
