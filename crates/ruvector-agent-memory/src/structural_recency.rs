//! Structural-time recency: an alternative "recency" signal for
//! [`crate::compaction::CoherencePolicy`], reusing `emergent-time`'s
//! [`emergent_time::StructuralProperTime`] clock instead of raw per-event tick
//! counting.
//!
//! ## Why
//!
//! `MemoryEntry::last_accessed_at` (see `memory.rs`) is a *logical clock*: every
//! insertion or access advances it by exactly one tick, regardless of what
//! happened. This is functionally identical to `emergent-time`'s [`WallClock`]
//! (`Clock::tick` returns `1.0` unconditionally) — the "null hypothesis" clock
//! that crate's own benchmarks are built to beat. The consequence for
//! `weighted_importance`'s `recency` term: a burst of near-duplicate,
//! low-information memories (retries, redundant re-observations) inflates the
//! tick count exactly as much as a burst of genuinely novel memories, so it can
//! make older-but-still-relevant memories look artificially "stale" purely
//! because unrelated churn happened after them — not because anything
//! obsoleted them.
//!
//! [`StructuralTimeWeights`] replaces the tick-count axis with the accumulated
//! *structural arc length* ([`emergent_time::StructuralProperTime`]) of the
//! memory-insertion trajectory: embedding movement (`Δv`) and coherence loss
//! against the active context window (`ΔC`). A run of near-duplicate insertions
//! moves the embedding barely at all, so it contributes ≈0 structural time —
//! memories inserted before it are not penalized for churn that carried no new
//! information. The entropy (`ΔS`), graph (`ΔG`), and prediction-error (`ΔE`)
//! channels are left at weight 0: this crate has no real observable for them
//! and fabricating one would not be honest.
//!
//! [`DedupGatedRecency`] is the fair, cheap competitor (no `emergent-time`
//! dependency, always compiled): a tick only counts if the entry's embedding is
//! more than `dedup_threshold` cosine-distance from the immediately preceding
//! entry in time order. This is the same "don't let the physics-flavoured clock
//! win by strawman" discipline `emergent-time::agentic_time::WindowedDeltaClock`
//! applies to itself — if structural time does not clearly beat this simple
//! baseline, that is reported, not hidden.
//!
//! ## A documented negative result: [`StructuralTimeRecency`] does not work as a score
//!
//! [`structural_recency`] turns cumulative structural time into a `[0, 1]`
//! score and [`StructuralTimeRecency`] ranks by it, mirroring how
//! [`crate::compaction::CoherencePolicy`] ranks by raw tick count. Measured on
//! the nightly benchmark (`examples/structural_time_recency_bench.rs`), this
//! **does not beat the tick-recency baseline**: cumulative structural time is,
//! by construction, monotone non-decreasing in insertion order — exactly like
//! the tick count it replaces — so ranking by it produces nearly the same
//! top-K selection (top-K by any strictly monotone reparametrization of the
//! same order is invariant except at ties, and ties resolve *in favor of* the
//! later/churn entry, since churn is inserted after the memory it duplicates).
//! This is retained as a working, tested reference implementation and an
//! informative negative result, not deleted.
//!
//! [`StructuralKeyframeRetention`] is the corrected integration: it uses
//! `emergent_time::structural_clock::keyframes`, the primitive the
//! `emergent-time` crate itself defines for *budget-constrained retention*
//! (compress a trajectory to `budget` samples spaced evenly in structural
//! time, not by rank). Because a churn burst contributes ≈0 structural time,
//! few or no keyframes land inside it — the mechanism naturally skips
//! redundant duplicates and keeps the regime-shift memories that preceded
//! them, which a rank-based score cannot do.

use crate::compaction::{CoherenceWeights, CompactionPolicy};
use crate::memory::MemoryEntry;
use crate::scoring::{coherence_score, cosine_sim};

/// Order entry indices by `last_accessed_at` (ties broken by `id` for a total,
/// deterministic order).
fn time_order(entries: &[MemoryEntry]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..entries.len()).collect();
    idx.sort_unstable_by_key(|&i| (entries[i].last_accessed_at, entries[i].id));
    idx
}

/// Normalize a slice of non-negative scores to `[0, 1]` by dividing by the max
/// (all-zero input maps to all-zero output).
fn normalize_unit(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().cloned().fold(0.0_f32, f32::max);
    if max < 1e-12 {
        vec![0.0; scores.len()]
    } else {
        scores.iter().map(|&s| s / max).collect()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Fair cheap baseline: dedup-gated tick recency (no emergent-time dependency)
// ────────────────────────────────────────────────────────────────────────────

/// Cosine-distance threshold below which a successive entry is treated as a
/// duplicate of its predecessor and does not advance the recency clock.
#[derive(Debug, Clone)]
pub struct DedupGatedWeights {
    pub base: CoherenceWeights,
    /// Entries within this cosine distance of the previous entry (in time
    /// order) contribute 0 ticks instead of 1.
    pub dedup_threshold: f32,
}

impl Default for DedupGatedWeights {
    fn default() -> Self {
        Self {
            base: CoherenceWeights::default(),
            dedup_threshold: 0.02,
        }
    }
}

/// Per-entry dedup-gated recency in `[0, 1]`: cumulative tick count where a
/// tick only fires when the entry's embedding differs from its time-order
/// predecessor by more than `dedup_threshold` cosine distance, normalized by
/// the maximum cumulative count observed.
pub fn dedup_gated_recency(entries: &[MemoryEntry], dedup_threshold: f32) -> Vec<f32> {
    if entries.is_empty() {
        return Vec::new();
    }
    let order = time_order(entries);
    let mut cum = vec![0.0f32; entries.len()];
    let mut acc = 0.0f32;
    for (pos, &i) in order.iter().enumerate() {
        if pos > 0 {
            let prev = order[pos - 1];
            let dist = 1.0 - cosine_sim(&entries[prev].vector, &entries[i].vector);
            if dist > dedup_threshold {
                acc += 1.0;
            }
        }
        cum[i] = acc;
    }
    normalize_unit(&cum)
}

/// Compaction policy: identical to [`crate::compaction::CoherencePolicy`]
/// except the `recency` term is [`dedup_gated_recency`] instead of raw tick
/// count. The fair, cheap competitor to [`StructuralTimeRecency`].
#[derive(Debug, Clone, Default)]
pub struct DedupGatedRecency {
    pub weights: DedupGatedWeights,
}

impl DedupGatedRecency {
    pub fn new(weights: DedupGatedWeights) -> Self {
        Self { weights }
    }
}

/// Per-entry weighted importance using dedup-gated recency in place of raw
/// tick recency; otherwise identical to [`crate::compaction::weighted_importance`].
pub fn weighted_importance_dedup(
    entries: &[MemoryEntry],
    weights: &DedupGatedWeights,
    context: &[Vec<f32>],
) -> Vec<f32> {
    if entries.is_empty() {
        return Vec::new();
    }
    let recency = dedup_gated_recency(entries, weights.dedup_threshold);
    let max_count = entries.iter().map(|e| e.access_count).max().unwrap_or(1);
    let max_count_f = max_count.max(1) as f32;
    entries
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let frequency = e.access_count as f32 / max_count_f;
            let coherence = if context.is_empty() {
                0.0
            } else {
                coherence_score(&e.vector, context)
            };
            weights.base.alpha * recency[i]
                + weights.base.beta * frequency
                + weights.base.gamma * coherence
        })
        .collect()
}

impl CompactionPolicy for DedupGatedRecency {
    fn name(&self) -> &str {
        "DedupGatedRecency"
    }

    fn select_survivors(
        &self,
        entries: &[MemoryEntry],
        target_size: usize,
        context: &[Vec<f32>],
    ) -> Vec<usize> {
        if entries.is_empty() {
            return Vec::new();
        }
        let importance = weighted_importance_dedup(entries, &self.weights, context);
        let mut scored: Vec<(usize, f32)> = importance.into_iter().enumerate().collect();
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored
            .into_iter()
            .take(target_size)
            .map(|(i, _)| i)
            .collect()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Structural-time recency (feature = "structural-time", depends on emergent-time)
// ────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "structural-time")]
mod structural {
    use super::*;
    use emergent_time::{Clock, StateSnapshot, StructuralMetric, StructuralProperTime};

    /// Weights for [`StructuralTimeRecency`]: the base recency/frequency/coherence
    /// weights plus the structural-clock metric applied to the embedding and
    /// coherence-loss channels only (entropy/graph/pred-error stay at 0 — this
    /// crate has no honest observable for them).
    #[derive(Debug, Clone)]
    pub struct StructuralTimeWeights {
        pub base: CoherenceWeights,
        pub metric: StructuralMetric,
    }

    impl Default for StructuralTimeWeights {
        fn default() -> Self {
            Self {
                base: CoherenceWeights::default(),
                metric: StructuralMetric {
                    w_embedding: 1.0,
                    w_entropy: 0.0,
                    w_graph: 0.0,
                    w_coherence: 1.0,
                    w_pred_error: 0.0,
                    gate: 0.0,
                },
            }
        }
    }

    /// Per-entry structural-time recency in `[0, 1]`: an entry's position along
    /// the structural-arc-length trajectory of memory insertions (embedding
    /// movement + coherence loss against `context`), normalized by the total
    /// accumulated structural time. Memories inserted just before a period of
    /// low structural movement (e.g. a burst of near-duplicate churn) keep a
    /// recency score close to that of memories inserted *after* the churn,
    /// because the churn itself contributes almost no structural time.
    pub fn structural_recency(
        entries: &[MemoryEntry],
        context: &[Vec<f32>],
        metric: StructuralMetric,
    ) -> Vec<f32> {
        if entries.is_empty() {
            return Vec::new();
        }
        let order = time_order(entries);
        let traj: Vec<StateSnapshot> = order
            .iter()
            .map(|&i| {
                let e = &entries[i];
                let embedding: Vec<f64> = e.vector.iter().map(|&x| x as f64).collect();
                let coherence = if context.is_empty() {
                    0.0
                } else {
                    coherence_score(&e.vector, context) as f64
                };
                StateSnapshot::full(embedding, 0.0, coherence, 0.0, 0.0)
            })
            .collect();
        let clock = StructuralProperTime::new(metric);
        let cum = clock.cumulative(&traj);
        let total = cum.last().copied().unwrap_or(0.0);
        let mut out = vec![0.0f32; entries.len()];
        if total < 1e-12 {
            // No structural movement at all: every entry is equally "recent".
            return out;
        }
        for (pos, &i) in order.iter().enumerate() {
            out[i] = (cum[pos] / total) as f32;
        }
        out
    }

    /// Compaction policy: identical to [`crate::compaction::CoherencePolicy`]
    /// except the `recency` term is [`structural_recency`] instead of raw tick
    /// count.
    #[derive(Debug, Clone, Default)]
    pub struct StructuralTimeRecency {
        pub weights: StructuralTimeWeights,
    }

    impl StructuralTimeRecency {
        pub fn new(weights: StructuralTimeWeights) -> Self {
            Self { weights }
        }
    }

    /// Per-entry weighted importance using structural-time recency in place of
    /// raw tick recency; otherwise identical to
    /// [`crate::compaction::weighted_importance`].
    pub fn weighted_importance_structural(
        entries: &[MemoryEntry],
        weights: &StructuralTimeWeights,
        context: &[Vec<f32>],
    ) -> Vec<f32> {
        if entries.is_empty() {
            return Vec::new();
        }
        let recency = structural_recency(entries, context, weights.metric);
        let max_count = entries.iter().map(|e| e.access_count).max().unwrap_or(1);
        let max_count_f = max_count.max(1) as f32;
        entries
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let frequency = e.access_count as f32 / max_count_f;
                let coherence = if context.is_empty() {
                    0.0
                } else {
                    coherence_score(&e.vector, context)
                };
                weights.base.alpha * recency[i]
                    + weights.base.beta * frequency
                    + weights.base.gamma * coherence
            })
            .collect()
    }

    impl CompactionPolicy for StructuralTimeRecency {
        fn name(&self) -> &str {
            "StructuralTimeRecency"
        }

        fn select_survivors(
            &self,
            entries: &[MemoryEntry],
            target_size: usize,
            context: &[Vec<f32>],
        ) -> Vec<usize> {
            if entries.is_empty() {
                return Vec::new();
            }
            let importance = weighted_importance_structural(entries, &self.weights, context);
            let mut scored: Vec<(usize, f32)> = importance.into_iter().enumerate().collect();
            scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            scored
                .into_iter()
                .take(target_size)
                .map(|(i, _)| i)
                .collect()
        }
    }

    // ------------------------------------------------------------------------
    // Structural keyframe retention: budget-constrained sampling in structural
    // time, using emergent-time's own intended primitive for this task.
    // ------------------------------------------------------------------------

    /// Retention policy: select `target_size` entries by sampling
    /// `emergent_time::structural_clock::keyframes` over the memory-insertion
    /// trajectory (evenly spaced in *structural* time, not wall time), then
    /// top up any shortfall (keyframe dedup can return fewer than `target_size`
    /// positions) with the remaining entries ranked by frequency + coherence
    /// (no recency term — a topped-up entry was, by definition, not chosen by
    /// the structural sampling).
    #[derive(Debug, Clone)]
    pub struct StructuralKeyframeRetention {
        pub base: CoherenceWeights,
        pub metric: StructuralMetric,
    }

    impl Default for StructuralKeyframeRetention {
        fn default() -> Self {
            Self {
                base: CoherenceWeights::default(),
                metric: StructuralTimeWeights::default().metric,
            }
        }
    }

    impl StructuralKeyframeRetention {
        pub fn new(base: CoherenceWeights, metric: StructuralMetric) -> Self {
            Self { base, metric }
        }
    }

    impl CompactionPolicy for StructuralKeyframeRetention {
        fn name(&self) -> &str {
            "StructuralKeyframeRetention"
        }

        fn select_survivors(
            &self,
            entries: &[MemoryEntry],
            target_size: usize,
            context: &[Vec<f32>],
        ) -> Vec<usize> {
            if entries.is_empty() {
                return Vec::new();
            }
            let order = time_order(entries);
            let traj: Vec<StateSnapshot> = order
                .iter()
                .map(|&i| {
                    let e = &entries[i];
                    let embedding: Vec<f64> = e.vector.iter().map(|&x| x as f64).collect();
                    let coherence = if context.is_empty() {
                        0.0
                    } else {
                        coherence_score(&e.vector, context) as f64
                    };
                    StateSnapshot::full(embedding, 0.0, coherence, 0.0, 0.0)
                })
                .collect();
            let clock = StructuralProperTime::new(self.metric);
            let frame_positions =
                emergent_time::structural_clock::keyframes(&clock, &traj, target_size);

            let mut selected: std::collections::BTreeSet<usize> =
                frame_positions.iter().map(|&pos| order[pos]).collect();

            if selected.len() > target_size {
                // keyframes() can return up to 2 extra positions to guarantee
                // the trajectory endpoints are included; trim the earliest
                // (by time order) excess first, since the whole point of this
                // policy is to bias toward *later* structurally-significant
                // memories, not toward the very first inserted entry.
                //
                // Known limitation: when a trailing run of near-identical
                // entries ties the trajectory's final cumulative value (e.g.
                // several duplicate observations of the very last topic),
                // `keyframes`'s nearest-sample search can resolve the
                // "total" target level to an *interior* tied index rather
                // than the forced last index, adding a redundant frame this
                // branch then trims — which can, in a degenerate all-exact-
                // duplicate tail, discard a genuine earlier keyframe instead
                // of a duplicate. Not observed on the (noisy, non-exact)
                // nightly benchmark corpus; documented here rather than
                // silently relied upon.
                let mut ordered: Vec<usize> = order
                    .iter()
                    .copied()
                    .filter(|i| selected.contains(i))
                    .collect();
                while ordered.len() > target_size {
                    ordered.remove(0);
                }
                selected = ordered.into_iter().collect();
            } else if selected.len() < target_size {
                // Top up the shortfall with the remaining entries ranked by
                // frequency + coherence (recency is meaningless here: these
                // entries were, by definition, not selected as keyframes).
                let max_count = entries.iter().map(|e| e.access_count).max().unwrap_or(1);
                let max_count_f = max_count.max(1) as f32;
                let mut remaining: Vec<(usize, f32)> = (0..entries.len())
                    .filter(|i| !selected.contains(i))
                    .map(|i| {
                        let e = &entries[i];
                        let frequency = e.access_count as f32 / max_count_f;
                        let coherence = if context.is_empty() {
                            0.0
                        } else {
                            coherence_score(&e.vector, context)
                        };
                        (i, self.base.beta * frequency + self.base.gamma * coherence)
                    })
                    .collect();
                remaining.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                for (i, _) in remaining.into_iter().take(target_size - selected.len()) {
                    selected.insert(i);
                }
            }

            selected.into_iter().collect()
        }
    }
}

#[cfg(feature = "structural-time")]
pub use structural::{
    structural_recency, weighted_importance_structural, StructuralKeyframeRetention,
    StructuralTimeRecency, StructuralTimeWeights,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryEntry;

    fn entry(id: u64, v: Vec<f32>, t: u64) -> MemoryEntry {
        let mut e = MemoryEntry::new(id, v, t);
        e.last_accessed_at = t;
        e
    }

    #[test]
    fn dedup_gated_recency_ignores_duplicate_bursts() {
        // entry 0: unique. entries 1..=5: near-duplicates of entry 0 (burst).
        // entry 6: unique, inserted after the burst.
        let mut entries = vec![entry(0, vec![1.0, 0.0], 0)];
        for t in 1..=5u64 {
            entries.push(entry(t, vec![1.0, 0.001 * t as f32], t));
        }
        entries.push(entry(6, vec![0.0, 1.0], 6));

        let recency = dedup_gated_recency(&entries, 0.02);
        // The burst (indices 1..=5) should not advance the clock relative to
        // entry 0: they all keep ~the same (low) recency as entry 0, while
        // entry 6 (a real change) is clearly highest.
        for i in 0..=5 {
            assert!(
                recency[i] < recency[6],
                "burst entry {i} ({}) should be less recent than the post-burst entry ({})",
                recency[i],
                recency[6]
            );
        }
        for i in 1..=5 {
            assert!(
                (recency[i] - recency[0]).abs() < 1e-6,
                "duplicate-burst entries should not advance the dedup clock: entry {i}={} vs entry 0={}",
                recency[i],
                recency[0]
            );
        }
    }

    #[test]
    fn dedup_gated_recency_is_deterministic() {
        let entries = vec![
            entry(0, vec![1.0, 0.0], 0),
            entry(1, vec![0.0, 1.0], 1),
            entry(2, vec![-1.0, 0.0], 2),
        ];
        let a = dedup_gated_recency(&entries, 0.02);
        let b = dedup_gated_recency(&entries, 0.02);
        assert_eq!(a, b);
    }
}

#[cfg(all(test, feature = "structural-time"))]
mod structural_tests {
    use super::structural::{structural_recency, StructuralKeyframeRetention};
    use crate::compaction::{CoherenceWeights, CompactionPolicy};
    use crate::memory::MemoryEntry;
    use emergent_time::StructuralMetric;

    fn entry(id: u64, v: Vec<f32>, t: u64) -> MemoryEntry {
        let mut e = MemoryEntry::new(id, v, t);
        e.last_accessed_at = t;
        e
    }

    /// Three regime-shift memories, each followed by a burst of *exact*
    /// duplicate churn (distance 0, the limit case of "near-duplicate" —
    /// contributes precisely 0 structural time, not just "very little"). This
    /// makes every churn entry's cumulative structural time an exact tie with
    /// its own regime-shift memory; `keyframes`' nearest-sample search only
    /// overwrites its best match on a *strictly* smaller distance
    /// (`structural_clock::keyframes`, `if d < bestd`), so a tie always
    /// resolves to the earliest (regime-shift) index, never a later churn
    /// duplicate. `emergent_time::structural_clock::keyframes` also always
    /// forces in the trajectory's first and last raw indices (documented
    /// behavior), so this test targets the *middle* segment, which is only
    /// kept via the structural-time tie-break, not raw position.
    #[test]
    fn keyframe_retention_prefers_regime_shift_over_churn() {
        let mut entries = vec![entry(0, vec![1.0, 0.0, 0.0], 0)];
        for t in 1..=4u64 {
            entries.push(entry(t, vec![1.0, 0.0, 0.0], t)); // exact duplicate of id 0
        }
        entries.push(entry(5, vec![0.0, 1.0, 0.0], 5));
        for t in 6..=9u64 {
            entries.push(entry(t, vec![0.0, 1.0, 0.0], t)); // exact duplicate of id 5
        }
        // Trajectory ends immediately on the third regime-shift (no trailing
        // churn) so the cumulative total is reached at exactly one index —
        // n-1 itself — instead of a tied plateau, which would otherwise let
        // the endpoint-forcing rule in `keyframes` add a second, redundant
        // "last index" frame and push the budget over 3.
        entries.push(entry(10, vec![-1.0, 0.0, 0.0], 10));

        let policy = StructuralKeyframeRetention::default();
        let survivors = policy.select_survivors(&entries, 3, &[]);
        let ids: Vec<u64> = survivors.iter().map(|&i| entries[i].id).collect();
        assert!(
            ids.contains(&0),
            "should keep the first regime-shift memory, got {ids:?}"
        );
        assert!(
            ids.contains(&5),
            "should keep the middle regime-shift memory (not a churn duplicate), got {ids:?}"
        );
        assert!(
            ids.contains(&10),
            "should keep the last regime-shift memory, got {ids:?}"
        );
    }

    #[test]
    fn keyframe_retention_is_deterministic() {
        let entries = vec![
            entry(0, vec![1.0, 0.0], 0),
            entry(1, vec![0.9, 0.1], 1),
            entry(2, vec![0.0, 1.0], 2),
            entry(3, vec![-1.0, 0.0], 3),
        ];
        let policy = StructuralKeyframeRetention::default();
        let a = policy.select_survivors(&entries, 2, &[]);
        let b = policy.select_survivors(&entries, 2, &[]);
        assert_eq!(a, b);
    }

    #[test]
    fn keyframe_retention_respects_target_size() {
        let entries: Vec<MemoryEntry> = (0..10u64)
            .map(|t| entry(t, vec![t as f32, 0.0], t))
            .collect();
        let policy = StructuralKeyframeRetention::new(
            CoherenceWeights::default(),
            StructuralMetric::default(),
        );
        for budget in [1, 3, 7, 10] {
            let survivors = policy.select_survivors(&entries, budget, &[]);
            assert_eq!(survivors.len(), budget, "budget {budget} not respected");
        }
    }

    #[test]
    fn structural_recency_score_matches_tick_recency_order_on_monotone_trajectory() {
        // Documents the B1 negative result: since cumulative structural time
        // is monotone non-decreasing in insertion order (same as raw ticks),
        // ranking by it should select the SAME top-K as plain tick order when
        // there is no churn to compress (strictly increasing embedding
        // movement at every step, no ties).
        let entries: Vec<MemoryEntry> = (0..10u64)
            .map(|t| entry(t, vec![t as f32, 0.0], t))
            .collect();
        let recency = structural_recency(&entries, &[], StructuralMetric::default());
        // Strictly increasing: later entries always score higher.
        for w in recency.windows(2) {
            assert!(
                w[1] > w[0],
                "structural recency should be monotone here: {recency:?}"
            );
        }
    }
}
