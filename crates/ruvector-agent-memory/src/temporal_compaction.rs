//! Structural-time-gated memory compaction (nightly research, 2026-08-29, ADR-340).
//!
//! `CoherencePolicy` (see [`crate::compaction`]) scores recency from
//! `last_accessed_at`, a wall/step clock: every insert or access ticks it by
//! exactly one, whether or not the agent was actually doing anything.
//! `CoherenceWeights`'s recency term is min-max normalized over the whole
//! store (`(t - min) / (max - min)`), so **a single long idle gap** — the
//! agent waiting on a human, a slow tool call, or a scheduled pause —
//! dominates that range. Every memory written *before* the gap gets crushed
//! toward `recency ≈ 0`, indistinguishable from memories that really are
//! stale, regardless of how semantically relevant they are. The coherence
//! term (`gamma`) can only partly compensate.
//!
//! This module swaps the recency signal for `emergent_time::structural_clock`
//! — **Structural Proper Time**: internal time that accumulates with
//! *embedding movement* through the memory stream, not wall-clock ticks. An
//! idle gap with zero new memories contributes zero structural time, so
//! memories written just before a gap keep a recency score reflecting their
//! true position in the write sequence instead of being flattened to zero.
//!
//! Two variants are provided, matching the crate's `baseline / candidate_A /
//! candidate_B` convention (baseline = `CoherencePolicy` in
//! [`crate::compaction`]):
//!
//! * [`StructuralTimePolicy`] (candidate A) — pure embedding-arc-length clock
//!   (`StructuralMetric` with only `w_embedding` set; this crate has no
//!   honest entropy/graph/prediction-error channel per memory entry, so those
//!   weights are left at zero rather than fabricated).
//! * [`GatedStructuralTimePolicy`] (candidate B) — the same clock with
//!   `StructuralMetric::gate` set above zero, so embedding jitter below the
//!   gate contributes no structural time. Tests whether ignoring
//!   micro-movements as idle-equivalent noise helps or hurts retention.

use crate::compaction::{CoherenceWeights, CompactionPolicy};
use crate::memory::MemoryEntry;
use crate::scoring::coherence_score;
use emergent_time::structural_clock::{
    Clock, StateSnapshot, StructuralMetric, StructuralProperTime,
};

/// Build the structural-time trajectory from entries in their current slice
/// order. Compaction always calls `select_survivors` with `store.entries()`
/// before any reordering happens, so slice order is write (insertion) order —
/// exactly the "worldline" Structural Proper Time is defined over. Only the
/// embedding channel is populated (`Δv`); entropy/coherence/graph/prediction
/// error are left at zero because this crate has no honest per-entry source
/// for them, and `StructuralMetric::w_embedding`-only weighting means the
/// zeroed channels contribute nothing to the clock regardless.
fn trajectory(entries: &[MemoryEntry]) -> Vec<StateSnapshot> {
    entries
        .iter()
        .map(|e| StateSnapshot::new(e.vector.iter().map(|&x| x as f64).collect(), 0.0, 0.0))
        .collect()
}

/// Recency score per entry (0 = earliest structural time seen, 1 = latest),
/// normalized the same way `CoherencePolicy` normalizes `last_accessed_at`.
fn structural_recency(entries: &[MemoryEntry], metric: StructuralMetric) -> Vec<f32> {
    let traj = trajectory(entries);
    let clock = StructuralProperTime::new(metric);
    let cum = clock.cumulative(&traj);
    let max_t = cum.iter().cloned().fold(0.0_f64, f64::max);
    let range = max_t.max(1e-9); // cumulative(traj)[0] == 0.0 always, so min is 0
    cum.iter().map(|&t| (t / range) as f32).collect()
}

fn weighted_select(
    entries: &[MemoryEntry],
    target_size: usize,
    context: &[Vec<f32>],
    weights: &CoherenceWeights,
    recency_scores: &[f32],
) -> Vec<usize> {
    if entries.is_empty() {
        return Vec::new();
    }
    let max_count = entries.iter().map(|e| e.access_count).max().unwrap_or(1);
    let max_count_f = max_count.max(1) as f32;

    let mut scored: Vec<(usize, f32)> = entries
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let recency = recency_scores[i];
            let frequency = e.access_count as f32 / max_count_f;
            let coherence = if context.is_empty() {
                0.0
            } else {
                coherence_score(&e.vector, context)
            };
            let importance =
                weights.alpha * recency + weights.beta * frequency + weights.gamma * coherence;
            (i, importance)
        })
        .collect();

    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored
        .into_iter()
        .take(target_size)
        .map(|(i, _)| i)
        .collect()
}

/// Candidate A: `CoherenceWeights`-style importance with recency from
/// Structural Proper Time (embedding arc length) instead of `last_accessed_at`.
pub struct StructuralTimePolicy {
    pub weights: CoherenceWeights,
    pub metric: StructuralMetric,
}

impl StructuralTimePolicy {
    pub fn new(weights: CoherenceWeights, metric: StructuralMetric) -> Self {
        Self { weights, metric }
    }
}

impl Default for StructuralTimePolicy {
    fn default() -> Self {
        Self {
            weights: CoherenceWeights::default(),
            metric: StructuralMetric {
                w_embedding: 1.0,
                w_entropy: 0.0,
                w_graph: 0.0,
                w_coherence: 0.0,
                w_pred_error: 0.0,
                gate: 0.0,
            },
        }
    }
}

impl CompactionPolicy for StructuralTimePolicy {
    fn name(&self) -> &str {
        "StructuralTime"
    }

    fn select_survivors(
        &self,
        entries: &[MemoryEntry],
        target_size: usize,
        context: &[Vec<f32>],
    ) -> Vec<usize> {
        let recency = structural_recency(entries, self.metric);
        weighted_select(entries, target_size, context, &self.weights, &recency)
    }
}

/// Candidate B: [`StructuralTimePolicy`] with `StructuralMetric::gate` set
/// above zero, so embedding movement smaller than the gate registers as no
/// structural time at all (treated as idle jitter, not real change).
pub struct GatedStructuralTimePolicy {
    pub weights: CoherenceWeights,
    pub metric: StructuralMetric,
}

impl GatedStructuralTimePolicy {
    pub fn new(weights: CoherenceWeights, gate: f64) -> Self {
        Self {
            weights,
            metric: StructuralMetric {
                w_embedding: 1.0,
                w_entropy: 0.0,
                w_graph: 0.0,
                w_coherence: 0.0,
                w_pred_error: 0.0,
                gate,
            },
        }
    }
}

impl Default for GatedStructuralTimePolicy {
    fn default() -> Self {
        // Gate at 0.05 embedding-L2 units: small enough not to swallow a real
        // topic switch (perturbation noise in the benchmark is 0.35), large
        // enough to absorb near-duplicate re-writes.
        Self::new(CoherenceWeights::default(), 0.05)
    }
}

impl CompactionPolicy for GatedStructuralTimePolicy {
    fn name(&self) -> &str {
        "GatedStructuralTime"
    }

    fn select_survivors(
        &self,
        entries: &[MemoryEntry],
        target_size: usize,
        context: &[Vec<f32>],
    ) -> Vec<usize> {
        let recency = structural_recency(entries, self.metric);
        weighted_select(entries, target_size, context, &self.weights, &recency)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryStore;

    /// An idle gap (encoded only in `last_accessed_at`, per `advance_clock`)
    /// must not move the structural-time recency ranking at all: Structural
    /// Proper Time depends only on the embedding trajectory, never on wall
    /// clock, by construction.
    #[test]
    fn structural_recency_ignores_wall_clock_gaps() {
        let mut store = MemoryStore::new(2);
        store.insert(vec![1.0, 0.0]);
        store.insert(vec![0.9, 0.1]);
        let recency_before =
            structural_recency(store.entries(), StructuralTimePolicy::default().metric);

        store.advance_clock(1_000_000);
        store.insert(vec![0.8, 0.2]);
        // Re-derive recency for just the first two entries (trajectory prefix
        // is unaffected by what's appended after it).
        let recency_after = structural_recency(
            &store.entries()[..2],
            StructuralTimePolicy::default().metric,
        );
        assert_eq!(recency_before, recency_after);
    }

    #[test]
    fn gate_absorbs_small_moves() {
        let entries = vec![
            MemoryEntry::new(0, vec![0.0, 0.0], 0),
            MemoryEntry::new(1, vec![0.001, 0.0], 0), // below gate
            MemoryEntry::new(2, vec![1.0, 0.0], 0),   // above gate
        ];
        let gated = GatedStructuralTimePolicy::new(CoherenceWeights::default(), 0.05);
        let recency = structural_recency(&entries, gated.metric);
        assert_eq!(
            recency[0], recency[1],
            "sub-gate move should not advance time"
        );
        assert!(
            recency[2] > recency[1],
            "above-gate move should advance time"
        );
    }

    #[test]
    fn empty_entries_returns_empty() {
        let policy = StructuralTimePolicy::default();
        assert!(policy.select_survivors(&[], 0, &[]).is_empty());
    }
}
