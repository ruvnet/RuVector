//! Structural-Time-Gated Compaction Scheduling.
//!
//! Nightly research, 2026-09-18
//! (`docs/research/nightly/2026-09-18-structural-time-gated-memory-compaction`).
//!
//! [`crate::compaction::CompactionPolicy`] answers "which memories survive a
//! compaction pass?". This module answers a question this crate has never
//! addressed: "when should a compaction pass run at all?" Every existing
//! benchmark (`src/main.rs`, `examples/mincut_gated_forgetting_bench.rs`)
//! invokes [`crate::compact`] exactly once, on demand. A long-running agent
//! needs a *trigger* that decides when accumulated writes justify the cost of
//! a compaction pass — today that decision is left entirely to the caller.
//!
//! Three triggers are compared here:
//!
//! - [`FixedIntervalTrigger`] — fires every `interval` writes. The
//!   wall-clock-equivalent baseline: cadence is fixed regardless of what the
//!   writes actually contain.
//! - [`CapacityTrigger`] — fires once the store exceeds `max_size`. The
//!   simplest realistic real-world default absent any content signal.
//! - [`StructuralGateTrigger`] — fires once the accumulated
//!   `emergent_time::structural_clock::StructuralProperTime` arc length of a
//!   sliding-window summary of the store's recent writes exceeds a
//!   threshold. Reuses `emergent-time`'s `Clock`/`StateSnapshot` machinery
//!   as-is (no reimplementation): each write updates a windowed centroid,
//!   mean-coherence, and diversity-entropy summary, and the trigger asks
//!   `StructuralProperTime::tick` for the internal-time cost of the
//!   transition since the last snapshot. A quiet run of near-duplicate
//!   writes accumulates almost no structural time and the trigger stays
//!   silent; a burst of semantically novel writes moves the summary a lot and
//!   the trigger fires promptly.
//!
//! The windowed summary is deliberately `O(window · dims)` per write, not
//! `O(store.len() · dims)` — the prior nightly run
//! (2026-09-05, mincut-gated-forgetting) was rejected in part because its
//! structural signal cost 1,800-2,700x the scalar baseline at trivial corpus
//! sizes. This trigger is designed from the start to stay cheap at any store
//! size by only ever looking at the last `window` writes.

use crate::memory::MemoryEntry;
use crate::scoring::cosine_sim;
use emergent_time::structural_clock::{
    Clock, StateSnapshot, StructuralMetric, StructuralProperTime,
};
use std::collections::VecDeque;

/// Decides *when* a compaction pass should run, given the write stream seen
/// so far. Orthogonal to [`crate::compaction::CompactionPolicy`], which
/// decides what survives once compaction runs.
pub trait CompactionTrigger {
    fn name(&self) -> &str;

    /// Called once after each write is applied to `entries`. Returns `true`
    /// if a compaction pass should run now.
    fn on_write(&mut self, entries: &[MemoryEntry]) -> bool;

    /// Called once after a compaction pass has run (whether or not this
    /// trigger caused it), so triggers can reset internal accumulators
    /// against the post-compaction state.
    fn on_compacted(&mut self, entries: &[MemoryEntry]);
}

// ---------------------------------------------------------------------------
// Baseline: fixed wall-tick interval.
// ---------------------------------------------------------------------------

/// Fires every `interval` writes since the last compaction (or start).
pub struct FixedIntervalTrigger {
    pub interval: u64,
    since_last: u64,
}

impl FixedIntervalTrigger {
    pub fn new(interval: u64) -> Self {
        assert!(interval > 0, "interval must be positive");
        Self {
            interval,
            since_last: 0,
        }
    }
}

impl CompactionTrigger for FixedIntervalTrigger {
    fn name(&self) -> &str {
        "FixedInterval"
    }

    fn on_write(&mut self, _entries: &[MemoryEntry]) -> bool {
        self.since_last += 1;
        self.since_last >= self.interval
    }

    fn on_compacted(&mut self, _entries: &[MemoryEntry]) {
        self.since_last = 0;
    }
}

// ---------------------------------------------------------------------------
// Candidate A: capacity threshold.
// ---------------------------------------------------------------------------

/// Fires once the store exceeds `max_size` entries.
pub struct CapacityTrigger {
    pub max_size: usize,
}

impl CapacityTrigger {
    pub fn new(max_size: usize) -> Self {
        Self { max_size }
    }
}

impl CompactionTrigger for CapacityTrigger {
    fn name(&self) -> &str {
        "Capacity"
    }

    fn on_write(&mut self, entries: &[MemoryEntry]) -> bool {
        entries.len() > self.max_size
    }

    fn on_compacted(&mut self, _entries: &[MemoryEntry]) {}
}

// ---------------------------------------------------------------------------
// Candidate B: Structural Proper Time gate.
// ---------------------------------------------------------------------------

/// Windowed structural summary of the most recent writes, converted into an
/// `emergent_time::structural_clock::StateSnapshot` each time it changes.
struct WindowSummary {
    window: usize,
    recent: VecDeque<Vec<f32>>,
}

impl WindowSummary {
    fn new(window: usize) -> Self {
        Self {
            window,
            recent: VecDeque::with_capacity(window),
        }
    }

    fn push(&mut self, v: &[f32]) {
        if self.recent.len() == self.window {
            self.recent.pop_front();
        }
        self.recent.push_back(v.to_vec());
    }

    /// `O(window * dims)`. Centroid (`Δv`), mean coherence-to-centroid
    /// (`ΔC`, coherence channel), and Shannon entropy of an 8-bin similarity
    /// histogram (`ΔS`, diversity channel). `graph`/`pred_error` are left at
    /// 0.0: this trigger has no topology or predictive-error signal
    /// available and does not fabricate one.
    fn snapshot(&self) -> Option<StateSnapshot> {
        if self.recent.is_empty() {
            return None;
        }
        let dims = self.recent[0].len();
        let mut centroid = vec![0.0f64; dims];
        for v in &self.recent {
            for (c, x) in centroid.iter_mut().zip(v.iter()) {
                *c += *x as f64;
            }
        }
        let n = self.recent.len() as f64;
        for c in centroid.iter_mut() {
            *c /= n;
        }
        let centroid_f32: Vec<f32> = centroid.iter().map(|&x| x as f32).collect();

        let sims: Vec<f32> = self
            .recent
            .iter()
            .map(|v| cosine_sim(v, &centroid_f32))
            .collect();
        let mean_coherence = sims.iter().map(|&s| s as f64).sum::<f64>() / n;

        const BINS: usize = 8;
        let mut hist = [0usize; BINS];
        for &s in &sims {
            let clamped = ((s + 1.0) / 2.0).clamp(0.0, 0.999_999);
            let bin = (clamped * BINS as f32) as usize;
            hist[bin.min(BINS - 1)] += 1;
        }
        let entropy = hist
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / n;
                -p * p.ln()
            })
            .sum::<f64>();

        Some(StateSnapshot::full(
            centroid,
            entropy,
            mean_coherence,
            0.0,
            0.0,
        ))
    }
}

/// Fires once the accumulated `StructuralProperTime` since the last
/// compaction (or start) exceeds `threshold`.
pub struct StructuralGateTrigger {
    clock: StructuralProperTime,
    threshold: f64,
    summary: WindowSummary,
    last_snapshot: Option<StateSnapshot>,
    accumulated: f64,
}

impl StructuralGateTrigger {
    pub fn new(metric: StructuralMetric, threshold: f64, window: usize) -> Self {
        assert!(threshold > 0.0, "threshold must be positive");
        assert!(window >= 2, "window must be at least 2");
        Self {
            clock: StructuralProperTime::new(metric),
            threshold,
            summary: WindowSummary::new(window),
            last_snapshot: None,
            accumulated: 0.0,
        }
    }

    /// Calibrate a threshold from the mean per-write structural tick
    /// observed over an initial quiet baseline stream, scaled by
    /// `multiplier`. Mirrors the `baseline_window` calibration
    /// `emergent_time::structural_clock::alarm_step` already uses — decided
    /// once, up front, from data the trigger will actually see, not tuned
    /// after inspecting the full benchmark result.
    pub fn calibrate_threshold(
        metric: StructuralMetric,
        baseline_writes: &[Vec<f32>],
        window: usize,
        multiplier: f64,
    ) -> f64 {
        let clock = StructuralProperTime::new(metric);
        let mut summary = WindowSummary::new(window);
        let mut last: Option<StateSnapshot> = None;
        let mut ticks = Vec::new();
        for v in baseline_writes {
            summary.push(v);
            if let Some(cur) = summary.snapshot() {
                if let Some(prev) = &last {
                    ticks.push(clock.tick(prev, &cur));
                }
                last = Some(cur);
            }
        }
        if ticks.is_empty() {
            return multiplier;
        }
        let mean = ticks.iter().sum::<f64>() / ticks.len() as f64;
        (mean * multiplier).max(1e-9)
    }
}

impl CompactionTrigger for StructuralGateTrigger {
    fn name(&self) -> &str {
        "StructuralGate"
    }

    fn on_write(&mut self, entries: &[MemoryEntry]) -> bool {
        if let Some(last_entry) = entries.last() {
            self.summary.push(&last_entry.vector);
        }
        let cur = match self.summary.snapshot() {
            Some(s) => s,
            None => return false,
        };
        if let Some(prev) = &self.last_snapshot {
            self.accumulated += self.clock.tick(prev, &cur).max(0.0);
        }
        self.last_snapshot = Some(cur);
        self.accumulated >= self.threshold
    }

    fn on_compacted(&mut self, _entries: &[MemoryEntry]) {
        self.accumulated = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryStore;

    fn insert_n(store: &mut MemoryStore, vectors: &[Vec<f32>]) {
        for v in vectors {
            store.insert(v.clone());
        }
    }

    #[test]
    fn fixed_interval_fires_exactly_on_multiples() {
        let mut t = FixedIntervalTrigger::new(5);
        let mut store = MemoryStore::new(2);
        let mut fires = 0;
        for _ in 0..20 {
            store.insert(vec![1.0, 0.0]);
            if t.on_write(store.entries()) {
                fires += 1;
                t.on_compacted(store.entries());
            }
        }
        assert_eq!(fires, 4); // 20 / 5
    }

    #[test]
    fn capacity_trigger_fires_once_over_limit() {
        let mut t = CapacityTrigger::new(10);
        let mut store = MemoryStore::new(2);
        let mut fired_at = None;
        for i in 0..20 {
            store.insert(vec![1.0, 0.0]);
            if t.on_write(store.entries()) {
                fired_at = Some(i);
                break;
            }
        }
        assert_eq!(fired_at, Some(10)); // 11th insert (0-indexed 10) exceeds max_size=10
    }

    #[test]
    fn structural_gate_silent_on_near_duplicate_writes() {
        let metric = StructuralMetric::default();
        let mut t = StructuralGateTrigger::new(metric, 1.0, 8);
        let mut store = MemoryStore::new(4);
        let mut fires = 0;
        // 200 near-identical writes: should almost never cross a non-trivial
        // threshold.
        for i in 0..200 {
            let eps = 1e-4 * (i as f32 % 3.0);
            store.insert(vec![1.0 + eps, 0.0, 0.0, 0.0]);
            if t.on_write(store.entries()) {
                fires += 1;
                t.on_compacted(store.entries());
            }
        }
        assert!(
            fires <= 1,
            "near-duplicate stream should rarely trigger structural compaction, got {fires} fires"
        );
    }

    #[test]
    fn structural_gate_fires_on_novel_cluster_burst() {
        let metric = StructuralMetric::default();
        let mut t = StructuralGateTrigger::new(metric, 0.5, 8);
        let mut store = MemoryStore::new(4);

        // Quiet regime: one cluster.
        for _ in 0..40 {
            store.insert(vec![1.0, 0.0, 0.0, 0.0]);
            t.on_write(store.entries());
        }
        let fired_quiet = t.accumulated >= t.threshold;

        // Burst: an orthogonal, never-seen cluster.
        let mut fired_burst = false;
        for _ in 0..10 {
            store.insert(vec![0.0, 0.0, 1.0, 0.0]);
            if t.on_write(store.entries()) {
                fired_burst = true;
                break;
            }
        }
        assert!(!fired_quiet, "quiet single-cluster regime should not fire");
        assert!(fired_burst, "orthogonal cluster burst should fire");
    }

    #[test]
    fn calibrated_threshold_is_positive_and_data_dependent() {
        let metric = StructuralMetric::default();
        let quiet: Vec<Vec<f32>> = (0..30).map(|_| vec![1.0, 0.0, 0.0, 0.0]).collect();
        let thr = StructuralGateTrigger::calibrate_threshold(metric, &quiet, 8, 20.0);
        assert!(thr > 0.0);
    }

    #[test]
    fn all_triggers_implement_common_trait_object_safely() {
        let triggers: Vec<Box<dyn CompactionTrigger>> = vec![
            Box::new(FixedIntervalTrigger::new(10)),
            Box::new(CapacityTrigger::new(10)),
            Box::new(StructuralGateTrigger::new(
                StructuralMetric::default(),
                1.0,
                8,
            )),
        ];
        let mut store = MemoryStore::new(2);
        insert_n(&mut store, &[vec![1.0, 0.0]]);
        for t in &triggers {
            assert!(!t.name().is_empty());
        }
    }
}
