//! Online auto-tuner machinery for SONA config (ADR-271, Ornith-1.0 borrow #4).
//!
//! Offline evolution tunes a config to a *fixed* benchmark. Real workloads drift
//! (non-stationary trajectory streams), so a fixed config goes stale. This module
//! provides the **staleness-weighted** primitives for an *online* tuner that
//! re-optimizes against the live stream, weighting recent observations over old
//! ones via Ornith-1.0's staleness weight `w(d_t)`:
//!
//! ```text
//! w(d) = 1                       if d <= k1        (fresh — full weight)
//!      = exp(-lambda*(d - k1))   if k1 < d <= k2   (decaying)
//!      = 0                       if d  > k2        (too stale — dropped)
//! ```
//!
//! where `d` is the *age* (clock ticks since the observation). The
//! [`StalenessWindow`] maintains a staleness-weighted running estimate of "how
//! well the current config is doing lately"; a `(1+1)`-ES on top of it (see
//! `examples/darwin_autotuner.rs`) accepts a perturbed config only when its
//! recent, freshness-weighted score beats the incumbent — so the tuner tracks a
//! drifting optimum instead of averaging over a stale past.

use std::collections::VecDeque;

/// Ornith-1.0 staleness schedule `w(d_t)`.
#[derive(Clone, Copy, Debug)]
pub struct StalenessSchedule {
    /// Ages `<= k1` keep full weight 1.0.
    pub k1: u64,
    /// Ages `> k2` are dropped (weight 0).
    pub k2: u64,
    /// Exponential decay rate in the `(k1, k2]` band.
    pub lambda: f32,
}

impl StalenessSchedule {
    /// A sensible default: full weight for 16 ticks, decay to ~0 by 64.
    #[must_use]
    pub fn new(k1: u64, k2: u64, lambda: f32) -> Self {
        Self { k1, k2, lambda }
    }

    /// `w(d)` for an observation of age `d` ticks.
    #[must_use]
    pub fn weight(&self, age: u64) -> f32 {
        if age <= self.k1 {
            1.0
        } else if age <= self.k2 {
            (-self.lambda * (age - self.k1) as f32).exp()
        } else {
            0.0
        }
    }
}

impl Default for StalenessSchedule {
    fn default() -> Self {
        Self {
            k1: 16,
            k2: 64,
            lambda: 0.08,
        }
    }
}

/// A staleness-weighted window of recent scalar observations (e.g. per-step loss
/// under the current config). `push` advances the clock; `weighted_mean` reports
/// the freshness-weighted average; observations past `k2` are evicted.
#[derive(Clone, Debug)]
pub struct StalenessWindow {
    schedule: StalenessSchedule,
    /// `(value, recorded_at_clock)` newest-last.
    samples: VecDeque<(f32, u64)>,
    clock: u64,
    cap: usize,
}

impl StalenessWindow {
    /// New window with the given schedule and a hard capacity cap.
    #[must_use]
    pub fn new(schedule: StalenessSchedule, cap: usize) -> Self {
        Self {
            schedule,
            samples: VecDeque::with_capacity(cap),
            clock: 0,
            cap: cap.max(1),
        }
    }

    /// Record an observation under the current config; advances the clock and
    /// evicts samples that are too stale (`age > k2`) or over capacity.
    pub fn push(&mut self, value: f32) {
        self.samples.push_back((value, self.clock));
        self.clock += 1;
        while self.samples.len() > self.cap {
            self.samples.pop_front();
        }
        // Evict fully-stale observations from the front.
        while let Some(&(_, t)) = self.samples.front() {
            if self.clock.saturating_sub(t) > self.schedule.k2 {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }

    /// Reset the recorded observations but keep the clock running — used after a
    /// config switch so the new config is scored on *its own* fresh samples.
    pub fn clear_samples(&mut self) {
        self.samples.clear();
    }

    /// Staleness-weighted mean of the window, or `None` if empty / all-stale.
    #[must_use]
    pub fn weighted_mean(&self) -> Option<f32> {
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        for &(v, t) in &self.samples {
            let w = self.schedule.weight(self.clock.saturating_sub(t));
            num += w * v;
            den += w;
        }
        if den > 0.0 {
            Some(num / den)
        } else {
            None
        }
    }

    /// Current clock (number of observations ever pushed).
    #[must_use]
    pub fn clock(&self) -> u64 {
        self.clock
    }

    /// Number of live (non-evicted) observations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the window holds no live observations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_is_fresh_then_decays_then_drops() {
        let s = StalenessSchedule::new(4, 10, 0.5);
        assert_eq!(s.weight(0), 1.0);
        assert_eq!(s.weight(4), 1.0);
        let w5 = s.weight(5);
        assert!(w5 < 1.0 && w5 > 0.0); // decaying
        assert!(s.weight(9) < w5); // monotone decay
        assert_eq!(s.weight(11), 0.0); // dropped past k2
    }

    #[test]
    fn weighted_mean_favors_recent() {
        // Old samples = 1.0, recent samples = 0.0; the weighted mean must sit
        // well below the unweighted 0.5 because recent dominates.
        let mut w = StalenessWindow::new(StalenessSchedule::new(2, 32, 0.3), 64);
        for _ in 0..20 {
            w.push(1.0);
        }
        for _ in 0..20 {
            w.push(0.0);
        }
        let m = w.weighted_mean().unwrap();
        assert!(
            m < 0.25,
            "recent-weighted mean {m} should be near the recent 0.0"
        );
    }

    #[test]
    fn stale_observations_are_evicted() {
        let mut w = StalenessWindow::new(StalenessSchedule::new(2, 8, 0.5), 1000);
        for _ in 0..50 {
            w.push(1.0);
        }
        // Only observations within k2=8 ticks of the clock survive.
        assert!(w.len() <= 9, "expected <=9 live samples, got {}", w.len());
        assert!(w.weighted_mean().is_some());
    }

    #[test]
    fn empty_window_has_no_mean() {
        let w = StalenessWindow::new(StalenessSchedule::default(), 8);
        assert!(w.weighted_mean().is_none());
        assert!(w.is_empty());
    }
}
