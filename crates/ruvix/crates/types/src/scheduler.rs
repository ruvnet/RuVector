//! Scheduler types for coherence-aware task scheduling.
//!
//! The RuVix scheduler combines deadline pressure, novelty signal,
//! and structural risk to determine task priority.

/// Scheduler score combining multiple signals.
///
/// The final priority is computed from:
/// - Deadline urgency (inverse of time remaining)
/// - Novelty boost (for tasks processing new information)
/// - Risk penalty (for tasks that would lower coherence)
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct SchedulerScore {
    /// Combined score (higher = higher priority).
    /// Range: typically 0.0 to 10.0, but can exceed bounds.
    pub score: f32,

    /// Deadline urgency component (0.0 to 5.0).
    pub deadline_urgency: f32,

    /// Novelty boost component (0.0 to 1.0).
    pub novelty_boost: f32,

    /// Risk penalty component (0.0 to 2.0, subtracted from score).
    pub risk_penalty: f32,
}

impl SchedulerScore {
    /// Creates a new scheduler score with explicit components.
    #[inline]
    #[must_use]
    pub const fn new(
        deadline_urgency: f32,
        novelty_boost: f32,
        risk_penalty: f32,
    ) -> Self {
        Self {
            score: deadline_urgency + novelty_boost - risk_penalty,
            deadline_urgency,
            novelty_boost,
            risk_penalty,
        }
    }

    /// Creates a default priority score (normal task, no urgency).
    #[inline]
    #[must_use]
    pub const fn normal() -> Self {
        Self {
            score: 1.0,
            deadline_urgency: 1.0,
            novelty_boost: 0.0,
            risk_penalty: 0.0,
        }
    }

    /// Creates a high priority score.
    #[inline]
    #[must_use]
    pub const fn high() -> Self {
        Self {
            score: 3.0,
            deadline_urgency: 3.0,
            novelty_boost: 0.0,
            risk_penalty: 0.0,
        }
    }

    /// Creates a critical priority score.
    #[inline]
    #[must_use]
    pub const fn critical() -> Self {
        Self {
            score: 5.0,
            deadline_urgency: 5.0,
            novelty_boost: 0.0,
            risk_penalty: 0.0,
        }
    }

    /// Returns true if this score is higher than another.
    #[inline]
    #[must_use]
    pub fn is_higher_than(&self, other: &Self) -> bool {
        self.score > other.score
    }

    /// Adds novelty boost to the score.
    #[inline]
    #[must_use]
    pub fn with_novelty(mut self, boost: f32) -> Self {
        let clamped = if boost < 0.0 {
            0.0
        } else if boost > 1.0 {
            1.0
        } else {
            boost
        };
        self.novelty_boost = clamped;
        self.score = self.deadline_urgency + self.novelty_boost - self.risk_penalty;
        self
    }

    /// Adds risk penalty to the score.
    #[inline]
    #[must_use]
    pub fn with_risk(mut self, penalty: f32) -> Self {
        let clamped = if penalty < 0.0 {
            0.0
        } else if penalty > 2.0 {
            2.0
        } else {
            penalty
        };
        self.risk_penalty = clamped;
        self.score = self.deadline_urgency + self.novelty_boost - self.risk_penalty;
        self
    }
}

impl Default for SchedulerScore {
    fn default() -> Self {
        Self::normal()
    }
}

impl PartialOrd for SchedulerScore {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

/// Task scheduling partition.
///
/// Tasks are grouped by their RVF mount origin. Each partition gets
/// a guaranteed time slice, preventing a misbehaving component from
/// starving others.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct SchedulerPartition {
    /// Partition ID (typically matches RVF mount ID).
    pub partition_id: u32,

    /// Time slice in microseconds per scheduling epoch.
    pub time_slice_us: u32,

    /// Current tasks in this partition.
    pub task_count: u32,

    /// Remaining time in current epoch.
    pub remaining_us: u32,
}

impl SchedulerPartition {
    /// Creates a new scheduler partition.
    #[inline]
    #[must_use]
    pub const fn new(partition_id: u32, time_slice_us: u32) -> Self {
        Self {
            partition_id,
            time_slice_us,
            task_count: 0,
            remaining_us: time_slice_us,
        }
    }

    /// Returns true if the partition has exhausted its time slice.
    #[inline]
    #[must_use]
    pub const fn is_exhausted(&self) -> bool {
        self.remaining_us == 0
    }

    /// Resets the partition for a new scheduling epoch.
    #[inline]
    pub fn reset(&mut self) {
        self.remaining_us = self.time_slice_us;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_score_ordering() {
        let low = SchedulerScore::normal();
        let high = SchedulerScore::high();
        let critical = SchedulerScore::critical();

        assert!(high.is_higher_than(&low));
        assert!(critical.is_higher_than(&high));
    }

    #[test]
    fn test_scheduler_score_with_novelty() {
        let base = SchedulerScore::normal();
        let boosted = base.with_novelty(0.5);

        assert!(boosted.is_higher_than(&base));
        assert!((boosted.novelty_boost - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_scheduler_score_with_risk() {
        let base = SchedulerScore::normal();
        let penalized = base.with_risk(1.0);

        assert!(!penalized.is_higher_than(&base));
        assert!((penalized.risk_penalty - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_scheduler_partition() {
        let mut partition = SchedulerPartition::new(1, 10000);
        assert!(!partition.is_exhausted());

        partition.remaining_us = 0;
        assert!(partition.is_exhausted());

        partition.reset();
        assert!(!partition.is_exhausted());
    }
}
