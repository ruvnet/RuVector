//! Task control block and state management.
//!
//! The `TaskControlBlock` contains all the information needed by the scheduler
//! to make scheduling decisions, including deadline, novelty, and coherence data.

use crate::{Duration, Instant};
use ruvix_cap::CapRights;
use ruvix_types::{SchedulerScore, TaskHandle, TaskPriority};

/// Task execution state.
///
/// Tasks transition through these states during their lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TaskState {
    /// Task is ready to run and is in a ready queue.
    Ready = 0,

    /// Task is currently executing on a CPU.
    Running = 1,

    /// Task is blocked waiting for a queue operation or timer.
    Blocked = 2,

    /// Task is suspended (explicitly paused, not in any queue).
    Suspended = 3,

    /// Task has terminated and is pending cleanup.
    Terminated = 4,
}

impl TaskState {
    /// Returns true if the task can be scheduled.
    #[inline]
    #[must_use]
    pub const fn is_schedulable(&self) -> bool {
        matches!(self, Self::Ready | Self::Running)
    }

    /// Returns true if the task can transition to the target state.
    #[inline]
    #[must_use]
    pub const fn can_transition_to(&self, target: Self) -> bool {
        match (*self, target) {
            // Ready can go to Running, Blocked, Suspended, or Terminated
            (Self::Ready, Self::Running)
            | (Self::Ready, Self::Blocked)
            | (Self::Ready, Self::Suspended)
            | (Self::Ready, Self::Terminated) => true,

            // Running can go to Ready, Blocked, Suspended, or Terminated
            (Self::Running, Self::Ready)
            | (Self::Running, Self::Blocked)
            | (Self::Running, Self::Suspended)
            | (Self::Running, Self::Terminated) => true,

            // Blocked can go to Ready, Suspended, or Terminated
            (Self::Blocked, Self::Ready)
            | (Self::Blocked, Self::Suspended)
            | (Self::Blocked, Self::Terminated) => true,

            // Suspended can go to Ready, Blocked, or Terminated
            (Self::Suspended, Self::Ready)
            | (Self::Suspended, Self::Blocked)
            | (Self::Suspended, Self::Terminated) => true,

            // Terminated is final
            (Self::Terminated, _) => false,

            // Same state is not a valid transition
            _ => false,
        }
    }
}

impl Default for TaskState {
    fn default() -> Self {
        Self::Ready
    }
}

/// Task Control Block (TCB).
///
/// Contains all information needed by the scheduler to manage a task.
/// This is analogous to seL4's TCB but extended with coherence-aware
/// scheduling signals.
#[derive(Debug, Clone)]
#[repr(C)]
pub struct TaskControlBlock {
    /// Handle to this task.
    pub task_id: TaskHandle,

    /// Capability rights held by this task (determines accessible resources).
    pub capability_set: CapRights,

    /// Base scheduling priority.
    pub priority: TaskPriority,

    /// Hard deadline for real-time tasks. If `Some`, the scheduler uses EDF
    /// within the task's partition.
    pub deadline: Option<Instant>,

    /// Novelty of pending input (0.0..1.0).
    /// Higher values indicate the task is processing genuinely new information.
    /// This provides a priority boost to encourage exploration.
    pub pending_input_novelty: f32,

    /// Expected coherence delta if this task's pending mutations are applied.
    /// Negative values indicate mutations that would lower the coherence score.
    /// The scheduler penalizes tasks with negative coherence deltas.
    pub pending_coherence_delta: f32,

    /// Current task state.
    pub state: TaskState,

    /// Partition ID (typically matches RVF mount ID).
    /// Tasks are grouped by partition for guaranteed time slice allocation.
    pub partition_id: u32,

    /// Time quantum remaining for this task (in microseconds).
    pub time_remaining_us: u32,

    /// Total CPU time consumed by this task (in microseconds).
    pub total_cpu_time_us: u64,

    /// Number of times this task has been scheduled.
    pub schedule_count: u64,

    /// Number of times this task has missed its deadline.
    pub deadline_miss_count: u32,

    /// Cached scheduler score (recomputed when inputs change).
    cached_score: Option<SchedulerScore>,
}

impl TaskControlBlock {
    /// Creates a new task control block.
    #[inline]
    #[must_use]
    pub const fn new(
        task_id: TaskHandle,
        capability_set: CapRights,
        priority: TaskPriority,
        partition_id: u32,
    ) -> Self {
        Self {
            task_id,
            capability_set,
            priority,
            deadline: None,
            pending_input_novelty: 0.0,
            pending_coherence_delta: 0.0,
            state: TaskState::Ready,
            partition_id,
            time_remaining_us: 0,
            total_cpu_time_us: 0,
            schedule_count: 0,
            deadline_miss_count: 0,
            cached_score: None,
        }
    }

    /// Creates a new task control block with a deadline.
    #[inline]
    #[must_use]
    pub const fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self.cached_score = None;
        self
    }

    /// Sets the novelty value (0.0..1.0).
    ///
    /// # Panics
    ///
    /// Does not panic; values are clamped to valid range.
    #[inline]
    #[must_use]
    pub fn with_novelty(mut self, novelty: f32) -> Self {
        self.pending_input_novelty = novelty.clamp(0.0, 1.0);
        self.cached_score = None;
        self
    }

    /// Sets the coherence delta.
    #[inline]
    #[must_use]
    pub fn with_coherence_delta(mut self, delta: f32) -> Self {
        self.pending_coherence_delta = delta;
        self.cached_score = None;
        self
    }

    /// Sets the time quantum for this task.
    #[inline]
    #[must_use]
    pub const fn with_time_quantum(mut self, quantum_us: u32) -> Self {
        self.time_remaining_us = quantum_us;
        self
    }

    /// Updates the novelty value.
    #[inline]
    pub fn set_novelty(&mut self, novelty: f32) {
        self.pending_input_novelty = novelty.clamp(0.0, 1.0);
        self.cached_score = None;
    }

    /// Updates the coherence delta.
    #[inline]
    pub fn set_coherence_delta(&mut self, delta: f32) {
        self.pending_coherence_delta = delta;
        self.cached_score = None;
    }

    /// Updates the deadline.
    #[inline]
    pub fn set_deadline(&mut self, deadline: Option<Instant>) {
        self.deadline = deadline;
        self.cached_score = None;
    }

    /// Transitions the task to a new state.
    ///
    /// Returns `true` if the transition was valid, `false` otherwise.
    #[inline]
    pub fn transition_to(&mut self, new_state: TaskState) -> bool {
        if self.state.can_transition_to(new_state) {
            self.state = new_state;
            true
        } else {
            false
        }
    }

    /// Consumes time from the task's quantum.
    ///
    /// Returns `true` if the task still has time remaining.
    #[inline]
    pub fn consume_time(&mut self, elapsed_us: u32) -> bool {
        self.time_remaining_us = self.time_remaining_us.saturating_sub(elapsed_us);
        self.total_cpu_time_us = self.total_cpu_time_us.saturating_add(u64::from(elapsed_us));
        self.time_remaining_us > 0
    }

    /// Resets the time quantum.
    #[inline]
    pub fn reset_quantum(&mut self, quantum_us: u32) {
        self.time_remaining_us = quantum_us;
    }

    /// Marks that this task has been scheduled.
    #[inline]
    pub fn mark_scheduled(&mut self) {
        self.schedule_count = self.schedule_count.saturating_add(1);
    }

    /// Marks a deadline miss.
    #[inline]
    pub fn mark_deadline_miss(&mut self) {
        self.deadline_miss_count = self.deadline_miss_count.saturating_add(1);
    }

    /// Returns the cached scheduler score, if available.
    #[inline]
    #[must_use]
    pub const fn cached_score(&self) -> Option<SchedulerScore> {
        self.cached_score
    }

    /// Updates the cached scheduler score.
    #[inline]
    pub fn set_cached_score(&mut self, score: SchedulerScore) {
        self.cached_score = Some(score);
    }

    /// Invalidates the cached scheduler score.
    #[inline]
    pub fn invalidate_score(&mut self) {
        self.cached_score = None;
    }

    /// Decays the novelty value by a factor.
    #[inline]
    pub fn decay_novelty(&mut self, factor: f32) {
        self.pending_input_novelty *= factor;
        if self.pending_input_novelty < 0.001 {
            self.pending_input_novelty = 0.0;
        }
        self.cached_score = None;
    }

    /// Returns true if this task has a hard deadline.
    #[inline]
    #[must_use]
    pub const fn has_deadline(&self) -> bool {
        self.deadline.is_some()
    }

    /// Returns true if the task's deadline has passed.
    #[inline]
    #[must_use]
    pub fn deadline_passed(&self, now: Instant) -> bool {
        self.deadline.map_or(false, |d| now >= d)
    }

    /// Returns the time until deadline, or None if no deadline.
    #[inline]
    #[must_use]
    pub fn time_until_deadline(&self, now: Instant) -> Option<Duration> {
        self.deadline.map(|d| d.saturating_duration_since(now))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_state_transitions() {
        assert!(TaskState::Ready.can_transition_to(TaskState::Running));
        assert!(TaskState::Running.can_transition_to(TaskState::Ready));
        assert!(TaskState::Running.can_transition_to(TaskState::Blocked));
        assert!(TaskState::Blocked.can_transition_to(TaskState::Ready));
        assert!(!TaskState::Terminated.can_transition_to(TaskState::Ready));
    }

    #[test]
    fn test_task_state_schedulable() {
        assert!(TaskState::Ready.is_schedulable());
        assert!(TaskState::Running.is_schedulable());
        assert!(!TaskState::Blocked.is_schedulable());
        assert!(!TaskState::Suspended.is_schedulable());
        assert!(!TaskState::Terminated.is_schedulable());
    }

    #[test]
    fn test_tcb_creation() {
        let task = TaskControlBlock::new(
            TaskHandle::new(1, 0),
            CapRights::READ,
            TaskPriority::Normal,
            0,
        );

        assert_eq!(task.state, TaskState::Ready);
        assert_eq!(task.partition_id, 0);
        assert!(!task.has_deadline());
    }

    #[test]
    fn test_tcb_with_deadline() {
        let deadline = Instant::from_micros(10000);
        let task = TaskControlBlock::new(
            TaskHandle::new(1, 0),
            CapRights::READ,
            TaskPriority::Normal,
            0,
        )
        .with_deadline(deadline);

        assert!(task.has_deadline());
        assert_eq!(task.deadline, Some(deadline));
    }

    #[test]
    fn test_tcb_novelty_clamping() {
        let task = TaskControlBlock::new(
            TaskHandle::new(1, 0),
            CapRights::READ,
            TaskPriority::Normal,
            0,
        )
        .with_novelty(1.5);

        assert!((task.pending_input_novelty - 1.0).abs() < 0.001);

        let task2 = task.with_novelty(-0.5);
        assert!(task2.pending_input_novelty.abs() < 0.001);
    }

    #[test]
    fn test_tcb_time_consumption() {
        let mut task = TaskControlBlock::new(
            TaskHandle::new(1, 0),
            CapRights::READ,
            TaskPriority::Normal,
            0,
        )
        .with_time_quantum(1000);

        assert!(task.consume_time(500));
        assert_eq!(task.time_remaining_us, 500);
        assert_eq!(task.total_cpu_time_us, 500);

        assert!(!task.consume_time(600));
        assert_eq!(task.time_remaining_us, 0);
        assert_eq!(task.total_cpu_time_us, 1100);
    }

    #[test]
    fn test_tcb_deadline_tracking() {
        let now = Instant::from_micros(1000);
        let deadline = Instant::from_micros(2000);

        let task = TaskControlBlock::new(
            TaskHandle::new(1, 0),
            CapRights::READ,
            TaskPriority::Normal,
            0,
        )
        .with_deadline(deadline);

        assert!(!task.deadline_passed(now));
        assert!(task.deadline_passed(Instant::from_micros(2001)));

        let duration = task.time_until_deadline(now).unwrap();
        assert_eq!(duration.as_micros(), 1000);
    }

    #[test]
    fn test_tcb_state_transition() {
        let mut task = TaskControlBlock::new(
            TaskHandle::new(1, 0),
            CapRights::READ,
            TaskPriority::Normal,
            0,
        );

        assert!(task.transition_to(TaskState::Running));
        assert_eq!(task.state, TaskState::Running);

        assert!(task.transition_to(TaskState::Blocked));
        assert_eq!(task.state, TaskState::Blocked);

        assert!(task.transition_to(TaskState::Terminated));
        assert_eq!(task.state, TaskState::Terminated);

        // Cannot transition from Terminated
        assert!(!task.transition_to(TaskState::Ready));
    }

    #[test]
    fn test_tcb_novelty_decay() {
        let mut task = TaskControlBlock::new(
            TaskHandle::new(1, 0),
            CapRights::READ,
            TaskPriority::Normal,
            0,
        )
        .with_novelty(1.0);

        task.decay_novelty(0.5);
        assert!((task.pending_input_novelty - 0.5).abs() < 0.001);

        // Decay to near zero
        for _ in 0..20 {
            task.decay_novelty(0.5);
        }
        assert!(task.pending_input_novelty.abs() < 0.001);
    }
}
