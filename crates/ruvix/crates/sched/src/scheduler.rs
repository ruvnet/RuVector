//! Coherence-aware scheduler implementation.
//!
//! The scheduler combines deadline pressure, novelty signal, and structural
//! risk to determine task priority, as specified in ADR-087 Section 5.
//!
//! # Design
//!
//! The scheduler maintains:
//! - Ready queues per partition
//! - A partition manager for round-robin partition scheduling
//! - Priority computation using the three scheduling signals
//!
//! # Scheduling Flow
//!
//! 1. Select next partition (round-robin with time slices)
//! 2. Within partition, select highest-priority ready task
//! 3. Priority = deadline_urgency + novelty_boost - risk_penalty
//! 4. Preemption occurs only at queue boundaries

use crate::{
    compute_priority, Instant, PartitionManager, PriorityConfig, SchedError, SchedResult,
    TaskControlBlock, TaskState, DEFAULT_TIME_QUANTUM_US,
};
use ruvix_types::{SchedulerScore, TaskHandle};

/// Scheduler configuration.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Priority computation configuration.
    pub priority_config: PriorityConfig,

    /// Default time quantum per task (in microseconds).
    pub default_quantum_us: u32,

    /// Default time slice per partition (in microseconds).
    pub default_partition_slice_us: u32,

    /// Novelty decay factor per scheduling tick.
    pub novelty_decay: f32,

    /// Whether to automatically create partitions for unknown partition IDs.
    pub auto_create_partitions: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            priority_config: PriorityConfig::default(),
            default_quantum_us: DEFAULT_TIME_QUANTUM_US,
            default_partition_slice_us: 10_000, // 10ms per partition
            novelty_decay: 0.95,
            auto_create_partitions: true,
        }
    }
}

/// Scheduler statistics.
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    /// Total number of scheduling decisions made.
    pub schedule_count: u64,

    /// Total number of tasks scheduled.
    pub tasks_scheduled: u64,

    /// Total number of deadline misses detected.
    pub deadline_misses: u64,

    /// Total number of preemptions.
    pub preemptions: u64,

    /// Total number of voluntary yields.
    pub voluntary_yields: u64,

    /// Total number of epoch resets.
    pub epoch_resets: u64,

    /// Number of times no task was available.
    pub idle_count: u64,
}

/// Coherence-aware scheduler.
///
/// # Type Parameters
///
/// * `N` - Maximum number of tasks per partition.
/// * `M` - Maximum number of partitions.
///
/// # Example
///
/// ```
/// use ruvix_sched::{Scheduler, SchedulerConfig, TaskControlBlock, Instant};
/// use ruvix_types::{TaskHandle, TaskPriority};
/// use ruvix_cap::CapRights;
///
/// let config = SchedulerConfig::default();
/// let mut scheduler: Scheduler<64, 8> = Scheduler::new(config);
///
/// // Add a task
/// let task = TaskControlBlock::new(
///     TaskHandle::new(1, 0),
///     CapRights::READ,
///     TaskPriority::Normal,
///     0,
/// );
/// scheduler.add_task(task).unwrap();
///
/// // Schedule
/// let now = Instant::from_micros(0);
/// if let Some(handle) = scheduler.select_next_task_at(now) {
///     println!("Selected task: {:?}", handle);
/// }
/// ```
#[derive(Debug)]
pub struct Scheduler<const N: usize, const M: usize> {
    /// Configuration.
    config: SchedulerConfig,

    /// Task storage (indexed by slot, not task ID).
    tasks: [Option<TaskControlBlock>; N],

    /// Number of tasks in the scheduler.
    task_count: usize,

    /// Partition manager.
    partitions: PartitionManager<M>,

    /// Currently running task handle.
    current_task: Option<TaskHandle>,

    /// Current time (updated on each scheduling decision).
    current_time: Instant,

    /// Statistics.
    stats: SchedulerStats,
}

impl<const N: usize, const M: usize> Scheduler<N, M> {
    /// Creates a new scheduler with the given configuration.
    #[must_use]
    pub fn new(config: SchedulerConfig) -> Self {
        const NONE: Option<TaskControlBlock> = None;
        Self {
            partitions: PartitionManager::new(config.default_partition_slice_us),
            config,
            tasks: [NONE; N],
            task_count: 0,
            current_task: None,
            current_time: Instant::default(),
            stats: SchedulerStats::default(),
        }
    }

    /// Creates a new scheduler with default configuration.
    #[inline]
    #[must_use]
    pub fn with_default_config() -> Self {
        Self::new(SchedulerConfig::default())
    }

    /// Adds a task to the scheduler.
    ///
    /// # Errors
    ///
    /// Returns `SchedError::QueueFull` if the scheduler is full.
    /// Returns `SchedError::TaskAlreadyExists` if the task ID exists.
    pub fn add_task(&mut self, mut task: TaskControlBlock) -> SchedResult<()> {
        // Check for duplicate
        if self.find_task_idx(task.task_id).is_some() {
            return Err(SchedError::TaskAlreadyExists);
        }

        // Ensure partition exists
        if self.partitions.get_partition(task.partition_id).is_none() {
            if self.config.auto_create_partitions {
                self.partitions.add_partition_default(task.partition_id)?;
            } else {
                return Err(SchedError::PartitionNotFound);
            }
        }

        // Find empty slot
        for slot in &mut self.tasks {
            if slot.is_none() {
                // Set initial quantum
                task.reset_quantum(self.config.default_quantum_us);

                // Update partition task count
                if task.state == TaskState::Ready {
                    if let Some(partition) = self.partitions.get_partition_mut(task.partition_id) {
                        partition.task_ready();
                    }
                }

                *slot = Some(task);
                self.task_count += 1;
                return Ok(());
            }
        }

        Err(SchedError::QueueFull)
    }

    /// Removes a task from the scheduler.
    ///
    /// # Errors
    ///
    /// Returns `SchedError::TaskNotFound` if the task doesn't exist.
    pub fn remove_task(&mut self, task_id: TaskHandle) -> SchedResult<TaskControlBlock> {
        let idx = self
            .find_task_idx(task_id)
            .ok_or(SchedError::TaskNotFound)?;

        let task = self.tasks[idx].take().unwrap();

        // Update partition
        if task.state == TaskState::Ready {
            if let Some(partition) = self.partitions.get_partition_mut(task.partition_id) {
                partition.task_unready();
            }
        } else if task.state == TaskState::Blocked {
            if let Some(partition) = self.partitions.get_partition_mut(task.partition_id) {
                partition.task_unblocked();
            }
        }

        self.task_count -= 1;

        // Clear current task if it was removed
        if self.current_task == Some(task_id) {
            self.current_task = None;
        }

        Ok(task)
    }

    /// Gets a task by ID.
    #[must_use]
    pub fn get_task(&self, task_id: TaskHandle) -> Option<&TaskControlBlock> {
        self.find_task_idx(task_id)
            .and_then(|idx| self.tasks[idx].as_ref())
    }

    /// Gets a mutable task by ID.
    #[must_use]
    pub fn get_task_mut(&mut self, task_id: TaskHandle) -> Option<&mut TaskControlBlock> {
        self.find_task_idx(task_id)
            .and_then(|idx| self.tasks[idx].as_mut())
    }

    /// Selects the next task to run.
    ///
    /// This is the main scheduling entry point. It:
    /// 1. Selects the next partition (round-robin)
    /// 2. Computes priorities for all ready tasks in that partition
    /// 3. Selects the highest-priority task
    ///
    /// # Returns
    ///
    /// The `TaskHandle` of the selected task, or `None` if no task is runnable.
    pub fn select_next_task(&mut self) -> Option<TaskHandle> {
        self.select_next_task_at(self.current_time)
    }

    /// Selects the next task to run at a specific time.
    pub fn select_next_task_at(&mut self, now: Instant) -> Option<TaskHandle> {
        self.current_time = now;
        self.stats.schedule_count += 1;

        // Decay novelty for all tasks
        self.decay_all_novelty();

        // Select partition
        let partition_id = self.partitions.select_next_partition(now)?;

        // Find highest priority task in this partition
        let mut best_task: Option<(TaskHandle, SchedulerScore)> = None;

        for slot in &mut self.tasks {
            if let Some(ref mut task) = slot {
                if task.partition_id != partition_id || task.state != TaskState::Ready {
                    continue;
                }

                // Check for deadline miss
                if task.deadline_passed(now) {
                    task.mark_deadline_miss();
                    self.stats.deadline_misses += 1;
                }

                // Compute priority
                let score = compute_priority(task, now, &self.config.priority_config);
                task.set_cached_score(score);

                if best_task
                    .as_ref()
                    .map_or(true, |(_, best_score)| score.is_higher_than(best_score))
                {
                    best_task = Some((task.task_id, score));
                }
            }
        }

        if let Some((task_id, _)) = best_task {
            // Transition task to Running
            if let Some(task) = self.get_task_mut(task_id) {
                task.transition_to(TaskState::Running);
                task.mark_scheduled();
            }

            // Update partition
            if let Some(partition) = self.partitions.get_partition_mut(partition_id) {
                partition.task_unready();
            }

            self.current_task = Some(task_id);
            self.stats.tasks_scheduled += 1;

            Some(task_id)
        } else {
            self.stats.idle_count += 1;
            None
        }
    }

    /// Yields the current task (voluntary yield).
    ///
    /// The task is moved back to the Ready state.
    pub fn yield_task(&mut self) -> SchedResult<()> {
        let task_id = self.current_task.ok_or(SchedError::TaskNotFound)?;
        let default_quantum = self.config.default_quantum_us;

        let partition_id = {
            let task = self
                .get_task_mut(task_id)
                .ok_or(SchedError::TaskNotFound)?;
            if !task.transition_to(TaskState::Ready) {
                return Err(SchedError::InvalidStateTransition);
            }
            // Reset quantum
            task.reset_quantum(default_quantum);
            task.partition_id
        };

        // Update partition (now that task borrow is dropped)
        if let Some(partition) = self.partitions.get_partition_mut(partition_id) {
            partition.task_ready();
        }

        self.current_task = None;
        self.stats.voluntary_yields += 1;

        Ok(())
    }

    /// Blocks the current task.
    ///
    /// The task will not be scheduled until `unblock_task` is called.
    pub fn block_task(&mut self) -> SchedResult<()> {
        let task_id = self.current_task.ok_or(SchedError::TaskNotFound)?;

        let partition_id = {
            let task = self
                .get_task_mut(task_id)
                .ok_or(SchedError::TaskNotFound)?;
            if !task.transition_to(TaskState::Blocked) {
                return Err(SchedError::InvalidStateTransition);
            }
            task.partition_id
        };

        // Update partition (now that task borrow is dropped)
        if let Some(partition) = self.partitions.get_partition_mut(partition_id) {
            partition.task_blocked();
        }

        self.current_task = None;

        Ok(())
    }

    /// Unblocks a task.
    ///
    /// The task is moved to the Ready state and can be scheduled.
    pub fn unblock_task(&mut self, task_id: TaskHandle) -> SchedResult<()> {
        let default_quantum = self.config.default_quantum_us;

        let partition_id = {
            let task = self
                .get_task_mut(task_id)
                .ok_or(SchedError::TaskNotFound)?;

            if task.state != TaskState::Blocked {
                return Err(SchedError::InvalidStateTransition);
            }

            if !task.transition_to(TaskState::Ready) {
                return Err(SchedError::InvalidStateTransition);
            }

            // Reset quantum
            task.reset_quantum(default_quantum);
            task.partition_id
        };

        // Update partition (now that task borrow is dropped)
        if let Some(partition) = self.partitions.get_partition_mut(partition_id) {
            partition.task_unblocked();
            partition.task_ready();
        }

        Ok(())
    }

    /// Preempts the current task.
    ///
    /// Called when a higher-priority task becomes runnable or when the
    /// task's quantum expires.
    pub fn preempt_current(&mut self) -> SchedResult<()> {
        let task_id = self.current_task.ok_or(SchedError::TaskNotFound)?;

        let partition_id = {
            let task = self
                .get_task_mut(task_id)
                .ok_or(SchedError::TaskNotFound)?;
            if !task.transition_to(TaskState::Ready) {
                return Err(SchedError::InvalidStateTransition);
            }
            task.partition_id
        };

        // Update partition (now that task borrow is dropped)
        if let Some(partition) = self.partitions.get_partition_mut(partition_id) {
            partition.task_ready();
        }

        self.current_task = None;
        self.stats.preemptions += 1;

        Ok(())
    }

    /// Notifies that time has elapsed for the current task.
    ///
    /// # Returns
    ///
    /// `true` if the task still has quantum remaining, `false` if it should
    /// be preempted.
    pub fn tick(&mut self, elapsed_us: u32) -> bool {
        if let Some(task_id) = self.current_task {
            if let Some(task) = self.get_task_mut(task_id) {
                let has_quantum = task.consume_time(elapsed_us);

                // Also consume partition time
                let partition_id = task.partition_id;
                self.partitions.consume_time(partition_id, elapsed_us);

                return has_quantum;
            }
        }
        false
    }

    /// Updates the novelty value for a task.
    pub fn update_task_novelty(&mut self, task_id: TaskHandle, novelty: f32) -> SchedResult<()> {
        let task = self
            .get_task_mut(task_id)
            .ok_or(SchedError::TaskNotFound)?;
        task.set_novelty(novelty);
        Ok(())
    }

    /// Updates the coherence delta for a task.
    pub fn update_task_coherence(
        &mut self,
        task_id: TaskHandle,
        coherence_delta: f32,
    ) -> SchedResult<()> {
        let task = self
            .get_task_mut(task_id)
            .ok_or(SchedError::TaskNotFound)?;
        task.set_coherence_delta(coherence_delta);
        Ok(())
    }

    /// Sets the deadline for a task.
    pub fn set_task_deadline(
        &mut self,
        task_id: TaskHandle,
        deadline: Option<Instant>,
    ) -> SchedResult<()> {
        let task = self
            .get_task_mut(task_id)
            .ok_or(SchedError::TaskNotFound)?;
        task.set_deadline(deadline);
        Ok(())
    }

    /// Returns the currently running task.
    #[inline]
    #[must_use]
    pub const fn current_task(&self) -> Option<TaskHandle> {
        self.current_task
    }

    /// Returns the number of tasks in the scheduler.
    #[inline]
    #[must_use]
    pub const fn task_count(&self) -> usize {
        self.task_count
    }

    /// Returns the scheduler statistics.
    #[inline]
    #[must_use]
    pub const fn stats(&self) -> &SchedulerStats {
        &self.stats
    }

    /// Returns an iterator over all tasks.
    pub fn iter_tasks(&self) -> impl Iterator<Item = &TaskControlBlock> {
        self.tasks.iter().filter_map(|t| t.as_ref())
    }

    /// Returns the partition manager.
    #[inline]
    #[must_use]
    pub const fn partitions(&self) -> &PartitionManager<M> {
        &self.partitions
    }

    /// Returns a mutable reference to the partition manager.
    #[inline]
    #[must_use]
    pub fn partitions_mut(&mut self) -> &mut PartitionManager<M> {
        &mut self.partitions
    }

    /// Adds a partition explicitly.
    pub fn add_partition(&mut self, partition_id: u32, time_slice_us: u32) -> SchedResult<()> {
        self.partitions.add_partition(partition_id, time_slice_us)
    }

    /// Finds the index of a task by ID.
    fn find_task_idx(&self, task_id: TaskHandle) -> Option<usize> {
        self.tasks
            .iter()
            .position(|slot| slot.as_ref().map_or(false, |t| t.task_id == task_id))
    }

    /// Decays novelty for all tasks.
    fn decay_all_novelty(&mut self) {
        for slot in &mut self.tasks {
            if let Some(ref mut task) = slot {
                task.decay_novelty(self.config.novelty_decay);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvix_cap::CapRights;
    use ruvix_types::TaskPriority;

    fn make_task(id: u32, partition: u32) -> TaskControlBlock {
        TaskControlBlock::new(
            TaskHandle::new(id, 0),
            CapRights::READ,
            TaskPriority::Normal,
            partition,
        )
    }

    #[test]
    fn test_scheduler_creation() {
        let scheduler: Scheduler<64, 8> = Scheduler::with_default_config();
        assert_eq!(scheduler.task_count(), 0);
        assert!(scheduler.current_task().is_none());
    }

    #[test]
    fn test_add_remove_task() {
        let mut scheduler: Scheduler<64, 8> = Scheduler::with_default_config();

        let task = make_task(1, 0);
        scheduler.add_task(task).unwrap();
        assert_eq!(scheduler.task_count(), 1);

        // Duplicate should fail
        let task2 = make_task(1, 0);
        assert!(matches!(
            scheduler.add_task(task2),
            Err(SchedError::TaskAlreadyExists)
        ));

        // Remove task
        let removed = scheduler.remove_task(TaskHandle::new(1, 0)).unwrap();
        assert_eq!(removed.task_id, TaskHandle::new(1, 0));
        assert_eq!(scheduler.task_count(), 0);
    }

    #[test]
    fn test_select_next_task() {
        let mut scheduler: Scheduler<64, 8> = Scheduler::with_default_config();

        scheduler.add_task(make_task(1, 0)).unwrap();
        scheduler.add_task(make_task(2, 0)).unwrap();

        let now = Instant::from_micros(0);
        let selected = scheduler.select_next_task_at(now);

        assert!(selected.is_some());
        assert!(scheduler.current_task().is_some());
    }

    #[test]
    fn test_deadline_priority() {
        let mut scheduler: Scheduler<64, 8> = Scheduler::with_default_config();
        let now = Instant::from_micros(0);

        // Task with tight deadline
        let tight = make_task(1, 0).with_deadline(Instant::from_micros(1000));
        scheduler.add_task(tight).unwrap();

        // Task with loose deadline
        let loose = make_task(2, 0).with_deadline(Instant::from_micros(1_000_000));
        scheduler.add_task(loose).unwrap();

        // Should select task with tighter deadline
        let selected = scheduler.select_next_task_at(now);
        assert_eq!(selected, Some(TaskHandle::new(1, 0)));
    }

    #[test]
    fn test_novelty_priority() {
        let mut scheduler: Scheduler<64, 8> = Scheduler::with_default_config();
        let now = Instant::from_micros(0);

        // Low novelty task
        let low = make_task(1, 0).with_novelty(0.1);
        scheduler.add_task(low).unwrap();

        // High novelty task
        let high = make_task(2, 0).with_novelty(0.9);
        scheduler.add_task(high).unwrap();

        // Should select high novelty task (both have same priority otherwise)
        let selected = scheduler.select_next_task_at(now);
        assert_eq!(selected, Some(TaskHandle::new(2, 0)));
    }

    #[test]
    fn test_risk_penalty() {
        let mut scheduler: Scheduler<64, 8> = Scheduler::with_default_config();
        let now = Instant::from_micros(0);

        // Risky task (negative coherence delta)
        let risky = make_task(1, 0).with_coherence_delta(-0.5);
        scheduler.add_task(risky).unwrap();

        // Safe task
        let safe = make_task(2, 0).with_coherence_delta(0.1);
        scheduler.add_task(safe).unwrap();

        // Should select safe task due to risk penalty
        let selected = scheduler.select_next_task_at(now);
        assert_eq!(selected, Some(TaskHandle::new(2, 0)));
    }

    #[test]
    fn test_yield_task() {
        let mut scheduler: Scheduler<64, 8> = Scheduler::with_default_config();
        let now = Instant::from_micros(0);

        scheduler.add_task(make_task(1, 0)).unwrap();
        scheduler.select_next_task_at(now);

        assert!(scheduler.current_task().is_some());

        scheduler.yield_task().unwrap();
        assert!(scheduler.current_task().is_none());

        // Task should be ready again
        let task = scheduler.get_task(TaskHandle::new(1, 0)).unwrap();
        assert_eq!(task.state, TaskState::Ready);
    }

    #[test]
    fn test_block_unblock_task() {
        let mut scheduler: Scheduler<64, 8> = Scheduler::with_default_config();
        let now = Instant::from_micros(0);

        scheduler.add_task(make_task(1, 0)).unwrap();
        scheduler.select_next_task_at(now);

        // Block task
        scheduler.block_task().unwrap();
        assert!(scheduler.current_task().is_none());

        let task = scheduler.get_task(TaskHandle::new(1, 0)).unwrap();
        assert_eq!(task.state, TaskState::Blocked);

        // Unblock task
        scheduler.unblock_task(TaskHandle::new(1, 0)).unwrap();

        let task = scheduler.get_task(TaskHandle::new(1, 0)).unwrap();
        assert_eq!(task.state, TaskState::Ready);
    }

    #[test]
    fn test_partition_scheduling() {
        let mut scheduler: Scheduler<64, 8> = Scheduler::with_default_config();
        let now = Instant::from_micros(0);

        // Tasks in different partitions
        scheduler.add_task(make_task(1, 0)).unwrap();
        scheduler.add_task(make_task(2, 1)).unwrap();

        // Select from partition 0
        let first = scheduler.select_next_task_at(now);
        assert!(first.is_some());
        scheduler.yield_task().unwrap();

        // Exhaust partition 0
        scheduler.partitions_mut().consume_time(0, 10000);

        // Select from partition 1
        let second = scheduler.select_next_task_at(now);
        assert_eq!(second, Some(TaskHandle::new(2, 0)));
    }

    #[test]
    fn test_tick() {
        let mut scheduler: Scheduler<64, 8> = Scheduler::with_default_config();
        let now = Instant::from_micros(0);

        let task = make_task(1, 0).with_time_quantum(1000);
        scheduler.add_task(task).unwrap();
        scheduler.select_next_task_at(now);

        // Consume some time
        assert!(scheduler.tick(500));

        // Consume remaining time
        assert!(!scheduler.tick(600));
    }

    #[test]
    fn test_deadline_miss_tracking() {
        let mut scheduler: Scheduler<64, 8> = Scheduler::with_default_config();

        // Task with deadline in the past
        let late = make_task(1, 0).with_deadline(Instant::from_micros(500));
        scheduler.add_task(late).unwrap();

        let now = Instant::from_micros(1000);
        scheduler.select_next_task_at(now);

        assert_eq!(scheduler.stats().deadline_misses, 1);
    }

    #[test]
    fn test_stats() {
        let mut scheduler: Scheduler<64, 8> = Scheduler::with_default_config();
        let now = Instant::from_micros(0);

        scheduler.add_task(make_task(1, 0)).unwrap();

        // Initial stats
        assert_eq!(scheduler.stats().schedule_count, 0);

        scheduler.select_next_task_at(now);
        assert_eq!(scheduler.stats().schedule_count, 1);
        assert_eq!(scheduler.stats().tasks_scheduled, 1);

        scheduler.yield_task().unwrap();
        assert_eq!(scheduler.stats().voluntary_yields, 1);
    }
}
