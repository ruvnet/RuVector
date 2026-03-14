//! Task scheduler for the RuVix Cognition Kernel.
//!
//! The scheduler manages task execution with support for:
//!
//! - **Priority-based scheduling**: Higher priority tasks run first
//! - **Deadline scheduling**: Real-time tasks with strict deadlines
//! - **Fair sharing**: Background tasks get CPU time when system is idle

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
use alloc::collections::BTreeMap;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::{Duration, Result, TaskHandle, TaskPriority};
use ruvix_types::KernelError;

/// Maximum tasks (for no_std).
#[cfg(not(feature = "alloc"))]
const MAX_TASKS: usize = 256;

/// Configuration for the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerConfig {
    /// Time slice for normal priority tasks (nanoseconds).
    pub time_slice_ns: u64,
    /// Maximum tasks to run before checking deadlines.
    pub deadline_check_interval: u32,
    /// Enable preemption.
    pub preemption_enabled: bool,
}

impl SchedulerConfig {
    /// Creates a new configuration with default values.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            time_slice_ns: 10_000_000, // 10ms
            deadline_check_interval: 10,
            preemption_enabled: true,
        }
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// State of a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TaskState {
    /// Task is ready to run.
    Ready = 0,
    /// Task is currently running.
    Running = 1,
    /// Task is blocked waiting for an event.
    Blocked = 2,
    /// Task is blocked waiting for a timer.
    Sleeping = 3,
    /// Task has terminated.
    Terminated = 4,
}

impl TaskState {
    /// Returns the state as a string.
    #[inline]
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Running => "running",
            Self::Blocked => "blocked",
            Self::Sleeping => "sleeping",
            Self::Terminated => "terminated",
        }
    }
}

/// Task control block.
#[derive(Debug, Clone, Copy)]
pub struct TaskControlBlock {
    /// Task handle.
    pub handle: TaskHandle,
    /// Current state.
    pub state: TaskState,
    /// Priority level.
    pub priority: TaskPriority,
    /// Optional deadline (nanoseconds since boot).
    pub deadline_ns: Option<u64>,
    /// Time slice remaining (nanoseconds).
    pub time_remaining_ns: u64,
    /// Wake time for sleeping tasks (nanoseconds since boot).
    pub wake_time_ns: u64,
    /// Total CPU time consumed (nanoseconds).
    pub cpu_time_ns: u64,
    /// Number of times scheduled.
    pub schedule_count: u64,
}

impl TaskControlBlock {
    /// Creates a new task control block.
    #[inline]
    #[must_use]
    pub fn new(
        handle: TaskHandle,
        priority: TaskPriority,
        deadline: Option<Duration>,
        time_slice_ns: u64,
    ) -> Self {
        Self {
            handle,
            state: TaskState::Ready,
            priority,
            deadline_ns: deadline.map(|d| {
                #[cfg(feature = "std")]
                {
                    d.as_nanos() as u64
                }
                #[cfg(not(feature = "std"))]
                {
                    d.as_nanos()
                }
            }),
            time_remaining_ns: time_slice_ns,
            wake_time_ns: 0,
            cpu_time_ns: 0,
            schedule_count: 0,
        }
    }

    /// Returns true if the task has a deadline.
    #[inline]
    #[must_use]
    pub const fn has_deadline(&self) -> bool {
        self.deadline_ns.is_some()
    }

    /// Returns true if the deadline has passed.
    #[inline]
    #[must_use]
    pub fn deadline_missed(&self, current_time_ns: u64) -> bool {
        self.deadline_ns
            .map(|d| current_time_ns > d)
            .unwrap_or(false)
    }
}

/// The task scheduler.
pub struct Scheduler {
    /// Configuration.
    config: SchedulerConfig,

    /// Task table.
    #[cfg(feature = "alloc")]
    tasks: BTreeMap<u64, TaskControlBlock>,
    #[cfg(not(feature = "alloc"))]
    tasks: [Option<TaskControlBlock>; MAX_TASKS],
    #[cfg(not(feature = "alloc"))]
    task_count: usize,

    /// Currently running task.
    current_task: Option<TaskHandle>,

    /// Next task ID.
    next_task_id: u32,

    /// Current time (nanoseconds since boot).
    current_time_ns: u64,

    /// Statistics.
    stats: SchedulerStats,
}

/// Statistics about scheduler operations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SchedulerStats {
    /// Total tasks created.
    pub tasks_created: u64,
    /// Total tasks terminated.
    pub tasks_terminated: u64,
    /// Total context switches.
    pub context_switches: u64,
    /// Total deadlines missed.
    pub deadlines_missed: u64,
    /// Total timer wakes.
    pub timer_wakes: u64,
}

impl Scheduler {
    /// Creates a new scheduler.
    #[must_use]
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "alloc")]
            tasks: BTreeMap::new(),
            #[cfg(not(feature = "alloc"))]
            tasks: [None; MAX_TASKS],
            #[cfg(not(feature = "alloc"))]
            task_count: 0,
            current_task: None,
            next_task_id: 1,
            current_time_ns: 0,
            stats: SchedulerStats::default(),
        }
    }

    /// Creates a scheduler with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(SchedulerConfig::default())
    }

    /// Sets the current time.
    #[inline]
    pub fn set_current_time(&mut self, time_ns: u64) {
        self.current_time_ns = time_ns;
    }

    /// Returns the configuration.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    /// Returns the statistics.
    #[inline]
    #[must_use]
    pub const fn stats(&self) -> &SchedulerStats {
        &self.stats
    }

    /// Returns the currently running task.
    #[inline]
    #[must_use]
    pub const fn current_task(&self) -> Option<TaskHandle> {
        self.current_task
    }

    /// Returns the number of tasks.
    #[inline]
    #[must_use]
    pub fn task_count(&self) -> usize {
        #[cfg(feature = "alloc")]
        {
            self.tasks.len()
        }
        #[cfg(not(feature = "alloc"))]
        {
            self.task_count
        }
    }

    /// Returns an iterator over all task control blocks.
    #[cfg(feature = "alloc")]
    pub fn iter_tasks(&self) -> impl Iterator<Item = &TaskControlBlock> {
        self.tasks.values()
    }

    /// Returns an iterator over all task control blocks.
    #[cfg(not(feature = "alloc"))]
    pub fn iter_tasks(&self) -> impl Iterator<Item = &TaskControlBlock> {
        self.tasks[..self.task_count]
            .iter()
            .filter_map(|t| t.as_ref())
    }

    /// Creates a new task.
    pub fn create_task(
        &mut self,
        priority: TaskPriority,
        deadline: Option<Duration>,
    ) -> Result<TaskHandle> {
        let id = self.next_task_id;
        self.next_task_id = self.next_task_id.wrapping_add(1);

        let handle = TaskHandle::new(id, 0);
        let tcb = TaskControlBlock::new(handle, priority, deadline, self.config.time_slice_ns);

        #[cfg(feature = "alloc")]
        {
            self.tasks.insert(handle.raw().id as u64, tcb);
        }
        #[cfg(not(feature = "alloc"))]
        {
            if self.task_count >= MAX_TASKS {
                return Err(KernelError::LimitExceeded);
            }
            self.tasks[self.task_count] = Some(tcb);
            self.task_count += 1;
        }

        self.stats.tasks_created += 1;
        Ok(handle)
    }

    /// Terminates a task.
    pub fn terminate_task(&mut self, handle: TaskHandle) -> Result<()> {
        let tcb = self.get_task_mut(handle)?;
        tcb.state = TaskState::Terminated;
        self.stats.tasks_terminated += 1;

        // If this was the current task, clear it
        if self.current_task == Some(handle) {
            self.current_task = None;
        }

        Ok(())
    }

    /// Blocks a task.
    pub fn block_task(&mut self, handle: TaskHandle) -> Result<()> {
        let tcb = self.get_task_mut(handle)?;
        if tcb.state != TaskState::Running && tcb.state != TaskState::Ready {
            return Err(KernelError::NotPermitted);
        }
        tcb.state = TaskState::Blocked;

        if self.current_task == Some(handle) {
            self.current_task = None;
        }

        Ok(())
    }

    /// Unblocks a task.
    pub fn unblock_task(&mut self, handle: TaskHandle) -> Result<()> {
        let time_slice = self.config.time_slice_ns;
        let tcb = self.get_task_mut(handle)?;
        if tcb.state != TaskState::Blocked {
            return Err(KernelError::NotPermitted);
        }
        tcb.state = TaskState::Ready;
        tcb.time_remaining_ns = time_slice;
        Ok(())
    }

    /// Puts a task to sleep until the specified wake time.
    pub fn sleep_task(&mut self, handle: TaskHandle, wake_time_ns: u64) -> Result<()> {
        let tcb = self.get_task_mut(handle)?;
        if tcb.state != TaskState::Running && tcb.state != TaskState::Ready {
            return Err(KernelError::NotPermitted);
        }
        tcb.state = TaskState::Sleeping;
        tcb.wake_time_ns = wake_time_ns;

        if self.current_task == Some(handle) {
            self.current_task = None;
        }

        Ok(())
    }

    /// Wakes up sleeping tasks whose wake time has passed.
    pub fn wake_sleeping_tasks(&mut self) {
        #[cfg(feature = "alloc")]
        {
            for tcb in self.tasks.values_mut() {
                if tcb.state == TaskState::Sleeping && tcb.wake_time_ns <= self.current_time_ns {
                    tcb.state = TaskState::Ready;
                    tcb.time_remaining_ns = self.config.time_slice_ns;
                    self.stats.timer_wakes += 1;
                }
            }
        }
        #[cfg(not(feature = "alloc"))]
        {
            for i in 0..self.task_count {
                if let Some(ref mut tcb) = self.tasks[i] {
                    if tcb.state == TaskState::Sleeping && tcb.wake_time_ns <= self.current_time_ns
                    {
                        tcb.state = TaskState::Ready;
                        tcb.time_remaining_ns = self.config.time_slice_ns;
                        self.stats.timer_wakes += 1;
                    }
                }
            }
        }
    }

    /// Selects the next task to run.
    ///
    /// Selection priority:
    /// 1. Real-time tasks with approaching deadlines
    /// 2. High priority tasks
    /// 3. Normal priority tasks
    /// 4. Background tasks
    pub fn schedule(&mut self) -> Option<TaskHandle> {
        // Wake any sleeping tasks first
        self.wake_sleeping_tasks();

        // Find best candidate
        let mut best: Option<(TaskHandle, TaskPriority, Option<u64>)> = None;

        #[cfg(feature = "alloc")]
        let iter = self.tasks.values();
        #[cfg(not(feature = "alloc"))]
        let iter = self.tasks[..self.task_count]
            .iter()
            .filter_map(|t| t.as_ref());

        for tcb in iter {
            if tcb.state != TaskState::Ready {
                continue;
            }

            // Check deadline miss
            if tcb.deadline_missed(self.current_time_ns) {
                self.stats.deadlines_missed += 1;
                continue;
            }

            // Compare with current best
            let should_select = match &best {
                None => true,
                Some((_, best_priority, best_deadline)) => {
                    // Prefer tasks with deadlines
                    match (tcb.deadline_ns, *best_deadline) {
                        (Some(d), Some(bd)) => d < bd,
                        (Some(_), None) => true,
                        (None, Some(_)) => false,
                        (None, None) => tcb.priority > *best_priority,
                    }
                }
            };

            if should_select {
                best = Some((tcb.handle, tcb.priority, tcb.deadline_ns));
            }
        }

        if let Some((handle, _, _)) = best {
            // Context switch
            if let Some(prev) = self.current_task {
                if prev != handle {
                    // Mark previous task as ready
                    if let Ok(prev_tcb) = self.get_task_mut(prev) {
                        if prev_tcb.state == TaskState::Running {
                            prev_tcb.state = TaskState::Ready;
                        }
                    }
                    self.stats.context_switches += 1;
                }
            }

            // Mark new task as running
            if let Ok(tcb) = self.get_task_mut(handle) {
                tcb.state = TaskState::Running;
                tcb.schedule_count += 1;
            }

            self.current_task = Some(handle);
            Some(handle)
        } else {
            self.current_task = None;
            None
        }
    }

    /// Gets a task by handle.
    pub fn get_task(&self, handle: TaskHandle) -> Result<&TaskControlBlock> {
        #[cfg(feature = "alloc")]
        {
            self.tasks
                .get(&(handle.raw().id as u64))
                .ok_or(KernelError::NotFound)
        }
        #[cfg(not(feature = "alloc"))]
        {
            for i in 0..self.task_count {
                if let Some(ref tcb) = self.tasks[i] {
                    if tcb.handle == handle {
                        return Ok(tcb);
                    }
                }
            }
            Err(KernelError::NotFound)
        }
    }

    /// Gets a mutable reference to a task by handle.
    fn get_task_mut(&mut self, handle: TaskHandle) -> Result<&mut TaskControlBlock> {
        #[cfg(feature = "alloc")]
        {
            self.tasks
                .get_mut(&(handle.raw().id as u64))
                .ok_or(KernelError::NotFound)
        }
        #[cfg(not(feature = "alloc"))]
        {
            // Find index first, then return mutable reference
            let mut found_idx: Option<usize> = None;
            for i in 0..self.task_count {
                if let Some(ref tcb) = self.tasks[i] {
                    if tcb.handle == handle {
                        found_idx = Some(i);
                        break;
                    }
                }
            }
            match found_idx {
                Some(idx) => Ok(self.tasks[idx].as_mut().unwrap()),
                None => Err(KernelError::NotFound),
            }
        }
    }

    /// Consumes time for the current task.
    pub fn consume_time(&mut self, elapsed_ns: u64) {
        if let Some(handle) = self.current_task {
            if let Ok(tcb) = self.get_task_mut(handle) {
                tcb.cpu_time_ns += elapsed_ns;
                tcb.time_remaining_ns = tcb.time_remaining_ns.saturating_sub(elapsed_ns);
            }
        }
    }

    /// Checks if the current task needs preemption.
    pub fn needs_preemption(&self) -> bool {
        if !self.config.preemption_enabled {
            return false;
        }

        if let Some(handle) = self.current_task {
            if let Ok(tcb) = self.get_task(handle) {
                // Preempt if time slice exhausted
                if tcb.time_remaining_ns == 0 {
                    return true;
                }
            }
        }

        false
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_creation() {
        let scheduler = Scheduler::with_defaults();
        assert_eq!(scheduler.task_count(), 0);
        assert!(scheduler.current_task().is_none());
    }

    #[test]
    fn test_create_task() {
        let mut scheduler = Scheduler::with_defaults();

        let handle = scheduler.create_task(TaskPriority::Normal, None).unwrap();
        assert!(!handle.is_null());
        assert_eq!(scheduler.task_count(), 1);

        let tcb = scheduler.get_task(handle).unwrap();
        assert_eq!(tcb.state, TaskState::Ready);
        assert_eq!(tcb.priority, TaskPriority::Normal);
    }

    #[test]
    fn test_schedule_single_task() {
        let mut scheduler = Scheduler::with_defaults();

        let handle = scheduler.create_task(TaskPriority::Normal, None).unwrap();
        let scheduled = scheduler.schedule();

        assert_eq!(scheduled, Some(handle));
        assert_eq!(scheduler.current_task(), Some(handle));

        let tcb = scheduler.get_task(handle).unwrap();
        assert_eq!(tcb.state, TaskState::Running);
    }

    #[test]
    fn test_priority_scheduling() {
        let mut scheduler = Scheduler::with_defaults();

        let low = scheduler.create_task(TaskPriority::Background, None).unwrap();
        let high = scheduler.create_task(TaskPriority::High, None).unwrap();
        let normal = scheduler.create_task(TaskPriority::Normal, None).unwrap();

        // High priority should be scheduled first
        let scheduled = scheduler.schedule();
        assert_eq!(scheduled, Some(high));

        // Block high, normal should be next
        scheduler.block_task(high).unwrap();
        let scheduled = scheduler.schedule();
        assert_eq!(scheduled, Some(normal));

        // Block normal, low should be next
        scheduler.block_task(normal).unwrap();
        let scheduled = scheduler.schedule();
        assert_eq!(scheduled, Some(low));
    }

    #[test]
    fn test_deadline_scheduling() {
        let mut scheduler = Scheduler::with_defaults();
        scheduler.set_current_time(0);

        // Task with near deadline
        let urgent = scheduler
            .create_task(TaskPriority::Normal, Some(Duration::from_millis(10)))
            .unwrap();

        // Task with far deadline
        let _later = scheduler
            .create_task(TaskPriority::High, Some(Duration::from_millis(100)))
            .unwrap();

        // Urgent should be scheduled despite lower priority
        let scheduled = scheduler.schedule();
        assert_eq!(scheduled, Some(urgent));
    }

    #[test]
    fn test_task_sleep_wake() {
        let mut scheduler = Scheduler::with_defaults();
        scheduler.set_current_time(0);

        let handle = scheduler.create_task(TaskPriority::Normal, None).unwrap();
        scheduler.schedule();

        // Sleep until time 1000
        scheduler.sleep_task(handle, 1000).unwrap();

        let tcb = scheduler.get_task(handle).unwrap();
        assert_eq!(tcb.state, TaskState::Sleeping);

        // Should not schedule sleeping task
        assert!(scheduler.schedule().is_none());

        // Advance time past wake time
        scheduler.set_current_time(1001);
        scheduler.wake_sleeping_tasks();

        let tcb = scheduler.get_task(handle).unwrap();
        assert_eq!(tcb.state, TaskState::Ready);

        // Should schedule now
        assert_eq!(scheduler.schedule(), Some(handle));
    }

    #[test]
    fn test_task_termination() {
        let mut scheduler = Scheduler::with_defaults();

        let handle = scheduler.create_task(TaskPriority::Normal, None).unwrap();
        scheduler.schedule();

        scheduler.terminate_task(handle).unwrap();

        let tcb = scheduler.get_task(handle).unwrap();
        assert_eq!(tcb.state, TaskState::Terminated);

        // Terminated task should not be scheduled
        assert!(scheduler.schedule().is_none());
    }

    #[test]
    fn test_context_switch_counting() {
        let mut scheduler = Scheduler::with_defaults();

        // Use different priorities to ensure deterministic scheduling order
        let task1 = scheduler.create_task(TaskPriority::Normal, None).unwrap();
        let _task2 = scheduler.create_task(TaskPriority::Normal, None).unwrap();

        // First schedule - one of the tasks is selected, no switch (no previous task)
        scheduler.schedule();
        assert_eq!(scheduler.stats().context_switches, 0);

        // Context switch is counted when:
        // 1. current_task is Some (there's a running task)
        // 2. A DIFFERENT task is selected
        //
        // Create a higher priority task - it should preempt the current one
        let task3 = scheduler.create_task(TaskPriority::High, None).unwrap();

        // schedule() should preempt current task for task3 (higher priority)
        scheduler.schedule();
        assert_eq!(scheduler.stats().context_switches, 1);

        // Create an even higher priority task to cause another switch
        let _task4 = scheduler.create_task(TaskPriority::RealTime, None).unwrap();

        // schedule() should preempt task3 for task4 (real-time priority)
        scheduler.schedule();
        assert_eq!(scheduler.stats().context_switches, 2);

        // Verify the correct task is current
        assert!(scheduler.current_task().is_some());

        // Blocking task1 (which isn't running) shouldn't affect current_task
        scheduler.block_task(task1).unwrap();
        assert!(scheduler.current_task().is_some());

        // Blocking task3 (which isn't running) shouldn't affect current_task
        scheduler.block_task(task3).unwrap();
        assert!(scheduler.current_task().is_some());
    }
}
