//! Partition scheduling for capability-isolated task groups.
//!
//! Tasks are grouped by their RVF mount origin into partitions. Each partition
//! receives a guaranteed time slice, preventing a misbehaving component from
//! starving others.
//!
//! # Scheduling Guarantees
//!
//! - Each partition gets its configured time slice per epoch
//! - Within a partition, tasks are scheduled by EDF (earliest deadline first)
//! - Partitions are scheduled round-robin when their time slice is available
//! - When a partition exhausts its time slice, the scheduler moves to the next

use crate::{Instant, SchedError, SchedResult};
use ruvix_types::SchedulerPartition;

/// State of a partition within the scheduler.
#[derive(Debug, Clone)]
pub struct PartitionState {
    /// The partition configuration.
    pub partition: SchedulerPartition,

    /// Number of ready tasks in this partition.
    pub ready_task_count: u32,

    /// Number of blocked tasks in this partition.
    pub blocked_task_count: u32,

    /// Total CPU time consumed by this partition.
    pub total_cpu_time_us: u64,

    /// Number of scheduling epochs this partition has participated in.
    pub epoch_count: u64,

    /// Number of times this partition was skipped (no ready tasks).
    pub skip_count: u64,

    /// Last time this partition was scheduled.
    pub last_scheduled: Option<Instant>,

    /// Whether this partition is enabled.
    pub enabled: bool,
}

impl PartitionState {
    /// Creates a new partition state from a partition configuration.
    #[inline]
    #[must_use]
    pub const fn new(partition: SchedulerPartition) -> Self {
        Self {
            partition,
            ready_task_count: 0,
            blocked_task_count: 0,
            total_cpu_time_us: 0,
            epoch_count: 0,
            skip_count: 0,
            last_scheduled: None,
            enabled: true,
        }
    }

    /// Returns true if the partition has ready tasks.
    #[inline]
    #[must_use]
    pub const fn has_ready_tasks(&self) -> bool {
        self.ready_task_count > 0
    }

    /// Returns true if the partition has exhausted its time slice.
    #[inline]
    #[must_use]
    pub const fn is_exhausted(&self) -> bool {
        self.partition.is_exhausted()
    }

    /// Resets the partition for a new scheduling epoch.
    #[inline]
    pub fn reset_epoch(&mut self) {
        self.partition.reset();
        self.epoch_count = self.epoch_count.saturating_add(1);
    }

    /// Consumes time from the partition's time slice.
    ///
    /// Returns `true` if the partition still has time remaining.
    #[inline]
    pub fn consume_time(&mut self, elapsed_us: u32) -> bool {
        self.partition.remaining_us = self.partition.remaining_us.saturating_sub(elapsed_us);
        self.total_cpu_time_us = self.total_cpu_time_us.saturating_add(u64::from(elapsed_us));
        self.partition.remaining_us > 0
    }

    /// Records that a task became ready in this partition.
    #[inline]
    pub fn task_ready(&mut self) {
        self.ready_task_count = self.ready_task_count.saturating_add(1);
    }

    /// Records that a task became unready (blocked, suspended, terminated).
    #[inline]
    pub fn task_unready(&mut self) {
        self.ready_task_count = self.ready_task_count.saturating_sub(1);
    }

    /// Records that a task became blocked.
    #[inline]
    pub fn task_blocked(&mut self) {
        self.blocked_task_count = self.blocked_task_count.saturating_add(1);
    }

    /// Records that a task became unblocked.
    #[inline]
    pub fn task_unblocked(&mut self) {
        self.blocked_task_count = self.blocked_task_count.saturating_sub(1);
    }

    /// Records that this partition was skipped (no ready tasks).
    #[inline]
    pub fn mark_skipped(&mut self) {
        self.skip_count = self.skip_count.saturating_add(1);
    }

    /// Records that this partition was scheduled.
    #[inline]
    pub fn mark_scheduled(&mut self, now: Instant) {
        self.last_scheduled = Some(now);
    }

    /// Returns the partition ID.
    #[inline]
    #[must_use]
    pub const fn partition_id(&self) -> u32 {
        self.partition.partition_id
    }
}

/// Partition manager for round-robin scheduling with guaranteed time slices.
///
/// # Type Parameters
///
/// * `M` - Maximum number of partitions (compile-time constant).
#[derive(Debug)]
pub struct PartitionManager<const M: usize> {
    /// Partition states.
    partitions: [Option<PartitionState>; M],

    /// Number of active partitions.
    partition_count: usize,

    /// Index of the currently scheduled partition.
    current_partition_idx: usize,

    /// Default time slice for new partitions (in microseconds).
    default_time_slice_us: u32,

    /// Whether we are in a new epoch (all partitions exhausted).
    new_epoch: bool,
}

impl<const M: usize> PartitionManager<M> {
    /// Creates a new partition manager.
    #[must_use]
    pub fn new(default_time_slice_us: u32) -> Self {
        const NONE: Option<PartitionState> = None;
        Self {
            partitions: [NONE; M],
            partition_count: 0,
            current_partition_idx: 0,
            default_time_slice_us,
            new_epoch: true,
        }
    }

    /// Adds a new partition.
    ///
    /// # Errors
    ///
    /// Returns `SchedError::MaxPartitionsReached` if the manager is full.
    /// Returns `SchedError::PartitionAlreadyExists` if the partition ID exists.
    pub fn add_partition(&mut self, partition_id: u32, time_slice_us: u32) -> SchedResult<()> {
        // Check if partition already exists
        if self.find_partition_idx(partition_id).is_some() {
            return Err(SchedError::PartitionAlreadyExists);
        }

        // Find empty slot
        for slot in &mut self.partitions {
            if slot.is_none() {
                let partition = SchedulerPartition::new(partition_id, time_slice_us);
                *slot = Some(PartitionState::new(partition));
                self.partition_count += 1;
                return Ok(());
            }
        }

        Err(SchedError::MaxPartitionsReached)
    }

    /// Adds a partition with the default time slice.
    ///
    /// # Errors
    ///
    /// Same as `add_partition`.
    pub fn add_partition_default(&mut self, partition_id: u32) -> SchedResult<()> {
        self.add_partition(partition_id, self.default_time_slice_us)
    }

    /// Removes a partition.
    ///
    /// # Errors
    ///
    /// Returns `SchedError::PartitionNotFound` if the partition doesn't exist.
    pub fn remove_partition(&mut self, partition_id: u32) -> SchedResult<()> {
        let idx = self
            .find_partition_idx(partition_id)
            .ok_or(SchedError::PartitionNotFound)?;

        self.partitions[idx] = None;
        self.partition_count = self.partition_count.saturating_sub(1);

        // Adjust current index if needed
        if self.current_partition_idx >= self.partition_count && self.partition_count > 0 {
            self.current_partition_idx = 0;
        }

        Ok(())
    }

    /// Gets a partition by ID.
    #[must_use]
    pub fn get_partition(&self, partition_id: u32) -> Option<&PartitionState> {
        self.find_partition_idx(partition_id)
            .and_then(|idx| self.partitions[idx].as_ref())
    }

    /// Gets a mutable partition by ID.
    #[must_use]
    pub fn get_partition_mut(&mut self, partition_id: u32) -> Option<&mut PartitionState> {
        self.find_partition_idx(partition_id)
            .and_then(|idx| self.partitions[idx].as_mut())
    }

    /// Selects the next partition to schedule.
    ///
    /// This implements round-robin selection among partitions with ready tasks
    /// and remaining time slices.
    ///
    /// # Returns
    ///
    /// The partition ID of the selected partition, or `None` if no partition
    /// is schedulable.
    pub fn select_next_partition(&mut self, now: Instant) -> Option<u32> {
        if self.partition_count == 0 {
            return None;
        }

        // Check if we need a new epoch (all partitions exhausted)
        if self.all_exhausted() {
            self.start_new_epoch();
        }

        // Find next partition with ready tasks and remaining time
        let start_idx = self.current_partition_idx;
        let mut attempts = 0;

        loop {
            if let Some(ref mut state) = self.partitions[self.current_partition_idx] {
                if state.enabled && state.has_ready_tasks() && !state.is_exhausted() {
                    state.mark_scheduled(now);
                    return Some(state.partition_id());
                }

                if state.enabled && !state.has_ready_tasks() {
                    state.mark_skipped();
                }
            }

            // Move to next partition
            self.current_partition_idx = (self.current_partition_idx + 1) % M;
            attempts += 1;

            // Wrapped around to start - no schedulable partition found
            if attempts >= M || self.current_partition_idx == start_idx {
                break;
            }
        }

        None
    }

    /// Notifies that time has been consumed from the current partition.
    ///
    /// Returns `true` if the partition still has time remaining.
    pub fn consume_time(&mut self, partition_id: u32, elapsed_us: u32) -> bool {
        if let Some(ref mut state) = self.get_partition_mut(partition_id) {
            state.consume_time(elapsed_us)
        } else {
            false
        }
    }

    /// Starts a new scheduling epoch (resets all partition time slices).
    pub fn start_new_epoch(&mut self) {
        for slot in &mut self.partitions {
            if let Some(ref mut state) = slot {
                state.reset_epoch();
            }
        }
        self.new_epoch = true;
    }

    /// Returns true if all partitions have exhausted their time slices.
    #[must_use]
    pub fn all_exhausted(&self) -> bool {
        for slot in &self.partitions {
            if let Some(ref state) = slot {
                if state.enabled && state.has_ready_tasks() && !state.is_exhausted() {
                    return false;
                }
            }
        }
        true
    }

    /// Returns the number of active partitions.
    #[inline]
    #[must_use]
    pub const fn partition_count(&self) -> usize {
        self.partition_count
    }

    /// Returns an iterator over active partitions.
    pub fn iter(&self) -> impl Iterator<Item = &PartitionState> {
        self.partitions.iter().filter_map(|p| p.as_ref())
    }

    /// Returns a mutable iterator over active partitions.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut PartitionState> {
        self.partitions.iter_mut().filter_map(|p| p.as_mut())
    }

    /// Finds the index of a partition by ID.
    fn find_partition_idx(&self, partition_id: u32) -> Option<usize> {
        self.partitions.iter().position(|slot| {
            slot.as_ref()
                .map_or(false, |s| s.partition_id() == partition_id)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_state_creation() {
        let partition = SchedulerPartition::new(1, 10000);
        let state = PartitionState::new(partition);

        assert_eq!(state.partition_id(), 1);
        assert!(!state.has_ready_tasks());
        assert!(!state.is_exhausted());
    }

    #[test]
    fn test_partition_time_consumption() {
        let partition = SchedulerPartition::new(1, 1000);
        let mut state = PartitionState::new(partition);

        assert!(state.consume_time(500));
        assert_eq!(state.partition.remaining_us, 500);
        assert_eq!(state.total_cpu_time_us, 500);

        assert!(!state.consume_time(600));
        assert!(state.is_exhausted());
    }

    #[test]
    fn test_partition_task_tracking() {
        let partition = SchedulerPartition::new(1, 1000);
        let mut state = PartitionState::new(partition);

        state.task_ready();
        state.task_ready();
        assert_eq!(state.ready_task_count, 2);
        assert!(state.has_ready_tasks());

        state.task_unready();
        assert_eq!(state.ready_task_count, 1);

        state.task_blocked();
        assert_eq!(state.blocked_task_count, 1);
    }

    #[test]
    fn test_partition_manager_creation() {
        let manager: PartitionManager<8> = PartitionManager::new(10000);
        assert_eq!(manager.partition_count(), 0);
    }

    #[test]
    fn test_partition_add_remove() {
        let mut manager: PartitionManager<8> = PartitionManager::new(10000);

        manager.add_partition(1, 5000).unwrap();
        manager.add_partition(2, 3000).unwrap();
        assert_eq!(manager.partition_count(), 2);

        // Duplicate should fail
        assert!(matches!(
            manager.add_partition(1, 5000),
            Err(SchedError::PartitionAlreadyExists)
        ));

        manager.remove_partition(1).unwrap();
        assert_eq!(manager.partition_count(), 1);

        // Removing non-existent should fail
        assert!(matches!(
            manager.remove_partition(1),
            Err(SchedError::PartitionNotFound)
        ));
    }

    #[test]
    fn test_partition_selection() {
        let mut manager: PartitionManager<8> = PartitionManager::new(10000);
        let now = Instant::from_micros(0);

        manager.add_partition(1, 5000).unwrap();
        manager.add_partition(2, 3000).unwrap();

        // No ready tasks - should return None
        assert!(manager.select_next_partition(now).is_none());

        // Add ready task to partition 1
        manager.get_partition_mut(1).unwrap().task_ready();

        // Should select partition 1
        let selected = manager.select_next_partition(now);
        assert_eq!(selected, Some(1));
    }

    #[test]
    fn test_round_robin_selection() {
        let mut manager: PartitionManager<8> = PartitionManager::new(10000);
        let now = Instant::from_micros(0);

        manager.add_partition(1, 5000).unwrap();
        manager.add_partition(2, 3000).unwrap();

        manager.get_partition_mut(1).unwrap().task_ready();
        manager.get_partition_mut(2).unwrap().task_ready();

        // First selection should be partition 1
        let first = manager.select_next_partition(now);
        assert_eq!(first, Some(1));

        // Exhaust partition 1
        manager.consume_time(1, 5000);

        // Next selection should be partition 2
        let second = manager.select_next_partition(now);
        assert_eq!(second, Some(2));
    }

    #[test]
    fn test_epoch_reset() {
        let mut manager: PartitionManager<8> = PartitionManager::new(1000);
        let now = Instant::from_micros(0);

        manager.add_partition(1, 1000).unwrap();
        manager.get_partition_mut(1).unwrap().task_ready();

        // Exhaust the partition
        manager.consume_time(1, 1000);
        assert!(manager.all_exhausted());

        // New epoch should reset
        manager.start_new_epoch();
        assert!(!manager.get_partition(1).unwrap().is_exhausted());
    }

    #[test]
    fn test_max_partitions() {
        let mut manager: PartitionManager<2> = PartitionManager::new(10000);

        manager.add_partition(1, 5000).unwrap();
        manager.add_partition(2, 3000).unwrap();

        assert!(matches!(
            manager.add_partition(3, 2000),
            Err(SchedError::MaxPartitionsReached)
        ));
    }
}
