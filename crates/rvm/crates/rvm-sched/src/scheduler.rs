//! Main scheduler: ties together per-CPU schedulers, epoch management,
//! mode selection, and degraded mode handling.

use crate::epoch::{EpochSummary, EpochTracker};
use crate::modes::SchedulerMode;
use crate::per_cpu::PerCpuScheduler;
use crate::priority::compute_priority;
use rvm_types::{CutPressure, PartitionId};

/// Maximum entries in a per-CPU run queue.
pub const MAX_RUN_QUEUE: usize = 32;

/// An entry in a per-CPU run queue.
///
/// Uses `PartitionId::HYPERVISOR` (id 0) as the empty sentinel instead
/// of `Option<RunQueueEntry>`, eliminating the discriminant byte and
/// associated padding. The hypervisor partition is never schedulable,
/// so `partition_id == PartitionId::HYPERVISOR` unambiguously means
/// the slot is empty.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct RunQueueEntry {
    /// Partition identifier.
    /// `PartitionId::HYPERVISOR` (0) means this slot is empty.
    pub partition_id: PartitionId,
    /// Deadline urgency (higher = more urgent).
    pub deadline_urgency: u16,
    /// Cut pressure from the coherence engine.
    pub cut_pressure: CutPressure,
    /// Computed priority (cached).
    pub priority: u32,
}

impl RunQueueEntry {
    /// The empty sentinel entry (hypervisor partition, zero priority).
    pub const EMPTY: Self = Self {
        partition_id: PartitionId::HYPERVISOR,
        deadline_urgency: 0,
        cut_pressure: CutPressure::ZERO,
        priority: 0,
    };

    /// Returns true if this entry is the empty sentinel.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.partition_id.is_hypervisor()
    }
}

/// The top-level scheduler for all CPUs.
///
/// # Type Parameters
///
/// * `MAX_CPUS` - Maximum number of physical CPUs.
/// * `MAX_PARTITIONS` - Maximum number of partitions in the system.
pub struct Scheduler<const MAX_CPUS: usize, const MAX_PARTITIONS: usize> {
    /// Per-CPU scheduler metadata.
    per_cpu: [PerCpuScheduler; MAX_CPUS],
    /// Per-CPU run queues (sentinel-based: `RunQueueEntry::EMPTY` = unused).
    run_queues: [[RunQueueEntry; MAX_RUN_QUEUE]; MAX_CPUS],
    /// Per-CPU run queue lengths.
    queue_lens: [usize; MAX_CPUS],
    /// Current scheduling mode.
    mode: SchedulerMode,
    /// Epoch tracker.
    epoch: EpochTracker,
    /// Whether the system is in degraded mode (DC-6).
    degraded: bool,
}

impl<const MAX_CPUS: usize, const MAX_PARTITIONS: usize> Scheduler<MAX_CPUS, MAX_PARTITIONS> {
    /// Empty run queue (all sentinel entries).
    const EMPTY_QUEUE: [RunQueueEntry; MAX_RUN_QUEUE] = [RunQueueEntry::EMPTY; MAX_RUN_QUEUE];

    /// Create a new scheduler in Flow mode.
    #[must_use]
    pub fn new() -> Self {
        Self {
            per_cpu: core::array::from_fn(|i| PerCpuScheduler::new(i as u16)),
            run_queues: [Self::EMPTY_QUEUE; MAX_CPUS],
            queue_lens: [0; MAX_CPUS],
            mode: SchedulerMode::Flow,
            epoch: EpochTracker::new(),
            degraded: false,
        }
    }

    /// Return the current scheduling mode.
    #[must_use]
    pub const fn mode(&self) -> SchedulerMode {
        self.mode
    }

    /// Switch to a new scheduling mode.
    pub fn set_mode(&mut self, mode: SchedulerMode) {
        self.mode = mode;
    }

    /// Return the current epoch number.
    #[must_use]
    pub const fn current_epoch(&self) -> u32 {
        self.epoch.current_epoch()
    }

    /// Return whether the system is in degraded mode.
    #[must_use]
    pub const fn is_degraded(&self) -> bool {
        self.degraded
    }

    /// Enter degraded mode (DC-6). In degraded mode, cut_pressure = 0.
    pub fn enter_degraded(&mut self) {
        self.degraded = true;
    }

    /// Exit degraded mode.
    pub fn exit_degraded(&mut self) {
        self.degraded = false;
    }

    /// Advance the scheduler epoch. Returns the completed epoch summary.
    #[inline]
    pub fn tick_epoch(&mut self) -> EpochSummary {
        let runnable: u32 = self.queue_lens.iter().map(|&l| l as u32).sum();
        // Clamp to u16::MAX to fit EpochSummary::runnable_count.
        let clamped = if runnable > u16::MAX as u32 { u16::MAX } else { runnable as u16 };
        self.epoch.advance(clamped)
    }

    /// Enqueue a partition on a specific CPU.
    ///
    /// Uses a binary max-heap for O(log n) insertion instead of O(n) sorted insert.
    /// In degraded mode (DC-6), `cut_pressure` is zeroed automatically.
    ///
    /// Returns `false` (rejecting the request) for an out-of-range CPU,
    /// a full queue, or `PartitionId::HYPERVISOR`: the hypervisor id is
    /// the empty-slot sentinel and is never schedulable — accepting it
    /// would permanently wedge a queue slot.
    #[inline]
    pub fn enqueue(
        &mut self,
        cpu: usize,
        partition_id: PartitionId,
        deadline_urgency: u16,
        cut_pressure: CutPressure,
    ) -> bool {
        // Reject the empty-slot sentinel: enqueuing the hypervisor id
        // would create an entry indistinguishable from an empty slot.
        if partition_id.is_hypervisor() {
            return false;
        }
        if cpu >= MAX_CPUS || self.queue_lens[cpu] >= MAX_RUN_QUEUE {
            return false;
        }

        let effective_pressure = if self.degraded {
            CutPressure::ZERO
        } else {
            cut_pressure
        };

        let priority = compute_priority(deadline_urgency, effective_pressure);
        let entry = RunQueueEntry {
            partition_id,
            deadline_urgency,
            cut_pressure: effective_pressure,
            priority,
        };

        // Binary max-heap push: O(log n).
        let len = self.queue_lens[cpu];
        let queue = &mut self.run_queues[cpu];
        queue[len] = entry;
        self.queue_lens[cpu] = len + 1;

        // Sift up: bubble the new entry toward the root.
        let mut pos = len;
        while pos > 0 {
            let parent = (pos - 1) / 2;
            if queue[pos].priority > queue[parent].priority {
                queue.swap(pos, parent);
                pos = parent;
            } else {
                break;
            }
        }

        true
    }

    /// Pick the next partition on a specific CPU and switch to it.
    ///
    /// Uses a binary max-heap for O(log n) pop instead of O(n) shift.
    /// Returns `(old_partition, new_partition)` if a switch occurred.
    #[inline]
    pub fn switch_next(&mut self, cpu: usize) -> Option<(Option<PartitionId>, PartitionId)> {
        if cpu >= MAX_CPUS || self.queue_lens[cpu] == 0 {
            return None;
        }

        let queue = &mut self.run_queues[cpu];
        let len = self.queue_lens[cpu];

        // Extract the max (root of the heap). The sentinel can never be
        // enqueued (see `enqueue`), so a sentinel root with len > 0 is
        // corrupted state; we still remove it below (defensive cleanup)
        // so the slot is not wedged forever.
        let entry = queue[0];
        debug_assert!(
            !entry.is_empty(),
            "sentinel entry in run queue with non-zero length (corrupted heap)"
        );

        // Move the last element to the root and clear the vacated slot.
        if len > 1 {
            queue[0] = queue[len - 1];
        }
        queue[len - 1] = RunQueueEntry::EMPTY;
        self.queue_lens[cpu] = len - 1;

        // Sift down: restore heap property.
        let new_len = len - 1;
        let mut pos = 0;
        loop {
            let left = 2 * pos + 1;
            let right = 2 * pos + 2;
            let mut largest = pos;

            if left < new_len && queue[left].priority > queue[largest].priority {
                largest = left;
            }
            if right < new_len && queue[right].priority > queue[largest].priority {
                largest = right;
            }

            if largest != pos {
                queue.swap(pos, largest);
                pos = largest;
            } else {
                break;
            }
        }

        if entry.is_empty() {
            // Defensive cleanup path (release builds): the corrupt
            // sentinel entry has been removed from the heap above, but
            // we do not "switch" to the hypervisor sentinel.
            return None;
        }

        let old = self.per_cpu[cpu].current;
        self.per_cpu[cpu].current = Some(entry.partition_id);
        self.per_cpu[cpu].idle = false;
        self.epoch.record_switch();

        Some((old, entry.partition_id))
    }

    /// Return a reference to the per-CPU scheduler for the given CPU.
    #[must_use]
    pub fn per_cpu(&self, cpu: usize) -> Option<&PerCpuScheduler> {
        self.per_cpu.get(cpu)
    }

    /// Return the run queue length for a specific CPU.
    #[must_use]
    pub fn queue_len(&self, cpu: usize) -> usize {
        self.queue_lens.get(cpu).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pid(id: u32) -> PartitionId {
        PartitionId::new(id)
    }

    #[test]
    fn test_scheduler_creation() {
        let sched: Scheduler<4, 256> = Scheduler::new();
        assert_eq!(sched.mode(), SchedulerMode::Flow);
        assert!(!sched.is_degraded());
        assert_eq!(sched.current_epoch(), 0);
    }

    #[test]
    fn test_mode_switch() {
        let mut sched: Scheduler<4, 256> = Scheduler::new();
        sched.set_mode(SchedulerMode::Reflex);
        assert_eq!(sched.mode(), SchedulerMode::Reflex);
        sched.set_mode(SchedulerMode::Recovery);
        assert_eq!(sched.mode(), SchedulerMode::Recovery);
    }

    #[test]
    fn test_enqueue_and_switch() {
        let mut sched: Scheduler<4, 256> = Scheduler::new();
        assert!(sched.enqueue(0, pid(1), 100, CutPressure::ZERO));
        assert!(sched.enqueue(0, pid(2), 200, CutPressure::ZERO));

        // Should switch to pid(2) which has higher deadline urgency.
        let (old, new) = sched.switch_next(0).unwrap();
        assert!(old.is_none());
        assert_eq!(new, pid(2));
    }

    #[test]
    fn test_degraded_mode_zeroes_pressure() {
        let mut sched: Scheduler<4, 256> = Scheduler::new();
        sched.enter_degraded();

        // In degraded mode, cut_pressure is zeroed.
        let high_pressure = CutPressure::from_fixed(u32::MAX);
        sched.enqueue(0, pid(1), 100, high_pressure); // effective pressure = 0
        sched.enqueue(0, pid(2), 150, CutPressure::ZERO);

        // pid(2) should win because 150 > 100 (pressure zeroed).
        let (_, new) = sched.switch_next(0).unwrap();
        assert_eq!(new, pid(2));
    }

    #[test]
    fn test_degraded_mode_deadline_only() {
        let mut sched: Scheduler<4, 256> = Scheduler::new();
        sched.enter_degraded();

        sched.enqueue(0, pid(1), 50, CutPressure::ZERO);
        sched.enqueue(0, pid(2), 200, CutPressure::ZERO);

        // In degraded mode, priority = deadline_urgency only.
        let (_, new) = sched.switch_next(0).unwrap();
        assert_eq!(new, pid(2));
    }

    #[test]
    fn test_epoch_tick() {
        let mut sched: Scheduler<4, 256> = Scheduler::new();
        sched.enqueue(0, pid(1), 100, CutPressure::ZERO);
        sched.switch_next(0);

        let summary = sched.tick_epoch();
        assert_eq!(summary.epoch, 0);
        assert_eq!(summary.switch_count, 1);
        assert_eq!(sched.current_epoch(), 1);
    }

    #[test]
    fn test_epoch_summary_degraded() {
        let mut sched: Scheduler<4, 256> = Scheduler::new();
        sched.enter_degraded();
        let summary = sched.tick_epoch();
        assert_eq!(summary.switch_count, 0);
    }

    #[test]
    fn test_invalid_cpu() {
        let mut sched: Scheduler<4, 256> = Scheduler::new();
        assert!(!sched.enqueue(99, pid(1), 100, CutPressure::ZERO));
        assert!(sched.switch_next(99).is_none());
    }

    #[test]
    fn test_queue_full() {
        let mut sched: Scheduler<4, 256> = Scheduler::new();
        for i in 0..MAX_RUN_QUEUE {
            // Start at 1: partition 0 (HYPERVISOR) is never schedulable.
            assert!(sched.enqueue(0, pid(i as u32 + 1), 100, CutPressure::ZERO));
        }
        // Queue is full.
        assert!(!sched.enqueue(0, pid(999), 100, CutPressure::ZERO));
    }

    #[test]
    fn test_enqueue_hypervisor_sentinel_rejected() {
        let mut sched: Scheduler<4, 256> = Scheduler::new();

        // The hypervisor sentinel must be rejected and must not consume
        // a queue slot.
        assert!(!sched.enqueue(0, PartitionId::HYPERVISOR, 100, CutPressure::ZERO));
        assert_eq!(sched.queue_len(0), 0);
        assert!(sched.switch_next(0).is_none());

        // The queue keeps working normally afterwards.
        assert!(sched.enqueue(0, pid(1), 100, CutPressure::ZERO));
        assert_eq!(sched.queue_len(0), 1);
        let (_, new) = sched.switch_next(0).unwrap();
        assert_eq!(new, pid(1));
        assert_eq!(sched.queue_len(0), 0);
    }

    #[test]
    fn test_priority_ordering() {
        let mut sched: Scheduler<4, 256> = Scheduler::new();
        sched.enqueue(0, pid(1), 50, CutPressure::ZERO);
        sched.enqueue(0, pid(2), 100, CutPressure::ZERO);
        sched.enqueue(0, pid(3), 75, CutPressure::ZERO);

        // Should dequeue in priority order: 100, 75, 50.
        let (_, first) = sched.switch_next(0).unwrap();
        assert_eq!(first, pid(2));

        let (_, second) = sched.switch_next(0).unwrap();
        assert_eq!(second, pid(3));

        let (_, third) = sched.switch_next(0).unwrap();
        assert_eq!(third, pid(1));
    }
}
