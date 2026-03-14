//! Coherence-aware scheduler for the RuVix Cognition Kernel.
//!
//! This crate implements the scheduler specified in ADR-087 Section 5. The scheduler
//! combines three signals to determine task priority:
//!
//! 1. **Deadline pressure**: EDF (Earliest Deadline First) scheduling within
//!    capability partitions.
//! 2. **Novelty signal**: Priority boost for tasks processing genuinely new
//!    information (measured by vector distance from recent inputs).
//! 3. **Structural risk**: Deprioritization of tasks whose pending mutations
//!    would lower the coherence score.
//!
//! # Scheduling Guarantees (ADR-087 Section 5.2)
//!
//! - **No priority inversion**: Capability-based access prevents tasks from
//!   blocking on resources they do not hold capabilities for.
//! - **Bounded preemption**: Preemption occurs only at queue boundaries
//!   (after `queue_send` or `queue_recv` completes).
//! - **Partition scheduling**: Tasks are grouped by RVF mount origin, with
//!   each partition receiving a guaranteed time slice.
//!
//! # Example
//!
//! ```
//! use ruvix_sched::{Scheduler, SchedulerConfig, TaskControlBlock, TaskState};
//! use ruvix_types::{TaskHandle, TaskPriority, SchedulerPartition};
//! use ruvix_cap::CapRights;
//!
//! // Create a scheduler with default configuration
//! let config = SchedulerConfig::default();
//! let mut scheduler: Scheduler<64, 8> = Scheduler::new(config);
//!
//! // Create a task control block
//! let task_handle = TaskHandle::new(1, 0);
//! let partition = SchedulerPartition::new(0, 10000); // 10ms time slice
//! let mut tcb = TaskControlBlock::new(
//!     task_handle,
//!     CapRights::READ | CapRights::WRITE,
//!     TaskPriority::Normal,
//!     partition.partition_id,
//! );
//!
//! // Add task to scheduler
//! scheduler.add_task(tcb).unwrap();
//!
//! // Select next task to run
//! if let Some(next_task) = scheduler.select_next_task() {
//!     println!("Next task: {:?}", next_task);
//! }
//! ```

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

mod error;
mod novelty;
mod partition;
mod priority;
mod scheduler;
mod task;

pub use error::{SchedError, SchedResult};
pub use novelty::{NoveltyConfig, NoveltyTracker};
pub use partition::{PartitionManager, PartitionState};
pub use priority::{compute_priority, PriorityConfig, RISK_WEIGHT};
pub use scheduler::{Scheduler, SchedulerConfig, SchedulerStats};
pub use task::{TaskControlBlock, TaskState};

// Re-export commonly used types from ruvix-types
pub use ruvix_types::{SchedulerPartition, SchedulerScore, TaskHandle, TaskPriority};

// Re-export capability types
pub use ruvix_cap::CapRights;

/// Default maximum tasks per partition.
pub const DEFAULT_MAX_TASKS_PER_PARTITION: usize = 64;

/// Default maximum number of partitions.
pub const DEFAULT_MAX_PARTITIONS: usize = 8;

/// Default time quantum in microseconds (1ms).
pub const DEFAULT_TIME_QUANTUM_US: u32 = 1000;

/// Default novelty decay factor (per scheduling tick).
pub const DEFAULT_NOVELTY_DECAY: f32 = 0.95;

/// Default coherence risk weight.
pub const DEFAULT_RISK_WEIGHT: f32 = 2.0;

/// Preemption boundary marker.
///
/// Preemption in RuVix occurs only at queue boundaries, not at arbitrary
/// instruction boundaries. This eliminates the need for kernel-level spinlocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PreemptionBoundary {
    /// After a successful `queue_send` operation.
    QueueSendComplete = 0,
    /// After a successful `queue_recv` operation.
    QueueRecvComplete = 1,
    /// After a timer deadline has elapsed.
    TimerExpired = 2,
    /// Voluntary yield by the task.
    VoluntaryYield = 3,
}

/// Instant type for deadline tracking.
///
/// In a no_std environment, this is a simple wrapper around a microsecond counter.
/// When the `std` feature is enabled, this uses `std::time::Instant`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Instant(u64);

impl Instant {
    /// Creates a new instant from microseconds since boot.
    #[inline]
    #[must_use]
    pub const fn from_micros(micros: u64) -> Self {
        Self(micros)
    }

    /// Returns the number of microseconds since boot.
    #[inline]
    #[must_use]
    pub const fn as_micros(&self) -> u64 {
        self.0
    }

    /// Returns the duration since another instant, or zero if `other` is later.
    #[inline]
    #[must_use]
    pub const fn saturating_duration_since(&self, other: Self) -> Duration {
        if self.0 > other.0 {
            Duration::from_micros(self.0 - other.0)
        } else {
            Duration::from_micros(0)
        }
    }

    /// Adds a duration to this instant.
    #[inline]
    #[must_use]
    pub const fn add_duration(&self, duration: Duration) -> Self {
        Self(self.0.saturating_add(duration.as_micros()))
    }
}

impl Default for Instant {
    fn default() -> Self {
        Self(0)
    }
}

/// Duration type for time intervals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct Duration(u64);

impl Duration {
    /// Creates a new duration from microseconds.
    #[inline]
    #[must_use]
    pub const fn from_micros(micros: u64) -> Self {
        Self(micros)
    }

    /// Creates a new duration from milliseconds.
    #[inline]
    #[must_use]
    pub const fn from_millis(millis: u64) -> Self {
        Self(millis.saturating_mul(1000))
    }

    /// Creates a new duration from seconds.
    #[inline]
    #[must_use]
    pub const fn from_secs(secs: u64) -> Self {
        Self(secs.saturating_mul(1_000_000))
    }

    /// Returns the duration in microseconds.
    #[inline]
    #[must_use]
    pub const fn as_micros(&self) -> u64 {
        self.0
    }

    /// Returns the duration in milliseconds (rounded down).
    #[inline]
    #[must_use]
    pub const fn as_millis(&self) -> u64 {
        self.0 / 1000
    }
}
