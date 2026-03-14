//! Error types for the scheduler.

use core::fmt;

/// Result type for scheduler operations.
pub type SchedResult<T> = Result<T, SchedError>;

/// Scheduler errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedError {
    /// Task was not found in the scheduler.
    TaskNotFound,

    /// Partition was not found.
    PartitionNotFound,

    /// The ready queue is full (maximum tasks reached).
    QueueFull,

    /// The partition has no more capacity for tasks.
    PartitionFull,

    /// Invalid task state transition.
    InvalidStateTransition,

    /// Task is already in the scheduler.
    TaskAlreadyExists,

    /// Partition already exists.
    PartitionAlreadyExists,

    /// The task's deadline has already passed.
    DeadlineMissed,

    /// Invalid priority value.
    InvalidPriority,

    /// Invalid coherence delta value.
    InvalidCoherenceDelta,

    /// Invalid novelty value (must be 0.0..1.0).
    InvalidNovelty,

    /// Maximum number of partitions reached.
    MaxPartitionsReached,

    /// Task cannot be scheduled (blocked or suspended).
    TaskNotSchedulable,
}

impl fmt::Display for SchedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TaskNotFound => write!(f, "task not found in scheduler"),
            Self::PartitionNotFound => write!(f, "partition not found"),
            Self::QueueFull => write!(f, "ready queue is full"),
            Self::PartitionFull => write!(f, "partition has no more capacity"),
            Self::InvalidStateTransition => write!(f, "invalid task state transition"),
            Self::TaskAlreadyExists => write!(f, "task already exists in scheduler"),
            Self::PartitionAlreadyExists => write!(f, "partition already exists"),
            Self::DeadlineMissed => write!(f, "task deadline has passed"),
            Self::InvalidPriority => write!(f, "invalid priority value"),
            Self::InvalidCoherenceDelta => write!(f, "invalid coherence delta value"),
            Self::InvalidNovelty => write!(f, "invalid novelty value (must be 0.0..1.0)"),
            Self::MaxPartitionsReached => write!(f, "maximum number of partitions reached"),
            Self::TaskNotSchedulable => write!(f, "task cannot be scheduled"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SchedError {}
