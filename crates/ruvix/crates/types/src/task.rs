//! Task management types.
//!
//! A Task is the unit of concurrent execution in RuVix. Each task has an
//! explicit capability set and runs with a scheduling priority.

use crate::handle::Handle;

/// Handle to a task (unit of concurrent execution).
///
/// Tasks are analogous to seL4 TCBs (Thread Control Blocks). Each task has:
/// - An entry point (RVF component)
/// - A capability set (what resources it can access)
/// - A scheduling priority
/// - An optional hard deadline
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct TaskHandle(pub Handle);

impl TaskHandle {
    /// Creates a new task handle.
    #[inline]
    #[must_use]
    pub const fn new(id: u32, generation: u32) -> Self {
        Self(Handle::new(id, generation))
    }

    /// Creates a null (invalid) task handle.
    #[inline]
    #[must_use]
    pub const fn null() -> Self {
        Self(Handle::null())
    }

    /// Checks if this handle is null.
    #[inline]
    #[must_use]
    pub const fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Returns the raw handle.
    #[inline]
    #[must_use]
    pub const fn raw(&self) -> Handle {
        self.0
    }
}

impl Default for TaskHandle {
    fn default() -> Self {
        Self::null()
    }
}

/// Task scheduling priority.
///
/// Priority is used by the coherence-aware scheduler in combination with
/// deadline pressure, novelty signal, and structural risk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum TaskPriority {
    /// Lowest priority. Background tasks, housekeeping.
    Idle = 0,

    /// Normal priority. Default for most tasks.
    Normal = 1,

    /// High priority. Time-sensitive operations.
    High = 2,

    /// Real-time priority. Hard deadline tasks.
    RealTime = 3,

    /// Critical priority. Kernel-level urgency.
    Critical = 4,
}

impl TaskPriority {
    /// Returns the priority as a numeric weight (0-100).
    #[inline]
    #[must_use]
    pub const fn weight(&self) -> u8 {
        match self {
            Self::Idle => 0,
            Self::Normal => 25,
            Self::High => 50,
            Self::RealTime => 75,
            Self::Critical => 100,
        }
    }

    /// Converts from a raw u8 value.
    #[inline]
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Idle),
            1 => Some(Self::Normal),
            2 => Some(Self::High),
            3 => Some(Self::RealTime),
            4 => Some(Self::Critical),
            _ => None,
        }
    }
}

impl Default for TaskPriority {
    fn default() -> Self {
        Self::Normal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_handle() {
        let h = TaskHandle::new(1, 2);
        assert!(!h.is_null());
        assert_eq!(h.raw().id, 1);
        assert_eq!(h.raw().generation, 2);
    }

    #[test]
    fn test_task_priority_ordering() {
        assert!(TaskPriority::Idle < TaskPriority::Normal);
        assert!(TaskPriority::Normal < TaskPriority::High);
        assert!(TaskPriority::High < TaskPriority::RealTime);
        assert!(TaskPriority::RealTime < TaskPriority::Critical);
    }

    #[test]
    fn test_task_priority_weight() {
        assert_eq!(TaskPriority::Critical.weight(), 100);
        assert_eq!(TaskPriority::Idle.weight(), 0);
    }
}
