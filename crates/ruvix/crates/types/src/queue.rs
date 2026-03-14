//! Queue IPC types.
//!
//! All inter-task communication in RuVix goes through queues. There are no
//! synchronous IPC calls, no shared memory without explicit region grants,
//! and no signals. Queues use io_uring-style ring buffers.

use crate::handle::Handle;

/// Handle to a kernel queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct QueueHandle(pub Handle);

impl QueueHandle {
    /// Creates a new queue handle.
    #[inline]
    #[must_use]
    pub const fn new(id: u32, generation: u32) -> Self {
        Self(Handle::new(id, generation))
    }

    /// Creates a null (invalid) queue handle.
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

impl Default for QueueHandle {
    fn default() -> Self {
        Self::null()
    }
}

/// Message priority for queue operations.
///
/// Higher priority messages are delivered before lower priority messages
/// when multiple messages are available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum MsgPriority {
    /// Lowest priority. Background messages, bulk transfers.
    Low = 0,

    /// Normal priority. Default for most messages.
    Normal = 1,

    /// High priority. Time-sensitive messages.
    High = 2,

    /// Urgent priority. Control messages, error notifications.
    Urgent = 3,
}

impl MsgPriority {
    /// Converts from a raw u8 value.
    #[inline]
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Low),
            1 => Some(Self::Normal),
            2 => Some(Self::High),
            3 => Some(Self::Urgent),
            _ => None,
        }
    }
}

impl Default for MsgPriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Queue ring buffer configuration.
///
/// Used internally by the kernel to manage queue state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct QueueConfig {
    /// Ring buffer size (must be a power of 2).
    pub ring_size: u32,

    /// Maximum message size in bytes.
    pub max_msg_size: u32,

    /// RVF WIT type identifier for message validation.
    /// Zero means no schema validation (raw bytes).
    pub schema_id: u32,
}

impl QueueConfig {
    /// Creates a new queue configuration.
    #[inline]
    #[must_use]
    pub const fn new(ring_size: u32, max_msg_size: u32) -> Self {
        Self {
            ring_size,
            max_msg_size,
            schema_id: 0,
        }
    }

    /// Creates a queue configuration with schema validation.
    #[inline]
    #[must_use]
    pub const fn with_schema(ring_size: u32, max_msg_size: u32, schema_id: u32) -> Self {
        Self {
            ring_size,
            max_msg_size,
            schema_id,
        }
    }

    /// Checks if the ring size is valid (power of 2).
    #[inline]
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.ring_size > 0 && (self.ring_size & (self.ring_size - 1)) == 0
    }
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            ring_size: 64,
            max_msg_size: 4096,
            schema_id: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_handle() {
        let h = QueueHandle::new(7, 3);
        assert!(!h.is_null());
        assert_eq!(h.raw().id, 7);
    }

    #[test]
    fn test_msg_priority_ordering() {
        assert!(MsgPriority::Low < MsgPriority::Normal);
        assert!(MsgPriority::Normal < MsgPriority::High);
        assert!(MsgPriority::High < MsgPriority::Urgent);
    }

    #[test]
    fn test_queue_config_valid() {
        let config = QueueConfig::new(64, 4096);
        assert!(config.is_valid());

        let invalid = QueueConfig::new(63, 4096);
        assert!(!invalid.is_valid());
    }
}
