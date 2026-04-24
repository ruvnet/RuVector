//! io_uring-style Ring Buffer IPC for RuVix Cognition Kernel.
//!
//! This crate implements the Queue primitive from ADR-087 Section 7. All inter-task
//! communication in RuVix goes through queues. There are no synchronous IPC calls,
//! no shared memory without explicit region grants, and no signals.
//!
//! # Architecture
//!
//! Queues use io_uring-style ring buffers with separate submission (SQ) and
//! completion (CQ) queues. This enables:
//!
//! - **Lock-free operation**: Using atomic head/tail pointers
//! - **Zero-copy semantics**: When sender and receiver share a region
//! - **Typed messages**: WIT schema validation at send time
//! - **Priority support**: Higher priority messages are delivered first
//!
//! # Zero-Copy Semantics
//!
//! When sender and receiver share a region, `queue_send` places a descriptor
//! (offset + length) in the ring rather than copying bytes. This is critical
//! for high-throughput vector streaming where copying 768-dimensional f32
//! vectors would be prohibitive.
//!
//! **TOCTOU Protection**: Only Immutable or AppendOnly regions can use descriptors
//! (ADR-087 Section 20.5). The kernel rejects descriptors pointing into Slab
//! regions to prevent time-of-check-to-time-of-use attacks.
//!
//! # Example
//!
//! ```rust,ignore
//! use ruvix_queue::{KernelQueue, QueueConfig};
//! use ruvix_types::MsgPriority;
//!
//! // Create a queue
//! let config = QueueConfig::new(64, 4096); // 64 entries, 4KB max message
//! let mut queue = KernelQueue::new(config, region_handle)?;
//!
//! // Send a message
//! queue.send(b"hello", MsgPriority::Normal)?;
//!
//! // Receive with timeout
//! let mut buf = [0u8; 4096];
//! let len = queue.recv(&mut buf, Duration::from_millis(100))?;
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

#[cfg(feature = "alloc")]
extern crate alloc;

mod descriptor;
mod kernel_queue;
mod ring;
mod ring_optimized;

pub use descriptor::{DescriptorValidator, MessageDescriptor};
pub use kernel_queue::{KernelQueue, QueueConfig};
pub use ring::{RingBuffer, RingEntry, RingStats};
pub use ring_optimized::{OptimizedRingBuffer, OptimizedRingEntry, OptimizedRingSlot};

use ruvix_types::KernelError;

/// Result type for queue operations.
pub type Result<T> = core::result::Result<T, KernelError>;

/// Duration type for timeouts (re-export for convenience).
#[cfg(feature = "std")]
pub type Duration = std::time::Duration;

/// Duration in nanoseconds for no_std environments.
#[cfg(not(feature = "std"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Duration(u64);

#[cfg(not(feature = "std"))]
impl Duration {
    /// Zero duration.
    pub const ZERO: Self = Self(0);

    /// Maximum duration.
    pub const MAX: Self = Self(u64::MAX);

    /// Create from nanoseconds.
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }

    /// Create from microseconds.
    pub const fn from_micros(micros: u64) -> Self {
        Self(micros.saturating_mul(1_000))
    }

    /// Create from milliseconds.
    pub const fn from_millis(millis: u64) -> Self {
        Self(millis.saturating_mul(1_000_000))
    }

    /// Create from seconds.
    pub const fn from_secs(secs: u64) -> Self {
        Self(secs.saturating_mul(1_000_000_000))
    }

    /// Get nanoseconds.
    pub const fn as_nanos(&self) -> u64 {
        self.0
    }

    /// Check if zero.
    pub const fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_module_compiles() {
        assert!(true);
    }
}
