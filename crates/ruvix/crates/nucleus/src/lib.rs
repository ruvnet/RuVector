//! RuVix Nucleus - Integration crate for the RuVix Cognition Kernel.
//!
//! This crate brings all RuVix subsystems together and provides:
//!
//! - **Syscall dispatch table** for all 12 syscalls defined in ADR-087
//! - **Kernel struct** coordinating all subsystems
//! - **Deterministic replay** support for checkpoint/restore
//! - **Witness log** for attestation and auditability
//!
//! # Architecture
//!
//! The Nucleus is the integration point where:
//!
//! ```text
//!                    +-------------------+
//!                    |    RuVix Nucleus  |
//!                    |  (this crate)     |
//!                    +--------+----------+
//!                             |
//!         +-------------------+-------------------+
//!         |           |           |               |
//!   +-----v-----+ +---v---+ +-----v-----+ +-------v-------+
//!   | RegionMgr | | CapMgr| | QueueMgr  | | ProofEngine   |
//!   +-----------+ +-------+ +-----------+ +---------------+
//!         |           |           |               |
//!   +-----v-----+ +---v---+ +-----v-----+ +-------v-------+
//!   | VectorMgr | |GraphMgr| | Scheduler | | WitnessLog    |
//!   +-----------+ +-------+ +-----------+ +---------------+
//! ```
//!
//! # Syscall Invariants (ADR-087 Section 3.2)
//!
//! 1. **Capability-gated**: No syscall succeeds without appropriate capability
//! 2. **Proof-required for mutation**: `vector_put_proved`, `graph_apply_proved`, `rvf_mount`
//! 3. **Bounded latency**: No unbounded loops in syscall path
//! 4. **Witness-logged**: Every mutation emits a witness record
//! 5. **No allocation in syscall path**: Pre-allocated structures only
//!
//! # Example
//!
//! ```rust,ignore
//! use ruvix_nucleus::{Kernel, KernelConfig, Syscall};
//!
//! // Create kernel with default configuration
//! let mut kernel = Kernel::new(KernelConfig::default());
//!
//! // Execute a syscall
//! let result = kernel.dispatch(Syscall::RegionMap {
//!     size: 4096,
//!     policy: RegionPolicy::AppendOnly { max_size: 4096 },
//!     cap: root_cap,
//! });
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

#[cfg(feature = "alloc")]
extern crate alloc;

mod kernel;
mod proof_engine;
mod scheduler;
mod syscall;
mod witness_log;
mod vector_store;
mod graph_store;
mod checkpoint;

#[cfg(feature = "shell")]
mod shell_backend;

pub use kernel::{Kernel, KernelConfig, KernelStats};
pub use proof_engine::{ProofEngine, ProofEngineConfig, ProofVerifyResult};
pub use scheduler::{Scheduler, SchedulerConfig, TaskState};
pub use syscall::{Syscall, SyscallResult, AttestPayload};
pub use witness_log::{WitnessLog, WitnessRecord, WitnessRecordKind};
pub use vector_store::{VectorStore, VectorStoreEntry};
pub use graph_store::{GraphStore, GraphNode, GraphEdge};
pub use checkpoint::{Checkpoint, CheckpointConfig, ReplayEngine};

// Re-export commonly used types from dependencies
pub use ruvix_types::{
    CapHandle, CapRights, GraphHandle, GraphMutation, KernelError, MsgPriority,
    ProofAttestation, ProofPayload, ProofTier, ProofToken, QueueHandle, RegionHandle,
    RegionPolicy, RvfComponentId, RvfMountHandle, SensorDescriptor, SubscriptionHandle,
    TaskHandle, TimerSpec, VectorKey, VectorStoreConfig, VectorStoreHandle,
};

use ruvix_types::KernelError as Error;

/// Result type for kernel operations.
pub type Result<T> = core::result::Result<T, Error>;

/// Duration type for timeouts.
#[cfg(feature = "std")]
pub type Duration = std::time::Duration;

/// Duration in nanoseconds for no_std environments.
#[cfg(not(feature = "std"))]
pub use ruvix_queue::Duration;

/// Task priority levels for scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum TaskPriority {
    /// Background tasks with no deadline.
    Background = 0,
    /// Normal priority tasks.
    Normal = 1,
    /// High priority tasks.
    High = 2,
    /// Real-time tasks with strict deadlines.
    RealTime = 3,
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
    fn test_module_compiles() {
        // Basic smoke test
        let _ = TaskPriority::default();
    }

    #[test]
    fn test_task_priority_ordering() {
        assert!(TaskPriority::Background < TaskPriority::Normal);
        assert!(TaskPriority::Normal < TaskPriority::High);
        assert!(TaskPriority::High < TaskPriority::RealTime);
    }
}
