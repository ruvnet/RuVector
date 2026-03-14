//! Syscall dispatch table for the RuVix Cognition Kernel.
//!
//! This module defines all 12 syscalls from ADR-087 Section 3.1 and provides
//! the dispatch mechanism for the kernel.

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::{
    CapHandle, CapRights, Duration, GraphHandle, GraphMutation, MsgPriority, ProofToken,
    QueueHandle, RegionHandle, RegionPolicy, Result, RvfComponentId, SensorDescriptor,
    TaskHandle, TaskPriority, TimerSpec, VectorKey, VectorStoreHandle,
};

/// All 12 syscalls defined in ADR-087 Section 3.1.
///
/// Syscalls are the only way for tasks to interact with kernel resources.
/// Each syscall is capability-gated and may require proofs for mutations.
#[derive(Debug, Clone)]
pub enum Syscall {
    // =========================================================================
    // Task Management (Section 4)
    // =========================================================================
    /// Spawn a new task from an RVF component.
    ///
    /// Creates a new task with the specified entry point, capabilities, priority,
    /// and optional deadline. The task inherits no capabilities by default;
    /// all capabilities must be explicitly granted.
    TaskSpawn {
        /// RVF component to execute.
        entry: RvfComponentId,
        /// Capabilities to grant to the new task.
        #[cfg(feature = "alloc")]
        caps: Vec<CapHandle>,
        #[cfg(not(feature = "alloc"))]
        caps: [Option<CapHandle>; 16],
        /// Task priority level.
        priority: TaskPriority,
        /// Optional deadline for real-time scheduling.
        deadline: Option<Duration>,
    },

    // =========================================================================
    // Capability Management (Section 6)
    // =========================================================================
    /// Grant a capability to another task.
    ///
    /// The caller must have GRANT or GRANT_ONCE right on the source capability.
    /// Granted rights must be a subset of the source's rights.
    CapGrant {
        /// Target task to receive the capability.
        target: TaskHandle,
        /// Capability to grant (from caller's table).
        cap: CapHandle,
        /// Rights to grant (must be subset of cap's rights).
        rights: CapRights,
    },

    // =========================================================================
    // Memory Management (Section 5)
    // =========================================================================
    /// Map a new memory region.
    ///
    /// Creates a region with the specified policy. The region is physically
    /// backed at creation time (no demand paging).
    RegionMap {
        /// Size of the region in bytes.
        size: usize,
        /// Region access policy.
        policy: RegionPolicy,
        /// Capability authorizing the mapping.
        cap: CapHandle,
    },

    // =========================================================================
    // Queue IPC (Section 7)
    // =========================================================================
    /// Send a message to a queue.
    ///
    /// Messages are prioritized; higher priority messages are delivered first.
    QueueSend {
        /// Queue to send to.
        queue: QueueHandle,
        /// Message payload.
        #[cfg(feature = "alloc")]
        msg: Vec<u8>,
        #[cfg(not(feature = "alloc"))]
        msg: [u8; 4096],
        #[cfg(not(feature = "alloc"))]
        msg_len: usize,
        /// Message priority.
        priority: MsgPriority,
    },

    /// Receive a message from a queue.
    ///
    /// Blocks until a message is available or timeout expires.
    QueueRecv {
        /// Queue to receive from.
        queue: QueueHandle,
        /// Maximum buffer size.
        buf_size: usize,
        /// Timeout duration.
        timeout: Duration,
    },

    // =========================================================================
    // Timer (Section 8)
    // =========================================================================
    /// Wait until a timer expires.
    ///
    /// Suspends the calling task until the deadline.
    TimerWait {
        /// Timer specification (absolute or relative).
        deadline: TimerSpec,
    },

    // =========================================================================
    // RVF Management (Section 9)
    // =========================================================================
    /// Mount an RVF package.
    ///
    /// **Proof-required**: Requires a valid proof token to mount.
    /// Verifies the package signature before mounting.
    RvfMount {
        /// RVF package data.
        #[cfg(feature = "alloc")]
        rvf_data: Vec<u8>,
        #[cfg(not(feature = "alloc"))]
        rvf_data: [u8; 65536],
        #[cfg(not(feature = "alloc"))]
        rvf_len: usize,
        /// Mount point in the namespace.
        #[cfg(feature = "alloc")]
        mount_point: alloc::string::String,
        #[cfg(not(feature = "alloc"))]
        mount_point: [u8; 256],
        #[cfg(not(feature = "alloc"))]
        mount_point_len: usize,
        /// Capability authorizing the mount.
        cap: CapHandle,
    },

    // =========================================================================
    // Attestation (Section 10)
    // =========================================================================
    /// Emit an attestation record.
    ///
    /// Records a proof verification result in the witness log.
    AttestEmit {
        /// Attestation operation payload.
        operation: AttestPayload,
        /// Proof token authorizing the attestation.
        proof: ProofToken,
    },

    // =========================================================================
    // Vector Store (Section 11)
    // =========================================================================
    /// Get a vector from a kernel vector store.
    ///
    /// Returns the vector data and coherence metadata.
    VectorGet {
        /// Vector store handle.
        store: VectorStoreHandle,
        /// Vector key.
        key: VectorKey,
    },

    /// Put a vector with proof verification.
    ///
    /// **Proof-required**: Requires a valid proof token to mutate.
    VectorPutProved {
        /// Vector store handle.
        store: VectorStoreHandle,
        /// Vector key.
        key: VectorKey,
        /// Vector data (f32 components).
        #[cfg(feature = "alloc")]
        data: Vec<f32>,
        #[cfg(not(feature = "alloc"))]
        data: [f32; 768],
        #[cfg(not(feature = "alloc"))]
        data_len: usize,
        /// Proof token authorizing the mutation.
        proof: ProofToken,
    },

    // =========================================================================
    // Graph Store (Section 12)
    // =========================================================================
    /// Apply a graph mutation with proof verification.
    ///
    /// **Proof-required**: Requires a valid proof token to mutate.
    GraphApplyProved {
        /// Graph handle.
        graph: GraphHandle,
        /// Graph mutation to apply.
        mutation: GraphMutation,
        /// Proof token authorizing the mutation.
        proof: ProofToken,
    },

    // =========================================================================
    // Sensor Subscription (Section 13)
    // =========================================================================
    /// Subscribe to sensor events.
    ///
    /// Events from the sensor will be delivered to the target queue.
    SensorSubscribe {
        /// Sensor descriptor.
        sensor: SensorDescriptor,
        /// Queue to receive events.
        target_queue: QueueHandle,
        /// Capability authorizing the subscription.
        cap: CapHandle,
    },
}

impl Syscall {
    /// Returns the syscall number (for dispatch table).
    #[inline]
    #[must_use]
    pub const fn number(&self) -> u8 {
        match self {
            Self::TaskSpawn { .. } => 0,
            Self::CapGrant { .. } => 1,
            Self::RegionMap { .. } => 2,
            Self::QueueSend { .. } => 3,
            Self::QueueRecv { .. } => 4,
            Self::TimerWait { .. } => 5,
            Self::RvfMount { .. } => 6,
            Self::AttestEmit { .. } => 7,
            Self::VectorGet { .. } => 8,
            Self::VectorPutProved { .. } => 9,
            Self::GraphApplyProved { .. } => 10,
            Self::SensorSubscribe { .. } => 11,
        }
    }

    /// Returns the syscall name.
    #[inline]
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::TaskSpawn { .. } => "task_spawn",
            Self::CapGrant { .. } => "cap_grant",
            Self::RegionMap { .. } => "region_map",
            Self::QueueSend { .. } => "queue_send",
            Self::QueueRecv { .. } => "queue_recv",
            Self::TimerWait { .. } => "timer_wait",
            Self::RvfMount { .. } => "rvf_mount",
            Self::AttestEmit { .. } => "attest_emit",
            Self::VectorGet { .. } => "vector_get",
            Self::VectorPutProved { .. } => "vector_put_proved",
            Self::GraphApplyProved { .. } => "graph_apply_proved",
            Self::SensorSubscribe { .. } => "sensor_subscribe",
        }
    }

    /// Returns true if this syscall requires proof verification.
    #[inline]
    #[must_use]
    pub const fn requires_proof(&self) -> bool {
        matches!(
            self,
            Self::RvfMount { .. }
                | Self::AttestEmit { .. }
                | Self::VectorPutProved { .. }
                | Self::GraphApplyProved { .. }
        )
    }

    /// Returns true if this syscall mutates kernel state.
    #[inline]
    #[must_use]
    pub const fn is_mutation(&self) -> bool {
        matches!(
            self,
            Self::TaskSpawn { .. }
                | Self::CapGrant { .. }
                | Self::RegionMap { .. }
                | Self::QueueSend { .. }
                | Self::RvfMount { .. }
                | Self::AttestEmit { .. }
                | Self::VectorPutProved { .. }
                | Self::GraphApplyProved { .. }
                | Self::SensorSubscribe { .. }
        )
    }
}

/// Attestation payload for the `attest_emit` syscall.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttestPayload {
    /// Boot attestation (first record in witness log).
    Boot {
        /// Kernel version hash.
        kernel_hash: [u8; 32],
        /// Boot time (nanoseconds since epoch).
        boot_time_ns: u64,
    },

    /// RVF mount attestation.
    Mount {
        /// Mount handle.
        mount: RvfMountHandle,
        /// Package content hash.
        package_hash: [u8; 32],
    },

    /// Vector mutation attestation.
    VectorMutation {
        /// Store handle.
        store: VectorStoreHandle,
        /// Vector key.
        key: VectorKey,
        /// New data hash.
        data_hash: [u8; 32],
    },

    /// Graph mutation attestation.
    GraphMutation {
        /// Graph handle.
        graph: GraphHandle,
        /// Mutation hash.
        mutation_hash: [u8; 32],
    },

    /// Checkpoint attestation.
    Checkpoint {
        /// Checkpoint sequence number.
        sequence: u64,
        /// State hash at checkpoint.
        state_hash: [u8; 32],
    },
}

impl AttestPayload {
    /// Returns the attestation kind.
    #[inline]
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Boot { .. } => "boot",
            Self::Mount { .. } => "mount",
            Self::VectorMutation { .. } => "vector_mutation",
            Self::GraphMutation { .. } => "graph_mutation",
            Self::Checkpoint { .. } => "checkpoint",
        }
    }
}

/// Result of a syscall execution.
#[derive(Debug, Clone)]
pub enum SyscallResult {
    /// Task was spawned successfully.
    TaskSpawned(TaskHandle),

    /// Capability was granted successfully.
    CapGranted(CapHandle),

    /// Region was mapped successfully.
    RegionMapped(RegionHandle),

    /// Message was sent successfully.
    MessageSent,

    /// Message was received successfully.
    #[cfg(feature = "alloc")]
    MessageReceived {
        /// Message data.
        data: Vec<u8>,
        /// Message priority.
        priority: MsgPriority,
    },

    /// Message was received successfully (no_std version).
    #[cfg(not(feature = "alloc"))]
    MessageReceived {
        /// Message length.
        len: usize,
        /// Message priority.
        priority: MsgPriority,
    },

    /// Timer wait completed.
    TimerExpired,

    /// RVF package was mounted successfully.
    RvfMounted(RvfMountHandle),

    /// Attestation was emitted successfully.
    AttestEmitted {
        /// Witness log sequence number.
        sequence: u64,
    },

    /// Vector was retrieved successfully.
    #[cfg(feature = "alloc")]
    VectorRetrieved {
        /// Vector data.
        data: Vec<f32>,
        /// Coherence score (0.0-1.0).
        coherence: f32,
    },

    #[cfg(not(feature = "alloc"))]
    VectorRetrieved {
        /// Vector length.
        len: usize,
        /// Coherence score (0.0-1.0).
        coherence: f32,
    },

    /// Vector was stored successfully.
    VectorStored,

    /// Graph mutation was applied successfully.
    GraphApplied,

    /// Sensor subscription was created successfully.
    SensorSubscribed(crate::SubscriptionHandle),
}

impl SyscallResult {
    /// Returns true if the result indicates success.
    #[inline]
    #[must_use]
    pub const fn is_success(&self) -> bool {
        true // All variants represent success; errors use Result::Err
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syscall_numbers() {
        // Verify syscall numbers are sequential
        for i in 0..12 {
            let syscall = match i {
                0 => Syscall::TaskSpawn {
                    entry: RvfComponentId::root(RvfMountHandle::null()),
                    #[cfg(feature = "alloc")]
                    caps: Vec::new(),
                    #[cfg(not(feature = "alloc"))]
                    caps: [None; 16],
                    priority: TaskPriority::Normal,
                    deadline: None,
                },
                1 => Syscall::CapGrant {
                    target: TaskHandle::new(0, 0),
                    cap: CapHandle::null(),
                    rights: CapRights::READ,
                },
                2 => Syscall::RegionMap {
                    size: 4096,
                    policy: RegionPolicy::Immutable,
                    cap: CapHandle::null(),
                },
                3 => Syscall::QueueSend {
                    queue: QueueHandle::null(),
                    #[cfg(feature = "alloc")]
                    msg: Vec::new(),
                    #[cfg(not(feature = "alloc"))]
                    msg: [0u8; 4096],
                    #[cfg(not(feature = "alloc"))]
                    msg_len: 0,
                    priority: MsgPriority::Normal,
                },
                4 => Syscall::QueueRecv {
                    queue: QueueHandle::null(),
                    buf_size: 4096,
                    timeout: Duration::from_millis(100),
                },
                5 => Syscall::TimerWait {
                    deadline: TimerSpec::from_millis(100),
                },
                6 => Syscall::RvfMount {
                    #[cfg(feature = "alloc")]
                    rvf_data: Vec::new(),
                    #[cfg(not(feature = "alloc"))]
                    rvf_data: [0u8; 65536],
                    #[cfg(not(feature = "alloc"))]
                    rvf_len: 0,
                    #[cfg(feature = "alloc")]
                    mount_point: alloc::string::String::new(),
                    #[cfg(not(feature = "alloc"))]
                    mount_point: [0u8; 256],
                    #[cfg(not(feature = "alloc"))]
                    mount_point_len: 0,
                    cap: CapHandle::null(),
                },
                7 => Syscall::AttestEmit {
                    operation: AttestPayload::Boot {
                        kernel_hash: [0u8; 32],
                        boot_time_ns: 0,
                    },
                    proof: ProofToken::default(),
                },
                8 => Syscall::VectorGet {
                    store: VectorStoreHandle::null(),
                    key: VectorKey::new(0),
                },
                9 => Syscall::VectorPutProved {
                    store: VectorStoreHandle::null(),
                    key: VectorKey::new(0),
                    #[cfg(feature = "alloc")]
                    data: Vec::new(),
                    #[cfg(not(feature = "alloc"))]
                    data: [0.0; 768],
                    #[cfg(not(feature = "alloc"))]
                    data_len: 0,
                    proof: ProofToken::default(),
                },
                10 => Syscall::GraphApplyProved {
                    graph: GraphHandle::null(),
                    mutation: GraphMutation::add_node(0),
                    proof: ProofToken::default(),
                },
                11 => Syscall::SensorSubscribe {
                    sensor: SensorDescriptor::default(),
                    target_queue: QueueHandle::null(),
                    cap: CapHandle::null(),
                },
                _ => unreachable!(),
            };
            assert_eq!(syscall.number(), i);
        }
    }

    #[test]
    fn test_proof_required_syscalls() {
        // Verify proof-required syscalls
        let proof_syscalls = [
            Syscall::RvfMount {
                #[cfg(feature = "alloc")]
                rvf_data: Vec::new(),
                #[cfg(not(feature = "alloc"))]
                rvf_data: [0u8; 65536],
                #[cfg(not(feature = "alloc"))]
                rvf_len: 0,
                #[cfg(feature = "alloc")]
                mount_point: alloc::string::String::new(),
                #[cfg(not(feature = "alloc"))]
                mount_point: [0u8; 256],
                #[cfg(not(feature = "alloc"))]
                mount_point_len: 0,
                cap: CapHandle::null(),
            },
            Syscall::AttestEmit {
                operation: AttestPayload::Boot {
                    kernel_hash: [0u8; 32],
                    boot_time_ns: 0,
                },
                proof: ProofToken::default(),
            },
            Syscall::VectorPutProved {
                store: VectorStoreHandle::null(),
                key: VectorKey::new(0),
                #[cfg(feature = "alloc")]
                data: Vec::new(),
                #[cfg(not(feature = "alloc"))]
                data: [0.0; 768],
                #[cfg(not(feature = "alloc"))]
                data_len: 0,
                proof: ProofToken::default(),
            },
            Syscall::GraphApplyProved {
                graph: GraphHandle::null(),
                mutation: GraphMutation::add_node(0),
                proof: ProofToken::default(),
            },
        ];

        for syscall in &proof_syscalls {
            assert!(
                syscall.requires_proof(),
                "{} should require proof",
                syscall.name()
            );
        }
    }
}

use crate::RvfMountHandle;
