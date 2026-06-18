//! # RVM Core Types
//!
//! Foundation types for the RVM (`RuVix` Virtual Machine) coherence-native
//! microhypervisor, as specified in ADR-132, ADR-133, and ADR-134. This
//! crate has minimal external dependencies and provides the type vocabulary
//! shared by all RVM crates.
//!
//! ## First-Class Objects (ADR-132)
//!
//! | Type | Purpose |
//! |------|---------|
//! | [`PartitionId`] | Coherence domain container; unit of scheduling, isolation, and migration |
//! | [`Capability`] | Unforgeable authority token; grants specific rights over specific objects |
//! | [`WitnessRecord`] | 64-byte audit record emitted by every privileged action |
//! | [`MemoryRegion`] | Typed, tiered, owned memory range with explicit lifetime |
//! | [`CommEdge`] | Inter-partition communication channel; weighted edge in the coherence graph |
//! | [`DeviceLease`] | Time-bounded, revocable access grant to a hardware device |
//! | [`CoherenceScore`] | Locality and coupling metric derived from the coherence graph |
//! | [`CutPressure`] | Graph-derived isolation signal; high pressure triggers migration or split |
//! | [`RecoveryCheckpoint`] | State snapshot for rollback and reconstruction |
//!
//! ## Design Constraints
//!
//! - `#![no_std]` with zero heap allocation in the default configuration
//! - `#![forbid(unsafe_code)]` -- all types are safe Rust
//! - All identifiers are `Copy + Clone + Eq + Hash`-compatible newtypes

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

mod addr;
mod capability;
mod coherence;
mod config;
mod device;
mod error;
mod ids;
mod memory;
mod partition;
mod proof;
mod recovery;
mod scheduler;
mod witness;

// --- Address types ---
pub use addr::{GuestPhysAddr, PhysAddr, VirtAddr};

// --- Identifier types ---
pub use ids::{PartitionId, VcpuId};

// --- Capability types ---
pub use capability::{
    CapRights, CapToken, CapType, Capability, CapabilityId, MAX_DELEGATION_DEPTH,
};

// --- Witness types ---
pub use witness::{
    ActionKind, WitnessHash, WitnessRecord, WitnessRecordV2, WIRE_VERSION_OFFSET,
    WITNESS_RECORD_SIZE, WITNESS_RING_CAPACITY, fnv1a_32, fnv1a_64,
};

// --- Coherence types ---
pub use coherence::{CoherenceScore, CommEdge, CommEdgeId, CutPressure, PhiValue};

// --- Partition types ---
pub use partition::{
    PartitionConfig, PartitionState, PartitionType, MAX_DEVICES_PER_PARTITION,
    MAX_EDGES_PER_PARTITION, MAX_PARTITIONS,
};

// --- Memory types ---
pub use memory::{MemoryRegion, MemoryTier, OwnedRegionId, RegionPlacementWeights, RegionPolicy};

// --- Device types ---
pub use device::{DeviceClass, DeviceLease, DeviceLeaseId};

// --- Proof types ---
pub use proof::{ProofResult, ProofTier, ProofToken};

// --- Scheduler types ---
pub use scheduler::{EpochConfig, EpochSummary, Priority, SchedulerMode};

// --- Recovery types ---
pub use recovery::{FailureClass, RecoveryCheckpoint, ReconstructionReceipt};

// --- Configuration ---
pub use config::RvmConfig;

// --- Error types ---
pub use error::{RvmError, RvmResult};
