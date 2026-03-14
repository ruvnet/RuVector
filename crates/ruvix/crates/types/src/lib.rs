//! # RuVix Kernel Interface Types
//!
//! This crate provides all kernel interface types for the RuVix Cognition Kernel
//! as specified in ADR-087. It is designed to be `no_std` compatible with zero
//! external dependencies, ensuring it can be used in both kernel code and RVF
//! component code.
//!
//! ## Core Primitives
//!
//! RuVix has exactly six kernel primitives:
//!
//! | Primitive | Purpose | Analog |
//! |-----------|---------|--------|
//! | **Task** | Unit of concurrent execution with capability set | seL4 TCB |
//! | **Capability** | Unforgeable typed token granting access to a resource | seL4 capability |
//! | **Region** | Contiguous memory with access policy | seL4 Untyped + frame |
//! | **Queue** | Typed ring buffer for inter-task communication | io_uring SQ/CQ |
//! | **Timer** | Deadline-driven scheduling primitive | POSIX timer_create |
//! | **Proof** | Cryptographic attestation gating state mutation | Novel (ADR-047) |
//!
//! ## Features
//!
//! - `std`: Enable standard library support
//! - `alloc`: Enable alloc crate support for heap allocation

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

mod capability;
mod error;
mod graph;
mod handle;
mod object;
mod proof;
mod proof_cache;
mod proof_cache_optimized;
mod queue;
mod region;
mod rvf;
mod scheduler;
mod sensor;
mod task;
mod timer;
mod vector;

pub use capability::{CapHandle, CapRights, Capability};
pub use error::KernelError;
pub use graph::{GraphHandle, GraphMutation, GraphMutationKind};
pub use handle::Handle;
pub use object::ObjectType;
pub use proof::{ProofAttestation, ProofPayload, ProofTier, ProofToken};
pub use proof_cache::{CacheError, ProofCache, ProofCacheEntry, ProofCacheStats};
pub use proof_cache_optimized::{OptimizedProofCache, OptimizedProofEntry};
pub use queue::{MsgPriority, QueueConfig, QueueHandle};
pub use region::{RegionHandle, RegionPolicy};
pub use rvf::{RvfComponentId, RvfMountHandle, RvfVerifyStatus, WitTypeId};
pub use scheduler::{SchedulerPartition, SchedulerScore};
pub use sensor::{SensorDescriptor, SensorType, SubscriptionHandle};
pub use task::{TaskHandle, TaskPriority};
pub use timer::TimerSpec;
pub use vector::{CoherenceMeta, VectorKey, VectorStoreConfig, VectorStoreHandle};

/// Re-export proof cache module for direct access.
pub mod proof_cache_mod {
    pub use crate::proof_cache::*;
}

/// The witness size in bytes for proof attestations (ADR-047 compatible).
pub const ATTESTATION_SIZE: usize = 82;

/// Maximum capability delegation depth (Section 20.2).
pub const MAX_DELEGATION_DEPTH: u8 = 8;

/// Default Reflex proof cache TTL in milliseconds (Section 20.4).
/// See `proof_cache::PROOF_CACHE_TTL_MS` for the canonical value.
pub const REFLEX_CACHE_TTL_MS: u32 = proof_cache::PROOF_CACHE_TTL_MS;

/// Default Reflex proof cache size (Section 20.4).
/// See `proof_cache::PROOF_CACHE_MAX_ENTRIES` for the canonical value.
pub const REFLEX_CACHE_SIZE: usize = proof_cache::PROOF_CACHE_MAX_ENTRIES;
