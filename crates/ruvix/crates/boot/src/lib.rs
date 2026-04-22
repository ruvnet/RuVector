//! # RVF Boot Loading for RuVix Cognition Kernel
//!
//! This crate provides the RVF (RuVector Format) boot loading infrastructure
//! for the RuVix Cognition Kernel as specified in ADR-087.
//!
//! ## Boot Sequence (ADR-087 Section 9.1)
//!
//! The kernel boot follows a five-stage process:
//!
//! | Stage | Name | Description |
//! |-------|------|-------------|
//! | **0** | Hardware Init | Platform-specific initialization (mocked in Phase A) |
//! | **1** | RVF Verify | Parse manifest + ML-DSA-65 signature verification |
//! | **2** | Object Create | Create root task, regions, queues, witness log |
//! | **3** | Component Mount | Mount components + distribute capabilities |
//! | **4** | First Attestation | Boot attestation to witness log |
//!
//! ## Security (SEC-001)
//!
//! This crate implements critical security fixes:
//!
//! - **Signature failure**: PANIC IMMEDIATELY, no fallback boot path
//! - **Root task capability drop**: After Stage 3, root task drops to minimum set
//! - **Witness log integrity**: Append-only, cryptographically linked
//!
//! ## Features
//!
//! - `std`: Enable standard library support (default)
//! - `alloc`: Enable alloc crate support
//! - `metrics`: Enable boot metrics collection
//! - `verbose`: Enable verbose boot logging
//! - `baremetal`: Phase B bare metal (no std, no libc)
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvix_boot::{BootLoader, BootConfig};
//!
//! let config = BootConfig::default();
//! let mut loader = BootLoader::new(config);
//!
//! // Load and verify the RVF boot image
//! let manifest_bytes = include_bytes!("boot.rvf.manifest");
//! let signature = include_bytes!("boot.rvf.sig");
//!
//! // This will PANIC if signature verification fails (SEC-001)
//! loader.boot(manifest_bytes, signature)?;
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

mod attestation;
mod boot_loader;
mod capability_distribution;
mod manifest;
mod mount;
mod signature;
mod stages;
mod witness_log;

pub use attestation::{AttestationEntry, BootAttestation};
pub use boot_loader::{BootConfig, BootLoader, BootResult, BootStage};
pub use capability_distribution::{
    CapabilityDistribution, MinimumCapabilitySet, RootCapabilityDrop,
};
pub use manifest::{
    ComponentDecl, ComponentGraph, MemorySchema, ProofPolicy, QueueWiring, RollbackHook,
    RvfManifest, WitnessLogPolicy,
};
pub use mount::{MountConfig, MountResult, RvfMount};
pub use signature::{SignatureVerifier, VerifyResult};
pub use stages::{Stage0Hardware, Stage1Verify, Stage2Create, Stage3Mount, Stage4Attest};
pub use witness_log::{WitnessLog, WitnessLogConfig, WitnessLogEntry};

// Re-export commonly used types from dependencies
pub use ruvix_cap::{BootCapabilitySet, InitialCapability};
pub use ruvix_types::{
    KernelError, ProofAttestation, ProofTier, RegionHandle, RegionPolicy, RvfMountHandle,
    RvfVerifyStatus, TaskHandle, TaskPriority,
};

/// Result type for boot operations.
pub type Result<T> = core::result::Result<T, KernelError>;

/// Boot stage constants.
pub mod stage {
    /// Stage 0: Hardware initialization (mocked in Phase A).
    pub const HARDWARE_INIT: u8 = 0;

    /// Stage 1: RVF manifest parse + ML-DSA-65 signature verification.
    pub const RVF_VERIFY: u8 = 1;

    /// Stage 2: Kernel object creation (root task, regions, queues, witness log).
    pub const OBJECT_CREATE: u8 = 2;

    /// Stage 3: Component mount + capability distribution.
    pub const COMPONENT_MOUNT: u8 = 3;

    /// Stage 4: First attestation (boot attestation to witness log).
    pub const FIRST_ATTESTATION: u8 = 4;
}

/// ML-DSA-65 signature size in bytes (NIST FIPS 204).
pub const ML_DSA_65_SIGNATURE_SIZE: usize = 3309;

/// ML-DSA-65 public key size in bytes.
pub const ML_DSA_65_PUBLIC_KEY_SIZE: usize = 1952;

/// Maximum manifest size in bytes.
pub const MAX_MANIFEST_SIZE: usize = 1024 * 1024; // 1 MiB

/// Maximum number of components in an RVF package.
pub const MAX_COMPONENTS: usize = 256;

/// Maximum queue wiring connections per manifest.
pub const MAX_QUEUE_WIRINGS: usize = 1024;

/// Maximum region declarations per manifest.
pub const MAX_REGION_DECLS: usize = 256;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_constants() {
        assert_eq!(stage::HARDWARE_INIT, 0);
        assert_eq!(stage::RVF_VERIFY, 1);
        assert_eq!(stage::OBJECT_CREATE, 2);
        assert_eq!(stage::COMPONENT_MOUNT, 3);
        assert_eq!(stage::FIRST_ATTESTATION, 4);
    }

    #[test]
    fn test_signature_constants() {
        // ML-DSA-65 (NIST FIPS 204) signature size
        assert_eq!(ML_DSA_65_SIGNATURE_SIZE, 3309);
        assert_eq!(ML_DSA_65_PUBLIC_KEY_SIZE, 1952);
    }
}
