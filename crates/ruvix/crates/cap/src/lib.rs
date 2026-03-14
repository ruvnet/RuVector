//! seL4-inspired capability management for the RuVix Cognition Kernel.
//!
//! This crate provides the capability manager that enforces all access control
//! in RuVix. Every kernel object is accessed exclusively through capabilities,
//! following the principle: "No syscall succeeds without an appropriate
//! capability handle."
//!
//! # Core Concepts
//!
//! - **Capability**: An unforgeable kernel-managed token comprising object ID,
//!   type, rights bitmap, badge, and epoch.
//! - **Derivation Tree**: Capabilities can be derived with equal or fewer rights.
//!   Revoking a capability invalidates all derived capabilities.
//! - **Delegation Depth**: Maximum depth of 8 (configurable) to prevent
//!   unbounded delegation chains.
//!
//! # Design Principles (from ADR-087 Section 6)
//!
//! 1. A task can only grant capabilities it holds
//! 2. Granted rights must be equal or fewer than held rights
//! 3. Revocation propagates through the derivation tree
//! 4. GRANT_ONCE provides non-transitive delegation
//! 5. Epoch-based invalidation detects stale handles
//!
//! # Example
//!
//! ```
//! use ruvix_cap::{CapabilityManager, CapManagerConfig};
//! use ruvix_types::{ObjectType, CapRights, TaskHandle};
//!
//! let config = CapManagerConfig::default();
//! let mut manager: CapabilityManager<64> = CapabilityManager::new(config);
//!
//! // Create a root capability for a new vector store
//! let task = TaskHandle::new(1, 0);
//! let cap_handle = manager.create_root_capability(
//!     0x1000,  // object_id
//!     ObjectType::VectorStore,
//!     0,       // badge
//!     task,
//! ).unwrap();
//!
//! // Grant a read-only derived capability
//! let _derived = manager.grant(
//!     cap_handle,
//!     CapRights::READ,
//!     42,  // new badge
//!     task,
//!     TaskHandle::new(2, 0),  // target task
//! ).unwrap();
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

mod audit;
mod boot;
mod derivation;
mod error;
mod grant;
mod manager;
mod optimized;
mod revoke;
mod security;
mod table;

pub use audit::{AuditConfig, AuditEntry, AuditFlags, AuditResult, CapabilityAuditor};
pub use boot::{BootCapabilitySet, InitialCapability};
pub use derivation::{DerivationNode, DerivationTree};
pub use error::{CapError, CapResult};
pub use grant::{can_grant, validate_grant, GrantRequest, GrantResult};
pub use manager::{CapManagerConfig, CapabilityManager, ManagerStats};
pub use revoke::{can_revoke, validate_revoke, RevokeRequest, RevokeResult, RevokeStats};
pub use security::{
    verify_boot_signature_or_panic, verify_signature, BootSignature, BootVerifier,
    SignatureAlgorithm, SignatureVerifyResult, TrustedKey, TrustedKeyStore,
};
pub use table::{CapTableEntry, CapabilityTable};
pub use optimized::{OptimizedCapSlot, OptimizedCapTable};

// Re-export commonly used types from ruvix-types
pub use ruvix_types::{CapHandle, CapRights, Capability, ObjectType, TaskHandle};

/// Default maximum delegation depth (Section 20.2 of ADR-087).
pub const DEFAULT_MAX_DELEGATION_DEPTH: u8 = 8;

/// Maximum capacity of the capability table (per task).
pub const DEFAULT_CAP_TABLE_CAPACITY: usize = 1024;

/// Audit warning threshold for delegation chains (Section 20.2).
/// The audit system flags chains deeper than this value.
pub const AUDIT_DEPTH_WARNING_THRESHOLD: u8 = 4;
