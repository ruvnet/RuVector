//! Federated RVF transfer learning.
//!
//! This crate implements the federation protocol described in ADR-057:
//! - **PII stripping**: Three-stage pipeline (detect, redact, attest)
//! - **Differential privacy**: Gaussian/Laplace noise, RDP accountant, gradient clipping
//! - **Federation protocol**: Export builder, import merger, version-aware conflict resolution
//! - **Federated aggregation**: FedAvg, FedProx, Byzantine-tolerant weighted averaging
//! - **Segment types**: FederatedManifest, DiffPrivacyProof, RedactionLog, AggregateWeights

pub mod aggregate;
pub mod diff_privacy;
pub mod error;
pub mod federation;
pub mod pii_strip;
pub mod policy;
pub mod types;

pub use aggregate::{AggregationStrategy, FederatedAggregator};
pub use diff_privacy::{DiffPrivacyEngine, PrivacyAccountant};
pub use error::FederationError;
pub use federation::{ExportBuilder, ImportMerger};
pub use pii_strip::PiiStripper;
pub use policy::FederationPolicy;
pub use types::*;
