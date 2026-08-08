//! RVForge registry: a file-backed, content-addressed release registry
//! (ADR-294, `docs/research/rvf-forge/registry-model.md`).
//!
//! This crate is the MVP registry: the object model, the publication rules, the
//! trust ladder, revocation, and the append-only Merkle transparency log. The
//! hosted registry exposes the same objects over `/v1`; this one puts them in a
//! directory.
//!
//! # What a release has to survive to get published
//!
//! ```text
//! trustLevel is `published`   the publisher may not claim evidence
//! publisher signature         verifies against a live key in the PublisherRecord
//! rvfIdentity                 present and well-formed
//! capabilityManifest          resolves in the object store
//! predecessor                 null, or a published release of the same package
//! ```
//!
//! Each failure has its own error code, so a CLI can name the rule that broke.
//!
//! # Three invariants that shape the API
//!
//! - **Revocation never deletes.** [`Registry::revoke`] writes a
//!   [`Revocation`] and logs it. The revoked release stays readable through
//!   [`Registry::store`] forever; only [`Registry::is_executable`] changes its
//!   answer. Remote deletion of a user's local property is not a control this
//!   platform holds (ADR-294 §9).
//! - **Trust is raised by the registry, never claimed by the publisher.** A
//!   release may only be submitted at [`TrustLevel::Published`]; every higher
//!   level needs a registry-signed [`TrustUpdate`] (ADR-294 §7).
//! - **Absence from the log means unpublished.** [`Registry::is_published`]
//!   consults the transparency log, not the object store, so an object written
//!   to disk out of band is not a published release.
//!
//! # Example
//!
//! ```
//! use rvforge_registry::{
//!     objects::{SignatureRole, TrustLevel},
//!     testkit, Registry,
//! };
//! # fn main() -> Result<(), rvforge_registry::RegistryError> {
//! let dir = tempfile::tempdir().unwrap();
//! let registry = Registry::open(dir.path())?;
//! let now = "2026-08-03T00:00:00Z";
//!
//! let publisher_key = testkit::signing_key(1);
//! let publisher = registry.put_publisher(
//!     testkit::publisher_record("Example Inc", &[&publisher_key]),
//!     now,
//! )?;
//! let manifest = registry.put_capability_manifest(testkit::capability_manifest())?;
//!
//! let release = testkit::sign(
//!     testkit::release(&publisher.publisher_id, "example-agent", "1.0.0", &manifest.manifest_id),
//!     &publisher_key,
//!     SignatureRole::Publisher,
//! )?;
//! let published = registry.publish_release(release, now)?;
//!
//! assert!(registry.is_executable(&published.release.release_id)?);
//! assert_eq!(
//!     registry.effective_trust_level(&published.release.release_id)?,
//!     TrustLevel::Published,
//! );
//!
//! // The log entry is provable against the current tree head.
//! let proof = registry.inclusion_proof(&published.release.release_id)?;
//! assert!(proof.verify(&registry.tree_head()?.root_hash));
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod crypto;
pub mod error;
pub mod log;
pub mod merkle;
pub mod objects;
pub mod registry;
pub mod store;
pub mod testkit;

pub use crypto::{key_id_for, Signer};
pub use error::{RegistryError, RegistryErrorCode, Result};
pub use log::{InclusionProof, TransparencyLog};
pub use objects::{
    compute_id, CapabilityManifest, PublisherRecord, RegistryObject, Release, Revocation,
    Signature, SignatureRole, TransparencyLogEntry, TreeHead, TrustLevel, TrustUpdate,
    WitnessReceipt, SCHEMA_VERSION,
};
pub use registry::{
    ExecutionStatus, PackageIndexEntry, PublishOutcome, ReceiptRequest, Registry, RegistryKey,
    RevokeOutcome, TrustOutcome,
};
pub use store::ObjectStore;

/// The version of this crate.
pub const REGISTRY_VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_object_type_string_is_distinct() {
        let types = [
            objects::TYPE_PUBLISHER,
            objects::TYPE_RELEASE,
            objects::TYPE_CAPABILITY_MANIFEST,
            objects::TYPE_WITNESS_RECEIPT,
            objects::TYPE_REVOCATION,
            objects::TYPE_TRUST_UPDATE,
            objects::TYPE_LOG_ENTRY,
            objects::TYPE_TREE_HEAD,
        ];
        let unique: std::collections::BTreeSet<_> = types.iter().collect();
        assert_eq!(unique.len(), types.len());
    }

    #[test]
    fn trust_levels_are_ordered_as_a_ladder() {
        assert!(TrustLevel::Published < TrustLevel::Tested);
        assert!(TrustLevel::Tested < TrustLevel::Reviewed);
        assert!(TrustLevel::Reviewed < TrustLevel::EnterpriseApproved);
        assert_eq!(TrustLevel::INITIAL, TrustLevel::Published);
    }
}
