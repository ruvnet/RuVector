//! Public publish (`POST …/{version}:publish`, `ruvector:publish` ∩ owner).
//!
//! The public key `public/blobs/sha256/{hex}` is global and shared by every
//! tenant that publishes the same bytes, so no per-scope index can know
//! whether it exists, and nobody may overwrite it. The contract:
//!
//! 1. [`Registry::publish_plan`] returns `from`, `to` and the expected
//!    digest and size.
//! 2. The Worker writes `to` **create-only** (R2 conditional put that fails
//!    if the key exists), streaming `from` and hashing the bytes it writes;
//!    if `to` already exists it streams `to` back and hashes that instead.
//!    Either way it never overwrites an existing public object.
//! 3. [`Registry::publish_commit`] takes the measured [`ObjectEvidence`] of
//!    the object now at `to` and refuses unless it equals the version's
//!    SHA-256 and size. A public object with other bytes (a poisoned or
//!    stale key) therefore can never back a public version.

use super::{k_pkg, target, version_blob, PackageRecord, Registry};
use crate::authz::{authorize, Action, Caller};
use crate::error::{RegistryError, Result};
use crate::keys::{blob_key, BlobKey};
use crate::manifest::{PackageManifest, Visibility};
use crate::name::PackageName;
use crate::ports::{Clock, EntropySource, KvStore, StoreError};
use crate::upload::ObjectEvidence;
use crate::version::Version;

/// What the Worker must do before [`Registry::publish_commit`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishPlan {
    /// Current (tenant) blob.
    pub from: BlobKey,
    /// Public blob: write create-only, never overwrite.
    pub to: BlobKey,
    /// The bytes at `to` must hash to this.
    pub sha256: [u8; 32],
    /// And have this size.
    pub size: u64,
    /// The version is already public: nothing to copy.
    pub already_public: bool,
}

/// Result of a public publish.
#[derive(Debug, Clone, PartialEq)]
pub struct PublishOutcome {
    /// Updated manifest.
    pub manifest: PackageManifest,
    /// The tenant blob no version references any more, queued for
    /// [`Registry::sweep`] (never delete it directly).
    pub queued_for_gc: Option<BlobKey>,
}

impl<S: KvStore, C: Clock, E: EntropySource> Registry<S, C, E> {
    fn publishable(
        &self,
        caller: &Caller,
        name: &PackageName,
        version: &Version,
    ) -> Result<PackageManifest> {
        let m = self.version_record(name, version)?;
        authorize(caller, Action::Publish, &target(&m))?;
        if m.yanked.is_some() {
            return Err(RegistryError::InvalidRequest(
                "a yanked version cannot be published",
            ));
        }
        Ok(m)
    }

    /// Plan a public publish.
    pub fn publish_plan(
        &self,
        caller: &Caller,
        name: &PackageName,
        version: &Version,
    ) -> Result<PublishPlan> {
        let m = self.publishable(caller, name, version)?;
        Ok(PublishPlan {
            from: version_blob(&m),
            to: blob_key(&m.owner, name.scope(), Visibility::Public, &m.sha256),
            sha256: m.sha256,
            size: m.total_size,
            already_public: m.visibility == Visibility::Public,
        })
    }

    /// Make a version public, given the Worker's measurement of the public
    /// object. Idempotent. Visibility only widens (yank instead).
    pub fn publish_commit(
        &self,
        caller: &Caller,
        name: &PackageName,
        version: &Version,
        evidence: ObjectEvidence,
    ) -> Result<PublishOutcome> {
        let mut m = self.publishable(caller, name, version)?;
        if m.visibility == Visibility::Public {
            return Ok(PublishOutcome {
                manifest: m,
                queued_for_gc: None,
            });
        }
        if evidence.sha256 != m.sha256 || evidence.size != m.total_size {
            return Err(RegistryError::EvidenceMismatch);
        }
        let old = version_blob(&m);
        let mut pkg: PackageRecord = self
            .load(&k_pkg(name))?
            .ok_or(StoreError::Corrupt("package record"))?;
        pkg.public_versions = pkg.public_versions.saturating_add(1);
        m.visibility = Visibility::Public;
        self.write_version(&m)?;
        self.save(&k_pkg(name), &pkg)?;
        let queued_for_gc = self.release_blob(&old)?;
        Ok(PublishOutcome {
            manifest: m,
            queued_for_gc,
        })
    }
}
