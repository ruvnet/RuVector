//! The registry service over the ports: scope adoption, upload flow
//! (`flow`), reads, yank / unyank, public publish (`publish`), listings
//! (`list`), blob reference counts (`blobs`) and maintenance (`sweep`).
//!
//! Index layout in the [`KvStore`] (one registry DO per scope; blob keys
//! carry the scope, so this index is authoritative for its blobs):
//!
//! | Key | Value |
//! |---|---|
//! | `scope/{scope}` | [`ScopeRecord`]: owning tenant, adopted from the scope directory |
//! | `pkg/{scope}/{name}` | [`PackageRecord`]: owner, version counts |
//! | `ver/{scope}/{name}/{version}` | [`PackageManifest`] (immutable content; read one at a time) |
//! | `vidx/{scope}/{name}/{version}` | [`VersionIndex`]: compact row for resolution and listings |
//! | `upload/{upload_id}` | [`crate::upload::UploadSession`] |
//! | `blob/{blob_key}` | [`BlobRef`]: references (versions + pinned uploads) and whether the bytes are stored |
//! | `gc/{blob_key}` | a released blob awaiting [`Registry::sweep`] |
//!
//! Each operation validates and authorizes fully before its first write, and
//! issues its writes back to back (a DO coalesces them into one commit).

mod blobs;
mod flow;
mod index;
mod list;
mod publish;
mod sweep;

pub use blobs::BlobRef;
pub use flow::{FinalizePlan, FinalizeReport};
pub use index::VersionIndex;
pub use list::{PackageSummary, Page, PageRequest, VersionSummary};
pub use publish::{PublishOutcome, PublishPlan};
pub use sweep::SweepReport;

use crate::authz::{authorize, Action, Caller, Target};
use crate::error::{RegistryError, Result};
use crate::keys::{blob_key, BlobKey};
use crate::manifest::{tenant_key_serde, PackageManifest, Yank, MAX_YANK_REASON};
use crate::name::{PackageName, Scope};
use crate::ports::{Clock, EntropySource, KvStore, StoreError};
use crate::scopes::ScopeClaim;
use crate::upload::UploadLimits;
use crate::validate::ValidationLimits;
use crate::version::Version;
use crate::view::ManifestView;
use ruvector_edge_auth::Capability;
use ruvector_edge_tenancy::TenantKey;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// Service configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegistryConfig {
    /// Upload session limits.
    pub upload: UploadLimits,
    /// Validator limits.
    pub validation: ValidationLimits,
    /// Most versions per package.
    pub max_versions_per_package: u32,
    /// Largest page.
    pub max_page: usize,
    /// Most index entries scanned by one listing call.
    pub list_scan_budget: usize,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        RegistryConfig {
            upload: UploadLimits::default(),
            validation: ValidationLimits::default(),
            max_versions_per_package: 1000,
            max_page: 100,
            list_scan_budget: 1000,
        }
    }
}

/// Scope ownership, as adopted from the scope directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopeRecord {
    /// Owning tenant.
    #[serde(with = "tenant_key_serde")]
    pub owner: TenantKey,
    /// Who claimed it.
    pub claimed_by: String,
    /// When (seconds).
    pub claimed_at: u64,
}

/// Per-package bookkeeping.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackageRecord {
    /// `@scope/name`.
    pub name: PackageName,
    /// Owning tenant.
    #[serde(with = "tenant_key_serde")]
    pub owner: TenantKey,
    /// First version's finalize time.
    pub created_at: u64,
    /// Versions ever finalized.
    pub versions: u32,
    /// Versions made public.
    pub public_versions: u32,
}

/// Where a pull reads from.
#[derive(Debug, Clone, PartialEq)]
pub struct PullTicket {
    /// The manifest as this caller may see it (check `yanked` before
    /// installing).
    pub manifest: ManifestView,
    /// R2 key of the bytes.
    pub blob: BlobKey,
}

/// The registry service.
pub struct Registry<S, C, E> {
    store: S,
    clock: C,
    entropy: E,
    config: RegistryConfig,
}

pub(crate) fn k_scope(s: &Scope) -> String {
    format!("scope/{}", s.as_str())
}
pub(crate) fn k_pkg(n: &PackageName) -> String {
    format!("pkg/{}/{}", n.scope().as_str(), n.name())
}
pub(crate) fn k_ver(n: &PackageName, v: &Version) -> String {
    format!("ver/{}/{}/{}", n.scope().as_str(), n.name(), v)
}

pub(crate) fn target(m: &PackageManifest) -> Target<'_> {
    Target {
        owner: &m.owner,
        created_by: &m.created_by,
        visibility: m.visibility,
    }
}

/// The blob a version's bytes live in.
pub(crate) fn version_blob(m: &PackageManifest) -> BlobKey {
    blob_key(&m.owner, m.name.scope(), m.visibility, &m.sha256)
}

impl<S: KvStore, C: Clock, E: EntropySource> Registry<S, C, E> {
    /// A service over the given ports.
    pub fn new(store: S, clock: C, entropy: E, config: RegistryConfig) -> Self {
        Registry {
            store,
            clock,
            entropy,
            config,
        }
    }

    /// The configuration.
    pub fn config(&self) -> &RegistryConfig {
        &self.config
    }

    /// The store (tests, maintenance).
    pub fn store(&self) -> &S {
        &self.store
    }

    fn now(&self) -> u64 {
        self.clock.now_unix()
    }

    pub(crate) fn load<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        match self.store.get(key)? {
            None => Ok(None),
            Some(bytes) => serde_json::from_slice(&bytes)
                .map(Some)
                .map_err(|_| StoreError::Corrupt("registry record").into()),
        }
    }

    pub(crate) fn save<T: Serialize>(&self, key: &str, value: &T) -> Result<()> {
        let bytes = serde_json::to_vec(value).map_err(|_| StoreError::Corrupt("encode"))?;
        Ok(self.store.put(key, &bytes)?)
    }

    fn version_record(&self, name: &PackageName, version: &Version) -> Result<PackageManifest> {
        self.load(&k_ver(name, version))?
            .ok_or(RegistryError::NotFound)
    }

    /// Record a directory claim in this scope's DO (the gateway calls this
    /// right after [`crate::scopes::ScopeDirectory::claim`]). Idempotent for
    /// the same owner; a different owner is `ScopeTaken`.
    pub fn adopt_scope(&self, claim: &ScopeClaim) -> Result<ScopeRecord> {
        let scope = claim.scope()?;
        if let Some(existing) = self.scope_record(&scope)? {
            if existing.owner != claim.owner {
                return Err(RegistryError::ScopeTaken);
            }
            return Ok(existing);
        }
        let rec = ScopeRecord {
            owner: claim.owner.clone(),
            claimed_by: claim.claimed_by.clone(),
            claimed_at: claim.claimed_at,
        };
        self.save(&k_scope(&scope), &rec)?;
        Ok(rec)
    }

    /// The scope's owner record, if adopted.
    pub fn scope_record(&self, scope: &Scope) -> Result<Option<ScopeRecord>> {
        self.load(&k_scope(scope))
    }

    /// The manifest of one version (`GET /v1/rvf/{name}/{version}`), as the
    /// caller may see it.
    pub fn get(
        &self,
        caller: &Caller,
        name: &PackageName,
        version: &Version,
    ) -> Result<ManifestView> {
        let m = self.version_record(name, version)?;
        authorize(caller, Action::Pull, &target(&m))?;
        Ok(ManifestView::for_caller(&m, caller))
    }

    /// Where to stream the bytes from (`GET …/{version}:pull`). Yanked
    /// versions remain pullable by exact version.
    pub fn pull(
        &self,
        caller: &Caller,
        name: &PackageName,
        version: &Version,
    ) -> Result<PullTicket> {
        let m = self.version_record(name, version)?;
        authorize(caller, Action::Pull, &target(&m))?;
        Ok(PullTicket {
            blob: version_blob(&m),
            manifest: ManifestView::for_caller(&m, caller),
        })
    }

    /// Highest visible, non-yanked version (pre-releases only if asked).
    /// Resolution reads only the compact index rows, then loads the winner.
    pub fn resolve_latest(
        &self,
        caller: &Caller,
        name: &PackageName,
        include_prerelease: bool,
    ) -> Result<ManifestView> {
        let best = self
            .visible_versions(caller, name)?
            .into_iter()
            .filter(|r| !r.yanked && (include_prerelease || !r.version.is_prerelease()))
            .max_by(|a, b| a.version.cmp(&b.version))
            .ok_or(RegistryError::NotFound)?;
        self.get(caller, name, &best.version)
    }

    /// Yank a version. Idempotent: an existing yank is kept as-is.
    pub fn yank(
        &self,
        caller: &Caller,
        name: &PackageName,
        version: &Version,
        reason: &str,
    ) -> Result<PackageManifest> {
        let mut m = self.version_record(name, version)?;
        authorize(caller, Action::Yank, &target(&m))?;
        if reason.len() > MAX_YANK_REASON || reason.chars().any(char::is_control) {
            return Err(RegistryError::InvalidRequest(
                "yank reason too long or has control characters",
            ));
        }
        if m.yanked.is_none() {
            m.yanked = Some(Yank {
                at: self.now(),
                by: caller.sub.clone(),
                reason: reason.to_string(),
            });
            self.write_version(&m)?;
        }
        Ok(m)
    }

    /// Clear a yank. Only whoever set it, or a tenant admin, may clear it:
    /// an uploader cannot undo an admin's yank. Content is untouched.
    pub fn unyank(
        &self,
        caller: &Caller,
        name: &PackageName,
        version: &Version,
    ) -> Result<PackageManifest> {
        let mut m = self.version_record(name, version)?;
        authorize(caller, Action::Yank, &target(&m))?;
        if let Some(y) = &m.yanked {
            if y.by != caller.sub && !caller.caps.contains(Capability::Admin) {
                return Err(RegistryError::NotOwner);
            }
        }
        if m.yanked.take().is_some() {
            self.write_version(&m)?;
        }
        Ok(m)
    }
}
