//! The compact per-version index. A full [`PackageManifest`] can be close
//! to a megabyte (one [`crate::validate::SegmentEntry`] per segment, up to
//! `max_segments`), so resolution and listings read only [`VersionIndex`]
//! rows (a few hundred bytes each) and load a full manifest only for the
//! one version being returned.

use super::{k_ver, Registry};
use crate::authz::{authorize, Action, Caller, Target};
use crate::error::{RegistryError, Result};
use crate::manifest::{tenant_key_serde, PackageManifest, Visibility};
use crate::name::PackageName;
use crate::ports::{Clock, EntropySource, KvStore, StoreError};
use crate::version::Version;
use ruvector_edge_tenancy::TenantKey;
use serde::{Deserialize, Serialize};

/// One version's compact row: everything authorization, resolution and
/// listings need.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionIndex {
    /// Version.
    pub version: Version,
    /// Current visibility.
    pub visibility: Visibility,
    /// Yanked.
    pub yanked: bool,
    /// Owning tenant.
    #[serde(with = "tenant_key_serde")]
    pub owner: TenantKey,
    /// Uploader.
    pub created_by: String,
    /// Object SHA-256 (hex).
    pub sha256: String,
    /// Object size.
    pub total_size: u64,
    /// Finalize time.
    pub created_at: u64,
}

impl From<&PackageManifest> for VersionIndex {
    fn from(m: &PackageManifest) -> Self {
        VersionIndex {
            version: m.version.clone(),
            visibility: m.visibility,
            yanked: m.yanked.is_some(),
            owner: m.owner.clone(),
            created_by: m.created_by.clone(),
            sha256: m.sha256_hex(),
            total_size: m.total_size,
            created_at: m.created_at,
        }
    }
}

impl VersionIndex {
    fn target(&self) -> Target<'_> {
        Target {
            owner: &self.owner,
            created_by: &self.created_by,
            visibility: self.visibility,
        }
    }
}

pub(crate) fn k_vidx_prefix(n: &PackageName) -> String {
    format!("vidx/{}/{}/", n.scope().as_str(), n.name())
}

pub(crate) fn k_vidx(n: &PackageName, v: &Version) -> String {
    format!("{}{}", k_vidx_prefix(n), v)
}

impl<S: KvStore, C: Clock, E: EntropySource> Registry<S, C, E> {
    /// Write a version's full record and its index row together.
    pub(crate) fn write_version(&self, m: &PackageManifest) -> Result<()> {
        self.save(&k_ver(&m.name, &m.version), m)?;
        self.save(&k_vidx(&m.name, &m.version), &VersionIndex::from(m))
    }

    /// Index rows of every version of `name` the caller may see
    /// (unsorted). `NotFound` if none; `Forbidden(Read)` if some are visible
    /// but the caller lacks read.
    pub(crate) fn visible_versions(
        &self,
        caller: &Caller,
        name: &PackageName,
    ) -> Result<Vec<VersionIndex>> {
        let cap = self.config.max_versions_per_package as usize + 1;
        let rows = self.store.list(&k_vidx_prefix(name), None, cap)?;
        let mut out = Vec::new();
        let mut denied = None;
        for (_, bytes) in rows {
            let r: VersionIndex =
                serde_json::from_slice(&bytes).map_err(|_| StoreError::Corrupt("version index"))?;
            match authorize(caller, Action::Pull, &r.target()) {
                Ok(()) => out.push(r),
                Err(crate::authz::Denial::NotFound) => {}
                Err(d) => denied = Some(d),
            }
        }
        match (out.is_empty(), denied) {
            (true, Some(d)) => Err(d.into()),
            (true, None) => Err(RegistryError::NotFound),
            _ => Ok(out),
        }
    }
}
