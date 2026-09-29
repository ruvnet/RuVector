//! What a read returns to one caller.
//!
//! A public version is readable by every tenant, but the identities in its
//! manifest are not public: `owner` (the tenant key used in storage paths),
//! `created_by` and `yanked.by` (edge subjects, which are stable across
//! every tenant a person belongs to and would let a foreign tenant link one
//! user's activity). Callers outside the owning tenant get only
//! [`PublicManifest`]; the owning tenant also gets [`OwnerFields`]. The
//! publisher is identified by the package scope, which is already public.

use crate::authz::Caller;
use crate::hexser;
use crate::manifest::{tenant_key_serde, PackageManifest, Provenance, Visibility};
use crate::name::PackageName;
use crate::validate::SegmentEntry;
use crate::version::Version;
use core::ops::Deref;
use ruvector_edge_store::Metric;
use ruvector_edge_tenancy::TenantKey;
use serde::{Deserialize, Serialize};

/// A yank as shown to everyone: when and why, not who.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicYank {
    /// When, seconds since the epoch.
    pub at: u64,
    /// Reason.
    pub reason: String,
}

/// The content of a version, with no tenant or subject identifiers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PublicManifest {
    /// `@scope/name` (the scope is the publisher handle).
    pub name: PackageName,
    /// Version.
    pub version: Version,
    /// Vector dimension.
    pub dim: u16,
    /// Distance metric.
    pub metric: Metric,
    /// Live vectors per the RVF manifest.
    pub total_vectors: u64,
    /// Every segment with its SHA-256.
    pub segments: Vec<SegmentEntry>,
    /// Object size in bytes.
    pub total_size: u64,
    /// SHA-256 of the whole object.
    #[serde(with = "hexser")]
    pub sha256: [u8; 32],
    /// Finalize time.
    pub created_at: u64,
    /// Current visibility.
    pub visibility: Visibility,
    /// Optional witness root.
    pub provenance: Option<Provenance>,
    /// Set while yanked.
    pub yanked: Option<PublicYank>,
}

impl From<&PackageManifest> for PublicManifest {
    fn from(m: &PackageManifest) -> Self {
        PublicManifest {
            name: m.name.clone(),
            version: m.version.clone(),
            dim: m.dim,
            metric: m.metric,
            total_vectors: m.total_vectors,
            segments: m.segments.clone(),
            total_size: m.total_size,
            sha256: m.sha256,
            created_at: m.created_at,
            visibility: m.visibility,
            provenance: m.provenance.clone(),
            yanked: m.yanked.as_ref().map(|y| PublicYank {
                at: y.at,
                reason: y.reason.clone(),
            }),
        }
    }
}

/// Identity fields shown only inside the owning tenant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OwnerFields {
    /// Owning tenant.
    #[serde(with = "tenant_key_serde")]
    pub owner: TenantKey,
    /// Uploader.
    pub created_by: String,
    /// Who yanked it, while yanked.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub yanked_by: Option<String>,
}

/// A manifest as one caller may see it. Dereferences to the content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManifestView {
    /// Content, visible to anyone who may pull.
    #[serde(flatten)]
    pub content: PublicManifest,
    /// Present only for callers in the owning tenant.
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub owner_fields: Option<OwnerFields>,
}

impl ManifestView {
    /// Project `m` for `caller`.
    pub fn for_caller(m: &PackageManifest, caller: &Caller) -> Self {
        let owner_fields = (caller.tenant == m.owner).then(|| OwnerFields {
            owner: m.owner.clone(),
            created_by: m.created_by.clone(),
            yanked_by: m.yanked.as_ref().map(|y| y.by.clone()),
        });
        ManifestView {
            content: PublicManifest::from(m),
            owner_fields,
        }
    }
}

impl Deref for ManifestView {
    type Target = PublicManifest;
    fn deref(&self) -> &PublicManifest {
        &self.content
    }
}
