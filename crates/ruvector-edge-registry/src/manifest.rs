//! The package manifest: what a pull returns and what the index stores.
//!
//! Content fields (`name` … `provenance`) are fixed when the version is
//! finalized and never change. Only [`Visibility`] (widen-only, via public
//! publish) and [`Yank`] (set / cleared) mutate afterwards.

use crate::hexser;
use crate::name::PackageName;
use crate::validate::{SegmentEntry, ValidatedRvf};
use crate::version::Version;
use ruvector_edge_store::Metric;
use ruvector_edge_tenancy::TenantKey;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Longest witness root accepted as provenance.
pub const MAX_WITNESS_ROOT: usize = 128;
/// Longest yank reason.
pub const MAX_YANK_REASON: usize = 200;

/// Who may see a version.
///
/// - `Private`: the uploader (`created_by`) and tenant members holding
///   `Admin`.
/// - `Tenant`: every member of the owning tenant with `Read`.
/// - `Public`: any authenticated caller with `Read`, from any tenant. Only
///   reachable through public publish (`ruvector:publish` ∩ owner role).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Visibility {
    /// Uploader + admins.
    Private,
    /// Whole tenant.
    Tenant,
    /// Everyone.
    Public,
}

impl Visibility {
    /// Every visibility, for exhaustive tests.
    pub const ALL: [Visibility; 3] = [Visibility::Private, Visibility::Tenant, Visibility::Public];
}

/// Serde for [`TenantKey`] (stored strictly re-parsed).
pub(crate) mod tenant_key_serde {
    use super::*;
    pub fn serialize<S: Serializer>(k: &TenantKey, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(k.as_str())
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<TenantKey, D::Error> {
        let s = String::deserialize(d)?;
        TenantKey::parse(&s).map_err(serde::de::Error::custom)
    }
}

/// Provenance: the M3-style audit/witness chain head at finalize time, as
/// opaque bytes (hex on the wire). The registry never interprets it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// Opaque witness root, at most [`MAX_WITNESS_ROOT`] bytes.
    #[serde(with = "hex_vec")]
    pub witness_root: Vec<u8>,
}

mod hex_vec {
    use super::*;
    pub fn serialize<S: Serializer>(v: &[u8], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&hex::encode(v))
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let s = String::deserialize(d)?;
        let v = hex::decode(s).map_err(serde::de::Error::custom)?;
        if v.len() > MAX_WITNESS_ROOT {
            return Err(serde::de::Error::custom("witness root too long"));
        }
        Ok(v)
    }
}

/// A yank marker. Yanked versions stay pullable by exact version (existing
/// lockfiles keep working) but are skipped by latest-version resolution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Yank {
    /// When, seconds since the epoch.
    pub at: u64,
    /// Actor (edge subject).
    pub by: String,
    /// Free-text reason, at most [`MAX_YANK_REASON`] bytes.
    pub reason: String,
}

/// The manifest of one published version.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackageManifest {
    /// `@scope/name`.
    pub name: PackageName,
    /// Immutable version.
    pub version: Version,
    /// Vector dimension.
    pub dim: u16,
    /// Distance metric.
    pub metric: Metric,
    /// Live vectors per the RVF manifest (before deletions are applied).
    pub total_vectors: u64,
    /// Every segment with its SHA-256.
    pub segments: Vec<SegmentEntry>,
    /// Object size in bytes.
    pub total_size: u64,
    /// SHA-256 of the whole object: its content address.
    #[serde(with = "hexser")]
    pub sha256: [u8; 32],
    /// Owning tenant.
    #[serde(with = "tenant_key_serde")]
    pub owner: TenantKey,
    /// Uploader (edge subject).
    pub created_by: String,
    /// Finalize time, seconds since the epoch.
    pub created_at: u64,
    /// Current visibility.
    pub visibility: Visibility,
    /// Optional witness root.
    pub provenance: Option<Provenance>,
    /// Set while yanked.
    pub yanked: Option<Yank>,
}

impl PackageManifest {
    /// Build a manifest from a validated object. Content fields come only
    /// from the validator, never from the client.
    #[allow(clippy::too_many_arguments)]
    pub fn from_validated(
        name: PackageName,
        version: Version,
        v: &ValidatedRvf,
        owner: TenantKey,
        created_by: String,
        created_at: u64,
        visibility: Visibility,
        provenance: Option<Provenance>,
    ) -> Self {
        PackageManifest {
            name,
            version,
            dim: v.dim,
            metric: v.metric,
            total_vectors: v.total_vectors,
            segments: v.segments.clone(),
            total_size: v.total_size,
            sha256: v.sha256,
            owner,
            created_by,
            created_at,
            visibility,
            provenance,
            yanked: None,
        }
    }

    /// The object's content address as lowercase hex.
    pub fn sha256_hex(&self) -> String {
        hex::encode(self.sha256)
    }

    /// `true` if content fields (everything except visibility and yank)
    /// are identical: the immutability invariant.
    pub fn same_content(&self, other: &PackageManifest) -> bool {
        let strip = |m: &PackageManifest| PackageManifest {
            visibility: Visibility::Private,
            yanked: None,
            ..m.clone()
        };
        strip(self) == strip(other)
    }
}
