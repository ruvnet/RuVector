//! Storage keys (ADR-351 §6.3). Every component is server-derived or comes
//! out of a strict parser ([`TenantKey`], [`Scope`], [`PackageName`],
//! [`Version`], [`UploadId`]), so none is attacker-shaped.
//!
//! **Blobs are content-addressed** by the object's SHA-256:
//!
//! | Key | Holds | Dedupe domain |
//! |---|---|---|
//! | `rvf/{tenant_key}/{scope}/blobs/sha256/{hex}` | private and tenant-visible objects | one scope = one registry DO, whose `blob/` refcounts are authoritative for exactly this domain (a cross-tenant "already present" would be an existence oracle) |
//! | `public/blobs/sha256/{hex}` | public objects | global and **write-once**: a public version is never deleted, so these objects are never released, and every publish re-proves the bytes ([`crate::registry::PublishPlan`]) |
//! | `staging/{tenant_key}/{upload_id}` | the in-flight upload (R2 multipart) | — |
//!
//! The ADR's per-version paths (`rvf/{tenant_key}/{name}/{version}`,
//! `public/{publisher_tenant_key}/{name}/{version}`) hold the manifest JSON
//! ([`manifest_key`]) as a designed listable R2 mirror, NOT yet written by the
//! gateway (the index is the `RegistryScope` DO); bytes live once per domain.

use crate::manifest::Visibility;
use crate::name::{PackageName, Scope};
use crate::upload::UploadId;
use crate::version::Version;
use core::fmt;
use ruvector_edge_tenancy::TenantKey;
use sha2::{Digest, Sha256};

/// An R2 object key.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlobKey(String);

impl BlobKey {
    /// The key text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl BlobKey {
    /// Rebuild a key this crate wrote into its own index.
    pub(crate) fn from_index(s: &str) -> Self {
        BlobKey(s.to_string())
    }
}

impl fmt::Display for BlobKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// The blob key for an object in `scope`, owned by `owner`, with
/// `visibility`.
pub fn blob_key(
    owner: &TenantKey,
    scope: &Scope,
    visibility: Visibility,
    sha256: &[u8; 32],
) -> BlobKey {
    let hex = hex::encode(sha256);
    match visibility {
        Visibility::Public => BlobKey(format!("public/blobs/sha256/{hex}")),
        Visibility::Private | Visibility::Tenant => BlobKey(format!(
            "rvf/{}/{}/blobs/sha256/{hex}",
            owner.as_str(),
            scope.as_str()
        )),
    }
}

/// Staging object of an upload session.
pub fn staging_key(tenant: &TenantKey, upload: &UploadId) -> BlobKey {
    BlobKey(format!("staging/{}/{}", tenant.as_str(), upload.as_str()))
}

/// Manifest mirror path of a version (ADR §6.3 layout).
pub fn manifest_key(
    owner: &TenantKey,
    visibility: Visibility,
    name: &PackageName,
    version: &Version,
) -> BlobKey {
    let root = match visibility {
        Visibility::Public => "public",
        Visibility::Private | Visibility::Tenant => "rvf",
    };
    BlobKey(format!(
        "{root}/{}/{name}/{version}/manifest.json",
        owner.as_str()
    ))
}

/// Durable Object name of the registry index shard for `scope`:
/// `hex(sha256("v1|registry|" + scope))`. One index per scope keeps every
/// name's versions, scope claim and listings in one single-threaded object.
pub fn registry_do_name(scope: &Scope) -> String {
    hex::encode(Sha256::digest(format!("v1|registry|{}", scope.as_str())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tk(c: char) -> TenantKey {
        TenantKey::parse(&c.to_string().repeat(26)).unwrap()
    }

    #[test]
    fn layout_is_scope_scoped_for_private_and_global_for_public() {
        let sha = [0xab; 32];
        let (a, b) = (tk('a'), tk('b'));
        let (s1, s2) = (Scope::parse("acme").unwrap(), Scope::parse("beta").unwrap());
        let pa = blob_key(&a, &s1, Visibility::Private, &sha);
        assert_eq!(
            pa.as_str(),
            format!("rvf/{}/acme/blobs/sha256/{}", a.as_str(), "ab".repeat(32))
        );
        assert_eq!(pa, blob_key(&a, &s1, Visibility::Tenant, &sha));
        assert_ne!(pa, blob_key(&b, &s1, Visibility::Private, &sha));
        assert_ne!(pa, blob_key(&a, &s2, Visibility::Private, &sha));
        assert_eq!(
            blob_key(&a, &s1, Visibility::Public, &sha),
            blob_key(&b, &s2, Visibility::Public, &sha)
        );
        let n = PackageName::parse("@acme/x").unwrap();
        let v = Version::parse("1.0.0").unwrap();
        assert_eq!(
            manifest_key(&a, Visibility::Public, &n, &v).as_str(),
            format!("public/{}/@acme/x/1.0.0/manifest.json", a.as_str())
        );
    }

    #[test]
    fn registry_do_name_is_pinned() {
        let s = Scope::parse("acme").unwrap();
        assert_eq!(registry_do_name(&s).len(), 64);
        assert_eq!(
            registry_do_name(&s),
            hex::encode(Sha256::digest(b"v1|registry|acme"))
        );
    }
}
