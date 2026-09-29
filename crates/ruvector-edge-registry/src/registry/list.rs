//! Paginated listings (`GET /v1/rvf/{name}` versions, `GET /v1/rvf?prefix=`
//! packages). Cursors are opaque (`c1.` + hex) and bound to the listing that
//! issued them: a cursor from another name or prefix is `InvalidCursor`.
//!
//! Package listing requires a prefix of at least `@scope/`. Package *names*
//! are visible to members of the owning tenant; other tenants see a package
//! only once it has a public version. Version listings filter each version
//! through the pull rule, so a private version never appears to anyone who
//! could not pull it.

use super::{k_pkg, PackageRecord, Registry, VersionIndex};
use crate::authz::Caller;
use crate::error::{RegistryError, Result};
use crate::manifest::Visibility;
use crate::name::{PackageName, Scope};
use crate::ports::{Clock, EntropySource, KvStore, StoreError};
use crate::version::Version;
use ruvector_edge_auth::Capability;
use serde::{Deserialize, Serialize};

/// A page request.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PageRequest {
    /// Cursor from the previous page.
    pub cursor: Option<String>,
    /// Page size (clamped to `1..=max_page`; 0 means the maximum).
    pub limit: usize,
}

/// A page of results.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Page<T> {
    /// Items.
    pub items: Vec<T>,
    /// Present when more items may follow.
    pub next_cursor: Option<String>,
}

/// One row of a version listing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionSummary {
    /// Version.
    pub version: Version,
    /// Visibility.
    pub visibility: Visibility,
    /// Yanked.
    pub yanked: bool,
    /// Object SHA-256 (hex).
    pub sha256: String,
    /// Object size.
    pub total_size: u64,
    /// Finalize time.
    pub created_at: u64,
}

impl From<&VersionIndex> for VersionSummary {
    fn from(r: &VersionIndex) -> Self {
        VersionSummary {
            version: r.version.clone(),
            visibility: r.visibility,
            yanked: r.yanked,
            sha256: r.sha256.clone(),
            total_size: r.total_size,
            created_at: r.created_at,
        }
    }
}

/// One row of a package listing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackageSummary {
    /// `@scope/name`.
    pub name: PackageName,
    /// Versions ever finalized (owner tenant) or public versions (others).
    pub versions: u32,
}

fn encode_cursor(listing: &str, position: &str) -> String {
    format!("c1.{}", hex::encode(format!("{listing}\n{position}")))
}

fn decode_cursor(listing: &str, cursor: &str) -> Result<String> {
    let raw = cursor
        .strip_prefix("c1.")
        .filter(|h| h.len() <= 1024)
        .and_then(|h| hex::decode(h).ok())
        .and_then(|b| String::from_utf8(b).ok())
        .ok_or(RegistryError::InvalidCursor)?;
    let (l, pos) = raw.split_once('\n').ok_or(RegistryError::InvalidCursor)?;
    if l != listing {
        return Err(RegistryError::InvalidCursor);
    }
    Ok(pos.to_string())
}

/// Parse a listing prefix: `@scope/` followed by a (possibly empty) partial
/// name in the name charset.
fn parse_prefix(prefix: &str) -> Result<(Scope, &str)> {
    let rest = prefix
        .strip_prefix('@')
        .ok_or(RegistryError::InvalidRequest(
            "prefix must start with @scope/",
        ))?;
    let (scope, partial) = rest.split_once('/').ok_or(RegistryError::InvalidRequest(
        "prefix must start with @scope/",
    ))?;
    let scope = Scope::parse(scope)?;
    let ok = partial.len() <= crate::name::MAX_NAME_LEN
        && partial.bytes().all(|b| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'.' | b'_' | b'-')
        });
    if !ok {
        return Err(RegistryError::InvalidRequest(
            "prefix has a disallowed character",
        ));
    }
    Ok((scope, partial))
}

impl<S: KvStore, C: Clock, E: EntropySource> Registry<S, C, E> {
    fn page_size(&self, req: &PageRequest) -> usize {
        match req.limit {
            0 => self.config.max_page,
            n => n.min(self.config.max_page),
        }
    }

    /// Versions of `name` the caller may pull, newest first (SemVer order).
    pub fn list_versions(
        &self,
        caller: &Caller,
        name: &PackageName,
        req: &PageRequest,
    ) -> Result<Page<VersionSummary>> {
        let listing = format!("v:{name}");
        let after = match &req.cursor {
            Some(c) => Some(
                Version::parse(&decode_cursor(&listing, c)?)
                    .map_err(|_| RegistryError::InvalidCursor)?,
            ),
            None => None,
        };
        let mut all = self.visible_versions(caller, name)?;
        all.sort_by(|a, b| b.version.cmp(&a.version));
        let limit = self.page_size(req);
        let mut rest = all
            .iter()
            .filter(|m| after.as_ref().is_none_or(|a| m.version < *a))
            .peekable();
        let items: Vec<VersionSummary> = rest
            .by_ref()
            .take(limit)
            .map(VersionSummary::from)
            .collect();
        let next_cursor = match (rest.peek(), items.last()) {
            (Some(_), Some(last)) => Some(encode_cursor(&listing, last.version.as_str())),
            _ => None,
        };
        Ok(Page { items, next_cursor })
    }

    /// Packages under `prefix` (`@scope/` + partial name) the caller may see.
    pub fn list_packages(
        &self,
        caller: &Caller,
        prefix: &str,
        req: &PageRequest,
    ) -> Result<Page<PackageSummary>> {
        if !caller.caps.contains(Capability::Read) {
            return Err(RegistryError::Forbidden(Capability::Read));
        }
        let (scope, partial) = parse_prefix(prefix)?;
        let listing = format!("p:@{}/{partial}", scope.as_str());
        let key_prefix = format!("pkg/{}/{partial}", scope.as_str());
        let mut after = match &req.cursor {
            Some(c) => {
                let k = decode_cursor(&listing, c)?;
                if !k.starts_with(&key_prefix) {
                    return Err(RegistryError::InvalidCursor);
                }
                Some(k)
            }
            None => None,
        };
        let limit = self.page_size(req);
        let mut items = Vec::new();
        let mut scanned = 0usize;
        let mut exhausted = false;
        while items.len() < limit && scanned < self.config.list_scan_budget {
            let want = (limit - items.len()).min(self.config.list_scan_budget - scanned);
            let batch = self.store.list(&key_prefix, after.as_deref(), want)?;
            scanned += batch.len();
            if batch.len() < want {
                exhausted = true;
            }
            for (key, bytes) in &batch {
                let p: PackageRecord = serde_json::from_slice(bytes)
                    .map_err(|_| StoreError::Corrupt("package record"))?;
                debug_assert_eq!(*key, k_pkg(&p.name));
                if p.owner == caller.tenant {
                    items.push(PackageSummary {
                        name: p.name,
                        versions: p.versions,
                    });
                } else if p.public_versions > 0 {
                    items.push(PackageSummary {
                        name: p.name,
                        versions: p.public_versions,
                    });
                }
            }
            after = batch.last().map(|(k, _)| k.clone()).or(after);
            if exhausted {
                break;
            }
        }
        let next_cursor = match (&after, exhausted) {
            (Some(k), false) => Some(encode_cursor(&listing, k)),
            _ => None,
        };
        Ok(Page { items, next_cursor })
    }
}
