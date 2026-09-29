//! RFC 8707 resource allowlist with a **per-resource scope table**
//! (ADR-351 §5.3 grant rule, §5.7, §16.1). The minted `aud` is always one of
//! these canonical URLs, and the minted `scope` is always a subset of that
//! resource's scopes, e.g. `ruvector:admin` exists for
//! `https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev/v1`
//! but never for `…/v1/mcp`. Every resource — including an adapter resource
//! such as `https://team.ruv.io/mcp` — draws only on the §5.3 vocabulary
//! (`ruvector:*` + `offline_access`): the edge AS mints nothing else, and a
//! §16.1 exchange keeps `scope ⊆` the subject token's, so an adapter token
//! must carry `ruvector:*` scopes to be exchangeable for a `/v1` token.

use crate::error::{OAuthError, OAuthErrorCode};
use crate::params::{split_scope, strip_identity_scopes};
use crate::refresh::OFFLINE_ACCESS;
use ruvector_edge_auth::scopes::SCOPE_TABLE;
use ruvector_edge_auth::{AuthError, ResourceUrl};

/// Whether `scope` is in the ADR §5.3 vocabulary the edge AS mints:
/// a [`SCOPE_TABLE`] `ruvector:*` scope or `offline_access`.
pub fn is_edge_vocabulary(scope: &str) -> bool {
    scope == OFFLINE_ACCESS || SCOPE_TABLE.iter().any(|(s, _)| *s == scope)
}

/// Maximum scopes one resource may declare.
pub const MAX_RESOURCE_SCOPES: usize = 16;

/// One allowlisted resource and the scopes a token for it may carry. The
/// **first** scope is the default grant of an `/authorize` request that
/// omits `scope` (ADR §5.3: `ruvector:read` for the gateway resources).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceEntry {
    url: ResourceUrl,
    scopes: Vec<String>,
}

impl ResourceEntry {
    /// Build an entry.
    ///
    /// Contract: 1..=[`MAX_RESOURCE_SCOPES`] distinct RFC 6749 scope tokens,
    /// each in the §5.3 vocabulary ([`is_edge_vocabulary`]; e.g. `team:read`
    /// is refused); the first (the default grant) is not `offline_access`.
    /// Anything else is [`AuthError::InvalidConfig`].
    pub fn new(url: ResourceUrl, scopes: Vec<String>) -> Result<Self, AuthError> {
        let err = AuthError::InvalidConfig("resource scopes");
        let joined = scopes.join(" ");
        let parsed = split_scope(&joined).map_err(|_| err.clone())?;
        if parsed.len() != scopes.len()
            || scopes.len() > MAX_RESOURCE_SCOPES
            || scopes.first().map_or(true, |s| s == OFFLINE_ACCESS)
            || !scopes.iter().all(|s| is_edge_vocabulary(s))
        {
            return Err(err);
        }
        Ok(ResourceEntry { url, scopes })
    }

    /// The canonical resource URL (the future `aud`).
    pub fn url(&self) -> &ResourceUrl {
        &self.url
    }

    /// Scopes a token for this resource may carry.
    pub fn scopes(&self) -> &[String] {
        &self.scopes
    }

    /// Default grant when `/authorize` omits `scope` (the first scope).
    pub fn default_grant(&self) -> &str {
        &self.scopes[0]
    }

    /// Whether `scope` may be minted for this resource.
    pub fn allows(&self, scope: &str) -> bool {
        self.scopes.iter().any(|s| s == scope)
    }
}

/// Protected resources this AS will mint tokens for.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResourceAllowlist {
    entries: Vec<ResourceEntry>,
}

impl ResourceAllowlist {
    /// Build from entries (a repeated URL keeps the first entry).
    pub fn new(entries: Vec<ResourceEntry>) -> Self {
        let mut out: Vec<ResourceEntry> = Vec::with_capacity(entries.len());
        for e in entries {
            if !out.iter().any(|o| o.url == e.url) {
                out.push(e);
            }
        }
        ResourceAllowlist { entries: out }
    }

    /// Parse the wrangler `RESOURCE_ALLOWLIST` value: comma-separated
    /// entries, each `<canonical URL> <scope> [<scope> ...]` (whitespace
    /// separated; the first scope is the default grant), e.g.
    /// `https://gw/v1 ruvector:read ruvector:write ruvector:admin
    /// offline_access, https://gw/v1/mcp ruvector:read ruvector:write
    /// offline_access`. Any invalid entry, an entry without scopes, or a
    /// repeated URL fails the whole list.
    pub fn from_config(value: &str) -> Result<Self, AuthError> {
        let mut entries: Vec<ResourceEntry> = Vec::new();
        for raw in value.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let mut parts = raw.split_whitespace();
            let url = ResourceUrl::parse(parts.next().unwrap_or_default())?;
            let entry = ResourceEntry::new(url, parts.map(String::from).collect())?;
            if entries.iter().any(|e| e.url == entry.url) {
                return Err(AuthError::InvalidConfig("repeated resource"));
            }
            entries.push(entry);
        }
        Ok(ResourceAllowlist { entries })
    }

    /// All allowed resources with their scopes.
    pub fn entries(&self) -> &[ResourceEntry] {
        &self.entries
    }

    /// The entry for `resource`, if it is (still) allowlisted.
    pub fn get(&self, resource: &ResourceUrl) -> Option<&ResourceEntry> {
        self.entries.iter().find(|e| e.url == *resource)
    }

    /// Ordered union of every resource's scopes: the AS `scopes_supported`
    /// (RFC 8414 metadata and the DCR registrable set). Always ⊆ the §5.3
    /// vocabulary, since [`ResourceEntry::new`] refuses anything else.
    pub fn scopes_supported(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for s in self.entries.iter().flat_map(|e| e.scopes.iter()) {
            if !out.contains(s) {
                out.push(s.clone());
            }
        }
        out
    }

    /// Resolve a requested `resource` parameter to its entry.
    ///
    /// Contract: exactly one `resource` value (the caller rejects repeats);
    /// must parse as a [`ResourceUrl`] and be byte-equal to an allowlisted
    /// entry; otherwise `invalid_target`. A missing parameter is also
    /// `invalid_target` (no default audience).
    pub fn resolve_entry(&self, requested: Option<&str>) -> Result<&ResourceEntry, OAuthError> {
        let err = OAuthError::new(OAuthErrorCode::InvalidTarget, "resource not allowed");
        let requested = requested.ok_or(err.clone())?;
        let parsed = ResourceUrl::parse(requested).map_err(|_| err.clone())?;
        self.get(&parsed).ok_or(err)
    }

    /// [`ResourceAllowlist::resolve_entry`], returning only the URL.
    pub fn resolve(&self, requested: Option<&str>) -> Result<ResourceUrl, OAuthError> {
        self.resolve_entry(requested).map(|e| e.url.clone())
    }

    /// Whether `scope` is part of the vocabulary this AS understands: the
    /// §5.3 set ([`is_edge_vocabulary`], e.g. `ruvector:publish` before M5).
    /// Such scopes are **dropped** when they fall outside a grant; anything
    /// else is `invalid_scope`.
    pub fn is_vocabulary(&self, scope: &str) -> bool {
        is_edge_vocabulary(scope)
    }
}

/// The ADR §5.3 grant rule: `requested ∩ client ceiling ∩ resource scopes`.
///
/// Contract: `requested` is split per RFC 6749 §3.3 (malformed =>
/// `invalid_scope`); identity scopes (`openid profile email`) are dropped;
/// any remaining scope outside [`ResourceAllowlist::is_vocabulary`] =>
/// `invalid_scope`; an omitted (or identity-only) request asks for the
/// resource's [`ResourceEntry::default_grant`] plus `offline_access` when the
/// resource allows it (§5.3 table: both "granted by default"). Vocabulary
/// scopes outside the ceiling or the resource are **dropped**, not refused,
/// keeping request order. A result without any `ruvector:*` scope =>
/// `invalid_scope`.
pub fn grant_scopes(
    requested: Option<&str>,
    ceiling: &[String],
    entry: &ResourceEntry,
    allowlist: &ResourceAllowlist,
) -> Result<Vec<String>, OAuthError> {
    let asked = match requested {
        Some(s) => strip_identity_scopes(split_scope(s)?),
        None => Vec::new(),
    };
    if asked.iter().any(|s| !allowlist.is_vocabulary(s)) {
        return Err(OAuthError::new(
            OAuthErrorCode::InvalidScope,
            "unknown scope requested",
        ));
    }
    let asked = if asked.is_empty() {
        let mut d = vec![entry.default_grant().to_string()];
        if entry.allows(OFFLINE_ACCESS) {
            d.push(OFFLINE_ACCESS.to_string());
        }
        d
    } else {
        asked
    };
    let granted: Vec<String> = asked
        .into_iter()
        .filter(|s| ceiling.contains(s) && entry.allows(s))
        .collect();
    if !granted.iter().any(|s| s.starts_with("ruvector:")) {
        return Err(OAuthError::new(
            OAuthErrorCode::InvalidScope,
            "no requested scope can be granted for this resource",
        ));
    }
    Ok(granted)
}
