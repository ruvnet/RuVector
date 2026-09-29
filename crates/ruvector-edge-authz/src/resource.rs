//! RFC 8707 resource allowlist with **per-resource scope vocabularies**
//! (ADR-351 §5.3 grant rule, §5.7, §16.1). The minted `aud` is always one of
//! these canonical URLs, and the minted `scope` is always a subset of that
//! resource's scopes. Each resource draws on exactly one [`Vocabulary`]
//! (plus `offline_access`): the gateway resources on `ruvector:*`
//! (`ruvector:admin` for `…/v1` only, never `…/v1/mcp`), the RuFlo AI Team
//! adapter resource `https://team.ruv.io/mcp` on `team:*`. The URL → family
//! binding is **compiled** ([`RESOURCE_VOCABULARIES`]) and enforced when the
//! allowlist loads, so the `RESOURCE_ALLOWLIST` var can narrow a resource's
//! scopes but can neither add a resource nor give it another family. A scope
//! of another resource's vocabulary, or one that the resource or the client
//! ceiling does not offer, is dropped (never minted); nothing left but
//! `offline_access` is `invalid_scope`. `scopes_supported` is the union of
//! every resource's scopes.

use crate::error::{OAuthError, OAuthErrorCode};
use crate::params::{split_scope, strip_identity_scopes};
use crate::refresh::OFFLINE_ACCESS;
use ruvector_edge_auth::scopes::SCOPE_TABLE;
use ruvector_edge_auth::{AuthError, ResourceUrl};

/// The RuFlo AI Team adapter vocabulary (ADR-351 §5.3, §16.1). Minted only
/// for a resource that declares it (`https://team.ruv.io/mcp`); the gateway
/// never accepts such a token (its `aud` is never a gateway resource).
pub const TEAM_SCOPES: [&str; 3] = ["team:read", "team:write", "team:run"];

/// A scope vocabulary family (ADR-351 §5.3). Every allowlisted resource
/// declares scopes from exactly one family, plus `offline_access`, which
/// belongs to none and is valid everywhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Vocabulary {
    /// `ruvector:read/write/admin/publish` ([`SCOPE_TABLE`]): the gateway.
    Ruvector,
    /// `team:read/write/run` ([`TEAM_SCOPES`]): the team.ruv.io adapter.
    Team,
}

impl Vocabulary {
    /// Every family the edge AS knows.
    pub const ALL: [Vocabulary; 2] = [Vocabulary::Ruvector, Vocabulary::Team];

    /// Whether `scope` belongs to this family.
    pub fn contains(self, scope: &str) -> bool {
        match self {
            Vocabulary::Ruvector => SCOPE_TABLE.iter().any(|(s, _)| *s == scope),
            Vocabulary::Team => TEAM_SCOPES.contains(&scope),
        }
    }

    /// The family of `scope`; `None` for `offline_access` and unknown scopes.
    pub fn of(scope: &str) -> Option<Vocabulary> {
        Vocabulary::ALL.into_iter().find(|v| v.contains(scope))
    }

    /// The compiled family of canonical resource `url`
    /// ([`RESOURCE_VOCABULARIES`]); `None` for any other URL.
    pub fn for_resource(url: &ResourceUrl) -> Option<Vocabulary> {
        RESOURCE_VOCABULARIES
            .iter()
            .find(|(u, _)| *u == url.as_str())
            .map(|(_, v)| *v)
    }
}

/// The RuFlo AI Team adapter resource (ADR-351 §16.1): `team:*` only.
pub const TEAM_RESOURCE_URL: &str = "https://team.ruv.io/mcp";

/// team.ruv.io's compiled exchange map (ADR-351 §5.6, §16.1): the
/// `ruvector:*` scope an M1 token exchange derives from each `team:*` scope
/// of the subject token (`team:run` and `offline_access` map to nothing).
/// The consent page discloses exactly these entries whenever it shows a
/// matching `team:*` grant, so no family is consented without it.
pub const TEAM_EXCHANGE_MAP: [(&str, &str); 2] = [
    ("team:read", "ruvector:read"),
    ("team:write", "ruvector:write"),
];

/// The `(adapter scope, ruvector scope)` pairs of [`TEAM_EXCHANGE_MAP`]
/// that `granted` scopes for `resource` would carry into a `…/v1` token;
/// empty for every resource without an exchange map.
pub fn exchange_disclosure<'a>(
    resource: &ResourceUrl,
    granted: &'a [String],
) -> Vec<(&'a str, &'static str)> {
    if resource.as_str() != TEAM_RESOURCE_URL {
        return Vec::new();
    }
    granted
        .iter()
        .filter_map(|g| {
            TEAM_EXCHANGE_MAP
                .iter()
                .find(|(t, _)| *t == g.as_str())
                .map(|(_, r)| (g.as_str(), *r))
        })
        .collect()
}

/// Compiled resource → vocabulary bindings (ADR-351 §5.3, §16.1: an adapter
/// resource has its own vocabulary, never `ruvector:*`; the gateway
/// resources never carry an adapter vocabulary). [`ResourceEntry::new`]
/// refuses any URL not listed here and any entry whose scopes are of another
/// family, so a stale or overridden `RESOURCE_ALLOWLIST` fails the load
/// instead of minting e.g. team.ruv.io tokens with `ruvector:admin`. A new
/// resource or adapter is a reviewed code change here.
pub const RESOURCE_VOCABULARIES: [(&str, Vocabulary); 3] = [
    (
        "https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev/v1",
        Vocabulary::Ruvector,
    ),
    (
        "https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev/v1/mcp",
        Vocabulary::Ruvector,
    ),
    (TEAM_RESOURCE_URL, Vocabulary::Team),
];

/// Whether `scope` is edge vocabulary (ADR §5.3): a scope of some
/// [`Vocabulary`] or `offline_access`. Anything else is unknown.
pub fn is_edge_vocabulary(scope: &str) -> bool {
    scope == OFFLINE_ACCESS || Vocabulary::of(scope).is_some()
}

/// Maximum scopes one resource may declare.
pub const MAX_RESOURCE_SCOPES: usize = 16;

/// One allowlisted resource and the scopes a token for it may carry. The
/// **first** scope is the default grant of an `/authorize` request that
/// omits `scope` (ADR §5.3: `ruvector:read` for the gateway resources,
/// `team:read` for team.ruv.io).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceEntry {
    url: ResourceUrl,
    vocabulary: Vocabulary,
    scopes: Vec<String>,
}

impl ResourceEntry {
    /// Build an entry.
    ///
    /// Contract: 1..=[`MAX_RESOURCE_SCOPES`] distinct RFC 6749 scope tokens;
    /// every one except `offline_access` from the **same** [`Vocabulary`]
    /// (unknown scopes such as `mcp:invoke` and mixed entries such as
    /// `ruvector:read team:run` are refused); that family is the one
    /// compiled for `url` ([`RESOURCE_VOCABULARIES`]: an unlisted URL, or
    /// `ruvector:*` for team.ruv.io, or `team:*` for `…/v1`, is refused); the
    /// first (the default grant) is not `offline_access`. Anything else is
    /// [`AuthError::InvalidConfig`].
    pub fn new(url: ResourceUrl, scopes: Vec<String>) -> Result<Self, AuthError> {
        let err = AuthError::InvalidConfig("resource scopes");
        let joined = scopes.join(" ");
        let parsed = split_scope(&joined).map_err(|_| err.clone())?;
        if parsed.len() != scopes.len() || scopes.len() > MAX_RESOURCE_SCOPES {
            return Err(err);
        }
        let vocabulary = match scopes.first() {
            Some(s) if s != OFFLINE_ACCESS => Vocabulary::of(s).ok_or(err.clone())?,
            _ => return Err(err),
        };
        if Vocabulary::for_resource(&url) != Some(vocabulary) {
            return Err(AuthError::InvalidConfig("resource vocabulary"));
        }
        if !scopes
            .iter()
            .all(|s| s == OFFLINE_ACCESS || vocabulary.contains(s))
        {
            return Err(err);
        }
        Ok(ResourceEntry {
            url,
            vocabulary,
            scopes,
        })
    }

    /// The canonical resource URL (the future `aud`).
    pub fn url(&self) -> &ResourceUrl {
        &self.url
    }

    /// The vocabulary family this resource's scopes are drawn from.
    pub fn vocabulary(&self) -> Vocabulary {
        self.vocabulary
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

    /// Whether `scope` is edge vocabulary ([`is_edge_vocabulary`]): some
    /// resource family's scope (e.g. `ruvector:publish` before M5, or
    /// `team:run`) or `offline_access`. Anything else is `invalid_scope`.
    pub fn is_vocabulary(&self, scope: &str) -> bool {
        is_edge_vocabulary(scope)
    }
}

/// The ADR §5.3 grant rule: `requested ∩ client ceiling ∩ resource scopes`.
///
/// Contract: `requested` is split per RFC 6749 §3.3 (malformed =>
/// `invalid_scope`); identity scopes (`openid profile email`) are dropped;
/// any remaining scope outside [`ResourceAllowlist::is_vocabulary`] =>
/// `invalid_scope` ("unknown"). An omitted (or identity-only) request asks
/// for the resource's [`ResourceEntry::default_grant`] plus
/// `offline_access` when the resource allows it (§5.3 table: both "granted
/// by default"). Every other scope the ceiling or the resource does not
/// offer is **dropped**, not refused, keeping request order: this
/// resource's own family outside the ceiling or the resource (e.g.
/// `ruvector:admin` for `…/v1/mcp`, `ruvector:publish` before M5) and any
/// scope of **another resource's** [`Vocabulary`] (e.g. `ruvector:read` for
/// team.ruv.io, `team:read` for `…/v1/mcp`; never in the entry, since
/// [`ResourceEntry::new`] binds each URL to one family), so a client that
/// asks for the AS-metadata union still gets its resource's share (Q16(b)).
/// A result with nothing but `offline_access` => `invalid_scope`.
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
    if granted.iter().all(|s| s == OFFLINE_ACCESS) {
        return Err(OAuthError::new(
            OAuthErrorCode::InvalidScope,
            "no requested scope can be granted for this resource",
        ));
    }
    Ok(granted)
}
