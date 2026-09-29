//! Scope -> capability mapping and the default-deny route table
//! (ADR-351 §5.3). Versioned `const` tables; unit tests assert every route is
//! covered and unknown routes are denied.

use crate::claims::TokenKind;

/// Version of [`SCOPE_TABLE`] / [`ROUTE_TABLE`]; bump on any change.
/// v2: `ruvector:*` vocabulary, `Admin`, tenant/MCP/ops routes.
pub const SCOPE_TABLE_VERSION: u32 = 2;

/// Internal capability a route requires (always further ∩ role, §5.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Capability {
    /// Read vectors/collections/usage.
    Read,
    /// Upsert/delete vectors.
    Write,
    /// Create or drop collections.
    CreateCollection,
    /// Tenant administration: members, tenant deny entries, restore, audit.
    Admin,
    /// Publish public RVF packages (M5, never via `/v1/mcp`).
    PublishPublic,
}

impl Capability {
    const ALL: [Capability; 5] = [
        Capability::Read,
        Capability::Write,
        Capability::CreateCollection,
        Capability::Admin,
        Capability::PublishPublic,
    ];

    fn bit(self) -> u8 {
        1 << (self as u8)
    }

    /// The scope a client should request to obtain this capability (used in
    /// `WWW-Authenticate: ... error="insufficient_scope", scope="..."`).
    pub fn satisfying_scope(self) -> &'static str {
        match self {
            Capability::Read => "ruvector:read",
            Capability::Write | Capability::CreateCollection => "ruvector:write",
            Capability::Admin => "ruvector:admin",
            Capability::PublishPublic => "ruvector:publish",
        }
    }
}

/// Small set of [`Capability`] values.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct CapabilitySet(u8);

impl CapabilitySet {
    /// Empty set.
    pub const EMPTY: CapabilitySet = CapabilitySet(0);

    /// Add a capability.
    pub fn insert(&mut self, c: Capability) {
        self.0 |= c.bit();
    }

    /// Membership test.
    pub fn contains(&self, c: Capability) -> bool {
        self.0 & c.bit() != 0
    }

    /// Iterate members in declaration order.
    pub fn iter(&self) -> impl Iterator<Item = Capability> + '_ {
        Capability::ALL
            .into_iter()
            .filter(move |c| self.contains(*c))
    }
}

/// Which API surface the request arrived on (connectors are MCP-only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RouteSurface {
    /// `/v1/*` REST.
    Rest,
    /// `POST /v1/mcp`.
    Mcp,
}

/// `(scope, capabilities granted)` — the ADR §5.3 vocabulary. Scopes not
/// listed grant nothing: upstream product scopes (`mcp:*`, `swarm:*`,
/// `brains:*`, ...), identity scopes, `offline_access` (refresh only) and
/// `ruvector:publish` (not honoured before M5).
pub const SCOPE_TABLE: &[(&str, &[Capability])] = &[
    ("ruvector:read", &[Capability::Read]),
    (
        "ruvector:write",
        &[Capability::Write, Capability::CreateCollection],
    ),
    ("ruvector:admin", &[Capability::Admin]),
];

/// Fixed grant for upstream first-party tokens on REST (ADR §5.5: upstream
/// tokens carry no ruvector scopes, so they get `ruvector:read
/// ruvector:write ruvector:admin`, still ∩ role).
pub const UPSTREAM_FIRST_PARTY_CAPS: &[Capability] = &[
    Capability::Read,
    Capability::Write,
    Capability::CreateCollection,
    Capability::Admin,
];

/// Map granted scopes to capabilities (before the role intersection).
///
/// Contract: `EdgeIssued` — capabilities of the listed [`SCOPE_TABLE`]
/// scopes, unknown scopes ignored, `PublishPublic` never granted on
/// [`RouteSurface::Mcp`]. `UpstreamFirstParty` — the fixed
/// [`UPSTREAM_FIRST_PARTY_CAPS`] on [`RouteSurface::Rest`] (its scopes are
/// ignored) and nothing on any other surface (ADR §5.5: REST only).
pub fn capabilities_for(
    scopes: &[String],
    kind: TokenKind,
    surface: RouteSurface,
) -> CapabilitySet {
    let mut set = CapabilitySet::EMPTY;
    if kind == TokenKind::UpstreamFirstParty {
        if surface == RouteSurface::Rest {
            UPSTREAM_FIRST_PARTY_CAPS
                .iter()
                .for_each(|c| set.insert(*c));
        }
        return set;
    }
    for scope in scopes {
        if let Some((_, caps)) = SCOPE_TABLE.iter().find(|(s, _)| *s == scope.as_str()) {
            for cap in caps.iter() {
                if !(surface == RouteSurface::Mcp && *cap == Capability::PublishPublic) {
                    set.insert(*cap);
                }
            }
        }
    }
    set
}

/// HTTP method subset the API uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub enum Method {
    Get,
    Post,
    Put,
    Delete,
}

/// What a route requires (always further ∩ role and, for MCP/ops, per
/// tool/op, downstream).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RouteRequirement {
    /// No token (health + metadata documents only).
    Anonymous,
    /// Any valid token, no capability at the route level (`/v1/me`; and
    /// `POST /v1/mcp` / `POST /v1/ops`, whose tools/ops each check their own
    /// capability).
    AnyValidToken,
    /// A valid token carrying this capability.
    Capability(Capability),
    /// A valid token carrying at least one of these capabilities
    /// (`tenant:claim`: `ruvector:write` or `ruvector:admin`).
    AnyOf(&'static [Capability]),
}

impl RouteRequirement {
    /// Whether a token with `caps` (after the role intersection) meets the
    /// route requirement. `Anonymous`/`AnyValidToken` need no capability.
    pub fn satisfied_by(&self, caps: CapabilitySet) -> bool {
        match self {
            RouteRequirement::Anonymous | RouteRequirement::AnyValidToken => true,
            RouteRequirement::Capability(c) => caps.contains(*c),
            RouteRequirement::AnyOf(list) => list.iter().any(|c| caps.contains(*c)),
        }
    }
}

const CLAIM_CAPS: &[Capability] = &[Capability::Write, Capability::Admin];

/// Default-deny route table. Patterns use `{c}` / `{id}` / `{sub}` for one
/// path segment. M1 routes only; later milestones append rows.
pub const ROUTE_TABLE: &[(Method, &str, RouteRequirement)] = &[
    (
        Method::Get,
        "/.well-known/oauth-protected-resource",
        RouteRequirement::Anonymous,
    ),
    // RFC 9728 §3.1 path-inserted form for resource `https://<host>/v1`.
    (
        Method::Get,
        "/.well-known/oauth-protected-resource/v1",
        RouteRequirement::Anonymous,
    ),
    (
        Method::Get,
        "/.well-known/oauth-protected-resource/v1/mcp",
        RouteRequirement::Anonymous,
    ),
    (Method::Get, "/v1/health", RouteRequirement::Anonymous),
    (Method::Get, "/v1/me", RouteRequirement::AnyValidToken),
    (
        Method::Get,
        "/v1/usage",
        RouteRequirement::Capability(Capability::Read),
    ),
    (
        Method::Post,
        "/v1/collections",
        RouteRequirement::Capability(Capability::CreateCollection),
    ),
    (
        Method::Get,
        "/v1/collections",
        RouteRequirement::Capability(Capability::Read),
    ),
    (
        Method::Get,
        "/v1/collections/{c}",
        RouteRequirement::Capability(Capability::Read),
    ),
    (
        Method::Delete,
        "/v1/collections/{c}",
        RouteRequirement::Capability(Capability::CreateCollection),
    ),
    (
        Method::Post,
        "/v1/collections/{c}/vectors:upsert",
        RouteRequirement::Capability(Capability::Write),
    ),
    (
        Method::Post,
        "/v1/collections/{c}/query",
        RouteRequirement::Capability(Capability::Read),
    ),
    (
        Method::Get,
        "/v1/collections/{c}/vectors/{id}",
        RouteRequirement::Capability(Capability::Read),
    ),
    (
        Method::Post,
        "/v1/collections/{c}/vectors:fetch",
        RouteRequirement::Capability(Capability::Read),
    ),
    (
        Method::Post,
        "/v1/collections/{c}/vectors:delete",
        RouteRequirement::Capability(Capability::Write),
    ),
    // Tenant administration (§7.2, M1).
    (
        Method::Post,
        "/v1/tenant:claim",
        RouteRequirement::AnyOf(CLAIM_CAPS),
    ),
    (
        Method::Get,
        "/v1/tenant/members",
        RouteRequirement::Capability(Capability::Admin),
    ),
    (
        Method::Post,
        "/v1/tenant/members",
        RouteRequirement::Capability(Capability::Admin),
    ),
    (
        Method::Get,
        "/v1/tenant/members/{sub}",
        RouteRequirement::Capability(Capability::Admin),
    ),
    (
        Method::Delete,
        "/v1/tenant/members/{sub}",
        RouteRequirement::Capability(Capability::Admin),
    ),
    (
        Method::Post,
        "/v1/tenant/deny",
        RouteRequirement::Capability(Capability::Admin),
    ),
    // JSON-RPC / op envelopes: each tool/op enforces its own capability.
    (Method::Post, "/v1/mcp", RouteRequirement::AnyValidToken),
    (Method::Post, "/v1/ops", RouteRequirement::AnyValidToken),
];

/// Look up the requirement for `(method, path)`.
///
/// Contract: exact segment-wise match against [`ROUTE_TABLE`]; a `{x}`
/// placeholder matches exactly one non-empty segment; `None` means the route
/// does not exist and must be denied (404), never allowed.
pub fn route_requirement(method: Method, path: &str) -> Option<RouteRequirement> {
    ROUTE_TABLE
        .iter()
        .find(|(m, pattern, _)| *m == method && path_matches(pattern, path))
        .map(|(_, _, req)| *req)
}

/// Segment-wise match. A pattern segment `{x}` (optionally followed by a
/// literal suffix such as `:upsert`) matches one non-empty request segment
/// that ends with that suffix and whose variable part has no `:` and is not
/// `.` or `..`. Paths with empty segments never match.
fn path_matches(pattern: &str, path: &str) -> bool {
    if !path.starts_with('/') || path.contains("//") {
        return false;
    }
    let mut p = pattern.split('/');
    let mut r = path.split('/');
    loop {
        match (p.next(), r.next()) {
            (None, None) => return true,
            (Some(ps), Some(rs)) if segment_matches(ps, rs) => {}
            _ => return false,
        }
    }
}

fn segment_matches(pattern: &str, seg: &str) -> bool {
    if let Some(rest) = pattern.strip_prefix('{') {
        let Some(close) = rest.find('}') else {
            return false;
        };
        let suffix = &rest[close + 1..];
        match seg.strip_suffix(suffix) {
            Some(var) => !var.is_empty() && !var.contains(':') && var != "." && var != "..",
            None => false,
        }
    } else {
        pattern == seg
    }
}

#[cfg(test)]
mod tests;
