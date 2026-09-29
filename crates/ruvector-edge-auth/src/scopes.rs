//! Scope -> capability mapping and the default-deny route table
//! (ADR-351 §5.4). Versioned `const` tables; a unit test (later stage) asserts
//! every route is covered and unknown routes are denied.

use crate::claims::TokenKind;

/// Version of [`SCOPE_TABLE`] / [`ROUTE_TABLE`]; bump on any change.
pub const SCOPE_TABLE_VERSION: u32 = 1;

/// Internal capability a route requires.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Capability {
    /// Read vectors/collections/usage.
    Read,
    /// Upsert/delete vectors.
    Write,
    /// Create or drop collections.
    CreateCollection,
    /// Publish public RVF packages (M6, never via `/v1/mcp`).
    PublishPublic,
}

impl Capability {
    const ALL: [Capability; 4] = [
        Capability::Read,
        Capability::Write,
        Capability::CreateCollection,
        Capability::PublishPublic,
    ];

    fn bit(self) -> u8 {
        1 << (self as u8)
    }

    /// The scope a client should request to obtain this capability (used in
    /// `WWW-Authenticate: ... error="insufficient_scope", scope="..."`).
    pub fn satisfying_scope(self) -> &'static str {
        match self {
            Capability::Read => "mcp:read",
            Capability::Write | Capability::CreateCollection => "mcp:invoke",
            Capability::PublishPublic => "brains:contribute",
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

/// `(scope, capabilities granted)`. Scopes not listed grant nothing
/// (including the deliberately-unused product scopes in ADR §5.4).
pub const SCOPE_TABLE: &[(&str, &[Capability])] = &[
    ("mcp:read", &[Capability::Read]),
    (
        "mcp:invoke",
        &[Capability::Write, Capability::CreateCollection],
    ),
];

/// Map granted scopes to capabilities.
///
/// Contract: unknown scopes are ignored; `brains:*`, `namespaces:claim` and
/// `brains:contribute` are not honoured before M6 (absent from
/// [`SCOPE_TABLE`]); `PublishPublic` is never granted on
/// [`RouteSurface::Mcp`]. `kind` is accepted so per-kind restrictions can be
/// added without an API change.
pub fn capabilities_for(
    scopes: &[String],
    kind: TokenKind,
    surface: RouteSurface,
) -> CapabilitySet {
    let _ = kind;
    let mut set = CapabilitySet::EMPTY;
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

/// What a route requires.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RouteRequirement {
    /// No token (health + metadata documents only).
    Anonymous,
    /// Any valid token, no capability (`/v1/me`).
    AnyValidToken,
    /// A valid token carrying this capability.
    Capability(Capability),
}

/// Default-deny route table. Patterns use `{c}` / `{id}` for one path
/// segment. M1 routes only; later milestones append rows.
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
];

/// Look up the requirement for `(method, path)`.
///
/// Contract: exact segment-wise match against [`ROUTE_TABLE`]; a `{x}`
/// placeholder matches exactly one non-empty segment; `None` means the route
/// does not exist and must be denied (404), never allowed.
pub fn route_requirement(method: Method, path: &str) -> Option<RouteRequirement> {
    let _ = (method, path);
    None
}
