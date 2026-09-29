//! Dynamic Client Registration (RFC 7591) for public clients.

use crate::error::{OAuthError, OAuthErrorCode};
use crate::params::{ensure_subset, split_scope, strip_identity_scopes};
use crate::ports::{ClientStore, Clock, Rng};
use serde::{Deserialize, Serialize};
use url::{Host, Url};

/// Maximum redirect URIs per client.
pub const MAX_REDIRECT_URIS: usize = 8;
/// Maximum length of one redirect URI.
pub const MAX_REDIRECT_URI_LEN: usize = 512;
/// Maximum `client_name` length (bytes).
pub const MAX_CLIENT_NAME_LEN: usize = 128;
/// Maximum registration request body (bytes).
pub const MAX_REGISTRATION_BODY: usize = 16 * 1024;
/// Prefix of edge-minted client ids (distinct from upstream `dcr-`).
pub const CLIENT_ID_PREFIX: &str = "edc-";
/// Random bytes in a client id.
pub const CLIENT_ID_BYTES: usize = 16;

/// Known connector redirect targets `(host, path prefix)` shown as verified
/// on the consent page and eligible for direct error redirects.
pub const VERIFIED_REDIRECT_PREFIXES: [(&str, &str); 4] = [
    ("chatgpt.com", "/connector/oauth/"),
    ("chatgpt.com", "/aip/"),
    ("claude.ai", "/api/mcp/"),
    ("claude.com", "/api/mcp/"),
];

const GRANT_CODE: &str = "authorization_code";
const GRANT_REFRESH: &str = "refresh_token";

/// Registration request body. Unknown metadata members are ignored (RFC 7591
/// §2); only the members below influence behaviour.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub struct RegistrationRequest {
    /// Required, 1..=[`MAX_REDIRECT_URIS`].
    #[serde(default)]
    pub redirect_uris: Vec<String>,
    /// Must be `none` (or absent, meaning `none`).
    #[serde(default)]
    pub token_endpoint_auth_method: Option<String>,
    /// Subset of `["authorization_code","refresh_token"]`; default
    /// `["authorization_code"]`.
    #[serde(default)]
    pub grant_types: Option<Vec<String>>,
    /// Must be `["code"]` if present.
    #[serde(default)]
    pub response_types: Option<Vec<String>>,
    /// Display name (printable, <= [`MAX_CLIENT_NAME_LEN`]).
    #[serde(default)]
    pub client_name: Option<String>,
    /// Space-separated requested scope ceiling; must be a subset of the AS's
    /// `scopes_supported`. Absent -> the default public ceiling.
    #[serde(default)]
    pub scope: Option<String>,
}

impl RegistrationRequest {
    /// Parse a JSON body (<= [`MAX_REGISTRATION_BODY`]); anything else is
    /// `invalid_client_metadata`.
    pub fn from_json(body: &[u8]) -> Result<Self, OAuthError> {
        let err = OAuthError::new(
            OAuthErrorCode::InvalidClientMetadata,
            "malformed registration request",
        );
        if body.len() > MAX_REGISTRATION_BODY {
            return Err(err);
        }
        serde_json::from_slice(body).map_err(|_| err)
    }
}

/// A redirect URI that passed [`validate_redirect_uri`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RedirectUri(String);

/// Loopback per RFC 8252 §7.3, written with the literal IP (so parser
/// normalisations such as `0x7f.0.0.1` or `127.1` are refused).
fn is_loopback(raw: &str, url: &Url) -> bool {
    let literal = [
        "http://127.0.0.1:",
        "http://127.0.0.1/",
        "http://[::1]:",
        "http://[::1]/",
    ]
    .iter()
    .any(|p| raw.starts_with(p));
    literal
        && url.scheme() == "http"
        && match url.host() {
            Some(Host::Ipv4(ip)) => ip == std::net::Ipv4Addr::LOCALHOST,
            Some(Host::Ipv6(ip)) => ip == std::net::Ipv6Addr::LOCALHOST,
            _ => false,
        }
}

fn raw_ok(uri: &str) -> bool {
    !uri.is_empty()
        && uri.len() <= MAX_REDIRECT_URI_LEN
        && uri.bytes().all(|b| b.is_ascii_graphic())
        && !uri.contains('#')
}

impl RedirectUri {
    /// The exact registered string (authorization requests match byte-exact,
    /// except the loopback port rule in [`RedirectUri::matches`]).
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Whether this redirect target is **verified** (ADR-351 §5.6): loopback
    /// (RFC 8252 §7.3), or `https` on the default port with a host and path
    /// prefix from [`VERIFIED_REDIRECT_PREFIXES`]. Anyone can register any
    /// https URI through anonymous DCR, so the AS never redirects to an
    /// unverified target without user interaction (RFC 9700 §4.11.2).
    pub fn is_verified(&self) -> bool {
        let Ok(url) = Url::parse(&self.0) else {
            return false;
        };
        if is_loopback(&self.0, &url) {
            return true;
        }
        url.scheme() == "https"
            && url.port().is_none()
            && VERIFIED_REDIRECT_PREFIXES
                .iter()
                .any(|(host, path)| url.host_str() == Some(host) && url.path().starts_with(path))
    }

    /// Whether `presented` matches this registration: byte-exact, or, for
    /// loopback `http` URIs, equal after ignoring the port (RFC 8252 §7.3).
    pub fn matches(&self, presented: &str) -> bool {
        if presented == self.0 {
            return true;
        }
        if !raw_ok(presented) {
            return false;
        }
        let (Ok(reg), Ok(pre)) = (Url::parse(&self.0), Url::parse(presented)) else {
            return false;
        };
        is_loopback(&self.0, &reg)
            && is_loopback(presented, &pre)
            && reg.host() == pre.host()
            && reg.path() == pre.path()
            && reg.query() == pre.query()
            && pre.username().is_empty()
            && pre.password().is_none()
            && pre.fragment().is_none()
    }
}

/// Validate one redirect URI.
///
/// Contract: absolute URL <= [`MAX_REDIRECT_URI_LEN`], parsed with `url`;
/// no fragment, no userinfo; scheme `https` with a non-empty host, **or**
/// scheme `http` with host exactly `127.0.0.1` or `[::1]` (RFC 8252 §7.3;
/// `localhost` is refused). Everything else is `invalid_redirect_uri`.
/// The raw string is checked for whitespace/control bytes **before**
/// parsing, because the WHATWG parser silently strips tab and newline.
pub fn validate_redirect_uri(uri: &str) -> Result<RedirectUri, OAuthError> {
    let err = OAuthError::new(
        OAuthErrorCode::InvalidRedirectUri,
        "redirect_uri not allowed",
    );
    if !raw_ok(uri) {
        return Err(err);
    }
    let url = Url::parse(uri).map_err(|_| err.clone())?;
    if url.fragment().is_some() || !url.username().is_empty() || url.password().is_some() {
        return Err(err);
    }
    let https_ok = url.scheme() == "https" && url.host_str().is_some_and(|h| !h.is_empty());
    if https_ok || is_loopback(uri, &url) {
        Ok(RedirectUri(uri.to_string()))
    } else {
        Err(err)
    }
}

/// Server-side DCR policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DcrPolicy {
    /// Scopes a registration may request.
    pub scopes_supported: Vec<String>,
    /// Ceiling granted when `scope` is omitted.
    pub default_scope: Vec<String>,
}

/// Persisted client.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientRecord {
    /// Server-generated opaque id (`edc-` + random).
    pub client_id: String,
    /// Validated redirect URIs.
    pub redirect_uris: Vec<RedirectUri>,
    /// Allowed grant types.
    pub grant_types: Vec<String>,
    /// Scope ceiling.
    pub scope: Vec<String>,
    /// Display name.
    pub client_name: Option<String>,
    /// Registration time (unix seconds).
    pub client_id_issued_at: u64,
}

/// RFC 7591 §3.2.1 response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RegistrationResponse {
    /// Issued id.
    pub client_id: String,
    /// Issue time.
    pub client_id_issued_at: u64,
    /// Echo of registered redirect URIs.
    pub redirect_uris: Vec<String>,
    /// Always `none`.
    pub token_endpoint_auth_method: &'static str,
    /// Registered grant types.
    pub grant_types: Vec<String>,
    /// Always `["code"]`.
    pub response_types: Vec<&'static str>,
    /// Granted scope ceiling (space-separated).
    pub scope: String,
    /// Display name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_name: Option<String>,
}

fn meta_err(desc: &'static str) -> OAuthError {
    OAuthError::new(OAuthErrorCode::InvalidClientMetadata, desc)
}

/// `client_name` is shown on the consent page, so it must not be able to
/// reorder or hide text: 1..=[`MAX_CLIENT_NAME_LEN`] bytes, no leading or
/// trailing whitespace, and every char is visible ASCII, an ASCII space, or
/// an alphanumeric char. That allowlist excludes controls (Cc), format
/// characters (Cf: bidi overrides/isolates, zero-width, BOM), line and
/// paragraph separators (Zl/Zp), non-ASCII spaces and noncharacters.
pub fn is_display_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_CLIENT_NAME_LEN
        && name.trim() == name
        && name
            .chars()
            .all(|c| c.is_ascii_graphic() || c == ' ' || c.is_alphanumeric())
}

fn validate_grants(grants: Option<&Vec<String>>) -> Result<Vec<String>, OAuthError> {
    let Some(grants) = grants else {
        return Ok(vec![GRANT_CODE.to_string()]);
    };
    let mut out: Vec<String> = Vec::new();
    for g in grants {
        if g != GRANT_CODE && g != GRANT_REFRESH {
            return Err(meta_err("unsupported grant_type"));
        }
        if !out.contains(g) {
            out.push(g.clone());
        }
    }
    if !out.iter().any(|g| g == GRANT_CODE) {
        return Err(meta_err("grant_types must include authorization_code"));
    }
    Ok(out)
}

/// Validate a registration request into a record (id and time supplied by
/// the caller from the RNG/clock ports).
///
/// Contract: `redirect_uris` 1..=8, distinct, each via
/// [`validate_redirect_uri`]; `token_endpoint_auth_method` absent or `none`;
/// `response_types` absent or `["code"]`; `grant_types` subset of
/// `authorization_code`/`refresh_token` and must include
/// `authorization_code`; `scope` minus the identity scopes
/// (`openid profile email`, dropped) subset of `policy.scopes_supported`
/// (absent or nothing left -> `policy.default_scope`, never the full
/// supported set); `client_name` per [`is_display_name`].
pub fn validate_registration(
    req: &RegistrationRequest,
    policy: &DcrPolicy,
    client_id: String,
    now: u64,
) -> Result<ClientRecord, OAuthError> {
    if req.redirect_uris.is_empty() || req.redirect_uris.len() > MAX_REDIRECT_URIS {
        return Err(OAuthError::new(
            OAuthErrorCode::InvalidRedirectUri,
            "redirect_uris must hold 1 to 8 entries",
        ));
    }
    let mut redirect_uris: Vec<RedirectUri> = Vec::with_capacity(req.redirect_uris.len());
    for uri in &req.redirect_uris {
        let r = validate_redirect_uri(uri)?;
        if redirect_uris.contains(&r) {
            return Err(OAuthError::new(
                OAuthErrorCode::InvalidRedirectUri,
                "duplicate redirect_uri",
            ));
        }
        redirect_uris.push(r);
    }
    if req
        .token_endpoint_auth_method
        .as_deref()
        .is_some_and(|m| m != "none")
    {
        return Err(meta_err("only public clients (none) are supported"));
    }
    if let Some(rt) = &req.response_types {
        if rt.len() != 1 || rt[0] != "code" {
            return Err(meta_err("response_types must be [\"code\"]"));
        }
    }
    let grant_types = validate_grants(req.grant_types.as_ref())?;
    let scope = match req.scope.as_deref() {
        None => policy.default_scope.clone(),
        Some(s) => {
            let s = split_scope(s).map_err(|_| meta_err("malformed scope"))?;
            let s = strip_identity_scopes(s);
            ensure_subset(&s, &policy.scopes_supported)
                .map_err(|_| meta_err("scope not registrable"))?;
            if s.is_empty() {
                policy.default_scope.clone()
            } else {
                s
            }
        }
    };
    if let Some(name) = &req.client_name {
        if !is_display_name(name) {
            return Err(meta_err("invalid client_name"));
        }
    }
    Ok(ClientRecord {
        client_id,
        redirect_uris,
        grant_types,
        scope,
        client_name: req.client_name.clone(),
        client_id_issued_at: now,
    })
}

/// Full DCR: validate, mint `edc-<random>` id, persist, return the record.
/// RNG or storage failure is `server_error`; nothing is stored on a
/// validation failure.
pub fn register_client<S, R, C>(
    store: &S,
    rng: &R,
    clock: &C,
    policy: &DcrPolicy,
    req: &RegistrationRequest,
) -> Result<ClientRecord, OAuthError>
where
    S: ClientStore + ?Sized,
    R: Rng + ?Sized,
    C: Clock + ?Sized,
{
    let id = crate::random_secret(rng, CLIENT_ID_BYTES)?;
    let record = validate_registration(
        req,
        policy,
        format!("{CLIENT_ID_PREFIX}{id}"),
        clock.now_unix(),
    )?;
    store.insert_client(&record)?;
    Ok(record)
}

impl ClientRecord {
    /// Whether the client registered `grant`.
    pub fn allows_grant(&self, grant: &str) -> bool {
        self.grant_types.iter().any(|g| g == grant)
    }

    /// First registered redirect URI matching `presented`, if any.
    pub fn match_redirect(&self, presented: &str) -> Option<&RedirectUri> {
        self.redirect_uris.iter().find(|r| r.matches(presented))
    }

    /// Build the RFC 7591 response for this record.
    pub fn to_response(&self) -> RegistrationResponse {
        RegistrationResponse {
            client_id: self.client_id.clone(),
            client_id_issued_at: self.client_id_issued_at,
            redirect_uris: self
                .redirect_uris
                .iter()
                .map(|r| r.as_str().to_string())
                .collect(),
            token_endpoint_auth_method: "none",
            grant_types: self.grant_types.clone(),
            response_types: vec!["code"],
            scope: self.scope.join(" "),
            client_name: self.client_name.clone(),
        }
    }
}
