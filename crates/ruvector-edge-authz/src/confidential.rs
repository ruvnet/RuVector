//! Operator-registered **confidential** adapter clients (ADR-351 §5.6,
//! §16.1) and their RFC 7523 `private_key_jwt` authentication.
//!
//! These clients are never created by public DCR: the registry is reviewed
//! deploy configuration (the Worker's `CONFIDENTIAL_CLIENTS` var), validated
//! whole at load, and each client may only use the RFC 8693 exchange grant
//! toward [`GATEWAY_V1_URL`]. Their ids can never collide with DCR ids
//! (which all start with [`CLIENT_ID_PREFIX`]), and they are not in the
//! [`crate::ClientStore`], so the public grants and `/authorize` refuse them.

use crate::client::CLIENT_ID_PREFIX;
use crate::error::{OAuthError, OAuthErrorCode};
use crate::params::split_scope;
use crate::ports::AssertionReplayStore;
use crate::resource::{ResourceAllowlist, Vocabulary, GATEWAY_V1_URL};
use ruvector_edge_auth::claims::Audience;
use ruvector_edge_auth::jws::parse_compact;
use ruvector_edge_auth::{verify_es256, AuthError, Jwk, ResourceUrl, VerifyingKey};
use serde::Deserialize;

/// RFC 7523 §2.2 `client_assertion_type`.
pub const JWT_BEARER_ASSERTION: &str = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer";
/// Maximum client-assertion lifetime from `now` (and from `iat`, if sent).
pub const MAX_ASSERTION_LIFETIME_SECS: u64 = 300;
/// Clock skew tolerated on assertion `iat` / `nbf`.
pub const ASSERTION_SKEW_SECS: u64 = 60;
/// Maximum registered confidential clients.
pub const MAX_CONFIDENTIAL_CLIENTS: usize = 16;
/// Maximum subject audiences per client.
pub const MAX_SUBJECT_AUDIENCES: usize = 4;
/// Maximum `CONFIDENTIAL_CLIENTS` value (bytes).
pub const MAX_REGISTRY_BYTES: usize = 16 * 1024;
/// Maximum `client_id` / assertion `jti` length (bytes).
pub const MAX_ID_LEN: usize = 128;
/// Private (or symmetric) JWK members; a registered JWK must carry none.
const PRIVATE_JWK_MEMBERS: [&str; 7] = ["d", "p", "q", "dp", "dq", "qi", "k"];

/// One adapter client: its public ES256 key, the resources whose user
/// tokens it may exchange, and its `…/v1` scope ceiling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfidentialClient {
    client_id: String,
    kid: String,
    key: VerifyingKey,
    subject_audiences: Vec<ResourceUrl>,
    scope: Vec<String>,
}

impl ConfidentialClient {
    /// The client id (`act.sub` and `client_id` of exchanged tokens).
    pub fn client_id(&self) -> &str {
        &self.client_id
    }
    /// RFC 7638 thumbprint of the registered key (required assertion `kid`).
    pub fn kid(&self) -> &str {
        &self.kid
    }
    /// Resources whose tokens this client may present as `subject_token`.
    pub fn subject_audiences(&self) -> &[ResourceUrl] {
        &self.subject_audiences
    }
    /// Whether `aud` is one of [`ConfidentialClient::subject_audiences`].
    pub fn allows_subject_audience(&self, aud: &str) -> bool {
        self.subject_audiences.iter().any(|a| a.as_str() == aud)
    }
    /// Scope ceiling of exchanged tokens (`ruvector:*` only).
    pub fn scope(&self) -> &[String] {
        &self.scope
    }
}

/// The validated registry.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConfidentialClients {
    clients: Vec<ConfidentialClient>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawEntry {
    client_id: String,
    jwk: serde_json::Map<String, serde_json::Value>,
    subject_audiences: Vec<String>,
    scope: String,
}

fn cfg(what: &'static str) -> AuthError {
    AuthError::InvalidConfig(what)
}

fn is_id(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= MAX_ID_LEN
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}

impl ConfidentialClients {
    /// Parse and validate the operator registry: a JSON array (empty or
    /// blank value = no clients) of
    /// `{"client_id", "jwk", "subject_audiences": [..], "scope"}`.
    ///
    /// Contract (any violation fails the whole registry): at most
    /// [`MAX_REGISTRY_BYTES`] and [`MAX_CONFIDENTIAL_CLIENTS`]; no unknown
    /// members; `client_id` unique, `[A-Za-z0-9._-]{1,128}`, never with the
    /// DCR prefix [`CLIENT_ID_PREFIX`]; `jwk` a **public** EC P-256 key (no
    /// private members) whose `kid` equals its RFC 7638 thumbprint;
    /// 1..=[`MAX_SUBJECT_AUDIENCES`] distinct canonical subject audiences,
    /// each allowlisted and compiled to an **adapter** vocabulary (never a
    /// gateway resource: not the exchange target [`GATEWAY_V1_URL`], not
    /// `/v1/mcp`); `scope` a non-empty ceiling of `ruvector:*`
    /// scopes only. With any client registered, the target must itself be
    /// allowlisted.
    pub fn from_config(value: &str, allowlist: &ResourceAllowlist) -> Result<Self, AuthError> {
        if value.trim().is_empty() {
            return Ok(ConfidentialClients::default());
        }
        if value.len() > MAX_REGISTRY_BYTES {
            return Err(cfg("confidential clients too large"));
        }
        let raw: Vec<RawEntry> =
            serde_json::from_str(value).map_err(|_| cfg("confidential clients json"))?;
        if raw.len() > MAX_CONFIDENTIAL_CLIENTS {
            return Err(cfg("too many confidential clients"));
        }
        let mut clients: Vec<ConfidentialClient> = Vec::with_capacity(raw.len());
        for e in raw {
            let c = Self::entry(e, allowlist)?;
            if clients.iter().any(|o| o.client_id == c.client_id) {
                return Err(cfg("repeated confidential client"));
            }
            clients.push(c);
        }
        let target = ResourceUrl::parse(GATEWAY_V1_URL).map_err(|_| cfg("exchange target"))?;
        if !clients.is_empty() && allowlist.get(&target).is_none() {
            return Err(cfg("exchange target not allowlisted"));
        }
        Ok(ConfidentialClients { clients })
    }

    fn entry(e: RawEntry, allowlist: &ResourceAllowlist) -> Result<ConfidentialClient, AuthError> {
        if !is_id(&e.client_id) || e.client_id.starts_with(CLIENT_ID_PREFIX) {
            return Err(cfg("confidential client_id"));
        }
        if PRIVATE_JWK_MEMBERS.iter().any(|m| e.jwk.contains_key(*m)) {
            return Err(cfg("confidential jwk must be public"));
        }
        let jwk: Jwk = serde_json::from_value(serde_json::Value::Object(e.jwk))
            .map_err(|_| cfg("confidential jwk"))?;
        let key = jwk.to_verifying_key()?;
        let n = e.subject_audiences.len();
        if n == 0 || n > MAX_SUBJECT_AUDIENCES {
            return Err(cfg("subject audiences"));
        }
        let mut subject_audiences: Vec<ResourceUrl> = Vec::with_capacity(n);
        for a in &e.subject_audiences {
            let url = ResourceUrl::parse(a).map_err(|_| cfg("subject audience"))?;
            // Adapter resources only: a gateway (`ruvector:*`) resource, `/v1`
            // or `/v1/mcp`, is never a subject (they authorise separately).
            let adapter = Vocabulary::for_resource(&url).is_some_and(|v| v != Vocabulary::Ruvector);
            let ok = url.as_str() == a
                && url.as_str() != GATEWAY_V1_URL
                && adapter
                && allowlist.get(&url).is_some()
                && !subject_audiences.contains(&url);
            if !ok {
                return Err(cfg("subject audience"));
            }
            subject_audiences.push(url);
        }
        let scope = split_scope(&e.scope).map_err(|_| cfg("confidential scope"))?;
        if !scope.iter().all(|s| Vocabulary::Ruvector.contains(s)) {
            return Err(cfg("confidential scope"));
        }
        Ok(ConfidentialClient {
            client_id: e.client_id,
            kid: jwk.kid,
            key,
            subject_audiences,
            scope,
        })
    }

    /// The client registered as `client_id`.
    pub fn get(&self, client_id: &str) -> Option<&ConfidentialClient> {
        self.clients.iter().find(|c| c.client_id == client_id)
    }

    /// Every registered client.
    pub fn clients(&self) -> &[ConfidentialClient] {
        &self.clients
    }
}

/// RFC 7523 claims read from a client assertion (unknown claims ignored).
#[derive(Deserialize)]
struct AssertionClaims {
    iss: Option<String>,
    sub: Option<String>,
    aud: Option<Audience>,
    exp: Option<u64>,
    iat: Option<u64>,
    nbf: Option<u64>,
    jti: Option<String>,
}

fn invalid_client(desc: &'static str) -> OAuthError {
    OAuthError::new(OAuthErrorCode::InvalidClient, desc)
}

/// Authenticate a confidential client by `private_key_jwt` (RFC 7523 §3,
/// RFC 7521 §4.2).
///
/// Contract, in order (every failure is `invalid_client`, except a replay
/// store failure, `server_error`): compact ES256 JWS (`none`/HS*/RS*,
/// `jku`/`jwk`/`x5*`/`crit` refused); header `typ`, if present, `JWT` or
/// `client-authentication+jwt` (so an `at+jwt` access token is never an
/// assertion); `iss` names a registered client; a `client_id` parameter, if
/// sent, equals it; header `kid` equals the client's key thumbprint; the
/// signature verifies with that key; `sub == iss`; `aud` is exactly
/// `token_endpoint` (a string or a one-element array); `exp > now` and
/// `exp <= now + MAX_ASSERTION_LIFETIME_SECS`; `iat`, if sent, `<= now +
/// skew` and `exp - iat <= MAX_ASSERTION_LIFETIME_SECS`; `nbf`, if sent,
/// `<= now + skew`; `jti` required (visible ASCII, <= 128 bytes). Only a
/// fully valid assertion reaches the replay cache (so nobody can burn a
/// client's `jti` with a forged one); a live `jti` there is a replay.
pub fn authenticate<'c>(
    clients: &'c ConfidentialClients,
    replay: &dyn AssertionReplayStore,
    token_endpoint: &str,
    now: u64,
    assertion: &str,
    client_id_param: Option<&str>,
) -> Result<&'c ConfidentialClient, OAuthError> {
    let malformed = || invalid_client("malformed client assertion");
    let jws = parse_compact(assertion).map_err(|_| malformed())?;
    match jws.header.typ.as_deref() {
        None => {}
        Some(t) if t.eq_ignore_ascii_case("JWT") => {}
        Some(t) if t.eq_ignore_ascii_case("client-authentication+jwt") => {}
        Some(_) => return Err(invalid_client("client assertion typ not accepted")),
    }
    let payload = jws.payload_bytes().map_err(|_| malformed())?;
    let object: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&payload).map_err(|_| malformed())?;
    let claims: AssertionClaims =
        serde_json::from_value(serde_json::Value::Object(object)).map_err(|_| malformed())?;
    let iss = claims.iss.as_deref().ok_or_else(malformed)?;
    let client = clients
        .get(iss)
        .ok_or(invalid_client("unknown confidential client"))?;
    if client_id_param.is_some_and(|c| c != iss) {
        return Err(invalid_client("client_id does not match the assertion"));
    }
    if jws.header.kid != client.kid {
        return Err(invalid_client("client assertion kid not registered"));
    }
    verify_es256(&jws, &client.key)
        .map_err(|_| invalid_client("bad client assertion signature"))?;
    if claims.sub.as_deref() != Some(iss) {
        return Err(invalid_client("client assertion sub must equal iss"));
    }
    let aud_ok = match &claims.aud {
        Some(Audience::One(a)) => a == token_endpoint,
        Some(Audience::Many(v)) => v.len() == 1 && v[0] == token_endpoint,
        None => false,
    };
    if !aud_ok {
        return Err(invalid_client(
            "client assertion aud must be the token endpoint",
        ));
    }
    let exp = claims
        .exp
        .ok_or(invalid_client("client assertion exp required"))?;
    if exp <= now {
        return Err(invalid_client("client assertion expired"));
    }
    if exp > now.saturating_add(MAX_ASSERTION_LIFETIME_SECS) {
        return Err(invalid_client("client assertion lifetime too long"));
    }
    let horizon = now.saturating_add(ASSERTION_SKEW_SECS);
    if let Some(iat) = claims.iat {
        if iat > horizon || exp.saturating_sub(iat) > MAX_ASSERTION_LIFETIME_SECS {
            return Err(invalid_client("client assertion iat invalid"));
        }
    }
    if claims.nbf.is_some_and(|nbf| nbf > horizon) {
        return Err(invalid_client("client assertion not yet valid"));
    }
    let jti = claims
        .jti
        .as_deref()
        .filter(|j| {
            !j.is_empty() && j.len() <= MAX_ID_LEN && j.bytes().all(|b| b.is_ascii_graphic())
        })
        .ok_or(invalid_client("client assertion jti required"))?;
    let key = crate::secret_hash(&format!("{}|{jti}", client.client_id));
    if !replay.record_assertion(&key, exp, now)? {
        return Err(invalid_client("client assertion replayed"));
    }
    Ok(client)
}
