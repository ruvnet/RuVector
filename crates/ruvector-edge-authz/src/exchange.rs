//! RFC 8693 token exchange for operator-registered adapter clients
//! (ADR-351 §5.6, §16.1): a user's edge token for an adapter resource
//! (e.g. `https://team.ruv.io/mcp`) becomes a `…/v1` token for the same
//! user, acting as the adapter (`act.sub`), through the compiled exchange
//! map. No refresh token; never outlives the subject token.

use crate::confidential::{self, JWT_BEARER_ASSERTION};
use crate::error::{OAuthError, OAuthErrorCode};
use crate::grant::TokenEndpoint;
use crate::metadata::paths;
use crate::params::{split_scope, strip_identity_scopes, Params};
use crate::refresh::OFFLINE_ACCESS;
use crate::resource::{
    exchange_scope, is_edge_vocabulary, ResourceEntry, Vocabulary, GATEWAY_V1_URL,
};
use crate::token::{sign_claims, AccessTokenClaims, Act, TokenResponse};
use crate::token::{ACCESS_TOKEN_TTL_SECS, JTI_BYTES};
use ruvector_edge_auth::jws::parse_compact;
use ruvector_edge_auth::subject::is_edge_subject;
use ruvector_edge_auth::{verify_es256, Jwk, VerifyingKey};
use serde::Deserialize;

/// RFC 8693 §2.1 `grant_type`.
pub const TOKEN_EXCHANGE_GRANT: &str = "urn:ietf:params:oauth:grant-type:token-exchange";
/// RFC 8693 §3 token type of both the subject and the issued token.
pub const ACCESS_TOKEN_TYPE: &str = "urn:ietf:params:oauth:token-type:access_token";
/// Clock skew tolerated on the subject token's `iat`.
pub const SUBJECT_SKEW_SECS: u64 = 60;
/// Maximum length of a string claim kept from the subject token.
const MAX_CLAIM_LEN: usize = 256;

/// Parsed exchange request. `Debug` redacts both tokens.
#[derive(Clone, PartialEq, Eq)]
pub struct ExchangeRequest {
    /// RFC 7523 `client_assertion` (the client's `private_key_jwt`).
    pub client_assertion: String,
    /// Optional `client_id` (must equal the assertion's `iss`).
    pub client_id: Option<String>,
    /// The user's edge access token for the adapter resource.
    pub subject_token: String,
    /// Target resource (must be [`GATEWAY_V1_URL`]).
    pub resource: Option<String>,
    /// Optional narrowing `scope`.
    pub scope: Option<String>,
}

impl std::fmt::Debug for ExchangeRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExchangeRequest")
            .field("client_assertion", &crate::REDACTED)
            .field("client_id", &self.client_id)
            .field("subject_token", &crate::REDACTED)
            .field("resource", &self.resource)
            .field("scope", &self.scope)
            .finish()
    }
}

fn err(code: OAuthErrorCode, desc: &'static str) -> OAuthError {
    OAuthError::new(code, desc)
}

impl ExchangeRequest {
    /// Parse the exchange members of a `grant_type` = [`TOKEN_EXCHANGE_GRANT`]
    /// request.
    ///
    /// Contract, in order: `client_secret` => `invalid_client` (only
    /// `private_key_jwt`); `client_assertion_type` other than
    /// [`JWT_BEARER_ASSERTION`] or no `client_assertion` => `invalid_client`;
    /// `subject_token` missing, `subject_token_type` not
    /// [`ACCESS_TOKEN_TYPE`], `requested_token_type` (if sent) not
    /// [`ACCESS_TOKEN_TYPE`], or any `actor_token` / `actor_token_type`
    /// (delegation chains are not supported: the actor is the authenticated
    /// client) => `invalid_request`; an `audience` parameter =>
    /// `invalid_target` (targets are named by `resource` only).
    pub fn from_params(p: &Params) -> Result<Self, OAuthError> {
        use OAuthErrorCode::{InvalidClient, InvalidRequest, InvalidTarget};
        if p.get("client_secret").is_some() {
            return Err(err(
                InvalidClient,
                "use private_key_jwt client authentication",
            ));
        }
        if p.get("client_assertion_type") != Some(JWT_BEARER_ASSERTION) {
            return Err(err(
                InvalidClient,
                "client_assertion_type must be jwt-bearer",
            ));
        }
        let client_assertion = p
            .take("client_assertion")
            .ok_or(err(InvalidClient, "client_assertion required"))?;
        let subject_token = p.require("subject_token", "subject_token required")?;
        if p.get("subject_token_type") != Some(ACCESS_TOKEN_TYPE) {
            return Err(err(
                InvalidRequest,
                "subject_token_type must be access_token",
            ));
        }
        if p.get("requested_token_type")
            .is_some_and(|t| t != ACCESS_TOKEN_TYPE)
        {
            return Err(err(InvalidRequest, "only access tokens can be issued"));
        }
        if p.get("actor_token").is_some() || p.get("actor_token_type").is_some() {
            return Err(err(InvalidRequest, "actor tokens are not supported"));
        }
        if p.get("audience").is_some() {
            return Err(err(InvalidTarget, "use resource, not audience"));
        }
        Ok(ExchangeRequest {
            client_assertion,
            client_id: p.take("client_id"),
            subject_token,
            resource: p.take("resource"),
            scope: p.take("scope"),
        })
    }
}

/// The verified subject token's claims that the exchange carries over.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SubjectClaims {
    /// Issuer (this AS).
    pub iss: String,
    /// Adapter resource the user consented to.
    pub aud: String,
    /// Edge subject.
    pub sub: String,
    /// Tenant namespace.
    pub upstream_iss: String,
    /// The subject token's client (the user's DCR client).
    pub client_id: String,
    /// Tenant input.
    pub org_id: String,
    /// Tenant input.
    pub workspace_id: String,
    /// Grant / refresh family (revocation key).
    pub family_id: String,
    /// Space-separated scopes.
    pub scope: String,
    /// Token id.
    pub jti: String,
    /// Issued-at.
    pub iat: u64,
    /// Expiry.
    pub exp: u64,
}

fn bad_subject(desc: &'static str) -> OAuthError {
    // RFC 8693 §2.2.2: an invalid or unacceptable subject_token is
    // `invalid_request`.
    err(OAuthErrorCode::InvalidRequest, desc)
}

/// Verify a subject token minted by this AS.
///
/// Contract: compact ES256 JWS with header `typ = at+jwt`; `kid` equal to
/// the RFC 7638 thumbprint of one of `keys` (none configured =>
/// `server_error`); signature verifies; the payload has **no `act` claim**
/// (any value, even `null`: an exchanged token is never re-exchanged, so
/// actor chains cannot nest); every [`SubjectClaims`] member present with
/// the right type; `iss == issuer`; `sub` an edge subject; string claims
/// non-empty and bounded; `iat <= now + skew`, `iat < exp`, `exp - iat <=
/// ACCESS_TOKEN_TTL_SECS` and `now < exp` (no skew on expiry). Anything
/// else is `invalid_request`.
pub fn verify_subject_token(
    token: &str,
    issuer: &str,
    keys: &[VerifyingKey],
    now: u64,
) -> Result<SubjectClaims, OAuthError> {
    if keys.is_empty() {
        return Err(err(OAuthErrorCode::ServerError, "signing key unavailable"));
    }
    let jws = parse_compact(token).map_err(|_| bad_subject("malformed subject_token"))?;
    if jws.header.typ.as_deref() != Some("at+jwt") {
        return Err(bad_subject("subject_token is not an access token"));
    }
    let key = keys
        .iter()
        .find(|k| Jwk::from_verifying_key(k).kid == jws.header.kid)
        .ok_or(bad_subject("subject_token key unknown"))?;
    verify_es256(&jws, key).map_err(|_| bad_subject("subject_token signature invalid"))?;
    let payload = jws
        .payload_bytes()
        .map_err(|_| bad_subject("malformed subject_token"))?;
    let object: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&payload).map_err(|_| bad_subject("malformed subject_token"))?;
    if object.contains_key("act") {
        return Err(bad_subject("subject_token is already an exchanged token"));
    }
    let c: SubjectClaims = serde_json::from_value(serde_json::Value::Object(object))
        .map_err(|_| bad_subject("subject_token claims incomplete"))?;
    if c.iss != issuer {
        return Err(bad_subject("subject_token issuer mismatch"));
    }
    if !is_edge_subject(&c.sub) {
        return Err(bad_subject("subject_token sub invalid"));
    }
    let strings = [
        &c.aud,
        &c.upstream_iss,
        &c.client_id,
        &c.org_id,
        &c.workspace_id,
        &c.family_id,
        &c.jti,
    ];
    if strings
        .iter()
        .any(|s| s.is_empty() || s.len() > MAX_CLAIM_LEN)
    {
        return Err(bad_subject("subject_token claims invalid"));
    }
    if c.iat > now.saturating_add(SUBJECT_SKEW_SECS)
        || c.exp <= c.iat
        || c.exp - c.iat > ACCESS_TOKEN_TTL_SECS
    {
        return Err(bad_subject("subject_token times invalid"));
    }
    if now >= c.exp {
        return Err(bad_subject("subject_token expired"));
    }
    Ok(c)
}

/// The exchange scope rule (ADR-351 §5.6): the compiled map of the subject
/// token's vocabulary `vocab` ([`exchange_scope`]) of `subject` scopes, in subject order, ∩
/// `requested` (if sent; malformed or non-vocabulary => `invalid_scope`,
/// identity scopes dropped) ∩ the client `ceiling` ∩ the target `entry`'s
/// current scopes, never `offline_access` (no refresh token). Empty =>
/// `invalid_scope`.
pub fn exchange_scopes(
    vocab: Vocabulary,
    subject: &str,
    requested: Option<&str>,
    ceiling: &[String],
    entry: &ResourceEntry,
) -> Result<Vec<String>, OAuthError> {
    let requested = match requested {
        Some(s) => {
            let r = strip_identity_scopes(split_scope(s)?);
            if r.iter().any(|s| !is_edge_vocabulary(s)) {
                return Err(err(OAuthErrorCode::InvalidScope, "unknown scope requested"));
            }
            Some(r)
        }
        None => None,
    };
    let mut out: Vec<String> = Vec::new();
    for s in subject.split(' ').filter_map(|s| exchange_scope(vocab, s)) {
        let keep = s != OFFLINE_ACCESS
            && requested
                .as_ref()
                .map_or(true, |r| r.iter().any(|x| x == s))
            && ceiling.iter().any(|c| c == s)
            && entry.allows(s)
            && !out.iter().any(|o| o == s);
        if keep {
            out.push(s.to_string());
        }
    }
    if out.is_empty() {
        return Err(err(
            OAuthErrorCode::InvalidScope,
            "no subject scope maps to a grantable scope",
        ));
    }
    Ok(out)
}

/// Handle a token-exchange request on `ep`.
///
/// Contract, in order: [`confidential::authenticate`] against
/// `<issuer>/token` (`invalid_client`); `resource` resolves in the current
/// allowlist and is exactly [`GATEWAY_V1_URL`] (`invalid_target`);
/// [`verify_subject_token`] against the AS's published keys; the subject
/// `aud` is one of the client's subject audiences and still allowlisted
/// (`invalid_request`); its `family_id` is not revoked (`invalid_request`);
/// [`exchange_scopes`]. Mints a `…/v1` token with the subject's `sub`,
/// `upstream_iss`, `org_id`, `workspace_id` and `family_id`, `client_id` =
/// `act.sub` = the adapter, `exp = min(now + ACCESS_TOKEN_TTL_SECS,
/// subject exp)`; responds with `issued_token_type` and no refresh token.
pub fn exchange(
    ep: &TokenEndpoint<'_>,
    req: &ExchangeRequest,
) -> Result<TokenResponse, OAuthError> {
    let now = ep.clock.now_unix();
    let token_endpoint = format!("{}{}", ep.issuer, paths::TOKEN);
    let client = confidential::authenticate(
        ep.confidential,
        ep.assertions,
        &token_endpoint,
        now,
        &req.client_assertion,
        req.client_id.as_deref(),
    )?;
    let entry = ep.resources.resolve_entry(req.resource.as_deref())?;
    if entry.url().as_str() != GATEWAY_V1_URL {
        return Err(err(
            OAuthErrorCode::InvalidTarget,
            "token exchange targets the gateway /v1 resource only",
        ));
    }
    let subject = verify_subject_token(
        &req.subject_token,
        ep.issuer,
        &ep.signer.verifying_keys(),
        now,
    )?;
    // The subject must be a token for an allowlisted adapter resource this
    // client registered; a gateway (`ruvector:*`) token never is.
    let vocab = ruvector_edge_auth::ResourceUrl::parse(&subject.aud)
        .ok()
        .filter(|u| ep.resources.get(u).is_some())
        .and_then(|u| Vocabulary::for_resource(&u))
        .filter(|v| *v != Vocabulary::Ruvector && client.allows_subject_audience(&subject.aud));
    let Some(vocab) = vocab else {
        return Err(bad_subject(
            "subject_token audience not exchangeable by this client",
        ));
    };
    if ep.refresh.is_family_revoked(&subject.family_id)? {
        return Err(bad_subject("subject_token revoked"));
    }
    let scopes = exchange_scopes(
        vocab,
        &subject.scope,
        req.scope.as_deref(),
        client.scope(),
        entry,
    )?;
    let exp = now.saturating_add(ACCESS_TOKEN_TTL_SECS).min(subject.exp);
    let claims = AccessTokenClaims {
        iss: ep.issuer.to_string(),
        aud: entry.url().as_str().to_string(),
        sub: subject.sub,
        upstream_iss: subject.upstream_iss,
        client_id: client.client_id().to_string(),
        org_id: subject.org_id,
        workspace_id: subject.workspace_id,
        family_id: subject.family_id,
        scope: scopes.join(" "),
        jti: crate::random_secret(ep.rng, JTI_BYTES)?,
        iat: now,
        exp,
        act: Some(Act {
            sub: client.client_id().to_string(),
        }),
    };
    let access_token = sign_claims(ep.signer, &claims)?;
    let audit = ExchangeAudit {
        client_id: claims.client_id,
        sub: claims.sub,
        family_id: claims.family_id,
        jti: claims.jti,
        scope: claims.scope.clone(),
        exp,
    };
    Ok(TokenResponse {
        access_token,
        token_type: "Bearer",
        expires_in: exp - now,
        refresh_token: None,
        scope: claims.scope,
        issued_token_type: Some(ACCESS_TOKEN_TYPE),
        audit: Some(audit),
    })
}

/// What the AS records for every successful exchange (ADR-351 §5.6: `act`
/// audited): the adapter, the user it acted for, the grant family and the
/// new token's `jti`, so a gateway `ops.act_sub` row traces back to the
/// mint. No token, assertion or upstream value.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ExchangeAudit {
    /// Adapter `client_id` (= `act.sub`).
    pub client_id: String,
    /// Edge subject the adapter acts for.
    pub sub: String,
    /// Grant family of the subject token.
    pub family_id: String,
    /// `jti` of the minted `…/v1` token.
    pub jti: String,
    /// Granted `…/v1` scopes.
    pub scope: String,
    /// Expiry of the minted token.
    pub exp: u64,
}

impl ExchangeAudit {
    /// One structured log line (`{"event":"token_exchange",…}`).
    pub fn log_line(&self) -> String {
        serde_json::json!({ "event": "token_exchange", "exchange": self }).to_string()
    }
}
