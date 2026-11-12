//! Token endpoint: request parsing, access-token minting (RFC 9068 JWT,
//! ES256), and the response body.

use crate::error::{OAuthError, OAuthErrorCode};
use crate::federation::UpstreamIdentity;
use crate::params::Params;
use crate::ports::{Clock, Rng, Signer};
use ruvector_edge_auth::jws::b64url_encode;
use ruvector_edge_auth::ResourceUrl;
use serde::Serialize;

/// Access-token lifetime (seconds). Kept <= the RS `max_lifetime_secs` and
/// <= the 15 min the DECISION requires.
pub const ACCESS_TOKEN_TTL_SECS: u64 = 900;
/// Random bytes in `jti`.
pub const JTI_BYTES: usize = 16;

/// Parsed `application/x-www-form-urlencoded` token request. `Debug`
/// redacts the code, verifier and refresh token.
#[derive(Clone, PartialEq, Eq)]
pub enum TokenRequest {
    /// `grant_type=authorization_code`.
    AuthorizationCode {
        /// The code.
        code: String,
        /// Must equal the authorization request's.
        redirect_uri: String,
        /// Public client id.
        client_id: String,
        /// PKCE verifier.
        code_verifier: String,
        /// Optional RFC 8707 resource (must equal the authorized one).
        resource: Option<String>,
    },
    /// `grant_type=refresh_token`.
    RefreshToken {
        /// Presented refresh token.
        refresh_token: String,
        /// Public client id.
        client_id: String,
        /// Optional down-scoping.
        scope: Option<String>,
        /// Optional RFC 8707 resource (must equal the family's).
        resource: Option<String>,
    },
    /// `grant_type=urn:ietf:params:oauth:grant-type:token-exchange`
    /// (RFC 8693; operator-registered confidential clients only).
    TokenExchange(crate::exchange::ExchangeRequest),
}

impl std::fmt::Debug for TokenRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::REDACTED as R;
        match self {
            TokenRequest::AuthorizationCode {
                redirect_uri,
                client_id,
                resource,
                ..
            } => f
                .debug_struct("AuthorizationCode")
                .field("code", &R)
                .field("redirect_uri", redirect_uri)
                .field("client_id", client_id)
                .field("code_verifier", &R)
                .field("resource", resource)
                .finish(),
            TokenRequest::RefreshToken {
                client_id,
                scope,
                resource,
                ..
            } => f
                .debug_struct("RefreshToken")
                .field("refresh_token", &R)
                .field("client_id", client_id)
                .field("scope", scope)
                .field("resource", resource)
                .finish(),
            TokenRequest::TokenExchange(x) => x.fmt(f),
        }
    }
}

impl TokenRequest {
    /// Parse form pairs.
    ///
    /// Contract: parameters without a value count as omitted; repeated
    /// parameters => `invalid_request` (`invalid_target` for `resource`);
    /// each value <= 2 KiB; `client_secret` or `client_assertion` present =>
    /// `invalid_client` (public clients only); `client_id` required;
    /// `grant_type` missing => `invalid_request`, unknown =>
    /// `unsupported_grant_type`; grant-specific members required. The
    /// token-exchange grant is decided first and parsed by
    /// [`crate::exchange::ExchangeRequest::from_params`] (confidential
    /// clients: `client_assertion` required there).
    pub fn from_form(pairs: &[(String, String)]) -> Result<Self, OAuthError> {
        let p = Params::from_pairs(pairs)?;
        if p.get("grant_type") == Some(crate::exchange::TOKEN_EXCHANGE_GRANT) {
            return crate::exchange::ExchangeRequest::from_params(&p)
                .map(TokenRequest::TokenExchange);
        }
        if p.get("client_secret").is_some() || p.get("client_assertion").is_some() {
            return Err(OAuthError::new(
                OAuthErrorCode::InvalidClient,
                "only public clients are supported",
            ));
        }
        let grant = p.require("grant_type", "grant_type required")?;
        let client_id = p.require("client_id", "client_id required")?;
        match grant.as_str() {
            "authorization_code" => Ok(TokenRequest::AuthorizationCode {
                code: p.require("code", "code required")?,
                redirect_uri: p.require("redirect_uri", "redirect_uri required")?,
                client_id,
                code_verifier: p.require("code_verifier", "code_verifier required")?,
                resource: p.take("resource"),
            }),
            "refresh_token" => Ok(TokenRequest::RefreshToken {
                refresh_token: p.require("refresh_token", "refresh_token required")?,
                client_id,
                scope: p.take("scope"),
                resource: p.take("resource"),
            }),
            _ => Err(OAuthError::new(
                OAuthErrorCode::UnsupportedGrantType,
                "unsupported grant_type",
            )),
        }
    }

    /// The public client's `client_id` member; `None` for the exchange
    /// grant, whose client is known only once its assertion verifies.
    pub fn client_id(&self) -> Option<&str> {
        match self {
            TokenRequest::AuthorizationCode { client_id, .. }
            | TokenRequest::RefreshToken { client_id, .. } => Some(client_id),
            TokenRequest::TokenExchange(_) => None,
        }
    }
}

/// Claims of an edge-minted access token (RFC 9068).
///
/// Carries exactly the ADR-351 §5.2 claim set: `iss`, `aud`, `sub` (the edge
/// subject), `client_id`, `scope`, `jti`, `iat`, `exp`, `family_id`,
/// `upstream_iss`, `org_id`, `workspace_id`, plus `act` **only** on tokens
/// minted by the RFC 8693 exchange grant (M1). The raw upstream `sub` is
/// never carried.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AccessTokenClaims {
    /// Edge AS issuer.
    pub iss: String,
    /// Exact resource URL.
    pub aud: String,
    /// Edge subject `es1_…` ([`ruvector_edge_auth::subject::edge_subject`]).
    pub sub: String,
    /// Upstream issuer the identity was verified against (tenant namespace).
    pub upstream_iss: String,
    /// Downstream client.
    pub client_id: String,
    /// Tenant input, copied from the upstream identity.
    pub org_id: String,
    /// Tenant input, copied from the upstream identity.
    pub workspace_id: String,
    /// Grant / refresh family id (ADR §5.6 deny-list key).
    pub family_id: String,
    /// Space-separated granted scopes.
    pub scope: String,
    /// Unique token id (random).
    pub jti: String,
    /// Issued-at.
    pub iat: u64,
    /// Expiry (`iat + ACCESS_TOKEN_TTL_SECS`).
    pub exp: u64,
    /// RFC 8693 §4.1 actor: present only on exchanged tokens, omitted from
    /// the JSON otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub act: Option<Act>,
}

/// RFC 8693 §4.1 `act` claim: the adapter client acting for the user.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Act {
    /// The adapter's `client_id`.
    pub sub: String,
}

/// Inputs for [`mint_access_token`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MintRequest<'a> {
    /// Edge AS issuer URL.
    pub issuer: &'a str,
    /// Target resource (already resolved against the allowlist).
    pub resource: &'a ResourceUrl,
    /// Downstream client.
    pub client_id: &'a str,
    /// User identity.
    pub identity: &'a UpstreamIdentity,
    /// Grant / refresh family id.
    pub family_id: &'a str,
    /// Granted scopes.
    pub scopes: &'a [String],
    /// Acting adapter (`act.sub`) for exchanged tokens; `None` for the
    /// authorization-code and refresh grants.
    pub act: Option<&'a str>,
}

#[derive(Serialize)]
struct Header<'a> {
    alg: &'static str,
    typ: &'static str,
    kid: &'a str,
}

fn server_error(desc: &'static str) -> OAuthError {
    OAuthError::new(OAuthErrorCode::ServerError, desc)
}

/// Mint a compact JWS access token.
///
/// Contract: header `{"alg":"ES256","typ":"at+jwt","kid":signer.kid()}`
/// (empty `kid` => `server_error`); claims per [`AccessTokenClaims`] with
/// `aud = resource.as_str()`, `sub` = the identity's edge subject,
/// `upstream_iss` from the identity (empty issuer or `sub` =>
/// `server_error`), `iat = now`, `exp = now +
/// ACCESS_TOKEN_TTL_SECS`, `jti` from [`JTI_BYTES`] RNG bytes; signature via
/// [`Signer::sign_es256`] over `b64(header).b64(claims)`. Returns the token
/// and its claims.
pub fn mint_access_token<S, R, C>(
    signer: &S,
    rng: &R,
    clock: &C,
    req: &MintRequest<'_>,
) -> Result<(String, AccessTokenClaims), OAuthError>
where
    S: Signer + ?Sized,
    R: Rng + ?Sized,
    C: Clock + ?Sized,
{
    let kid = signer.kid();
    if kid.is_empty() {
        return Err(server_error("signing key unavailable"));
    }
    if req.identity.upstream_iss.is_empty() || req.identity.sub.is_empty() {
        return Err(server_error("identity incomplete"));
    }
    let now = clock.now_unix();
    let claims = AccessTokenClaims {
        iss: req.issuer.to_string(),
        aud: req.resource.as_str().to_string(),
        sub: req.identity.edge_subject(),
        upstream_iss: req.identity.upstream_iss.clone(),
        client_id: req.client_id.to_string(),
        org_id: req.identity.org_id.clone(),
        workspace_id: req.identity.workspace_id.clone(),
        family_id: req.family_id.to_string(),
        scope: req.scopes.join(" "),
        jti: crate::random_secret(rng, JTI_BYTES)?,
        iat: now,
        exp: now.saturating_add(ACCESS_TOKEN_TTL_SECS),
        act: req.act.map(|sub| Act {
            sub: sub.to_string(),
        }),
    };
    Ok((sign_claims(signer, &claims)?, claims))
}

/// Sign `claims` as a compact JWS with header `{"alg":"ES256","typ":
/// "at+jwt","kid":signer.kid()}` (empty `kid` => `server_error`). Shared by
/// every grant, so the JOSE header is defined once.
pub(crate) fn sign_claims<S: Signer + ?Sized>(
    signer: &S,
    claims: &AccessTokenClaims,
) -> Result<String, OAuthError> {
    let kid = signer.kid();
    if kid.is_empty() {
        return Err(server_error("signing key unavailable"));
    }
    let header = Header {
        alg: "ES256",
        typ: "at+jwt",
        kid: &kid,
    };
    let signing_input = format!("{}.{}", b64_json(&header)?, b64_json(claims)?);
    let sig = signer.sign_es256(signing_input.as_bytes())?;
    Ok(format!("{signing_input}.{}", b64url_encode(&sig)))
}

fn b64_json<T: Serialize>(v: &T) -> Result<String, OAuthError> {
    serde_json::to_vec(v)
        .map(|b| b64url_encode(&b))
        .map_err(|_| server_error("encode token"))
}

/// RFC 6749 §5.1 success body (`Cache-Control: no-store` is the Worker's job).
/// `Debug` redacts both tokens.
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct TokenResponse {
    /// The JWT.
    pub access_token: String,
    /// Always `Bearer`.
    pub token_type: &'static str,
    /// Seconds until expiry.
    pub expires_in: u64,
    /// Rotated refresh token, if the client has the grant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    /// Granted scope.
    pub scope: String,
    /// RFC 8693 §2.2.1 `issued_token_type`: only on exchange responses.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub issued_token_type: Option<&'static str>,
    /// Exchange only: the mint the AS audits (ADR-351 §5.6); never
    /// serialised to the client.
    #[serde(skip)]
    pub audit: Option<crate::exchange::ExchangeAudit>,
}

impl std::fmt::Debug for TokenResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenResponse")
            .field("access_token", &crate::REDACTED)
            .field("token_type", &self.token_type)
            .field("expires_in", &self.expires_in)
            .field(
                "refresh_token",
                &self.refresh_token.as_ref().map(|_| crate::REDACTED),
            )
            .field("scope", &self.scope)
            .finish()
    }
}
