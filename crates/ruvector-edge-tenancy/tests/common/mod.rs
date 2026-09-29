//! Shared claim fixtures. Claims are built through the **real** verifier
//! path (`ruvector_edge_auth::claims::validate`) so fixtures carry exactly
//! what the verifier would produce (jti, family_id, the per-kind `iss`).
//! If the auth crate's `validate` signature changes, fix it here only.

#![allow(dead_code)]

use ruvector_edge_auth::claims::{validate, Audience, ClaimsPolicy, RawClaims};
use ruvector_edge_auth::{TokenKind, VerifiedClaims};
use ruvector_edge_tenancy::UPSTREAM_ISSUER;

/// Edge AS issuer (dev hostname). Must never feed the tenant key.
pub const EDGE_ISS: &str = "https://ruvector-edge-auth.cognitum-consulting-mail.workers.dev";
/// Edge MCP resource.
pub const AUD: &str = "https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev/v1/mcp";
/// A well-formed edge subject.
pub const SUB: &str = "es1_abcdefghijklmnopqrstuvwxyz";
/// A second well-formed edge subject.
pub const SUB2: &str = "es1_234567abcdefghijklmnopqrst";
/// A raw upstream subject as `auth.cognitum.one` mints it.
pub const UPSTREAM_SUB: &str = "8d0c7a52-4f1e-4b8a-9c3d-2e5f6a7b8c9d";
/// 16-byte base64url ids (canonical: final char has zero low bits).
pub const JTI: &str = "abcdefghijklmnopqrstuA";
/// 16-byte base64url family id.
pub const FAMILY: &str = "fam0fam0fam0fam0fam0fQ";
/// First-party CLI client id on the upstream allowlist.
pub const CLI_CLIENT: &str = "cognitum-cli";
/// Scope granting read (M0 swaps to `ruvector:read`; change only here).
pub const READ_SCOPE: &str = "ruvector:read";
/// Scope granting write + create (M0 swaps to `ruvector:write`).
pub const WRITE_SCOPE: &str = "ruvector:write";
/// Both, as a `scope` claim.
pub const READ_WRITE_SCOPES: &str = "ruvector:read ruvector:write";
/// Verification time.
pub const NOW: u64 = 10_000;

/// Token fixture.
#[derive(Clone)]
pub struct Tok {
    pub kind: TokenKind,
    pub iss: &'static str,
    pub aud: &'static str,
    pub sub: String,
    pub client_id: &'static str,
    pub org: &'static str,
    pub ws: &'static str,
    pub scope: &'static str,
    pub jti: Option<&'static str>,
    pub family_id: Option<&'static str>,
}

/// An edge-issued access token as minted by `ruvector-edge-authz`.
pub fn edge() -> Tok {
    Tok {
        kind: TokenKind::EdgeIssued,
        iss: EDGE_ISS,
        aud: AUD,
        sub: SUB.into(),
        client_id: "dcr-client-123",
        org: "org-1",
        ws: "ws-1",
        scope: READ_WRITE_SCOPES,
        jti: Some(JTI),
        family_id: Some(FAMILY),
    }
}

/// An `auth.cognitum.one` token in §5.5 first-party mode, carrying the raw
/// upstream `sub` (a UUID); `TenantContext` normalises it.
pub fn upstream() -> Tok {
    Tok {
        kind: TokenKind::UpstreamFirstParty,
        iss: UPSTREAM_ISSUER,
        aud: CLI_CLIENT,
        sub: UPSTREAM_SUB.into(),
        client_id: CLI_CLIENT,
        org: "org-1",
        ws: "ws-1",
        scope: "openid profile",
        jti: Some("upstream-jti-1"),
        family_id: Some("upstream-family-1"),
    }
}

impl Tok {
    /// Run the fixture through the real claim validator.
    pub fn verify(self) -> VerifiedClaims {
        let raw = RawClaims {
            iss: Some(self.iss.into()),
            aud: Some(Audience::One(self.aud.into())),
            sub: Some(self.sub),
            exp: Some(NOW + 600),
            iat: Some(NOW - 10),
            jti: self.jti.map(Into::into),
            scope: Some(self.scope.into()),
            client_id: Some(self.client_id.into()),
            org_id: Some(self.org.into()),
            workspace_id: Some(self.ws.into()),
            family_id: self.family_id.map(Into::into),
            // The verifier requires `typ = "access"` on upstream tokens.
            typ: (self.kind == TokenKind::UpstreamFirstParty).then(|| "access".into()),
            ..RawClaims::default()
        };
        validate(raw, &ClaimsPolicy::with_defaults(self.iss), self.kind, NOW)
            .expect("fixture must pass the real verifier")
    }
}
