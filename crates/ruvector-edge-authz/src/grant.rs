//! Token endpoint orchestration: `authorization_code` and `refresh_token`
//! grants over the ports.

use crate::client::ClientRecord;
use crate::error::{OAuthError, OAuthErrorCode};
use crate::federation::UpstreamIdentity;
use crate::ports::{ClientStore, Clock, CodeStore, RefreshStore, Rng, Signer};
use crate::refresh::OFFLINE_ACCESS;
use crate::resource::ResourceAllowlist;
use crate::token::ACCESS_TOKEN_TTL_SECS;
use crate::token::{mint_access_token, MintRequest, TokenRequest, TokenResponse};
use ruvector_edge_auth::ResourceUrl;

/// Everything the token endpoint needs. Object-safe ports so the Worker can
/// pass one Durable Object handle for every store.
pub struct TokenEndpoint<'a> {
    /// Edge AS issuer (`iss`).
    pub issuer: &'a str,
    /// Current resource allowlist (re-checked at every issuance).
    pub resources: &'a ResourceAllowlist,
    /// Registered clients.
    pub clients: &'a dyn ClientStore,
    /// Authorization codes.
    pub codes: &'a dyn CodeStore,
    /// Refresh tokens.
    pub refresh: &'a dyn RefreshStore,
    /// Access-token signer.
    pub signer: &'a dyn Signer,
    /// Randomness.
    pub rng: &'a dyn Rng,
    /// Time.
    pub clock: &'a dyn Clock,
}

const GRANT_REFRESH: &str = "refresh_token";

impl TokenEndpoint<'_> {
    /// Handle a parsed token request.
    ///
    /// Contract: unknown `client_id` => `invalid_client`; the client must have
    /// registered the grant (`unauthorized_client`); code redemption via
    /// [`crate::code::redeem_code`] — a replay of an already redeemed code
    /// revokes the grant that code started, then `invalid_grant`; the new
    /// grant id is tombstoned against the code hash before any token is
    /// issued. Refresh: [`crate::refresh::prepare_rotation`], resource
    /// re-check and access-token minting, then
    /// [`crate::refresh::PendingRotation::commit`] (nothing is consumed
    /// unless the response can be produced). The bound resource must still
    /// be allowlisted (`invalid_target`). A refresh token is issued on code
    /// redemption only if `offline_access` was granted **and** the client
    /// registered `refresh_token` (ADR-351 §5.3). The access token's
    /// `family_id` is the refresh family (a fresh grant id when no refresh
    /// token is issued).
    pub fn handle(&self, req: &TokenRequest) -> Result<TokenResponse, OAuthError> {
        let client = self.client(req.client_id())?;
        match req {
            TokenRequest::AuthorizationCode {
                code,
                redirect_uri,
                client_id,
                code_verifier,
                resource,
            } => {
                let rec = match crate::code::redeem_code(
                    self.codes,
                    self.clock,
                    code,
                    client_id,
                    redirect_uri,
                    code_verifier,
                    resource.as_deref(),
                ) {
                    Ok(rec) => rec,
                    Err(e) => return Err(self.on_code_failure(code, e)),
                };
                let auth = &rec.authorization;
                self.ensure_allowed(&auth.resource)?;
                let family_id = crate::refresh::new_family_id(self.rng)?;
                crate::code::record_redemption(self.codes, &rec, &family_id)?;
                let offline = auth.scopes.iter().any(|s| s == OFFLINE_ACCESS);
                let refresh_token = if offline && client.allows_grant(GRANT_REFRESH) {
                    let (token, _) = crate::refresh::issue_refresh(
                        self.refresh,
                        self.rng,
                        self.clock,
                        &family_id,
                        client_id,
                        &rec.identity,
                        &auth.resource,
                        &auth.scopes,
                    )?;
                    Some(token)
                } else {
                    None
                };
                self.respond(
                    client_id,
                    &rec.identity,
                    &auth.resource,
                    &family_id,
                    &auth.scopes,
                    refresh_token,
                )
            }
            TokenRequest::RefreshToken {
                refresh_token,
                client_id,
                scope,
                resource,
            } => {
                if !client.allows_grant(GRANT_REFRESH) {
                    return Err(OAuthError::new(
                        OAuthErrorCode::UnauthorizedClient,
                        "client may not use refresh_token",
                    ));
                }
                // Validate and mint first; consume the old token last, so a
                // failure here never turns the client's retry into reuse.
                let pending = crate::refresh::prepare_rotation(
                    self.refresh,
                    self.rng,
                    self.clock,
                    refresh_token,
                    client_id,
                    scope.as_deref(),
                    resource.as_deref(),
                )?;
                let r = &pending.record;
                self.ensure_allowed(&r.resource)?;
                let mut response = self.respond(
                    client_id,
                    &r.identity,
                    &r.resource,
                    &r.family_id,
                    &pending.granted_scopes,
                    None,
                )?;
                let rotated = pending.commit(self.refresh)?;
                response.refresh_token = Some(rotated.token);
                Ok(response)
            }
        }
    }

    /// A failed code redemption: if the code is a replay of an already
    /// redeemed one, revoke the grant it started (RFC 6749 §4.1.2). The
    /// client still gets the original `invalid_grant`.
    fn on_code_failure(&self, code: &str, err: OAuthError) -> OAuthError {
        if err.error != OAuthErrorCode::InvalidGrant {
            return err;
        }
        match crate::code::replayed_family(self.codes, self.clock, code) {
            Ok(Some(family)) => match self.refresh.revoke_family(&family) {
                Ok(()) => err,
                Err(e) => e.into(),
            },
            Ok(None) => err,
            Err(e) => e,
        }
    }

    fn client(&self, client_id: &str) -> Result<ClientRecord, OAuthError> {
        self.clients.get_client(client_id)?.ok_or(OAuthError::new(
            OAuthErrorCode::InvalidClient,
            "unknown client",
        ))
    }

    fn ensure_allowed(&self, resource: &ResourceUrl) -> Result<(), OAuthError> {
        if self.resources.resources().contains(resource) {
            Ok(())
        } else {
            Err(OAuthError::new(
                OAuthErrorCode::InvalidTarget,
                "resource no longer allowed",
            ))
        }
    }

    fn respond(
        &self,
        client_id: &str,
        identity: &UpstreamIdentity,
        resource: &ResourceUrl,
        family_id: &str,
        scopes: &[String],
        refresh_token: Option<String>,
    ) -> Result<TokenResponse, OAuthError> {
        let (access_token, _) = mint_access_token(
            self.signer,
            self.rng,
            self.clock,
            &MintRequest {
                issuer: self.issuer,
                resource,
                client_id,
                identity,
                family_id,
                scopes,
            },
        )?;
        Ok(TokenResponse {
            access_token,
            token_type: "Bearer",
            expires_in: ACCESS_TOKEN_TTL_SECS,
            refresh_token,
            scope: scopes.join(" "),
        })
    }
}
