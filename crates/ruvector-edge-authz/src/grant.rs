//! Token endpoint orchestration: `authorization_code` and `refresh_token`
//! grants for public clients, and the RFC 8693 exchange grant for
//! operator-registered confidential clients ([`crate::exchange`]), over the
//! ports.

use crate::client::ClientRecord;
use crate::confidential::ConfidentialClients;
use crate::error::{OAuthError, OAuthErrorCode};
use crate::federation::UpstreamIdentity;
use crate::ports::{
    AssertionReplayStore, ClientStore, Clock, CodeStore, RefreshStore, Rng, Signer,
};
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
    /// Operator-registered confidential (adapter) clients: the only clients
    /// of the exchange grant.
    pub confidential: &'a ConfidentialClients,
    /// RFC 7523 client-assertion `jti` replay cache.
    pub assertions: &'a dyn AssertionReplayStore,
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
    /// be allowlisted (`invalid_target`) and the minted `scope` is
    /// re-intersected with that resource's current scopes. A refresh token
    /// is issued on code redemption iff the client registered the
    /// `refresh_token` grant (ADR-351 §5.3/§5.6: `offline_access` is only
    /// accepted and echoed, it does not gate refresh). The access token's
    /// `family_id` is the refresh family (a fresh grant id when no refresh
    /// token is issued).
    ///
    /// The exchange grant is dispatched first to
    /// [`crate::exchange::exchange`]: its clients are confidential and never
    /// in the [`ClientStore`] (and a DCR client is never confidential).
    pub fn handle(&self, req: &TokenRequest) -> Result<TokenResponse, OAuthError> {
        let client_id = match req {
            TokenRequest::TokenExchange(x) => return crate::exchange::exchange(self, x),
            TokenRequest::AuthorizationCode { client_id, .. }
            | TokenRequest::RefreshToken { client_id, .. } => client_id,
        };
        let client = self.client(client_id)?;
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
                let scopes = self.current_scopes(&auth.resource, &auth.scopes)?;
                let family_id = crate::refresh::new_family_id(self.rng)?;
                crate::code::record_redemption(self.codes, &rec, &family_id)?;
                let refresh_token = if client.allows_grant(GRANT_REFRESH) {
                    let (token, _) = crate::refresh::issue_refresh(
                        self.refresh,
                        self.rng,
                        self.clock,
                        &family_id,
                        client_id,
                        &rec.identity,
                        &auth.resource,
                        &scopes,
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
                    &scopes,
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
                let scopes = self.current_scopes(&r.resource, &pending.granted_scopes)?;
                let mut response = self.respond(
                    client_id,
                    &r.identity,
                    &r.resource,
                    &r.family_id,
                    &scopes,
                    None,
                )?;
                let rotated = pending.commit(self.refresh)?;
                response.refresh_token = Some(rotated.token);
                Ok(response)
            }
            // Dispatched above; kept total without a panic path.
            TokenRequest::TokenExchange(_) => Err(OAuthError::new(
                OAuthErrorCode::UnsupportedGrantType,
                "unsupported grant_type",
            )),
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

    /// `scopes` ∩ the bound resource's **current** scopes (config may have
    /// narrowed them since the grant). The resource must still be
    /// allowlisted (`invalid_target`) and something besides
    /// `offline_access` must remain (`invalid_scope`).
    fn current_scopes(
        &self,
        resource: &ResourceUrl,
        scopes: &[String],
    ) -> Result<Vec<String>, OAuthError> {
        let entry = self.resources.get(resource).ok_or(OAuthError::new(
            OAuthErrorCode::InvalidTarget,
            "resource no longer allowed",
        ))?;
        let kept: Vec<String> = scopes.iter().filter(|s| entry.allows(s)).cloned().collect();
        if kept.iter().all(|s| s == OFFLINE_ACCESS) {
            return Err(OAuthError::new(
                OAuthErrorCode::InvalidScope,
                "scope no longer allowed for this resource",
            ));
        }
        Ok(kept)
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
                act: None,
            },
        )?;
        Ok(TokenResponse {
            access_token,
            token_type: "Bearer",
            expires_in: ACCESS_TOKEN_TTL_SECS,
            refresh_token,
            scope: scopes.join(" "),
            issued_token_type: None,
            audit: None,
        })
    }
}
