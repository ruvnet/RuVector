//! `POST /token` (authorization_code + refresh_token grants for DCR
//! clients; the RFC 8693 token-exchange grant for operator-registered
//! confidential clients, `private_key_jwt`) and
//! `POST /revoke` (RFC 7009). Form-encoded bodies only; every response is
//! `Cache-Control: no-store`.

use super::{AuthStores, Ctx};
use crate::http::{read_form, Reply};
use ruvector_edge_authz::revoke::{revoke as revoke_token, RevocationRequest};
use ruvector_edge_authz::token::TokenRequest;
use ruvector_edge_authz::{Clock, Rng, Signer, TokenEndpoint};

/// Handle `POST /token`.
pub fn token<S: AuthStores, R: Rng, C: Clock, G: Signer>(
    ctx: &Ctx<'_, S, R, C, G>,
    content_type: Option<&str>,
    body: &[u8],
) -> Reply {
    let pairs = match read_form(content_type, body) {
        Ok(p) => p,
        Err(r) => return r,
    };
    let endpoint = TokenEndpoint {
        issuer: &ctx.cfg.issuer,
        resources: &ctx.cfg.resources,
        clients: ctx.store,
        codes: ctx.store,
        refresh: ctx.store,
        signer: ctx.signer,
        rng: ctx.rng,
        clock: ctx.clock,
        confidential: &ctx.cfg.confidential_clients,
        assertions: ctx.store,
    };
    let result = TokenRequest::from_form(&pairs).and_then(|req| {
        let resp = endpoint.handle(&req)?;
        // Idle-expiry bookkeeping (best effort): a DCR client that completes
        // token requests is live. Confidential (exchange) clients are
        // operator config and never expire.
        if let Some(client_id) = req.client_id() {
            let _ = ctx.store.touch_client(client_id, ctx.clock.now_unix());
        }
        Ok(resp)
    });
    match result {
        Ok(resp) => {
            let mut r = Reply::json(200, &resp);
            r.log = resp.audit.as_ref().map(|a| a.log_line());
            r.headers.push(("Pragma", "no-cache".to_string()));
            r
        }
        Err(e) => Reply::oauth_error(&e),
    }
}

/// Handle `POST /revoke`: 200 with an empty body for any well-formed request
/// (unknown tokens included, RFC 7009 §2.2).
pub fn revoke<S: AuthStores, R: Rng, C: Clock, G: Signer>(
    ctx: &Ctx<'_, S, R, C, G>,
    content_type: Option<&str>,
    body: &[u8],
) -> Reply {
    let pairs = match read_form(content_type, body) {
        Ok(p) => p,
        Err(r) => return r,
    };
    let result = RevocationRequest::from_form(&pairs).and_then(|req| {
        revoke_token(
            ctx.store,
            &req.token,
            &req.client_id,
            req.token_type_hint.as_deref(),
        )
    });
    match result {
        Ok(()) => Reply::empty(200),
        Err(e) => Reply::oauth_error(&e),
    }
}
