//! `POST /register` — RFC 7591 Dynamic Client Registration (public clients).

use super::{AuthStores, Ctx};
use crate::abuse::{self, CLIENT_IDLE_TTL_SECS, DCR_RATE_WINDOW_SECS, UNUSED_CLIENT_TTL_SECS};
use crate::http::{is_json, Reply, MAX_JSON_BODY};
use ruvector_edge_authz::client::{register_client, RegistrationRequest};
use ruvector_edge_authz::{Clock, OAuthError, OAuthErrorCode, Rng, Signer};

fn metadata_error(desc: &'static str) -> OAuthError {
    OAuthError::new(OAuthErrorCode::InvalidClientMetadata, desc)
}

/// Register a client: JSON body <= 16 KiB, a durable per-IP window
/// (`client_key` is the salted IP bucket, [`abuse::ip_bucket`]; 429 over
/// `DCR_RATE_PER_HOUR`), idle-client expiry, then the global cap over the
/// live clients, then the core's validation and `edc-` id minting.
/// 201 + RFC 7591 §3.2.1 body.
pub fn register<S: AuthStores, R: Rng, C: Clock, G: Signer>(
    ctx: &Ctx<'_, S, R, C, G>,
    client_key: &str,
    content_type: Option<&str>,
    body: &[u8],
) -> Reply {
    if !is_json(content_type) {
        return Reply::oauth_error(&metadata_error("application/json body required"));
    }
    if body.len() > MAX_JSON_BODY {
        let mut r = Reply::oauth_error(&metadata_error("registration body too large"));
        r.status = 413;
        return r;
    }
    let now = ctx.clock.now_unix();
    match ctx
        .store
        .dcr_rate_hit(client_key, now, DCR_RATE_WINDOW_SECS)
    {
        Ok(n) if n <= ctx.cfg.dcr_rate_per_hour => {}
        Ok(_) => return abuse::rate_limited(DCR_RATE_WINDOW_SECS),
        Err(e) => return Reply::oauth_error(&e.into()),
    }
    // Abandoned registrations must not hold the cap. The purge scans the
    // client and refresh tables, so it runs only when the cap is reached.
    let under_cap = |store: &S| store.client_count().map(|n| n < ctx.cfg.max_clients);
    let mut room = under_cap(ctx.store);
    if matches!(room, Ok(false)) {
        let _ = ctx.store.purge_idle_clients(
            now,
            now.saturating_sub(UNUSED_CLIENT_TTL_SECS),
            now.saturating_sub(CLIENT_IDLE_TTL_SECS),
        );
        room = under_cap(ctx.store);
    }
    match room {
        Ok(true) => {}
        Ok(false) => {
            return Reply::oauth_error(&OAuthError::new(
                OAuthErrorCode::TemporarilyUnavailable,
                "client registration limit reached",
            ))
        }
        Err(e) => return Reply::oauth_error(&e.into()),
    }
    let result = RegistrationRequest::from_json(body).and_then(|req| {
        register_client(ctx.store, ctx.rng, ctx.clock, &ctx.cfg.dcr_policy(), &req)
    });
    match result {
        Ok(record) => Reply::json(201, &record.to_response()),
        Err(e) => Reply::oauth_error(&e),
    }
}
