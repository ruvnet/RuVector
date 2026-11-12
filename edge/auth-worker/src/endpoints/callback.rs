//! `GET /callback` — return from `auth.cognitum.one`: consume the one-time
//! flow (state + browser-binding cookie), exchange the upstream code,
//! verify the upstream access token, issue our code, redirect to the client.

use super::{consent, deliver, AuthStores, Ctx};
use crate::http::{parse_query, Reply};
use crate::upstream::{
    check_id_token, exchange, revoke_refresh, verify_access_token, UpstreamHttp,
};
use ruvector_edge_auth::KeySource;
use ruvector_edge_authz::authorize::success_redirect;
use ruvector_edge_authz::code::issue_code;
use ruvector_edge_authz::federation::{
    complete_upstream, identity_from_upstream, upstream_token_form, UpstreamCallback,
};
use ruvector_edge_authz::{AuthorizeError, Clock, OAuthError, OAuthErrorCode, Rng, Signer};

fn with_cleared_cookie(mut r: Reply, name: Option<&str>) -> Reply {
    if let Some(n) = name {
        r.headers.push(("Set-Cookie", consent::clear_cookie(n)));
    }
    r
}

/// Handle `GET /callback?state=..&(code=..|error=..)[&iss=..]`.
///
/// Before the flow is consumed nothing is known about the client, so errors
/// go to the user agent. After it, every failure is redirected to the
/// client's verified redirect URI (`access_denied` for upstream refusals,
/// `server_error` for exchange or verification failures).
pub async fn callback<S, R, C, G, H, K>(
    ctx: &Ctx<'_, S, R, C, G>,
    query: Option<&str>,
    cookie_header: Option<&str>,
    http: &H,
    upstream_keys: K,
) -> Reply
where
    S: AuthStores,
    R: Rng,
    C: Clock,
    G: Signer,
    H: UpstreamHttp,
    K: KeySource,
{
    let issuer = ctx.cfg.issuer.as_str();
    let upstream = &ctx.cfg.upstream;
    let pairs = match parse_query(query) {
        Ok(p) => p,
        Err(e) => return Reply::error_page(&e),
    };
    let cb = match UpstreamCallback::from_pairs(&pairs) {
        Ok(cb) => cb,
        Err(e) => return Reply::error_page(&e),
    };
    let cookie = consent::cookie_name(&cb.state);
    let secret = cookie
        .as_deref()
        .and_then(|n| consent::read_cookie(cookie_header, n))
        .unwrap_or("");
    let flow = match complete_upstream(ctx.store, ctx.clock, &cb.state, secret) {
        Ok(f) => f,
        Err(e) => return with_cleared_cookie(Reply::error_page(&e), cookie.as_deref()),
    };
    let fail = |code: OAuthErrorCode, desc: &'static str| {
        let e = AuthorizeError::Redirect {
            redirect_uri: flow.downstream.redirect_uri.clone(),
            state: flow.downstream.state.clone(),
            error: OAuthError::new(code, desc),
        };
        with_cleared_cookie(deliver(&e, issuer), cookie.as_deref())
    };
    // RFC 9207 mix-up defence: if upstream names an issuer, it must be ours.
    let iss = pairs.iter().find(|(k, _)| k == "iss").map(|(_, v)| v);
    if iss.is_some_and(|v| *v != upstream.issuer) {
        return fail(OAuthErrorCode::AccessDenied, "upstream issuer mismatch");
    }
    let code = match &cb.outcome {
        Ok(code) => code,
        Err(_) => {
            return fail(
                OAuthErrorCode::AccessDenied,
                "upstream login was not completed",
            )
        }
    };
    let form = upstream_token_form(upstream, &flow, code);
    let tokens = match exchange(http, upstream, &form).await {
        Ok(t) => t,
        Err(_) => {
            return fail(
                OAuthErrorCode::ServerError,
                "upstream token exchange failed",
            )
        }
    };
    // ADR-351 §5.6 step 4: the upstream refresh token is never used, so it
    // is revoked at once (best effort), whatever happens next.
    if let Some(rt) = tokens.refresh_token.as_deref() {
        revoke_refresh(
            http,
            &ctx.cfg.upstream_revocation_endpoint,
            &upstream.client_id,
            rt,
        )
        .await;
    }
    let kids = &ctx.cfg.upstream_accepted_kids;
    let claims = match verify_access_token(
        &upstream_keys,
        ctx.clock,
        issuer,
        upstream,
        kids,
        &tokens.access_token,
    )
    .await
    {
        Ok(c) => c,
        Err(_) => return fail(OAuthErrorCode::ServerError, "upstream token rejected"),
    };
    // ADR-351 §5.6 step 3: an ID token, if returned, must carry our nonce.
    if let Some(id_token) = tokens.id_token.as_deref() {
        let checked = check_id_token(
            &upstream_keys,
            ctx.clock,
            upstream,
            kids,
            id_token,
            &flow.nonce,
        )
        .await;
        if let Err(e) = checked {
            return fail(e.error, e.error_description);
        }
    }
    let identity = match identity_from_upstream(&claims, &flow, upstream) {
        Ok(i) => i,
        Err(e) => return fail(e.error, e.error_description),
    };
    let our_code = match issue_code(
        ctx.store,
        ctx.rng,
        ctx.clock,
        flow.downstream.clone(),
        identity,
    ) {
        Ok(c) => c,
        Err(e) => return fail(e.error, e.error_description),
    };
    let location = success_redirect(&flow.downstream, &our_code, issuer);
    with_cleared_cookie(Reply::redirect(&location), cookie.as_deref())
}
