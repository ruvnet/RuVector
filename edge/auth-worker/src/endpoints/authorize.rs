//! `GET /authorize` — validate the downstream request, start the upstream
//! leg, and render the consent page (never an automatic upstream redirect);
//! `POST /authorize/consent` — the verified Continue click.

use super::{consent, deliver, AuthStores, Ctx};
use crate::http::{parse_query, read_form, Reply};
use ruvector_edge_authz::authorize::{
    consent_token, validate_authorization, verify_consent_token, AuthorizationRequest,
};
use ruvector_edge_authz::federation::{begin_upstream, UpstreamConfig, UpstreamFlowState};
use ruvector_edge_authz::pkce::challenge_s256;
use ruvector_edge_authz::{secret_hash, Clock, OAuthError, OAuthErrorCode, Rng, Signer};

fn page_error(code: OAuthErrorCode, desc: &'static str) -> Reply {
    Reply::error_page(&OAuthError::new(code, desc))
}

/// Handle `GET /authorize?...`.
///
/// Errors before the client and redirect URI are validated are shown to the
/// user agent. Later pre-consent errors (including `temporarily_unavailable`
/// while `UPSTREAM_CLIENT_ID` is unset) are redirected with `state` and
/// RFC 9207 `iss` **only** to a verified redirect URI
/// ([`ValidatedAuthorization::error`]); an unverified (anonymously
/// registered) target gets an error page, never a 302.
///
/// [`ValidatedAuthorization::error`]: ruvector_edge_authz::authorize::ValidatedAuthorization::error
pub fn authorize<S: AuthStores, R: Rng, C: Clock, G: Signer>(
    ctx: &Ctx<'_, S, R, C, G>,
    query: Option<&str>,
) -> Reply {
    let issuer = ctx.cfg.issuer.as_str();
    let pairs = match parse_query(query) {
        Ok(p) => p,
        Err(e) => return Reply::error_page(&e),
    };
    let req = match AuthorizationRequest::from_pairs(&pairs) {
        Ok(r) => r,
        Err(e) => return deliver(&e, issuer),
    };
    let Some(client_id) = req.client_id.as_deref() else {
        return page_error(OAuthErrorCode::InvalidRequest, "client_id required");
    };
    let client = match ctx.store.get_client(client_id) {
        Ok(Some(c)) => c,
        Ok(None) => return page_error(OAuthErrorCode::InvalidClient, "unknown client_id"),
        Err(e) => return Reply::error_page(&e.into()),
    };
    let validated = match validate_authorization(&req, &client, &ctx.cfg.resources) {
        Ok(v) => v,
        Err(e) => return deliver(&e, issuer),
    };
    // Pre-consent errors: redirected only to a verified redirect URI, else
    // shown to the user agent (no open redirect via anonymous DCR,
    // RFC 9700 §4.11.2).
    if let Err(e) = ctx.cfg.ensure_federation_ready() {
        return deliver(&validated.error(e), issuer);
    }
    // Best effort: abandoned flows and codes must not accumulate.
    let _ = ctx.store.purge_expired(ctx.clock.now_unix());
    let start = match begin_upstream(
        ctx.store,
        ctx.rng,
        ctx.clock,
        &ctx.cfg.upstream,
        validated.clone(),
    ) {
        Ok(s) => s,
        Err(e) => return deliver(&validated.error(e), issuer),
    };
    let flow = consent::state_of(&start.authorization_url);
    let Some((flow, name)) = flow.and_then(|f| consent::cookie_name(&f).map(|n| (f, n))) else {
        return deliver(
            &validated.error(OAuthError::new(
                OAuthErrorCode::ServerError,
                "flow state unavailable",
            )),
            issuer,
        );
    };
    // Cancel is followed only after the user saw the consent page.
    let cancel = validated
        .error_after_consent(OAuthError::new(
            OAuthErrorCode::AccessDenied,
            "the user cancelled",
        ))
        .redirect_url(issuer)
        .unwrap_or_default();
    let token = consent_token(&start.browser_secret, &validated);
    let upstream_origin = upstream_origin(&ctx.cfg.upstream.authorization_endpoint);
    let form = consent::ConsentForm {
        flow: &flow,
        token: &token,
        upstream_origin: &upstream_origin,
    };
    consent::page(
        &client,
        &validated,
        &form,
        &cancel,
        consent::set_cookie(&name, &start.browser_secret),
    )
}

fn upstream_origin(endpoint: &str) -> String {
    url::Url::parse(endpoint)
        .map(|u| u.origin().ascii_serialization())
        .unwrap_or_default()
}

/// The upstream authorization URL for a stored flow — the same parameters
/// `federation::begin_upstream` sends (a test pins the equality).
pub(crate) fn upstream_authorization_url(
    cfg: &UpstreamConfig,
    flow: &UpstreamFlowState,
) -> Option<String> {
    let mut url = url::Url::parse(&cfg.authorization_endpoint).ok()?;
    url.query_pairs_mut()
        .append_pair("response_type", "code")
        .append_pair("client_id", &cfg.client_id)
        .append_pair("redirect_uri", &cfg.redirect_uri)
        .append_pair("scope", &cfg.scopes.join(" "))
        .append_pair("state", &flow.state)
        .append_pair("nonce", &flow.nonce)
        .append_pair(
            "code_challenge",
            &challenge_s256(&flow.upstream_code_verifier),
        )
        .append_pair("code_challenge_method", "S256");
    Some(url.into())
}

/// Handle `POST /authorize/consent` (form: `flow`, `consent`): the user's
/// click on Continue. The flow is looked up without being consumed; the
/// `__Host-` cookie must match the flow's browser binding and the consent
/// token must verify for that cookie and request, else an error page (never
/// a redirect). On success, 303 to the upstream authorization URL.
pub fn consent_submit<S: AuthStores, R: Rng, C: Clock, G: Signer>(
    ctx: &Ctx<'_, S, R, C, G>,
    cookie_header: Option<&str>,
    content_type: Option<&str>,
    body: &[u8],
) -> Reply {
    let denied = || {
        page_error(
            OAuthErrorCode::AccessDenied,
            "consent could not be verified",
        )
    };
    let pairs = match read_form(content_type, body) {
        Ok(p) => p,
        Err(r) => return r,
    };
    let field = |k: &str| {
        pairs
            .iter()
            .find(|(n, _)| n == k)
            .map(|(_, v)| v.as_str())
            .filter(|v| !v.is_empty() && v.len() <= 256)
    };
    let (Some(flow_state), Some(presented)) = (field("flow"), field("consent")) else {
        return page_error(OAuthErrorCode::InvalidRequest, "consent form incomplete");
    };
    let Some(name) = consent::cookie_name(flow_state) else {
        return denied();
    };
    let cookie = consent::read_cookie(cookie_header, &name).unwrap_or("");
    let flow = match ctx.store.peek_flow(flow_state, ctx.clock.now_unix()) {
        Ok(Some(f)) => f,
        Ok(None) => return denied(),
        Err(e) => return Reply::error_page(&e.into()),
    };
    let bound = !cookie.is_empty() && secret_hash(cookie) == flow.browser_binding;
    if !bound || !verify_consent_token(cookie, &flow.downstream, presented) {
        return denied();
    }
    match upstream_authorization_url(&ctx.cfg.upstream, &flow) {
        Some(url) => {
            let mut r = Reply::redirect(&url);
            if r.status == 302 {
                r.status = 303;
            }
            r
        }
        None => page_error(OAuthErrorCode::ServerError, "upstream endpoint"),
    }
}
