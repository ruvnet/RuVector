//! `/token` (and `/revoke`, `/authorize` POST) form parsing (ADR-351 §5.3,
//! M0 fuzz criterion). The body is decoded exactly as the Worker's
//! `read_form` does (8 KiB cap, `application/x-www-form-urlencoded` via the
//! `url` crate) and driven through `TokenRequest::from_form`,
//! `RevocationRequest::from_form` and `AuthorizationRequest::from_pairs`;
//! accepted requests continue into the parse-level token-endpoint checks:
//! RFC 8707 `resource` resolution against the production allowlist and the
//! §5.3 scope grant rule.
//!
//! Invariants: no panic; a parsed request carries the values of the form
//! (no repeated name, every value <= 2 KiB); `Debug` never prints the code,
//! PKCE verifier, refresh token or exchanged tokens; a resolved resource is
//! allowlisted and canonical; granted scopes are a subset of the ceiling
//! and the resource's scopes and never only `offline_access`.
#![no_main]

#[path = "common.rs"]
mod common;

use libfuzzer_sys::fuzz_target;
use ruvector_edge_authz::authorize::AuthorizationRequest;
use ruvector_edge_authz::params::{split_scope, MAX_PARAM_LEN, MAX_SCOPE_TOKENS};
use ruvector_edge_authz::resource::grant_scopes;
use ruvector_edge_authz::revoke::RevocationRequest;
use ruvector_edge_authz::token::TokenRequest;
use ruvector_edge_authz::ResourceAllowlist;

/// `edge/auth-worker/src/http.rs` `MAX_FORM_BODY`.
const MAX_FORM_BODY: usize = 8 * 1024;

fn decode(body: &[u8]) -> Vec<(String, String)> {
    url::form_urlencoded::parse(body)
        .map(|(k, v)| (k.into_owned(), v.into_owned()))
        .collect()
}

/// `Debug` must not leak a secret. Only checked for secrets that cannot
/// appear through another printed field or an escape sequence.
fn assert_redacted(dbg: &str, secret: &str, printed: &[Option<&str>]) {
    let plain = secret.len() >= 12 && secret.bytes().all(|b| b.is_ascii_alphanumeric());
    let elsewhere = printed.iter().flatten().any(|p| p.contains(secret));
    if plain && !elsewhere {
        assert!(!dbg.contains(secret), "secret in Debug: {dbg}");
    }
}

fn scopes(allowlist: &ResourceAllowlist, resource: Option<&str>, scope: Option<&str>) {
    let Ok(entry) = allowlist.resolve_entry(resource) else {
        return;
    };
    // Only host case may differ (ResourceUrl lowercases it).
    assert!(resource.is_some_and(|r| r.eq_ignore_ascii_case(entry.url().as_str())));
    assert_eq!(allowlist.resolve(resource).unwrap(), *entry.url());
    if let Some(s) = scope {
        if let Ok(tokens) = split_scope(s) {
            assert!(!tokens.is_empty() && tokens.len() <= MAX_SCOPE_TOKENS);
            assert!(tokens.iter().all(|t| s.split(' ').any(|x| x == t)));
        }
    }
    // Ceilings: a default DCR client and one that registered everything.
    let full = allowlist.scopes_supported();
    let default = common::dcr_policy(allowlist).default_scope;
    for ceiling in [&default, &full] {
        if let Ok(granted) = grant_scopes(scope, ceiling, entry, allowlist) {
            assert!(granted.iter().any(|g| g != "offline_access"));
            for g in &granted {
                assert!(ceiling.contains(g) && entry.allows(g), "{g}");
            }
        }
    }
}

fn check(body: &[u8]) {
    if body.len() > MAX_FORM_BODY {
        return;
    }
    let pairs = decode(body);
    let allowlist = common::allowlist();
    let value = |name: &str| pairs.iter().find(|(k, v)| k == name && !v.is_empty());

    if let Ok(req) = TokenRequest::from_form(&pairs) {
        let dbg = format!("{req:?}");
        match &req {
            TokenRequest::AuthorizationCode {
                code,
                redirect_uri,
                client_id,
                code_verifier,
                resource,
            } => {
                assert_eq!(value("code").map(|p| &p.1), Some(code));
                assert!(code.len() <= MAX_PARAM_LEN && code_verifier.len() <= MAX_PARAM_LEN);
                let shown = [
                    Some(redirect_uri.as_str()),
                    Some(client_id.as_str()),
                    resource.as_deref(),
                ];
                assert_redacted(&dbg, code, &shown);
                assert_redacted(&dbg, code_verifier, &shown);
                scopes(&allowlist, resource.as_deref(), None);
            }
            TokenRequest::RefreshToken {
                refresh_token,
                client_id,
                scope,
                resource,
            } => {
                assert_eq!(value("refresh_token").map(|p| &p.1), Some(refresh_token));
                let shown = [
                    Some(client_id.as_str()),
                    scope.as_deref(),
                    resource.as_deref(),
                ];
                assert_redacted(&dbg, refresh_token, &shown);
                scopes(&allowlist, resource.as_deref(), scope.as_deref());
            }
            TokenRequest::TokenExchange(x) => {
                let shown = [
                    x.client_id.as_deref(),
                    x.resource.as_deref(),
                    x.scope.as_deref(),
                ];
                assert_redacted(&dbg, &x.client_assertion, &shown);
                assert_redacted(&dbg, &x.subject_token, &shown);
                assert!(req.client_id().is_none());
                scopes(&allowlist, x.resource.as_deref(), x.scope.as_deref());
            }
        }
    }
    if let Ok(req) = RevocationRequest::from_form(&pairs) {
        let dbg = format!("{req:?}");
        let shown = [Some(req.client_id.as_str()), req.token_type_hint.as_deref()];
        assert_redacted(&dbg, &req.token, &shown);
    }
    if let Ok(req) = AuthorizationRequest::from_pairs(&pairs) {
        scopes(&allowlist, req.resource.as_deref(), req.scope.as_deref());
    }
}

fuzz_target!(|data: &[u8]| check(data));
