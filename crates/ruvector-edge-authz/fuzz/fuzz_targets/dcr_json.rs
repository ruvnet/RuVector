//! Dynamic Client Registration body (RFC 7591, ADR-351 §5.3/§5.6, M0 fuzz
//! criterion): arbitrary bytes through `RegistrationRequest::from_json` and
//! `validate_registration` with the production DCR policy, exactly as
//! `POST /register` runs them.
//!
//! Invariants for every accepted registration: 1..=8 distinct redirect URIs,
//! each re-validating and matching itself; grant types a de-duplicated
//! subset of {authorization_code, refresh_token} that includes the code
//! grant; a non-empty scope ceiling inside `scopes_supported` with no
//! identity scope; a display-safe `client_name`; the RFC 7591 response and
//! the stored record serialise, and the record round-trips through JSON.
//! Presented redirect strings (the tail of the input) never panic
//! `matches`/`match_redirect`, and a match is always byte-exact or a
//! loopback port variant.
#![no_main]

#[path = "common.rs"]
mod common;

use libfuzzer_sys::fuzz_target;
use ruvector_edge_authz::client::{
    is_display_name, validate_redirect_uri, validate_registration, ClientRecord,
    RegistrationRequest, MAX_REDIRECT_URIS,
};
use ruvector_edge_authz::params::IDENTITY_SCOPES;

const NOW: u64 = 1_800_000_000;

fn check(body: &[u8], presented: &[&str]) {
    let Ok(req) = RegistrationRequest::from_json(body) else {
        return;
    };
    let allowlist = common::allowlist();
    let policy = common::dcr_policy(&allowlist);
    let Ok(rec) = validate_registration(&req, &policy, "edc-fuzz".into(), NOW) else {
        return;
    };
    assert!((1..=MAX_REDIRECT_URIS).contains(&rec.redirect_uris.len()));
    for (i, r) in rec.redirect_uris.iter().enumerate() {
        assert_eq!(validate_redirect_uri(r.as_str()).unwrap(), *r);
        assert!(r.matches(r.as_str()));
        assert!(!rec.redirect_uris[..i].contains(r));
        let _ = r.is_verified();
    }
    assert!(rec.allows_grant("authorization_code"));
    for (i, g) in rec.grant_types.iter().enumerate() {
        assert!(g == "authorization_code" || g == "refresh_token", "{g}");
        assert!(!rec.grant_types[..i].contains(g));
    }
    assert!(!rec.scope.is_empty());
    for s in &rec.scope {
        assert!(policy.scopes_supported.contains(s), "{s}");
        assert!(!IDENTITY_SCOPES.contains(&s.as_str()));
    }
    if let Some(name) = &rec.client_name {
        assert!(is_display_name(name));
    }
    let resp = rec.to_response();
    assert_eq!(resp.scope, rec.scope.join(" "));
    serde_json::to_vec(&resp).expect("response serialises");
    let stored = serde_json::to_vec(&rec).expect("record serialises");
    let back: ClientRecord = serde_json::from_slice(&stored).expect("record deserialises");
    assert_eq!(back, rec);
    for p in presented {
        if let Some(m) = rec.match_redirect(p) {
            let reg = m.as_str();
            assert!(
                reg == *p || (reg.starts_with("http://") && p.starts_with("http://")),
                "{reg:?} matched {p:?}"
            );
        }
    }
}

fuzz_target!(|data: &[u8]| {
    // Input: `<json body>\0<presented redirect>\0<presented redirect>...`.
    let mut parts = data.split(|&b| b == 0);
    let body = parts.next().unwrap_or_default();
    let presented: Vec<&str> = parts.filter_map(|p| std::str::from_utf8(p).ok()).collect();
    check(body, &presented);
    // The whole input as a body too (NULs are valid inside JSON strings as
    // escapes only, so this mostly exercises the error paths).
    if data.len() != body.len() {
        let _ = RegistrationRequest::from_json(data);
    }
});
