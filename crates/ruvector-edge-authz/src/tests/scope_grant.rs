//! ADR-351 §5.3 grant rule (requested ∩ client ceiling ∩ resource scopes),
//! the per-resource scope table (§5.7, §16.1), DCR defaults and the §5.2
//! minted claim set.

use super::assert_code;
use crate::authorize::{validate_authorization, AuthorizationRequest};
use crate::client::{validate_registration, ClientRecord, DEFAULT_CLIENT_SCOPE};
use crate::error::OAuthErrorCode as C;
use crate::resource::grant_scopes;
use crate::testing::*;
use crate::token::{mint_access_token, MintRequest};
use ruvector_edge_auth::jws::b64url_decode;

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

fn client_with(ceiling: &[&str]) -> ClientRecord {
    let mut c = client(&["authorization_code", "refresh_token"]);
    c.scope = s(ceiling);
    c
}

fn authorize(
    ceiling: &[&str],
    resource: &str,
    scope: Option<&str>,
) -> Result<Vec<String>, crate::OAuthError> {
    let req = AuthorizationRequest {
        response_type: Some("code".into()),
        client_id: Some(CLIENT_ID.into()),
        redirect_uri: Some(REDIRECT.into()),
        scope: scope.map(Into::into),
        state: None,
        code_challenge: Some(challenge()),
        code_challenge_method: Some("S256".into()),
        resource: Some(resource.into()),
    };
    validate_authorization(&req, &client_with(ceiling), &allowlist())
        .map(|v| v.scopes)
        .map_err(|e| e.oauth().clone())
}

const ALL_GW: [&str; 4] = [
    "ruvector:read",
    "ruvector:write",
    "ruvector:admin",
    "offline_access",
];

#[test]
fn allowlist_carries_per_resource_scopes() {
    let a = allowlist();
    let scopes = |r: &str| a.resolve_entry(Some(r)).unwrap().scopes().to_vec();
    assert_eq!(scopes(OTHER_RESOURCE), s(&ALL_GW));
    assert_eq!(
        scopes(RESOURCE),
        s(&["ruvector:read", "ruvector:write", "offline_access"])
    );
    assert_eq!(
        scopes(TEAM_RESOURCE),
        s(&["ruvector:read", "ruvector:write", "offline_access"])
    );
    assert_eq!(
        a.resolve_entry(Some(TEAM_RESOURCE))
            .unwrap()
            .default_grant(),
        "ruvector:read"
    );
    assert_eq!(
        a.scopes_supported(),
        s(&[
            "ruvector:read",
            "ruvector:write",
            "offline_access",
            "ruvector:admin",
        ])
    );
}

/// Regression (§5.3): `scopes_supported` (RFC 8414 + DCR) is exactly the
/// §5.3 vocabulary, so an allowlist entry with any other scope (e.g. an
/// adapter's own `team:*`) does not load.
#[test]
fn allowlist_refuses_scopes_outside_the_edge_vocabulary() {
    use crate::resource::ResourceAllowlist;
    for bad in [
        "https://team.ruv.io/mcp team:read offline_access",
        "https://team.ruv.io/mcp ruvector:read team:run",
        "https://gw.example/v1 ruvector:read mcp:invoke",
        "https://gw.example/v1 ruvector:read openid",
    ] {
        assert!(ResourceAllowlist::from_config(bad).is_err(), "{bad}");
    }
    assert!(ResourceAllowlist::from_config(
        "https://team.ruv.io/mcp ruvector:read ruvector:write offline_access"
    )
    .is_ok());
}

/// Regression (§5.3): out-of-ceiling vocabulary scopes are dropped, not
/// refused, and the granted scope reports what was granted.
#[test]
fn out_of_ceiling_scopes_are_dropped() {
    let got = authorize(
        &["ruvector:read", "offline_access"],
        RESOURCE,
        Some("ruvector:read ruvector:write offline_access"),
    )
    .unwrap();
    assert_eq!(got, s(&["ruvector:read", "offline_access"]));
    // Vocabulary not minted before M5 is dropped too.
    let got = authorize(
        &ALL_GW,
        OTHER_RESOURCE,
        Some("ruvector:publish ruvector:read"),
    )
    .unwrap();
    assert_eq!(got, s(&["ruvector:read"]));
}

/// Regression (§5.3/§5.7): `ruvector:admin` is minted for `/v1` only, never
/// for `/v1/mcp`, even when the client registered it.
#[test]
fn admin_only_for_the_rest_resource() {
    let asked = Some("ruvector:read ruvector:admin");
    assert_eq!(
        authorize(&ALL_GW, OTHER_RESOURCE, asked).unwrap(),
        s(&["ruvector:read", "ruvector:admin"])
    );
    assert_eq!(
        authorize(&ALL_GW, RESOURCE, asked).unwrap(),
        s(&["ruvector:read"])
    );
    assert_code(
        authorize(&ALL_GW, RESOURCE, Some("ruvector:admin offline_access")),
        C::InvalidScope,
    );
}

/// Regression (§5.3 table): omitted `scope` grants the resource default
/// `ruvector:read` plus `offline_access` (both "granted by default"), still
/// intersected with the client ceiling.
#[test]
fn omitted_scope_grants_the_resource_default() {
    for r in [RESOURCE, OTHER_RESOURCE, TEAM_RESOURCE] {
        assert_eq!(
            authorize(&ALL_GW, r, None).unwrap(),
            s(&["ruvector:read", "offline_access"])
        );
    }
    assert_eq!(
        authorize(&["ruvector:read"], RESOURCE, None).unwrap(),
        s(&["ruvector:read"])
    );
    assert_code(
        authorize(&["ruvector:write", "offline_access"], RESOURCE, None),
        C::InvalidScope,
    );
}

/// Unknown scopes are `invalid_scope`; `offline_access` alone grants nothing.
#[test]
fn unknown_or_empty_grants_are_invalid_scope() {
    for bad in ["brains:read", "ruvector:read mcp:invoke", "offline_access"] {
        assert_code(authorize(&ALL_GW, RESOURCE, Some(bad)), C::InvalidScope);
    }
}

/// Regression (§5.3, §16.1): the adapter resource uses the `ruvector:*`
/// vocabulary, so a default registration gets an adapter token whose scope
/// a later exchange for `/v1` can keep (`scope ⊆` the subject token's);
/// admin is never minted for it and `team:*` is not a scope at all.
#[test]
fn team_resource_uses_the_edge_vocabulary() {
    let default: Vec<&str> = DEFAULT_CLIENT_SCOPE.to_vec();
    assert_eq!(
        authorize(&default, TEAM_RESOURCE, None).unwrap(),
        s(&["ruvector:read", "offline_access"])
    );
    assert_eq!(
        authorize(
            &ALL_GW,
            TEAM_RESOURCE,
            Some("ruvector:write ruvector:admin offline_access")
        )
        .unwrap(),
        s(&["ruvector:write", "offline_access"])
    );
    for bad in ["team:read", "ruvector:read team:run"] {
        assert_code(
            authorize(&default, TEAM_RESOURCE, Some(bad)),
            C::InvalidScope,
        );
    }
}

#[test]
fn grant_scopes_is_order_preserving_and_deduplicated() {
    let a = allowlist();
    let e = a.resolve_entry(Some(OTHER_RESOURCE)).unwrap();
    let got = grant_scopes(
        Some("offline_access ruvector:write openid ruvector:read ruvector:write"),
        &s(&ALL_GW),
        e,
        &a,
    )
    .unwrap();
    assert_eq!(
        got,
        s(&["offline_access", "ruvector:write", "ruvector:read"])
    );
}

/// Regression (§5.3/§5.6): DCR without `scope` gets `ruvector:read
/// ruvector:write offline_access`; admin only on request; not-yet-minted
/// vocabulary is dropped; scopes outside §5.3 (`team:*`, `mcp:*`) are
/// refused.
#[test]
fn dcr_defaults_and_explicit_scopes() {
    let reg = |scope: Option<&str>| {
        let mut req = reg_request();
        req.scope = scope.map(Into::into);
        validate_registration(&req, &policy(), "edc-x".into(), T0)
    };
    assert_eq!(reg(None).unwrap().scope, s(&DEFAULT_CLIENT_SCOPE));
    assert_eq!(
        reg(Some("ruvector:read ruvector:admin")).unwrap().scope,
        s(&["ruvector:read", "ruvector:admin"])
    );
    assert_code(reg(Some("team:read team:run")), C::InvalidClientMetadata);
    assert_eq!(
        reg(Some("ruvector:publish ruvector:read")).unwrap().scope,
        s(&["ruvector:read"])
    );
    assert_code(reg(Some("mcp:read")), C::InvalidClientMetadata);
}

/// M0 acceptance (minted-token golden test): exactly the §5.2 claim set,
/// `exp = iat + 900`, `sub` the edge subject, and `act` only when exchanged.
#[test]
fn minted_claims_are_exactly_the_adr_set() {
    let (signer, rng, clock) = (TestSigner::default(), SeqRng::default(), FixedClock::at(T0));
    let (res, id, scopes) = (resource(), identity(), s(&["ruvector:read"]));
    let mint = |act: Option<&str>| {
        let req = MintRequest {
            issuer: ISSUER,
            resource: &res,
            client_id: CLIENT_ID,
            identity: &id,
            family_id: "fam-1",
            scopes: &scopes,
            act,
        };
        let (jwt, _) = mint_access_token(&signer, &rng, &clock, &req).unwrap();
        let p = jwt.split('.').nth(1).unwrap().to_string();
        serde_json::from_slice::<serde_json::Value>(&b64url_decode(&p).unwrap()).unwrap()
    };
    let claims = mint(None);
    let mut keys: Vec<&str> = claims
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            "aud",
            "client_id",
            "exp",
            "family_id",
            "iat",
            "iss",
            "jti",
            "org_id",
            "scope",
            "sub",
            "upstream_iss",
            "workspace_id",
        ]
    );
    assert_eq!(claims["exp"].as_u64().unwrap(), T0 + 900);
    assert_eq!(claims["iat"].as_u64().unwrap(), T0);
    assert_eq!(claims["sub"], id.edge_subject());
    assert!(ruvector_edge_auth::subject::is_edge_subject(
        claims["sub"].as_str().unwrap()
    ));
    let exchanged = mint(Some("edc-adapter"));
    assert_eq!(exchanged["act"], serde_json::json!({"sub": "edc-adapter"}));
}
