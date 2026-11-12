//! ADR-351 §5.3 per-resource vocabularies: the team.ruv.io adapter resource
//! (§16.1) mints `team:*` only, the gateway resources `ruvector:*` only; the
//! URL → family binding is compiled and enforced at load, and a scope of one
//! resource's vocabulary is dropped (never minted) for the other.

use super::assert_code;
use crate::authorize::{validate_authorization, AuthorizationRequest};
use crate::client::{register_client, validate_registration, ClientRecord, DEFAULT_CLIENT_SCOPE};
use crate::error::OAuthErrorCode as C;
use crate::federation::{begin_upstream, complete_upstream, identity_from_upstream};
use crate::grant::TokenEndpoint;
use crate::testing::*;
use crate::token::{TokenRequest, TokenResponse};
use ruvector_edge_auth::jws::b64url_decode;
use ruvector_edge_auth::{TokenKind, VerifiedClaims};

const TEAM_ALL: [&str; 4] = ["team:read", "team:write", "team:run", "offline_access"];
const BOTH: [&str; 6] = [
    "ruvector:read",
    "ruvector:write",
    "offline_access",
    "team:read",
    "team:write",
    "team:run",
];

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

fn authorize(
    ceiling: &[&str],
    resource: &str,
    scope: Option<&str>,
) -> Result<Vec<String>, crate::OAuthError> {
    let mut client = client(&["authorization_code", "refresh_token"]);
    client.scope = s(ceiling);
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
    validate_authorization(&req, &client, &allowlist())
        .map(|v| v.scopes)
        .map_err(|e| e.oauth().clone())
}

fn register(scope: Option<&str>) -> Result<ClientRecord, crate::OAuthError> {
    let mut req = reg_request();
    req.scope = scope.map(Into::into);
    validate_registration(&req, &policy(), "edc-x".into(), T0)
}

/// Team resource: `team:*` only; omitted `scope` grants `team:read` +
/// `offline_access`; `ruvector:admin`-style same-family narrowing still
/// drops (`team:run` outside the ceiling is dropped, not refused).
#[test]
fn team_resource_grants_team_scopes_only() {
    assert_eq!(
        authorize(&TEAM_ALL, TEAM_RESOURCE, None).unwrap(),
        s(&["team:read", "offline_access"])
    );
    assert_eq!(
        authorize(&BOTH, TEAM_RESOURCE, Some("team:read team:write team:run")).unwrap(),
        s(&["team:read", "team:write", "team:run"])
    );
    assert_eq!(
        authorize(
            &["team:read", "offline_access"],
            TEAM_RESOURCE,
            Some("team:read team:run offline_access")
        )
        .unwrap(),
        s(&["team:read", "offline_access"])
    );
}

/// Cross-vocabulary scopes are dropped in both directions (Q16(b)): never
/// minted, never a hard failure when a valid scope is left, `invalid_scope`
/// only when nothing but `offline_access` remains.
#[test]
fn other_resource_vocabulary_is_dropped() {
    for (resource, asked, want) in [
        (TEAM_RESOURCE, "team:read ruvector:read", &["team:read"][..]),
        (RESOURCE, "ruvector:read team:run", &["ruvector:read"][..]),
        (
            OTHER_RESOURCE,
            "team:write ruvector:write offline_access",
            &["ruvector:write", "offline_access"][..],
        ),
    ] {
        assert_eq!(
            authorize(&BOTH, resource, Some(asked)).unwrap(),
            s(want),
            "{resource} {asked}"
        );
    }
    for (resource, bad) in [
        (TEAM_RESOURCE, "ruvector:read"),
        (TEAM_RESOURCE, "ruvector:admin offline_access"),
        (RESOURCE, "team:read"),
        (OTHER_RESOURCE, "ruvector:admin team:write"),
    ] {
        let e = authorize(&BOTH, resource, Some(bad)).unwrap_err();
        assert_eq!(e.error, C::InvalidScope, "{resource} {bad}");
        assert_eq!(
            e.error_description, "no requested scope can be granted for this resource",
            "{bad}"
        );
    }
    // Unknown scopes keep their own description.
    let e = authorize(&BOTH, TEAM_RESOURCE, Some("team:admin")).unwrap_err();
    assert_eq!(e.error_description, "unknown scope requested");
}

/// Q16(b): a client that requests the AS-metadata `scopes_supported` (the
/// union of both families) gets each resource's own share, never the other
/// family.
#[test]
fn as_metadata_union_request_gets_each_resource_share() {
    let union = allowlist().scopes_supported().join(" ");
    let all: Vec<String> = allowlist().scopes_supported();
    let c: Vec<&str> = all.iter().map(String::as_str).collect();
    assert_eq!(
        authorize(&c, RESOURCE, Some(&union)).unwrap(),
        s(&["ruvector:read", "ruvector:write", "offline_access"])
    );
    assert_eq!(
        authorize(&c, OTHER_RESOURCE, Some(&union)).unwrap(),
        s(&[
            "ruvector:read",
            "ruvector:write",
            "offline_access",
            "ruvector:admin"
        ])
    );
    assert_eq!(
        authorize(&c, TEAM_RESOURCE, Some(&union)).unwrap(),
        s(&["offline_access", "team:read", "team:write", "team:run"])
    );
}

/// Regression (ADR-351 §5.3, §16.1): the URL → vocabulary binding is
/// compiled and enforced at load. The team.ruv.io entry live before this
/// change (`ruvector:*`), an unknown adapter with `ruvector:*` and a gateway
/// resource with `team:*` all fail the load; the compiled bindings load.
#[test]
fn allowlist_binds_each_url_to_its_compiled_vocabulary() {
    use crate::resource::{ResourceAllowlist, Vocabulary, RESOURCE_VOCABULARIES};
    for bad in [
        "https://team.ruv.io/mcp ruvector:read ruvector:write offline_access".to_string(),
        "https://team.ruv.io/mcp ruvector:read".to_string(),
        "https://other.example/mcp ruvector:read".to_string(),
        "https://any-adapter.example/mcp ruvector:read ruvector:admin".to_string(),
        "https://other.example/mcp team:read".to_string(),
        format!("{OTHER_RESOURCE} team:read team:write offline_access"),
        format!("{RESOURCE} team:read"),
        // One bad entry fails the whole list.
        format!("{RESOURCE} ruvector:read, https://team.ruv.io/mcp ruvector:read"),
    ] {
        assert!(ResourceAllowlist::from_config(&bad).is_err(), "{bad}");
    }
    // Canonicalisation still applies (host case), then the binding.
    let upper = TEAM_RESOURCE.replace("team.ruv.io", "TEAM.RUV.IO");
    assert!(ResourceAllowlist::from_config(&format!("{upper} team:read")).is_ok());
    assert!(ResourceAllowlist::from_config(&format!("{upper} ruvector:read")).is_err());
    let a = allowlist();
    let got: Vec<(&str, Vocabulary)> = a
        .entries()
        .iter()
        .map(|e| (e.url().as_str(), e.vocabulary()))
        .collect();
    for (url, v) in RESOURCE_VOCABULARIES {
        assert!(got.contains(&(url, v)), "{url}");
    }
}

/// A default registration (`ruvector:*` ceiling) cannot be granted team
/// scopes; one that registered `team:*` explicitly can, and a registration
/// naming both families gets their union and works on both resources.
#[test]
fn dcr_ceiling_is_the_union_of_requested_vocabularies() {
    let default = register(None).unwrap();
    assert_eq!(default.scope, s(&DEFAULT_CLIENT_SCOPE));
    let d: Vec<&str> = DEFAULT_CLIENT_SCOPE.to_vec();
    assert_code(authorize(&d, TEAM_RESOURCE, None), C::InvalidScope);

    let team = register(Some("team:read team:write team:run offline_access")).unwrap();
    assert_eq!(team.scope, s(&TEAM_ALL));
    let both = register(Some(
        "ruvector:read ruvector:write offline_access team:read",
    ))
    .unwrap();
    assert_eq!(
        both.scope,
        s(&[
            "ruvector:read",
            "ruvector:write",
            "offline_access",
            "team:read"
        ])
    );
    let c: Vec<&str> = both.scope.iter().map(String::as_str).collect();
    assert_eq!(
        authorize(&c, TEAM_RESOURCE, None).unwrap(),
        s(&["team:read", "offline_access"])
    );
    assert_eq!(
        authorize(&c, RESOURCE, None).unwrap(),
        s(&["ruvector:read", "offline_access"])
    );
    // Identity scopes are still dropped, unknown ones refused.
    assert_eq!(
        register(Some("openid team:run")).unwrap().scope,
        s(&["team:run"])
    );
    assert_code(
        register(Some("team:read team:admin")),
        C::InvalidClientMetadata,
    );
}

fn payload(jwt: &str) -> serde_json::Value {
    let p = jwt.split('.').nth(1).unwrap();
    serde_json::from_slice(&b64url_decode(p).unwrap()).unwrap()
}

/// End to end: a team.ruv.io token carries `aud` = the team resource and
/// only `team:*` scopes; refresh keeps them and cannot widen into
/// `ruvector:*`.
#[test]
fn team_token_carries_team_scopes_and_audience() {
    let (store, rng, clock, signer) = (
        MemStore::default(),
        SeqRng::default(),
        FixedClock::at(T0),
        TestSigner::default(),
    );
    let resources = allowlist();
    let mut req = reg_request();
    req.scope = Some(BOTH.join(" "));
    let client = register_client(&store, &rng, &clock, &policy(), &req).unwrap();
    let areq = AuthorizationRequest {
        response_type: Some("code".into()),
        client_id: Some(client.client_id.clone()),
        redirect_uri: Some(REDIRECT.into()),
        scope: Some("team:read team:run offline_access".into()),
        state: Some("st".into()),
        code_challenge: Some(challenge()),
        code_challenge_method: Some("S256".into()),
        resource: Some(TEAM_RESOURCE.into()),
    };
    let v = validate_authorization(&areq, &client, &resources).unwrap();
    let start = begin_upstream(&store, &rng, &clock, &upstream(), v).unwrap();
    let state = url::Url::parse(&start.authorization_url)
        .unwrap()
        .query_pairs()
        .find(|(k, _)| k == "state")
        .unwrap()
        .1
        .into_owned();
    let flow = complete_upstream(&store, &clock, &state, &start.browser_secret).unwrap();
    let up = upstream();
    let claims = VerifiedClaims::for_tests(
        TokenKind::UpstreamFirstParty,
        &up.issuer,
        &up.client_id,
        "user-42",
        &up.client_id,
        Some("org_1"),
        Some("ws_1"),
        &["openid"],
        T0,
        T0 + 900,
    );
    let id = identity_from_upstream(&claims, &flow, &up).unwrap();
    let code = crate::code::issue_code(&store, &rng, &clock, flow.downstream, id).unwrap();
    let ep = TokenEndpoint {
        issuer: ISSUER,
        resources: &resources,
        clients: &store,
        codes: &store,
        refresh: &store,
        signer: &signer,
        rng: &rng,
        clock: &clock,
        confidential: &crate::ConfidentialClients::default(),
        assertions: &store,
    };
    let t: TokenResponse = ep
        .handle(&TokenRequest::AuthorizationCode {
            code,
            redirect_uri: REDIRECT.into(),
            client_id: client.client_id.clone(),
            code_verifier: VERIFIER.into(),
            resource: Some(TEAM_RESOURCE.into()),
        })
        .unwrap();
    assert_eq!(t.scope, "team:read team:run offline_access");
    let p = payload(&t.access_token);
    assert_eq!(p["aud"], TEAM_RESOURCE);
    assert_eq!(p["scope"], "team:read team:run offline_access");
    let refresh = |rt: &str, scope: Option<&str>| {
        ep.handle(&TokenRequest::RefreshToken {
            refresh_token: rt.into(),
            client_id: client.client_id.clone(),
            scope: scope.map(Into::into),
            resource: None,
        })
    };
    let rt = t.refresh_token.unwrap();
    assert_code(refresh(&rt, Some("ruvector:read")), C::InvalidScope);
    let r = refresh(&rt, Some("team:read")).unwrap();
    assert_eq!(r.scope, "team:read");
    assert_eq!(payload(&r.access_token)["aud"], TEAM_RESOURCE);
}
