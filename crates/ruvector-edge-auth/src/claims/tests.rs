use super::*;
use crate::test_support::*;
use serde_json::{json, Value};

fn raw(v: Value) -> RawClaims {
    serde_json::from_value(v).unwrap()
}

fn edge_policy() -> ClaimsPolicy {
    ClaimsPolicy::edge(EDGE_ISS, UPSTREAM_ISS)
}

fn upstream_policy() -> ClaimsPolicy {
    ClaimsPolicy::upstream_access(UPSTREAM_ISS)
}

fn edge_with(k: &str, v: Value) -> RawClaims {
    let mut c = edge_claims();
    if v.is_null() {
        c.as_object_mut().unwrap().remove(k);
    } else {
        c[k] = v;
    }
    raw(c)
}

fn check_edge(r: RawClaims) -> Result<VerifiedClaims, AuthError> {
    validate(r, &edge_policy(), TokenKind::EdgeIssued, NOW)
}

#[test]
fn valid_edge_claims() {
    let v = check_edge(raw(edge_claims())).unwrap();
    assert_eq!(v.kind(), TokenKind::EdgeIssued);
    assert_eq!(v.iss(), EDGE_ISS);
    assert_eq!(v.aud(), RESOURCE);
    assert_eq!(v.sub(), "user-1");
    assert_eq!(v.client_id(), "edge-client-abc");
    assert_eq!(v.org_id(), Some("org-1"));
    assert_eq!(v.workspace_id(), Some("ws-1"));
    assert_eq!(v.scopes(), ["ruvector:read", "ruvector:write"]);
    assert_eq!(v.jti(), Some("jti-1"));
    assert_eq!(v.family_id(), Some("fam-edge-1"));
    assert_eq!(v.upstream_iss(), Some(UPSTREAM_ISS));
    assert_eq!(v.kid(), "", "validate alone never sees the header");
    assert_eq!((v.iat(), v.exp()), (NOW - 10, NOW + 890));
}

#[test]
fn time_rules() {
    let p = edge_policy();
    let t = |exp: u64, iat: u64, nbf: Option<u64>| {
        validate_time(
            &RawClaims {
                exp: Some(exp),
                iat: Some(iat),
                nbf,
                ..Default::default()
            },
            &p,
            NOW,
        )
    };
    assert_eq!(t(NOW + 100, NOW, None), Ok(()));
    // exp within skew still accepted; at/after skew boundary rejected.
    assert_eq!(t(NOW - 59, NOW - 600, None), Ok(()));
    assert_eq!(t(NOW - 60, NOW - 600, None), Err(AuthError::Expired));
    assert_eq!(t(NOW - 3000, NOW - 3500, None), Err(AuthError::Expired));
    // iat in the future.
    assert_eq!(t(NOW + 700, NOW + 60, None), Ok(()));
    assert_eq!(t(NOW + 700, NOW + 61, None), Err(AuthError::IssuedInFuture));
    // nbf in the future.
    assert_eq!(t(NOW + 100, NOW, Some(NOW + 60)), Ok(()));
    assert_eq!(
        t(NOW + 100, NOW, Some(NOW + 61)),
        Err(AuthError::NotYetValid)
    );
    // lifetime.
    assert_eq!(t(NOW + 900, NOW, None), Ok(()));
    assert_eq!(t(NOW + 901, NOW, None), Err(AuthError::LifetimeTooLong));
    assert_eq!(
        t(NOW + 10, NOW + 10, None),
        Err(AuthError::InvalidClaim("exp <= iat"))
    );
    // missing.
    let missing = RawClaims {
        iat: Some(NOW),
        ..Default::default()
    };
    assert_eq!(
        validate_time(&missing, &p, NOW),
        Err(AuthError::InvalidClaim("exp"))
    );
    // no overflow at the numeric edge.
    assert_eq!(
        t(u64::MAX, u64::MAX - 10, None),
        Err(AuthError::IssuedInFuture)
    );
}

#[test]
fn expired_and_nbf_through_validate() {
    assert_eq!(
        check_edge(edge_with("exp", json!(NOW - 61))),
        Err(AuthError::Expired)
    );
    assert_eq!(
        check_edge(edge_with("nbf", json!(NOW + 300))),
        Err(AuthError::NotYetValid)
    );
}

#[test]
fn wrong_or_missing_issuer() {
    assert_eq!(
        check_edge(edge_with("iss", json!(UPSTREAM_ISS))),
        Err(AuthError::WrongIssuer)
    );
    assert_eq!(
        check_edge(edge_with("iss", json!(format!("{EDGE_ISS}/")))),
        Err(AuthError::WrongIssuer)
    );
    assert_eq!(
        check_edge(edge_with("iss", Value::Null)),
        Err(AuthError::InvalidClaim("iss"))
    );
}

#[test]
fn required_string_claims() {
    for k in [
        "sub",
        "client_id",
        "jti",
        "org_id",
        "workspace_id",
        "scope",
        "aud",
    ] {
        assert!(
            matches!(
                check_edge(edge_with(k, Value::Null)),
                Err(AuthError::InvalidClaim(_))
            ),
            "missing {k}"
        );
    }
    for k in ["sub", "client_id", "jti", "org_id", "workspace_id"] {
        assert!(check_edge(edge_with(k, json!(""))).is_err(), "empty {k}");
        assert!(
            check_edge(edge_with(k, json!("x".repeat(MAX_CLAIM_LEN + 1)))).is_err(),
            "long {k}"
        );
        assert!(
            check_edge(edge_with(k, json!("a\u{0}b"))).is_err(),
            "ctrl {k}"
        );
    }
}

#[test]
fn aud_array_rejected_in_claims() {
    assert_eq!(
        check_edge(edge_with("aud", json!([RESOURCE]))),
        Err(AuthError::InvalidClaim("aud"))
    );
}

#[test]
fn forbidden_flags() {
    for f in ["exchanged", "setup", "workload"] {
        assert_eq!(
            check_edge(edge_with(f, json!(true))),
            Err(AuthError::InvalidClaim(f))
        );
        assert!(check_edge(edge_with(f, json!(false))).is_ok());
    }
}

#[test]
fn wrong_json_types_fail_to_decode() {
    for (k, v) in [
        ("exp", json!("123")),
        ("exp", json!(-1)),
        ("exp", json!(1.5)),
        ("sub", json!(5)),
        ("exchanged", json!("true")),
    ] {
        let mut c = edge_claims();
        c[k] = v;
        assert!(serde_json::from_value::<RawClaims>(c).is_err(), "{k}");
    }
}

#[test]
fn scope_parsing() {
    assert_eq!(parse_scope(""), Ok(vec![]));
    assert_eq!(parse_scope("a b:c"), Ok(vec!["a".into(), "b:c".into()]));
    for bad in [" a", "a ", "a  b", "a\tb", "a\"b", "a\\b", "é"] {
        assert_eq!(
            parse_scope(bad),
            Err(AuthError::InvalidClaim("scope")),
            "{bad:?}"
        );
    }
    let many = vec!["s"; MAX_SCOPES + 1].join(" ");
    assert!(parse_scope(&many).is_err());
    assert!(parse_scope(&"s".repeat(MAX_SCOPE_LEN + 1)).is_err());
}

#[test]
fn upstream_rules() {
    let ok = validate(
        raw(upstream_claims()),
        &upstream_policy(),
        TokenKind::UpstreamFirstParty,
        NOW,
    )
    .unwrap();
    assert_eq!(ok.family_id(), Some("fam-1"));
    assert_eq!(ok.kind(), TokenKind::UpstreamFirstParty);

    let with = |k: &str, v: Value| {
        let mut c = upstream_claims();
        if v.is_null() {
            c.as_object_mut().unwrap().remove(k);
        } else {
            c[k] = v;
        }
        validate(
            raw(c),
            &upstream_policy(),
            TokenKind::UpstreamFirstParty,
            NOW,
        )
    };
    // ID token shape: no typ claim.
    assert_eq!(
        with("typ", Value::Null),
        Err(AuthError::InvalidClaim("typ"))
    );
    assert_eq!(
        with("typ", json!("inference")),
        Err(AuthError::InvalidClaim("typ"))
    );
    assert_eq!(
        with("typ", json!("refresh")),
        Err(AuthError::InvalidClaim("typ"))
    );
    assert_eq!(
        with("family_id", Value::Null),
        Err(AuthError::InvalidClaim("family_id"))
    );
    assert_eq!(
        with("client_id", json!("other")),
        Err(AuthError::InvalidClaim("client_id != aud"))
    );
}

/// Regression: the kind's hard lifetime cap applies whatever the policy.
#[test]
fn kind_lifetime_caps_override_policy() {
    let legacy = ClaimsPolicy::with_defaults(EDGE_ISS);
    assert_eq!(
        legacy.lifetime_cap(TokenKind::EdgeIssued),
        EDGE_MAX_LIFETIME_SECS
    );
    assert_eq!(
        ClaimsPolicy {
            max_lifetime_secs: u64::MAX,
            ..legacy.clone()
        }
        .lifetime_cap(TokenKind::UpstreamFirstParty),
        UPSTREAM_MAX_LIFETIME_SECS
    );
    let mut c = edge_claims();
    c["iat"] = json!(NOW - 10);
    c["exp"] = json!(NOW - 10 + 901);
    assert_eq!(
        validate(raw(c.clone()), &legacy, TokenKind::EdgeIssued, NOW),
        Err(AuthError::LifetimeTooLong)
    );
    c["exp"] = json!(NOW - 10 + 900);
    assert!(validate(raw(c), &legacy, TokenKind::EdgeIssued, NOW).is_ok());
}

/// Regression: upstream tokens need `typ == "access"` even when the policy
/// sets no `typ_claim`.
#[test]
fn upstream_typ_required_without_policy_rule() {
    let legacy = ClaimsPolicy::with_defaults(UPSTREAM_ISS);
    let mut c = upstream_claims();
    c.as_object_mut().unwrap().remove("typ");
    assert_eq!(
        validate(raw(c.clone()), &legacy, TokenKind::UpstreamFirstParty, NOW),
        Err(AuthError::InvalidClaim("typ"))
    );
    c["typ"] = json!("refresh");
    assert_eq!(
        validate(raw(c.clone()), &legacy, TokenKind::UpstreamFirstParty, NOW),
        Err(AuthError::InvalidClaim("typ"))
    );
    c["typ"] = json!(UPSTREAM_ACCESS_TYP);
    let ok = validate(raw(c), &legacy, TokenKind::UpstreamFirstParty, NOW).unwrap();
    assert_eq!(ok.upstream_iss(), Some(UPSTREAM_ISS));
    assert_eq!(ok.sub(), "user-1", "raw upstream sub is kept verbatim");
}

#[test]
fn upstream_iss_rules() {
    assert_eq!(
        check_edge(edge_with("upstream_iss", Value::Null)),
        Err(AuthError::InvalidClaim("upstream_iss"))
    );
    assert_eq!(
        check_edge(edge_with("upstream_iss", json!("https://evil.example"))),
        Err(AuthError::InvalidClaim("upstream_iss"))
    );
    // A policy without an upstream issuer neither requires nor invents it.
    let legacy = ClaimsPolicy::with_defaults(EDGE_ISS);
    let v = validate(
        edge_with("upstream_iss", Value::Null),
        &legacy,
        TokenKind::EdgeIssued,
        NOW,
    )
    .unwrap();
    assert_eq!(v.upstream_iss(), None);
}

/// Regression: `aud` is bounded by the resource-URL limit (512), not the
/// generic 256-byte claim limit.
#[test]
fn aud_bounded_by_resource_url_limit() {
    use crate::resource::MAX_RESOURCE_URL_LEN;
    let at = |n: usize| format!("https://gw.example/{}", "a".repeat(n - 19));
    assert_eq!(at(300).len(), 300);
    assert!(check_edge(edge_with("aud", json!(at(300)))).is_ok());
    assert!(check_edge(edge_with("aud", json!(at(MAX_RESOURCE_URL_LEN)))).is_ok());
    assert_eq!(
        check_edge(edge_with("aud", json!(at(MAX_RESOURCE_URL_LEN + 1)))),
        Err(AuthError::InvalidClaim("aud"))
    );
}
