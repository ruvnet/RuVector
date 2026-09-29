use super::*;
use crate::test_support::*;

fn edge_policy() -> AudiencePolicy {
    AudiencePolicy::edge_only(EDGE_ISS, ResourceUrl::parse(RESOURCE).unwrap())
}

fn upstream_policy(kids: Vec<String>) -> UpstreamFirstPartyPolicy {
    UpstreamFirstPartyPolicy {
        issuer: UPSTREAM_ISS.into(),
        first_party_auds: vec![CLI_CLIENT.into()],
        accepted_kids: kids,
    }
}

fn with_upstream() -> AudiencePolicy {
    AudiencePolicy {
        upstream: Some(upstream_policy(vec![kid(&key(5))])),
        ..edge_policy()
    }
}

fn one(s: &str) -> Audience {
    Audience::One(s.into())
}

#[test]
fn classify_edge() {
    let p = edge_policy();
    assert_eq!(
        p.classify(Some(EDGE_ISS), Some("at+jwt")),
        Ok(TokenKind::EdgeIssued)
    );
    assert_eq!(
        p.classify(Some(EDGE_ISS), Some("application/at+JWT")),
        Ok(TokenKind::EdgeIssued)
    );
    for typ in [None, Some("JWT"), Some("id+jwt")] {
        assert_eq!(
            p.classify(Some(EDGE_ISS), typ),
            Err(AuthError::BadTyp),
            "{typ:?}"
        );
    }
    assert_eq!(
        p.classify(None, Some("at+jwt")),
        Err(AuthError::WrongIssuer)
    );
    assert_eq!(
        p.classify(Some("https://evil.example"), Some("at+jwt")),
        Err(AuthError::WrongIssuer)
    );
}

#[test]
fn upstream_path_only_when_configured() {
    assert_eq!(
        edge_policy().classify(Some(UPSTREAM_ISS), None),
        Err(AuthError::WrongIssuer)
    );
    let p = with_upstream();
    assert_eq!(
        p.classify(Some(UPSTREAM_ISS), None),
        Ok(TokenKind::UpstreamFirstParty)
    );
    assert_eq!(
        p.classify(Some(UPSTREAM_ISS), Some("JWT")),
        Ok(TokenKind::UpstreamFirstParty)
    );
    assert_eq!(
        p.classify(Some(UPSTREAM_ISS), Some("at+jwt")),
        Err(AuthError::BadTyp)
    );
    assert_eq!(
        edge_policy().check_audience(TokenKind::UpstreamFirstParty, Some(&one(CLI_CLIENT))),
        Err(AuthError::AudienceNotAllowed)
    );
}

#[test]
fn edge_audience_exact_match() {
    let p = edge_policy();
    assert_eq!(
        p.check_audience(TokenKind::EdgeIssued, Some(&one(RESOURCE))),
        Ok(())
    );
    for aud in [None, Some(Audience::Many(vec![]))] {
        let err = p
            .check_audience(TokenKind::EdgeIssued, aud.as_ref())
            .unwrap_err();
        assert_eq!(err, AuthError::InvalidClaim("aud"));
        assert_eq!(err.http_status(), 401);
    }
}

/// Regression (ADR §5.4.7, §8 delta 10): an edge token for the gateway's
/// *other* resource is 401 like every other mismatch (no 403 carve-out), so
/// the client gets the `resource_metadata` challenge and re-runs discovery.
#[test]
fn every_audience_mismatch_is_401() {
    let p = edge_policy();
    for bad in [
        SIBLING.to_string(),
        format!("{RESOURCE}/"),
        RESOURCE.to_uppercase(),
        format!("{RESOURCE}x"),
        format!("{SIBLING}/"),
        "https://team.ruv.io/mcp".to_string(),
        "https://api.cognitum.one/v1/mcp".to_string(),
        CLI_CLIENT.to_string(),
        String::new(),
    ] {
        let err = p
            .check_audience(TokenKind::EdgeIssued, Some(&one(&bad)))
            .unwrap_err();
        assert_eq!(err, AuthError::AudienceNotAllowed, "{bad}");
        assert_eq!(err.http_status(), 401, "{bad}");
        assert_eq!(err.rfc6750_error(), Some("invalid_token"), "{bad}");
        assert!(err.is_audience_mismatch(), "{bad}");
    }
}

/// Regression: arrays of any length are 401, never 403.
#[test]
fn audience_arrays_rejected_as_401() {
    let p = with_upstream();
    for aud in [
        Audience::Many(vec![RESOURCE.into()]),
        Audience::Many(vec![SIBLING.into()]),
        Audience::Many(vec![RESOURCE.into(), "https://other.example".into()]),
    ] {
        let err = p
            .check_audience(TokenKind::EdgeIssued, Some(&aud))
            .unwrap_err();
        assert_eq!(err, AuthError::InvalidClaim("aud"));
        assert_eq!(err.http_status(), 401);
    }
    assert_eq!(
        p.check_audience(
            TokenKind::UpstreamFirstParty,
            Some(&Audience::Many(vec![CLI_CLIENT.into()]))
        ),
        Err(AuthError::InvalidClaim("aud"))
    );
}

#[test]
fn upstream_allowlist_is_exact_and_misses_are_401() {
    let p = with_upstream();
    assert_eq!(
        p.check_audience(TokenKind::UpstreamFirstParty, Some(&one(CLI_CLIENT))),
        Ok(())
    );
    for bad in ["dcr-abc", "ruvector-edge-cli2", "ruvector-edge", RESOURCE] {
        let err = p
            .check_audience(TokenKind::UpstreamFirstParty, Some(&one(bad)))
            .unwrap_err();
        assert_eq!(err, AuthError::AudienceNotAllowed, "{bad}");
        assert_eq!(err.http_status(), 401);
    }
}

/// Regression (ADR §5.5): the upstream `kid` pin is mandatory and exact.
#[test]
fn upstream_kid_pin() {
    let p = with_upstream();
    assert_eq!(p.check_upstream_kid(&kid(&key(5))), Ok(()));
    assert_eq!(
        p.check_upstream_kid(&kid(&key(6))),
        Err(AuthError::UnknownKid)
    );
    let empty = AudiencePolicy {
        upstream: Some(upstream_policy(Vec::new())),
        ..edge_policy()
    };
    let err = empty.check_upstream_kid(&kid(&key(5))).unwrap_err();
    assert!(matches!(err, AuthError::InvalidConfig(_)));
    assert_eq!(err.http_status(), 500);
    assert_eq!(
        edge_policy().check_upstream_kid(&kid(&key(5))),
        Err(AuthError::WrongIssuer)
    );
}
