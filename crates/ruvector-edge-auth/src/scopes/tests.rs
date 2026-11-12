use super::*;

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| (*x).to_string()).collect()
}

fn set(caps: &[Capability]) -> CapabilitySet {
    let mut out = CapabilitySet::EMPTY;
    caps.iter().for_each(|c| out.insert(*c));
    out
}

#[test]
fn capabilities_from_ruvector_scopes() {
    let caps = capabilities_for(
        &s(&["ruvector:read"]),
        TokenKind::EdgeIssued,
        RouteSurface::Mcp,
    );
    assert!(caps.contains(Capability::Read));
    assert!(!caps.contains(Capability::Write));

    let caps = capabilities_for(
        &s(&["ruvector:read", "ruvector:write", "ruvector:admin"]),
        TokenKind::EdgeIssued,
        RouteSurface::Rest,
    );
    assert_eq!(
        caps.iter().collect::<Vec<_>>(),
        [
            Capability::Read,
            Capability::Write,
            Capability::CreateCollection,
            Capability::Admin,
        ]
    );
}

/// Regression (ADR §5.3): upstream product scopes, identity scopes and
/// `offline_access` grant nothing.
#[test]
fn legacy_unknown_and_reserved_scopes_grant_nothing() {
    let caps = capabilities_for(
        &s(&[
            "mcp:read",
            "mcp:invoke",
            "openid",
            "offline_access",
            "brains:contribute",
            "namespaces:claim",
            "RUVECTOR:READ",
            "ruvector:read2",
        ]),
        TokenKind::EdgeIssued,
        RouteSurface::Rest,
    );
    assert_eq!(caps, CapabilitySet::EMPTY);
}

/// Regression (ADR §5.5): upstream tokens get the fixed read/write/admin set
/// on REST only, independent of their (upstream) scopes.
#[test]
fn upstream_first_party_fixed_grant_on_rest_only() {
    let rest = capabilities_for(
        &s(&["openid", "mcp:invoke"]),
        TokenKind::UpstreamFirstParty,
        RouteSurface::Rest,
    );
    assert_eq!(rest, set(UPSTREAM_FIRST_PARTY_CAPS));
    assert!(rest.contains(Capability::Admin));
    assert!(!rest.contains(Capability::PublishPublic));
    let mcp = capabilities_for(
        &s(&["ruvector:read"]),
        TokenKind::UpstreamFirstParty,
        RouteSurface::Mcp,
    );
    assert_eq!(mcp, CapabilitySet::EMPTY);
}

#[test]
fn satisfying_scopes_use_ruvector_vocabulary() {
    for cap in Capability::ALL {
        let scope = cap.satisfying_scope();
        assert!(scope.starts_with("ruvector:"), "{cap:?} -> {scope}");
        assert!(SCOPE_TABLE.iter().any(|(s, _)| *s == scope), "{scope}");
        let caps = capabilities_for(&s(&[scope]), TokenKind::EdgeIssued, RouteSurface::Rest);
        assert!(caps.contains(cap), "{scope} must grant {cap:?}");
    }
}

/// Regression (ADR §5.3/§5.7): `ruvector:admin` and `ruvector:publish` are
/// `/v1`-only; an MCP-surface token carrying them gets neither capability.
#[test]
fn admin_and_publish_never_granted_on_mcp() {
    let all = s(&[
        "ruvector:read",
        "ruvector:write",
        "ruvector:admin",
        "ruvector:publish",
    ]);
    let mcp = capabilities_for(&all, TokenKind::EdgeIssued, RouteSurface::Mcp);
    assert_eq!(
        mcp,
        set(&[
            Capability::Read,
            Capability::Write,
            Capability::CreateCollection
        ])
    );
    let rest = capabilities_for(&all, TokenKind::EdgeIssued, RouteSurface::Rest);
    assert!(rest.contains(Capability::Admin) && rest.contains(Capability::PublishPublic));
}

#[test]
fn every_route_row_resolves_to_itself() {
    for (method, pattern, req) in ROUTE_TABLE {
        let concrete = pattern
            .replace("{c}", "coll_1")
            .replace("{id}", "vec-9")
            .replace("{sub}", "es1_abcdefghijklmnopqrstuvwxyz");
        assert_eq!(
            route_requirement(*method, &concrete),
            Some(*req),
            "{pattern}"
        );
    }
    assert_eq!(SCOPE_TABLE_VERSION, 3);
}

/// Regression: every ADR §7.2 M1 route has a row (MCP, ops and tenant
/// administration were missing and would have been default-denied).
#[test]
fn m1_routes_are_covered() {
    use RouteRequirement::*;
    let admin = Capability(super::Capability::Admin);
    for (m, p, want) in [
        (Method::Post, "/v1/mcp", AnyValidToken),
        (Method::Post, "/v1/ops", AnyValidToken),
        (
            Method::Post,
            "/v1/tenant:claim",
            AnyOf(&[super::Capability::Write, super::Capability::Admin]),
        ),
        (Method::Get, "/v1/tenant/members", admin),
        (Method::Post, "/v1/tenant/members", admin),
        (Method::Get, "/v1/tenant/members/es1_x", admin),
        (Method::Delete, "/v1/tenant/members/es1_x", admin),
        (Method::Post, "/v1/tenant/deny", admin),
    ] {
        assert_eq!(route_requirement(m, p), Some(want), "{m:?} {p}");
    }
}

#[test]
fn requirement_satisfaction() {
    let claim = route_requirement(Method::Post, "/v1/tenant:claim").unwrap();
    let write = capabilities_for(
        &s(&["ruvector:write"]),
        TokenKind::EdgeIssued,
        RouteSurface::Rest,
    );
    let admin = capabilities_for(
        &s(&["ruvector:admin"]),
        TokenKind::EdgeIssued,
        RouteSurface::Rest,
    );
    let read = capabilities_for(
        &s(&["ruvector:read"]),
        TokenKind::EdgeIssued,
        RouteSurface::Rest,
    );
    assert!(claim.satisfied_by(write));
    assert!(claim.satisfied_by(admin));
    assert!(!claim.satisfied_by(read));
    let members = route_requirement(Method::Get, "/v1/tenant/members").unwrap();
    assert!(members.satisfied_by(admin));
    assert!(!members.satisfied_by(write));
    assert!(RouteRequirement::AnyValidToken.satisfied_by(CapabilitySet::EMPTY));
}

#[test]
fn unknown_routes_denied() {
    for (m, p) in [
        (Method::Get, "/"),
        (Method::Get, "/v1"),
        (Method::Get, "/v1/"),
        (Method::Get, "/v1/me/"),
        (Method::Put, "/v1/me"),
        (Method::Delete, "/v1/collections"),
        (Method::Get, "/v1/collections//vectors/x"),
        (Method::Get, "/v1/collections/../usage"),
        (Method::Get, "/v1/collections/./x"),
        (Method::Post, "/v1/collections/c/vectors:drop"),
        (Method::Post, "/v1/collections/:upsert"),
        (Method::Post, "/v1/collections/a:b/vectors:upsert"),
        (Method::Get, "/v1/collections/c/vectors/id/extra"),
        (Method::Get, "v1/health"),
        (Method::Get, "/V1/health"),
        (Method::Get, "/v1/mcp"),
        (Method::Post, "/v1/mcp/"),
        (Method::Get, "/v1/tenant:claim"),
        (Method::Post, "/v1/tenant/members/"),
        (Method::Put, "/v1/tenant/members/es1_x"),
        (Method::Get, "/v1/tenant/deny"),
    ] {
        assert_eq!(route_requirement(m, p), None, "{m:?} {p}");
    }
}

#[test]
fn anonymous_routes_are_only_health_and_metadata() {
    for (_, pattern, req) in ROUTE_TABLE {
        if *req == RouteRequirement::Anonymous {
            assert!(
                *pattern == "/v1/health"
                    || pattern.starts_with("/.well-known/oauth-protected-resource"),
                "{pattern}"
            );
        }
    }
}
