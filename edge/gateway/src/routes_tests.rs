//! Route table: classification and per-route authentication targets.

use super::*;
use ruvector_edge_auth::prm;

fn cfg() -> GatewayConfig {
    GatewayConfig {
        edge_issuer: "https://as.example".into(),
        edge_jwks_url: "https://as.example/.well-known/jwks.json".into(),
        rest_resource: ResourceUrl::parse("https://gw.example/v1").unwrap(),
        mcp_resource: ResourceUrl::parse("https://gw.example/v1/mcp").unwrap(),
        upstream: None,
        upstream_jwks_url: String::new(),
    }
}

#[test]
fn classifies_m1_routes_exactly() {
    let g = Method::Get;
    assert_eq!(classify(&g, "/v1/health"), Route::Health);
    assert_eq!(
        classify(&g, &format!("{PRM_WELL_KNOWN}/v1")),
        Route::PrmRest
    );
    assert_eq!(
        classify(&g, &format!("{PRM_WELL_KNOWN}/v1/mcp")),
        Route::PrmMcp
    );
    assert_eq!(classify(&g, "/v1/me"), Route::Me);
    assert_eq!(classify(&Method::Post, "/v1/me"), Route::OtherV1);
    assert_eq!(classify(&g, "/v1/collections"), Route::OtherV1);
    assert_eq!(classify(&g, "/v1"), Route::OtherV1);
    assert_eq!(classify(&g, "/v10"), Route::NotFound);
    assert_eq!(
        classify(&g, &format!("{PRM_WELL_KNOWN}/v2")),
        Route::NotFound
    );
    assert_eq!(classify(&g, "/"), Route::NotFound);
}

/// Regression (RFC 9728 §3.3): the bare metadata path is 404, since its
/// `resource` would have to be the bare origin (ADR-351 §5.7).
#[test]
fn bare_protected_resource_metadata_is_not_served() {
    assert_eq!(classify(&Method::Get, PRM_WELL_KNOWN), Route::NotFound);
    assert_eq!(classify(&Method::Options, PRM_WELL_KNOWN), Route::NotFound);
}

/// Regression (/v1/mcp fell into OtherV1): every method but OPTIONS on the
/// MCP resource and below is `Mcp`, bound to the MCP resource and surface.
#[test]
fn mcp_paths_bind_to_the_mcp_resource_and_surface() {
    let c = cfg();
    for m in [Method::Post, Method::Get, Method::Delete] {
        for p in ["/v1/mcp", "/v1/mcp/sse"] {
            assert_eq!(classify(&m, p), Route::Mcp, "{m:?} {p}");
        }
    }
    assert_eq!(classify(&Method::Post, "/v1/mcpx"), Route::OtherV1);
    let (res, surface) = auth_target(Route::Mcp, &c).unwrap();
    assert_eq!(res, &c.mcp_resource);
    assert_eq!(surface, RouteSurface::Mcp);
    assert_eq!(
        prm::metadata_url(res),
        "https://gw.example/.well-known/oauth-protected-resource/v1/mcp"
    );
    for r in [Route::Me, Route::OtherV1] {
        assert_eq!(
            auth_target(r, &c),
            Some((&c.rest_resource, RouteSurface::Rest))
        );
    }
    for r in [
        Route::Health,
        Route::PrmRest,
        Route::PrmMcp,
        Route::Preflight,
        Route::NotFound,
    ] {
        assert!(auth_target(r, &c).is_none(), "{r:?}");
    }
}

/// Regression (browser MCP discovery): preflights on any `/v1` path and on
/// the metadata documents are answered, never authenticated.
#[test]
fn preflight_is_never_authenticated() {
    let o = Method::Options;
    for p in [
        "/v1/health",
        "/v1/me",
        "/v1/mcp",
        "/v1/collections/x",
        "/.well-known/oauth-protected-resource/v1",
        "/.well-known/oauth-protected-resource/v1/mcp",
    ] {
        assert_eq!(classify(&o, p), Route::Preflight, "{p}");
    }
    assert_eq!(classify(&o, "/elsewhere"), Route::NotFound);
    assert!(auth_target(Route::Preflight, &cfg()).is_none());
}

/// The data routes (REST table, `/v1/ops`, `/v1/claim`) are `OtherV1`,
/// bound to the `/v1` resource and REST surface.
#[test]
fn data_routes_bind_to_the_v1_resource() {
    let c = cfg();
    for (m, p) in [
        (Method::Post, "/v1/ops"),
        (Method::Post, "/v1/claim"),
        (Method::Get, "/v1/usage"),
        (Method::Get, "/v1/collections"),
        (Method::Post, "/v1/collections/d/query"),
        (Method::Delete, "/v1/collections/d/vectors"),
    ] {
        let r = classify(&m, p);
        assert_eq!(r, Route::OtherV1, "{p}");
        assert_eq!(
            auth_target(r, &c),
            Some((&c.rest_resource, RouteSurface::Rest))
        );
    }
}
