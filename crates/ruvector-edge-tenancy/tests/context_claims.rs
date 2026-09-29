//! ADR-351 §4.1 / §4.2 / §5.3 / §5.5: `TenantContext::from_verified`.

mod common;

use common::{edge, upstream, Tok, AUD, EDGE_ISS, FAMILY, JTI, SUB};
use ruvector_edge_auth::{Capability, CapabilitySet, RouteSurface, TokenKind, VerifiedClaims};
use ruvector_edge_tenancy::{
    derive_tenant_key, ProblemCode, Role, TenancyError, TenantContext, UPSTREAM_ISSUER,
};

/// `base32lower(sha256("v1|https://auth.cognitum.one|org-1|ws-1"))[0..26]`,
/// cross-checked with coreutils (`sha256sum | xxd -r -p | base32`).
const GOLDEN_UPSTREAM_TENANT_KEY: &str = "crqpcs6dpk7kjq463bcaqaqjvz";

fn build(t: Tok, role: Option<Role>, surface: RouteSurface) -> Result<TenantContext, TenancyError> {
    TenantContext::from_verified(t.verify(), role, surface)
}

fn caps(list: &[Capability]) -> CapabilitySet {
    let mut s = CapabilitySet::EMPTY;
    for c in list {
        s.insert(*c);
    }
    s
}

#[test]
fn tenant_key_is_keyed_on_upstream_iss_not_edge_iss() {
    let c = build(edge(), Some(Role::Editor), RouteSurface::Mcp).unwrap();
    assert_eq!(c.tenant_key().as_str(), GOLDEN_UPSTREAM_TENANT_KEY);
    assert_eq!(
        c.tenant_key(),
        &derive_tenant_key(UPSTREAM_ISSUER, "org-1", "ws-1").unwrap()
    );
    assert_ne!(
        c.tenant_key(),
        &derive_tenant_key(EDGE_ISS, "org-1", "ws-1").unwrap()
    );
    assert_eq!(c.upstream_iss(), UPSTREAM_ISSUER);
    assert_eq!(c.org_id(), "org-1");
    assert_eq!(c.workspace_id(), "ws-1");
    assert_eq!(c.sub(), SUB);
    assert_eq!(c.client_id(), "dcr-client-123");
    assert_eq!(c.family_id(), FAMILY);
    assert_eq!(c.jti(), JTI);
    assert_eq!(c.token_kind(), TokenKind::EdgeIssued);
    assert_eq!(c.role(), Some(Role::Editor));
    assert_eq!(
        c.capabilities(),
        caps(&[
            Capability::Read,
            Capability::Write,
            Capability::CreateCollection
        ])
    );
}

/// Regression: the edge path and the §5.5 first-party path must resolve the
/// same (org, workspace) to ONE tenant (one ledger, one `tenant:claim` race).
#[test]
fn edge_and_upstream_first_party_paths_share_the_tenant() {
    let e = build(edge(), None, RouteSurface::Rest).unwrap();
    let u = build(upstream(), None, RouteSurface::Rest).unwrap();
    assert_eq!(e.tenant_key(), u.tenant_key());
    assert_eq!(e.upstream_iss(), u.upstream_iss());
    // A different edge hostname (custom domain, dev vs prod) does not re-key.
    let mut moved = edge();
    moved.iss = "https://edge-auth.ruvector.example";
    let m = build(moved, None, RouteSurface::Rest).unwrap();
    assert_eq!(m.tenant_key(), e.tenant_key());
}

#[test]
fn upstream_token_with_foreign_issuer_is_401() {
    let mut t = upstream();
    t.iss = EDGE_ISS; // verifier-valid for its own policy, but not the upstream IdP
    let err = build(t, Some(Role::Owner), RouteSurface::Rest).unwrap_err();
    assert_eq!(err, TenancyError::InvalidTenantClaim("upstream_iss"));
    assert_eq!(err.problem_code(), ProblemCode::InvalidToken);
}

/// Regression (§5.5): first-party mode is REST `/v1` only, never `/v1/mcp`.
#[test]
fn upstream_first_party_rejected_off_rest_surface() {
    for role in [None, Some(Role::Viewer), Some(Role::Owner)] {
        let err = build(upstream(), role, RouteSurface::Mcp).unwrap_err();
        assert_eq!(err, TenancyError::InvalidTenantClaim("token_kind"));
        assert_eq!(err.problem_code(), ProblemCode::InvalidToken);
        assert!(build(upstream(), role, RouteSurface::Rest).is_ok());
    }
    // Edge tokens are fine on both surfaces.
    assert!(build(edge(), None, RouteSurface::Mcp).is_ok());
}

/// Regression (§5.3/§5.5): upstream product scopes grant nothing; the fixed
/// ruvector set applies, always ∩ role.
#[test]
fn upstream_capabilities_ignore_token_scopes_and_intersect_role() {
    let mut t = upstream();
    t.scope = common::READ_WRITE_SCOPES;
    assert_eq!(
        build(t.clone(), None, RouteSurface::Rest)
            .unwrap()
            .capabilities(),
        CapabilitySet::EMPTY
    );
    assert_eq!(
        build(t.clone(), Some(Role::Viewer), RouteSurface::Rest)
            .unwrap()
            .capabilities(),
        caps(&[Capability::Read])
    );
    let owner = build(upstream(), Some(Role::Owner), RouteSurface::Rest).unwrap();
    assert_eq!(
        owner.capabilities(),
        caps(&[
            Capability::Read,
            Capability::Write,
            Capability::CreateCollection
        ])
    );
    assert!(!owner.capabilities().contains(Capability::PublishPublic));
}

/// Regression (§4.2, §5.3): capability = scope ∩ role, default-deny.
#[test]
fn scope_without_role_and_role_without_scope_are_denied() {
    // Scope, no membership row.
    let c = build(edge(), None, RouteSurface::Rest).unwrap();
    assert_eq!(c.capabilities(), CapabilitySet::EMPTY);
    assert_eq!(c.role(), None);
    // Role, no scope.
    let mut t = edge();
    t.scope = "";
    let c = build(t, Some(Role::Owner), RouteSurface::Rest).unwrap();
    assert_eq!(c.capabilities(), CapabilitySet::EMPTY);
    // Viewer with write scope keeps only read.
    let c = build(edge(), Some(Role::Viewer), RouteSurface::Rest).unwrap();
    assert_eq!(c.capabilities(), caps(&[Capability::Read]));
    // Editor with read-only scope keeps only read.
    let mut t = edge();
    t.scope = common::READ_SCOPE;
    let c = build(t, Some(Role::Editor), RouteSurface::Rest).unwrap();
    assert_eq!(c.capabilities(), caps(&[Capability::Read]));
}

/// Regression: family_id is carried and required; edge ids are 16-byte b64url.
#[test]
fn family_id_and_jti_required_and_well_formed_on_edge_tokens() {
    let mut t = edge();
    t.family_id = None;
    assert_eq!(
        build(t, None, RouteSurface::Rest).unwrap_err(),
        TenancyError::InvalidTenantClaim("family_id")
    );
    for bad in [
        "fam-1",
        "fam0fam0fam0fam0fam0f",   // 21 chars
        "fam0fam0fam0fam0fam0fQQ", // 23 chars
        "fam0fam0fam0fam0fam0fB",  // non-canonical trailing bits
        "fam0fam0fam0fam0fam0f=",  // padding
        "fam0fam0fam0fam0fam0f+",  // standard alphabet
    ] {
        let mut t = edge();
        t.family_id = Some(bad);
        assert_eq!(
            build(t, None, RouteSurface::Rest).unwrap_err(),
            TenancyError::InvalidTenantClaim("family_id"),
            "{bad}"
        );
        let mut t = edge();
        t.jti = Some(bad);
        assert_eq!(
            build(t, None, RouteSurface::Rest).unwrap_err(),
            TenancyError::InvalidTenantClaim("jti"),
            "{bad}"
        );
    }
    // Upstream ids are not ours: any verifier-accepted value is carried.
    let u = build(upstream(), None, RouteSurface::Rest).unwrap();
    assert_eq!(u.family_id(), "upstream-family-1");
    assert_eq!(u.jti(), "upstream-jti-1");
}

/// Regression (§5.5 upstream path unusable): a real upstream token carries a
/// UUID `sub`; it is normalised with the edge-subject function, so the
/// upstream-first-party path yields the same actor the edge AS mints.
#[test]
fn upstream_sub_is_normalised_to_the_edge_subject() {
    let u = build(upstream(), None, RouteSurface::Rest).unwrap();
    assert_eq!(
        u.sub(),
        ruvector_edge_auth::subject::edge_subject(UPSTREAM_ISSUER, common::UPSTREAM_SUB)
    );
    assert!(ruvector_edge_auth::subject::is_edge_subject(u.sub()));
    let mut t = upstream();
    t.sub = "user-1".into();
    let other = build(t, None, RouteSurface::Rest).unwrap();
    assert_ne!(other.sub(), u.sub());
}

/// Regression (§5.2): an edge token's `sub` must already be an edge subject.
#[test]
fn raw_sub_is_rejected_on_edge_tokens() {
    for bad in [
        "8d0c7a52-user",
        "user-1",
        "es1_abcdefghijklmnopqrstuvwxy",   // 25
        "es1_abcdefghijklmnopqrstuvwxyza", // 27
        "es1_ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "es1_abcdefghijklmnopqrstuvwx01", // 0/1 not base32
        "es1_abcdefghijklmnopqrstuvwx89",
        "ES1_abcdefghijklmnopqrstuvwxyz",
        "es2_abcdefghijklmnopqrstuvwxyz",
    ] {
        let mut t = edge();
        t.sub = bad.into();
        let err = build(t, Some(Role::Owner), RouteSurface::Rest).unwrap_err();
        assert_eq!(err, TenancyError::InvalidTenantClaim("sub"), "{bad}");
    }
}

#[test]
fn sub_and_client_do_not_change_the_tenant() {
    let a = build(edge(), None, RouteSurface::Rest).unwrap();
    let mut t = edge();
    t.sub = common::SUB2.into();
    t.client_id = "other-client";
    t.aud = "https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev/v1";
    let b = build(t, None, RouteSurface::Rest).unwrap();
    assert_eq!(a.tenant_key(), b.tenant_key());
    assert_ne!(a.sub(), b.sub());
}

#[test]
fn missing_or_invalid_tenant_claims_are_401() {
    // Missing components cannot pass the real verifier; build them directly.
    let missing = |org: Option<&str>, ws: Option<&str>| {
        VerifiedClaims::for_tests(
            TokenKind::EdgeIssued,
            EDGE_ISS,
            AUD,
            SUB,
            "c",
            org,
            ws,
            &[],
            1,
            2,
        )
    };
    let mut cases: Vec<(VerifiedClaims, &str)> = vec![
        (missing(None, Some("w")), "org_id"),
        (missing(Some("o"), None), "workspace_id"),
    ];
    for (org, ws, what) in [
        ("o|x", "w", "org_id"),
        ("o", "w/x", "workspace_id"),
        ("\u{FF4F}", "w", "org_id"),
        (&*"o".repeat(65), "w", "org_id"),
    ] {
        let mut t = edge();
        t.org = Box::leak(org.to_string().into_boxed_str());
        t.ws = Box::leak(ws.to_string().into_boxed_str());
        cases.push((t.verify(), what));
    }
    for (c, what) in cases {
        let err = TenantContext::from_verified(c, None, RouteSurface::Rest).unwrap_err();
        assert_eq!(err, TenancyError::InvalidTenantClaim(what));
        assert_eq!(err.problem_code(), ProblemCode::InvalidToken);
    }
}
