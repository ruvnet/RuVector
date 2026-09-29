//! Tenant administration (members, deny entries) and the deny check, over
//! the in-process DO backend. Shared helpers for `drop_tests`.

use super::{handle, parse, ApiReply, ApiRoute, Caller};
use crate::api::guard;
use crate::api::ratelimit::mem::CountingLimiter;
use crate::api::ratelimit::Class;
use crate::backend::mem::MemBackend;
use crate::backend::Backend;
use crate::testkit::*;
use ruvector_edge_auth::scopes::{route_requirement, Method as RouteMethod};
use ruvector_edge_auth::{Capability, CapabilitySet};
use ruvector_edge_store::CallerContext;
use serde_json::{json, Value as Json};
use worker::Method;

pub(super) const JTI_B: &str = "ICEiIyQlJicoKSorLC0uLw";
pub(super) const FAM_B: &str = "MDEyMzQ1Njc4OTo7PD0-Pw";

pub(super) fn admin() -> CapabilitySet {
    caps(&[
        Capability::Read,
        Capability::Write,
        Capability::CreateCollection,
        Capability::Admin,
    ])
}

pub(super) fn who(org: &str, user: &str, scope: CapabilitySet) -> Caller {
    Caller {
        ctx: ctx(&tenant(org), user, scope),
        org_id: org.into(),
        workspace_id: "ws1".into(),
        scopes: vec![],
    }
}

/// `user` with token ids and client distinct from the owner's (so the
/// owner's own entries are never self-denials).
pub(super) fn other(org: &str, user: &str, scope: CapabilitySet) -> Caller {
    let client = format!("edc-{user}");
    let ctx = CallerContext::new(tenant(org), sub(user), client, JTI_B, FAM_B, None, scope);
    Caller {
        ctx,
        ..who(org, user, scope)
    }
}

pub(super) fn bob(org: &str, scope: CapabilitySet) -> Caller {
    other(org, "bob", scope)
}

pub(super) fn call<B: Backend>(
    b: &B,
    c: &Caller,
    m: Method,
    path: &str,
    body: Json,
    now: u64,
) -> ApiReply {
    let route = parse(&m, path).expect(path);
    let bytes = if body.is_null() {
        Vec::new()
    } else {
        body.to_string().into_bytes()
    };
    crate::testkit::block_on(handle(b, c, &route, &bytes, None, now, REST_MD))
}

pub(super) fn code(r: &ApiReply) -> (u16, String) {
    let v: Json = serde_json::from_str(&r.body).unwrap_or(Json::Null);
    let c = v["code"].as_str().unwrap_or_default().to_string();
    (r.status, c)
}

/// Guard result: `Ok` or `(status, code, www)`.
pub(super) fn g<B: Backend>(
    b: &B,
    l: &CountingLimiter,
    c: &Caller,
    now: u64,
) -> Result<(), (u16, String)> {
    block_on(guard(b, l, Class::Read, &c.ctx, now, REST_MD)).map_err(|r| {
        if r.reply.status == 401 {
            let www = r.reply.www_authenticate.clone().unwrap();
            assert!(
                www.contains("invalid_token") && www.contains(REST_MD),
                "{www}"
            );
        }
        code(&r.reply)
    })
}

#[test]
fn admin_routes_parse_and_are_in_the_auth_route_table() {
    let s = sub("bob");
    let cases = [
        (
            Method::Get,
            "/v1/tenant/members".to_string(),
            Some(ApiRoute::Members),
        ),
        (
            Method::Post,
            "/v1/tenant/members".into(),
            Some(ApiRoute::Invite),
        ),
        (
            Method::Delete,
            format!("/v1/tenant/members/{s}"),
            Some(ApiRoute::RemoveMember(s.clone())),
        ),
        (Method::Post, "/v1/tenant/deny".into(), Some(ApiRoute::Deny)),
        (
            Method::Delete,
            "/v1/collections/d".into(),
            Some(ApiRoute::Drop("d".into())),
        ),
        (Method::Delete, "/v1/tenant/members/".into(), None),
        (Method::Delete, "/v1/tenant/members".into(), None),
        (Method::Get, "/v1/tenant/deny".into(), None),
        (Method::Delete, "/v1/collections/a:b".into(), None),
    ];
    for (m, p, want) in cases {
        assert_eq!(parse(&m, &p), want, "{m:?} {p}");
    }
    for (m, p) in [
        (RouteMethod::Get, "/v1/tenant/members"),
        (RouteMethod::Post, "/v1/tenant/members"),
        (RouteMethod::Delete, "/v1/tenant/members/x"),
        (RouteMethod::Post, "/v1/tenant/deny"),
        (RouteMethod::Delete, "/v1/collections/d"),
    ] {
        assert!(route_requirement(m, p).is_some(), "{p}");
    }
}

#[test]
fn members_list_invite_remove_with_role_checks() {
    let b = MemBackend::new();
    let alice = who("org-a", "alice", admin());
    let bob_admin = bob("org-a", admin());
    let members = "/v1/tenant/members";
    // Unclaimed tenant: not_claimed, even with the admin scope.
    let r = call(&b, &alice, Method::Get, members, Json::Null, T0);
    assert_eq!(code(&r), (403, "not_claimed".into()));
    assert_eq!(
        call(&b, &alice, Method::Post, "/v1/claim", Json::Null, T0).status,
        201
    );
    // Owner without `ruvector:admin`: step-up naming it.
    let alice_rw = who("org-a", "alice", rw());
    let r = call(&b, &alice_rw, Method::Get, members, Json::Null, T0);
    assert_eq!(code(&r), (403, "insufficient_scope".into()));
    let www = r.www_authenticate.unwrap();
    assert!(
        www.contains("ruvector:admin") && www.contains(REST_MD),
        "{www}"
    );
    // Non-member with the admin scope: role_required.
    let r = call(&b, &bob_admin, Method::Get, members, Json::Null, T0);
    assert_eq!(code(&r), (403, "role_required".into()));
    // Invite: explicit editor, default viewer; owner/garbage refused.
    let inv = |c: &Caller, body: Json| call(&b, c, Method::Post, members, body, T0);
    let r = inv(&alice, json!({ "sub": sub("bob"), "role": "editor" }));
    assert_eq!(
        (r.status, r.body.contains("editor")),
        (201, true),
        "{}",
        r.body
    );
    let r = inv(&alice, json!({ "sub": sub("carol") }));
    assert!(r.status == 201 && r.body.contains("viewer"), "{}", r.body);
    assert_eq!(
        code(&inv(&alice, json!({ "sub": sub("dave"), "role": "owner" }))).0,
        400
    );
    assert_eq!(
        code(&inv(&alice, json!({ "sub": "not-an-edge-subject" }))).0,
        400
    );
    assert_eq!(
        code(&inv(&alice, json!({ "sub": sub("dave"), "x": 1 }))).0,
        400
    );
    // An editor holding the admin scope is still not the owner.
    for (m, p, body) in [
        (Method::Get, members.to_string(), Json::Null),
        (
            Method::Post,
            members.to_string(),
            json!({ "sub": sub("dave") }),
        ),
        (
            Method::Delete,
            format!("{members}/{}", sub("carol")),
            Json::Null,
        ),
    ] {
        let r = call(&b, &bob_admin, m, &p, body, T0);
        assert_eq!(code(&r), (403, "role_required".into()), "{p}");
    }
    let r = call(&b, &alice, Method::Get, members, Json::Null, T0);
    let v: Json = serde_json::from_str(&r.body).unwrap();
    let roles: Vec<_> = v["members"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["role"].clone())
        .collect();
    assert_eq!(roles.len(), 3, "{}", r.body);
    assert!(roles.contains(&json!("owner")) && roles.contains(&json!("editor")));
    // The owner (the tenant's only, hence last, owner) cannot be removed.
    let r = call(
        &b,
        &alice,
        Method::Delete,
        &format!("{members}/{}", sub("alice")),
        Json::Null,
        T0,
    );
    assert_eq!(code(&r), (409, "conflict".into()));
    let r = call(
        &b,
        &alice,
        Method::Delete,
        &format!("{members}/{}", sub("erin")),
        Json::Null,
        T0,
    );
    assert_eq!(code(&r).0, 404);
    // Removal also denies bob's sub for the rest of its tokens' life.
    let l = CountingLimiter::default();
    assert_eq!(g(&b, &l, &bob_admin, T0), Ok(()));
    let r = call(
        &b,
        &alice,
        Method::Delete,
        &format!("{members}/{}", sub("bob")),
        Json::Null,
        T0,
    );
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(
        g(&b, &l, &bob_admin, T0 + 1),
        Err((401, "invalid_token".into()))
    );
    assert_eq!(g(&b, &l, &alice, T0 + 1), Ok(()));
    assert_eq!(
        g(&b, &l, &bob_admin, T0 + 901),
        Ok(()),
        "removal deny lapses"
    );
    // Re-inviting clears the sub entry at once.
    let r = call(
        &b,
        &alice,
        Method::Delete,
        &format!("{members}/{}", sub("carol")),
        Json::Null,
        T0,
    );
    assert_eq!(r.status, 200);
    let carol = other("org-a", "carol", admin());
    assert!(g(&b, &l, &carol, T0 + 2).is_err());
    assert_eq!(inv(&alice, json!({ "sub": sub("carol") })).status, 201);
    assert_eq!(g(&b, &l, &carol, T0 + 3), Ok(()));
}

#[test]
fn deny_entries_are_owner_only_capped_and_enforced() {
    let b = MemBackend::new();
    let alice = who("org-a", "alice", admin());
    assert_eq!(
        call(&b, &alice, Method::Post, "/v1/claim", Json::Null, T0).status,
        201
    );
    b.invite(
        &tenant("org-a"),
        &sub("alice"),
        &sub("bob"),
        ruvector_edge_tenancy::Role::Viewer,
        T0,
    )
    .unwrap();
    let deny =
        |c: &Caller, body: Json, now| call(&b, c, Method::Post, "/v1/tenant/deny", body, now);
    let jti = |ttl: u64| json!({ "kind": "jti", "value": JTI_B, "ttl_s": ttl });
    // Viewer with the admin scope, and the owner without it.
    assert_eq!(
        code(&deny(&bob("org-a", admin()), jti(60), T0)),
        (403, "role_required".into())
    );
    let r = deny(&who("org-a", "alice", rw()), jti(60), T0);
    assert_eq!(code(&r), (403, "insufficient_scope".into()));
    // Validation: operator-only kinds, TTL caps, shapes, self-denial.
    for (body, why) in [
        (
            json!({ "kind": "org", "value": "org-a", "ttl_s": 60 }),
            "org",
        ),
        (json!({ "kind": "kid", "value": "k1", "ttl_s": 60 }), "kid"),
        (jti(901), "jti ttl"),
        (jti(0), "zero ttl"),
        (
            json!({ "kind": "family_id", "value": FAM_B, "ttl_s": 7 * 86_400 + 1 }),
            "family ttl",
        ),
        (
            json!({ "kind": "sub", "value": sub("bob"), "ttl_s": 30 * 86_400 + 1 }),
            "sub ttl",
        ),
        (
            json!({ "kind": "client_id", "value": "edc-bob", "ttl_s": 365 * 86_400 + 1 }),
            "client ttl",
        ),
        (
            json!({ "kind": "jti", "value": "short", "ttl_s": 60 }),
            "jti shape",
        ),
        (
            json!({ "kind": "client_id", "value": "a b", "ttl_s": 60 }),
            "client shape",
        ),
        (
            json!({ "kind": "sub", "value": sub("alice"), "ttl_s": 60 }),
            "self sub",
        ),
        (
            json!({ "kind": "jti", "value": "AAECAwQFBgcICQoLDA0ODw", "ttl_s": 60 }),
            "own jti",
        ),
        (
            json!({ "kind": "sub", "value": sub("zed"), "ttl_s": 60 }),
            "non-member sub",
        ),
        (
            json!({ "kind": "jti", "value": JTI_B, "ttl_s": 60, "reason": "x".repeat(257) }),
            "reason",
        ),
    ] {
        assert_eq!(
            code(&deny(&alice, body, T0)),
            (400, "invalid_request".into()),
            "{why}"
        );
    }
    let l = CountingLimiter::default();
    let bob_v = bob("org-a", rw());
    assert_eq!(g(&b, &l, &bob_v, T0), Ok(()));
    // jti: denied at once (write invalidates this isolate's cache).
    let r = deny(&alice, jti(900), T0);
    assert_eq!(r.status, 201, "{}", r.body);
    let v: Json = serde_json::from_str(&r.body).unwrap();
    assert_eq!(
        v,
        json!({ "kind": "jti", "value": JTI_B, "expires_at": T0 + 900 })
    );
    assert_eq!(
        g(&b, &l, &bob_v, T0 + 1),
        Err((401, "invalid_token".into()))
    );
    assert_eq!(g(&b, &l, &alice, T0 + 1), Ok(()), "owner unaffected");
    // Same jti in another tenant: tenant entries apply to their tenant only.
    assert_eq!(g(&b, &l, &bob("org-b", rw()), T0 + 1), Ok(()));
    // Expiry (past both the entry and the 30 s cache).
    assert_eq!(g(&b, &l, &bob_v, T0 + 931), Ok(()));
    // family_id, client_id and sub entries each catch bob's token.
    for (i, body) in [
        json!({ "kind": "family_id", "value": FAM_B, "ttl_s": 600 }),
        json!({ "kind": "client_id", "value": "edc-bob", "ttl_s": 600, "reason": "leaked app" }),
        json!({ "kind": "sub", "value": sub("bob"), "ttl_s": 600 }),
    ]
    .into_iter()
    .enumerate()
    {
        let now = T0 + 2_000 + 1_000 * i as u64;
        let r = deny(&alice, body.clone(), now);
        assert_eq!(r.status, 201, "{}", r.body);
        assert_eq!(
            g(&b, &l, &bob_v, now),
            Err((401, "invalid_token".into())),
            "{body}"
        );
        // Renewing an entry keeps one row; let it lapse before the next kind.
        assert_eq!(deny(&alice, body, now).status, 201);
        assert_eq!(g(&b, &l, &bob_v, now + 700), Ok(()));
    }
    // A role change does not lift a member's suspension (only a new
    // invite clears the removal entry).
    let now = T0 + 9_000;
    let body = json!({ "kind": "sub", "value": sub("bob"), "ttl_s": 600 });
    assert_eq!(deny(&alice, body, now).status, 201);
    let body = json!({ "sub": sub("bob"), "role": "editor" });
    let r = call(&b, &alice, Method::Post, "/v1/tenant/members", body, now);
    assert_eq!(r.status, 201, "{}", r.body);
    assert_eq!(
        g(&b, &l, &bob_v, now + 1),
        Err((401, "invalid_token".into()))
    );
}
