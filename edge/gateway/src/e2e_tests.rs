//! Cross-crate end to end (native): access tokens minted — and exchanged —
//! by the edge AS core (`ruvector-edge-authz`, configured from the deployed
//! auth-worker vars) are verified by the gateway's authenticator under its
//! compiled trust root, routed by the production route table and served by
//! the production handlers over the in-process Durable Object backend.
//! Only the Workers glue (`api::serve`: `Request` in, `Response` out) is
//! skipped.

use crate::config::EDGE_JWKS_PATH;
use crate::e2e_as::{payload, sign_jws, wrangler_var, ADAPTER, UPSTREAM};
use crate::e2e_world::World;
use crate::testkit::{Rng, T0};
use crate::trust_root;
use ruvector_edge_auth::prm::{self, MCP_SCOPES, REST_SCOPES};
use ruvector_edge_auth::subject::edge_subject;
use ruvector_edge_auth::Jwk;
use ruvector_edge_authz::resource::{GATEWAY_V1_URL, TEAM_RESOURCE_URL};
use ruvector_edge_tenancy::{derive_tenant_key, Role};
use serde_json::{json, Value as Json};
use worker::Method;

const DIM: usize = 384;
const N: usize = 1000;
const TEAM_ALL: [&str; 4] = ["team:read", "team:write", "team:run", "offline_access"];

fn ids(v: &Json) -> Vec<String> {
    v["matches"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect()
}

fn brute_top10(rows: &[(String, Vec<f32>)], q: &[f32]) -> Vec<String> {
    let mut all: Vec<(f64, &str)> = rows
        .iter()
        .map(|(id, v)| {
            let d = v
                .iter()
                .zip(q)
                .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
                .sum();
            (d, id.as_str())
        })
        .collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(b.1)));
    all.iter().take(10).map(|(_, id)| id.to_string()).collect()
}

#[test]
fn both_workers_agree_on_issuer_audiences_scopes_and_tenant_namespace() {
    let w = World::new("ruvector:read ruvector:write");
    assert_eq!(wrangler_var("ISSUER"), trust_root::EDGE_ISSUER);
    assert_eq!(w.cfg.edge_issuer, trust_root::EDGE_ISSUER);
    assert_eq!(
        w.cfg.edge_jwks_url,
        format!("{}{}", trust_root::EDGE_ISSUER, EDGE_JWKS_PATH)
    );
    assert_eq!(EDGE_JWKS_PATH, ruvector_edge_authz::metadata::paths::JWKS);
    // The exchange target is the gateway's `/v1` resource.
    assert_eq!(w.v1(), GATEWAY_V1_URL);
    // Every scope the AS may mint for a gateway resource is one its RFC 9728
    // document advertises, and vice versa.
    for (url, prm) in [(w.v1(), &REST_SCOPES[..]), (w.mcp(), &MCP_SCOPES[..])] {
        let entry = w
            .edge_as
            .resources
            .get(&ruvector_edge_auth::ResourceUrl::parse(&url).unwrap())
            .unwrap_or_else(|| panic!("{url} not in RESOURCE_ALLOWLIST"));
        assert_eq!(entry.scopes(), prm, "{url}");
    }
    // The upstream issuer that namespaces tenants.
    assert_eq!(wrangler_var("UPSTREAM_ISSUER"), trust_root::UPSTREAM_ISSUER);
    assert_eq!(ruvector_edge_tenancy::UPSTREAM_ISSUER, UPSTREAM);
}

#[test]
fn edge_token_claim_create_upsert_query_fetch_delete() {
    let w = World::new("ruvector:read ruvector:write");
    let owner = w.edge_as.user_token(
        "alice",
        "org-a",
        &w.v1(),
        &["ruvector:read", "ruvector:write"],
    );
    let me = w.ok(&owner, Method::Get, "/v1/me", Json::Null);
    let tenant = derive_tenant_key(UPSTREAM, "org-a", "ws1").unwrap();
    assert_eq!(me["tenant_key"], json!(tenant.as_str()));
    assert_eq!(me["sub"], json!(edge_subject(UPSTREAM, "alice")));
    assert_eq!(
        (me["client_id"].clone(), me["act_sub"].clone()),
        (json!("edc-e2e-client"), Json::Null)
    );

    let r = w.send(&owner, Method::Post, "/v1/claim", Json::Null, None);
    assert_eq!((r.status, r.body.as_str()), (201, r#"{"role":"owner"}"#));
    let spec = json!({ "name": "docs", "dim": DIM, "metric": "l2", "shards": 3 });
    let r = w.send(&owner, Method::Post, "/v1/collections", spec, None);
    assert_eq!(r.status, 201, "{}", r.body);

    let mut rng = Rng(351);
    let rows: Vec<(String, Vec<f32>)> = (0..N).map(|i| (format!("v{i}"), rng.vec(DIM))).collect();
    for chunk in rows.chunks(100) {
        let vectors: Vec<Json> = chunk
            .iter()
            .map(|(id, v)| json!({ "id": id, "values": v }))
            .collect();
        let body = json!({ "vectors": vectors });
        let up = w.ok(&owner, Method::Post, "/v1/collections/docs/vectors", body);
        assert_eq!(up["upserted"], json!(chunk.len()));
    }
    let g = w.ok(&owner, Method::Get, "/v1/collections/docs", Json::Null);
    assert_eq!(
        (g["count"].clone(), g["shards"].clone()),
        (json!(N), json!(3))
    );

    for _ in 0..3 {
        let q = rng.vec(DIM);
        let got = w.ok(
            &owner,
            Method::Post,
            "/v1/collections/docs/query",
            json!({ "vector": q, "top_k": 10 }),
        );
        assert_eq!(got["shards_queried"], json!(3));
        assert_eq!(
            ids(&got),
            brute_top10(&rows, &q),
            "exact top-10 = brute force"
        );
    }

    let f = w.ok(
        &owner,
        Method::Post,
        "/v1/collections/docs/fetch",
        json!({ "ids": ["v7", "v999", "nope"], "include_values": true }),
    );
    let got = f["vectors"].as_array().unwrap();
    assert_eq!(got.len(), 2);
    let v7 = got.iter().find(|v| v["id"] == json!("v7")).unwrap();
    // Values round-trip bit-exactly at f32 (the JSON text is f64-widened).
    let values: Vec<f32> = v7["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    assert_eq!(values, rows[7].1);

    let r = w.send(
        &owner,
        Method::Delete,
        "/v1/collections/docs/vectors",
        json!({ "ids": ["v7", "v999", "nope"] }),
        None,
    );
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(
        serde_json::from_str::<Json>(&r.body).unwrap()["deleted"],
        json!(2)
    );
    let f = w.ok(
        &owner,
        Method::Post,
        "/v1/collections/docs/fetch",
        json!({ "ids": ["v7", "v999"] }),
    );
    assert_eq!(f["vectors"], json!([]));
    let g = w.ok(&owner, Method::Get, "/v1/collections/docs", Json::Null);
    assert_eq!(g["count"], json!(N - 2));
    let u = w.ok(&owner, Method::Get, "/v1/usage", Json::Null);
    assert_eq!(u["usage"]["vectors"], json!(N - 2));
}

#[test]
fn foreign_tenant_gets_404_and_viewer_writes_get_403() {
    let w = World::new("ruvector:read ruvector:write");
    let rw = ["ruvector:read", "ruvector:write"];
    let alice = w.edge_as.user_token("alice", "org-a", &w.v1(), &rw);
    assert_eq!(
        w.send(&alice, Method::Post, "/v1/claim", Json::Null, None)
            .status,
        201
    );
    let spec = json!({ "name": "docs", "dim": 2, "metric": "l2" });
    assert_eq!(
        w.send(&alice, Method::Post, "/v1/collections", spec, None)
            .status,
        201
    );
    let row = json!({ "vectors": [{ "id": "a", "values": [1.0, 2.0] }] });
    w.ok(
        &alice,
        Method::Post,
        "/v1/collections/docs/vectors",
        row.clone(),
    );

    // Tenant B (another org), claimed: A's collection does not exist for it.
    let bob = w.edge_as.user_token("bob", "org-b", &w.v1(), &rw);
    assert_eq!(
        w.send(&bob, Method::Post, "/v1/claim", Json::Null, None)
            .status,
        201
    );
    for (m, path, body) in [
        (Method::Get, "/v1/collections/docs", Json::Null),
        (
            Method::Post,
            "/v1/collections/docs/query",
            json!({ "vector": [1.0, 2.0], "top_k": 1 }),
        ),
        (
            Method::Post,
            "/v1/collections/docs/fetch",
            json!({ "ids": ["a"] }),
        ),
        (Method::Post, "/v1/collections/docs/vectors", row.clone()),
    ] {
        let r = w.send(&bob, m, path, body, None);
        assert_eq!(r.status, 404, "{path}: {}", r.body);
    }
    assert_eq!(
        w.ok(&bob, Method::Get, "/v1/collections", Json::Null)["collections"],
        json!([])
    );

    // A viewer of tenant A: reads, but a write is 403 role_required even
    // with `ruvector:write`, and 403 insufficient_scope (step-up) without it.
    let tenant = derive_tenant_key(UPSTREAM, "org-a", "ws1").unwrap();
    let (owner_sub, val_sub) = (
        edge_subject(UPSTREAM, "alice"),
        edge_subject(UPSTREAM, "val"),
    );
    w.b.invite(&tenant, &owner_sub, &val_sub, Role::Viewer, T0)
        .unwrap();
    let val = w.edge_as.user_token("val", "org-a", &w.v1(), &rw);
    let q = json!({ "vector": [1.0, 2.0], "top_k": 1 });
    assert_eq!(
        ids(&w.ok(&val, Method::Post, "/v1/collections/docs/query", q)),
        ["a"]
    );
    let r = w.send(
        &val,
        Method::Post,
        "/v1/collections/docs/vectors",
        row.clone(),
        None,
    );
    assert_eq!(r.status, 403);
    assert!(r.body.contains("role_required"), "{}", r.body);
    assert_eq!(r.www_authenticate, None);
    let del = json!({ "ids": ["a"] });
    assert_eq!(
        w.send(
            &val,
            Method::Delete,
            "/v1/collections/docs/vectors",
            del,
            None
        )
        .status,
        403
    );
    let val_ro = w
        .edge_as
        .user_token("val", "org-a", &w.v1(), &["ruvector:read"]);
    let r = w.send(
        &val_ro,
        Method::Post,
        "/v1/collections/docs/vectors",
        row,
        None,
    );
    assert_eq!(r.status, 403);
    assert!(r.body.contains("insufficient_scope"), "{}", r.body);
    let www = r.www_authenticate.expect("step-up challenge");
    assert!(
        www.contains(r#"error="insufficient_scope""#) && www.contains("ruvector:write"),
        "{www}"
    );
}

#[test]
fn exchanged_team_token_is_a_v1_token_for_ops_but_never_for_mcp() {
    let w = World::new("ruvector:read ruvector:write");
    // Alice owns her tenant (her own `/v1` login).
    let own = w.edge_as.user_token(
        "alice",
        "org-a",
        &w.v1(),
        &["ruvector:read", "ruvector:write"],
    );
    assert_eq!(
        w.send(&own, Method::Post, "/v1/claim", Json::Null, None)
            .status,
        201
    );

    // Her team.ruv.io token is never accepted by the gateway itself.
    let team = w
        .edge_as
        .user_token("alice", "org-a", TEAM_RESOURCE_URL, &TEAM_ALL);
    let r = w.send(&team, Method::Get, "/v1/collections", Json::Null, None);
    assert_eq!(r.status, 401);
    assert!(r.www_authenticate.unwrap().contains(prm::AUDIENCE_MISMATCH));

    // team.ruv.io exchanges it: a `/v1` token acting for Alice.
    let x = w.edge_as.exchange(&team, &w.v1(), None).expect("exchange");
    assert_eq!(x.scope, "ruvector:read ruvector:write");
    assert_eq!((x.refresh_token.as_deref(), x.expires_in), (None, 900));
    assert!(x.issued_token_type.is_some());
    let (subject, minted) = (payload(&team), payload(&x.access_token));
    assert_eq!(minted["aud"], json!(GATEWAY_V1_URL));
    assert_eq!(minted["act"], json!({ "sub": ADAPTER }));
    assert_eq!(minted["client_id"], json!(ADAPTER));
    for claim in [
        "iss",
        "sub",
        "upstream_iss",
        "org_id",
        "workspace_id",
        "family_id",
    ] {
        assert_eq!(minted[claim], subject[claim], "{claim}");
    }
    let tok = x.access_token;

    // Accepted on `/v1` (REST) with the adapter recorded as the actor.
    let me = w.ok(&tok, Method::Get, "/v1/me", Json::Null);
    assert_eq!(
        (me["role"].clone(), me["act_sub"].clone()),
        (json!("owner"), json!(ADAPTER))
    );
    assert_eq!(me["scopes"], json!(["ruvector:read", "ruvector:write"]));
    let spec = json!({ "name": "team", "dim": 2, "metric": "l2" });
    assert_eq!(
        w.send(&tok, Method::Post, "/v1/collections", spec, None)
            .status,
        201
    );

    // Accepted on `/v1/ops`: identity, a mapped write, a read.
    let (s, r) = w.op(&tok, "01HZX00000000000000000E2E1", "tenant_me", json!({}));
    assert_eq!((s, r["result"]["act_sub"].clone()), (200, json!(ADAPTER)));
    assert_eq!(r["result"]["sub"], json!(edge_subject(UPSTREAM, "alice")));
    let up = json!({ "collection": "team", "vectors": [{ "id": "t1", "values": [0.5, 0.5] }] });
    let (s, r) = w.op(&tok, "01HZX00000000000000000E2E2", "vector_upsert", up);
    assert_eq!((s, r["result"]["upserted"].clone()), (200, json!(1)), "{r}");
    let q = json!({ "collection": "team", "vector": [0.5, 0.5], "top_k": 1 });
    let (s, r) = w.op(&tok, "01HZX00000000000000000E2E3", "vector_query", q);
    assert_eq!((s, ids(&r["result"])), (200, vec!["t1".to_string()]));

    // Rejected on `/v1/mcp`: 401 naming the MCP metadata, audience mismatch.
    let list = json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": { "name": "collection_list", "arguments": {} } });
    let r = w.send(&tok, Method::Post, "/v1/mcp", list.clone(), None);
    assert_eq!(r.status, 401, "{}", r.body);
    let www = r.www_authenticate.unwrap();
    assert!(
        www.contains(&prm::metadata_url(&w.cfg.mcp_resource)),
        "{www}"
    );
    assert!(www.contains(prm::AUDIENCE_MISMATCH), "{www}");
    // ...while Alice's own MCP token works there (the path itself is fine).
    let mcp = w
        .edge_as
        .user_token("alice", "org-a", &w.mcp(), &["ruvector:read"]);
    assert_eq!(
        w.send(&mcp, Method::Post, "/v1/mcp", list, None).status,
        200
    );

    // `team:read` alone maps to `ruvector:read`: writes are stepped up.
    let team_ro = w.edge_as.user_token(
        "alice",
        "org-a",
        TEAM_RESOURCE_URL,
        &["team:read", "team:run"],
    );
    let ro = w
        .edge_as
        .exchange(&team_ro, &w.v1(), None)
        .expect("exchange");
    assert_eq!(ro.scope, "ruvector:read");
    let up = json!({ "collection": "team", "vectors": [{ "id": "t2", "values": [1.0, 1.0] }] });
    let (s, r) = w.op(
        &ro.access_token,
        "01HZX00000000000000000E2E4",
        "vector_upsert",
        up,
    );
    assert_eq!(
        (s, r["error"]["code"].clone()),
        (403, json!("insufficient_scope"))
    );
    // The exchange never targets `/v1/mcp`, and a forged actor is refused.
    assert!(w.edge_as.exchange(&team, &w.mcp(), None).is_err());
    let mut forged = payload(&tok);
    forged["act"] = json!({ "sub": "someone-else" });
    let header = json!({ "alg": "ES256", "typ": "at+jwt", "kid": Jwk::from_verifying_key(w.edge_as.signer.0.verifying_key()).kid });
    let forged = sign_jws(&header, &forged, &w.edge_as.signer.0);
    assert_eq!(
        w.send(&forged, Method::Get, "/v1/me", Json::Null, None)
            .status,
        401
    );
}
