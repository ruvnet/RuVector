//! `/v1/ops` end to end: envelope checks, authorization, cross-tenant
//! isolation, op_id replay, quotas and dry runs.

mod common;

use common::*;
use ruvector_edge_auth::CapabilitySet;
use ruvector_edge_store::{ledger_meta_for, MemSqlStore};
use ruvector_edge_tenancy::{ledger_do_name, QuotaLimits, Role};
use serde_json::{json, Value as Json};

type H = Harness<MemSqlStore>;

fn setup() -> (H, ruvector_edge_tenancy::TenantKey) {
    let mut h = H::new();
    let t = tenant("org-a");
    h.claim(&t, "owner", &[("ed", Role::Editor), ("vi", Role::Viewer)]);
    let (s, r) = h.call(
        &ctx(&t, "owner", write_caps()),
        "collection_create",
        json!({"name": "c", "dim": 2, "metric": "l2"}),
    );
    assert_eq!(s, 200, "{r}");
    (h, t)
}

fn v(id: &str, x: f32) -> Json {
    json!({"id": id, "values": [x, 0.0]})
}

fn raw(
    h: &mut H,
    c: &ruvector_edge_store::CallerContext,
    body: &[u8],
    key: Option<&str>,
) -> (u16, Json) {
    let r = h
        .disp
        .dispatch(&mut h.cluster, c, body, key, &h.clock, &h.entropy);
    (r.status, serde_json::from_str(&r.body).unwrap())
}

#[test]
fn envelope_target_tenant_and_op_checks() {
    let (mut h, t) = setup();
    let c = ctx(&t, "owner", write_caps());
    let id = op_id(900);
    let mut env: Json =
        serde_json::from_slice(&envelope(&t, "collection_list", &id, json!({}), false)).unwrap();
    let send = |h: &mut H, e: &Json| raw(h, &c, &serde_json::to_vec(e).unwrap(), None);
    let mut bad = env.clone();
    bad["target"] = json!("https://evil.example/v1/ops");
    assert_eq!(
        (
            send(&mut h, &bad).0,
            code(&send(&mut h, &bad).1).to_string()
        ),
        (400, "target_mismatch".into())
    );
    let mut bad = env.clone();
    bad["tenant_key"] = json!(tenant("org-b").as_str());
    assert_eq!(code(&send(&mut h, &bad).1), "tenant_mismatch");
    assert_eq!(send(&mut h, &bad).0, 403);
    let mut bad = env.clone();
    bad["op"] = json!("collection_drop_everything");
    assert_eq!(code(&send(&mut h, &bad).1), "unknown_op");
    for (k, val) in [
        ("v", json!(2)),
        ("op_id", json!("short")),
        ("extra", json!(1)),
        ("args", json!([1])),
    ] {
        let mut bad = env.clone();
        bad[k] = val;
        let (s, r) = send(&mut h, &bad);
        assert_eq!((s, code(&r)), (400, "invalid_request"), "{k}");
    }
    assert_eq!(
        code(&raw(&mut h, &c, b"not json", None).1),
        "invalid_request"
    );
    let body = serde_json::to_vec(&env).unwrap();
    assert_eq!(
        code(&raw(&mut h, &c, &body, Some("DIFFERENT_KEY_0000000000000")).1),
        "invalid_request"
    );
    env["args"] = json!({"unexpected": true});
    assert_eq!(code(&send(&mut h, &env).1), "invalid_request");
}

#[test]
fn scope_and_role_checks() {
    let (mut h, t) = setup();
    // No upsert without ruvector:write, even as owner (step-up).
    let (s, r) = h.call(
        &ctx(&t, "owner", read_caps()),
        "vector_upsert",
        json!({"collection": "c", "vectors": [v("a", 1.0)]}),
    );
    assert_eq!((s, code(&r)), (403, "insufficient_scope"));
    assert_eq!(r["error"]["scope"], "ruvector:write");
    // Viewer with write scope still cannot write.
    let vi = ctx(&t, "vi", write_caps());
    let (s, r) = h.call(
        &vi,
        "vector_upsert",
        json!({"collection": "c", "vectors": [v("a", 1.0)]}),
    );
    assert_eq!((s, code(&r)), (403, "role_required"));
    assert_eq!(
        code(
            &h.call(
                &vi,
                "collection_create",
                json!({"name": "d", "dim": 2, "metric": "l2"})
            )
            .1
        ),
        "role_required"
    );
    assert_eq!(
        code(
            &h.call(
                &vi,
                "vector_delete",
                json!({"collection": "c", "ids": ["a"]})
            )
            .1
        ),
        "role_required"
    );
    // Viewer reads; editor writes.
    assert_eq!(
        h.call(
            &vi,
            "vector_query",
            json!({"collection": "c", "vector": [1.0, 0.0], "top_k": 1})
        )
        .0,
        200
    );
    let (s, r) = h.call(
        &ctx(&t, "ed", write_caps()),
        "vector_upsert",
        json!({"collection": "c", "vectors": [v("a", 1.0)]}),
    );
    assert_eq!(s, 200, "{r}");
    // Stranger: role_required; unclaimed tenant: not_claimed; tenant_me works for both.
    let stranger = ctx(&t, "stranger", write_caps());
    assert_eq!(
        code(&h.call(&stranger, "collection_list", json!({})).1),
        "role_required"
    );
    let t2 = tenant("org-new");
    let fresh = ctx(&t2, "someone", write_caps());
    assert_eq!(
        code(&h.call(&fresh, "collection_list", json!({})).1),
        "not_claimed"
    );
    let (s, me) = h.call(&fresh, "tenant_me", json!({}));
    assert_eq!(s, 200);
    assert_eq!(
        (
            me["result"]["claimed"].clone(),
            me["result"]["role"].clone()
        ),
        (json!(false), Json::Null)
    );
    let (_, me) = h.call(&ctx(&t, "vi", CapabilitySet::EMPTY), "tenant_me", json!({}));
    assert_eq!(me["result"]["role"], "viewer");
    assert_eq!(me["result"]["tenant_key"], t.as_str());
}

#[test]
fn cross_tenant_access_is_impossible() {
    let (mut h, ta) = setup();
    let owner_a = ctx(&ta, "owner", write_caps());
    h.call(
        &owner_a,
        "vector_upsert",
        json!({"collection": "c", "vectors": [v("secret", 1.0)]}),
    );
    let tb = tenant("org-b");
    h.claim(&tb, "mallory", &[]);
    let mallory = ctx(&tb, "mallory", write_caps());
    // Same collection name from tenant B resolves in B's own ledger: 404.
    for (op, args) in [
        (
            "vector_query",
            json!({"collection": "c", "vector": [1.0, 0.0], "top_k": 5}),
        ),
        (
            "vector_fetch",
            json!({"collection": "c", "ids": ["secret"]}),
        ),
        (
            "vector_upsert",
            json!({"collection": "c", "vectors": [v("x", 1.0)]}),
        ),
        (
            "vector_delete",
            json!({"collection": "c", "ids": ["secret"]}),
        ),
    ] {
        let (s, r) = h.call(&mallory, op, args);
        assert_eq!((s, code(&r)), (404, "not_found"), "{op}");
    }
    // Naming tenant A in the envelope with a tenant-B token: tenant_mismatch.
    let body = envelope(
        &ta,
        "vector_query",
        &op_id(777),
        json!({"collection": "c", "vector": [1.0, 0.0], "top_k": 5}),
        false,
    );
    assert_eq!(
        code(&raw(&mut h, &mallory, &body, None).1),
        "tenant_mismatch"
    );
    // B creating "c" gets its own collection; A's data is untouched and unseen.
    assert_eq!(
        h.call(
            &mallory,
            "collection_create",
            json!({"name": "c", "dim": 2, "metric": "l2"})
        )
        .0,
        200
    );
    let (_, r) = h.call(
        &mallory,
        "vector_query",
        json!({"collection": "c", "vector": [1.0, 0.0], "top_k": 5}),
    );
    assert!(r["result"]["matches"].as_array().unwrap().is_empty());
    let (_, r) = h.call(
        &owner_a,
        "vector_fetch",
        json!({"collection": "c", "ids": ["secret"]}),
    );
    assert_eq!(r["result"]["vectors"][0]["id"], "secret");
    assert_ne!(ledger_do_name(&ta), ledger_do_name(&tb));
}

#[test]
fn op_id_replay_semantics() {
    let (mut h, t) = setup();
    let ed = ctx(&t, "ed", write_caps());
    let id = op_id(4242);
    let args = json!({"collection": "c", "vectors": [v("a", 1.0), v("b", 2.0)]});
    let (s1, r1) = h.call_with(&ed, "vector_upsert", &id, args.clone(), false);
    assert_eq!(s1, 200, "{r1}");
    let seq = r1["result"]["write_seq"].clone();
    // Same body bytes ⇒ stored response, not re-applied (the binding hashes
    // the raw body; `json!` serializes keys sorted, so this is byte-equal).
    let reordered = json!({"vectors": [v("a", 1.0), v("b", 2.0)], "collection": "c"});
    let body = envelope(&t, "vector_upsert", &id, reordered, false);
    let rep = h
        .disp
        .dispatch(&mut h.cluster, &ed, &body, Some(&id), &h.clock, &h.entropy);
    assert!(rep.replayed);
    assert_eq!(serde_json::from_str::<Json>(&rep.body).unwrap(), r1);
    let (_, u) = h.call(&ed, "usage_get", json!({}));
    assert_eq!(
        u["result"]["usage"]["vectors"], 2,
        "replay did not double count"
    );
    // Different body under the same op_id ⇒ 409 op_replayed.
    let (s, r) = h.call_with(
        &ed,
        "vector_upsert",
        &id,
        json!({"collection": "c", "vectors": [v("z", 9.0)]}),
        false,
    );
    assert_eq!((s, code(&r)), (409, "op_replayed"));
    // Another subject may use the same op_id independently.
    let (s, _) = h.call_with(
        &ctx(&t, "owner", write_caps()),
        "collection_list",
        &id,
        json!({}),
        false,
    );
    assert_eq!(s, 200);
    // A replay is re-authorized: a removed member gets role_required.
    let lm = ledger_meta_for(&t).unwrap();
    {
        let (st, l) = h.cluster.ledger(&t).unwrap();
        l.remove_member(st, &lm, &sub("owner"), &sub("ed")).unwrap();
    }
    assert_eq!(
        code(
            &h.call_with(&ed, "vector_upsert", &id, args.clone(), false)
                .1
        ),
        "role_required"
    );
    // After 24 h the op_id is forgotten and the body executes again.
    {
        let (st, l) = h.cluster.ledger(&t).unwrap();
        l.put_member(st, &lm, &sub("owner"), &sub("ed"), Role::Editor, T0)
            .unwrap();
    }
    h.clock.advance(86_400);
    let (s, r) = h.call_with(&ed, "vector_upsert", &id, args, false);
    assert_eq!(s, 200);
    assert_ne!(r["result"]["write_seq"], seq);
}

#[test]
fn dimension_quota_and_dry_run() {
    let mut h = H::new();
    h.cluster = ruvector_edge_store::LocalCluster::new(QuotaLimits {
        max_vectors: 3,
        ..limits()
    });
    let t = tenant("org-q");
    h.claim(&t, "owner", &[]);
    let o = ctx(&t, "owner", write_caps());
    assert_eq!(
        h.call(
            &o,
            "collection_create",
            json!({"name": "c", "dim": 2, "metric": "cosine", "shards": 2})
        )
        .0,
        200
    );
    let (s, r) = h.call(
        &o,
        "vector_upsert",
        json!({"collection": "c", "vectors": [{"id": "a", "values": [1.0]}]}),
    );
    assert_eq!((s, code(&r)), (400, "dimension_mismatch"));
    let (s, r) = h.call(
        &o,
        "vector_query",
        json!({"collection": "c", "vector": [1.0, 2.0, 3.0], "top_k": 1}),
    );
    assert_eq!((s, code(&r)), (400, "dimension_mismatch"));
    // dry_run reports the effect and writes nothing.
    let id = op_id(55);
    let (s, r) = h.call_with(
        &o,
        "vector_upsert",
        &id,
        json!({"collection": "c", "vectors": [v("a", 1.0), v("b", 1.0)]}),
        true,
    );
    assert_eq!(s, 200, "{r}");
    assert_eq!(
        (
            r["result"]["dry_run"].clone(),
            r["result"]["delta"]["vectors"].clone()
        ),
        (json!(true), json!(2))
    );
    let (_, u) = h.call(&o, "usage_get", json!({}));
    assert_eq!(u["result"]["usage"]["vectors"], 0);
    let vecs: Vec<Json> = (0..4)
        .map(|i| v(&format!("r{i}"), 1.0 + i as f32))
        .collect();
    let (s, r) = h.call(
        &o,
        "vector_upsert",
        json!({"collection": "c", "vectors": vecs}),
    );
    assert_eq!((s, code(&r)), (413, "quota_exceeded"));
    let (_, f) = h.call(
        &o,
        "vector_fetch",
        json!({"collection": "c", "ids": ["r0", "r1", "r2", "r3"]}),
    );
    assert!(
        f["result"]["vectors"].as_array().unwrap().is_empty(),
        "a rejected batch writes nothing"
    );
    assert_eq!(
        h.call(
            &o,
            "vector_upsert",
            json!({"collection": "c", "vectors": [v("a", 1.0), v("b", 2.0), v("c", 3.0)]})
        )
        .0,
        200
    );
    // Delete releases quota; then the 4th vector fits.
    let (s, r) = h.call(
        &o,
        "vector_delete",
        json!({"collection": "c", "ids": ["a", "nope"]}),
    );
    assert_eq!((s, r["result"]["deleted"].clone()), (200, json!(1)));
    assert_eq!(
        h.call(
            &o,
            "vector_upsert",
            json!({"collection": "c", "vectors": [v("d", 4.0)]})
        )
        .0,
        200
    );
    let (_, u) = h.call(&o, "usage_get", json!({}));
    assert_eq!(u["result"]["usage"]["vectors"], 3);
    assert_eq!(u["result"]["usage"]["float_budget"], 6);
    let (_, q) = h.call(
        &o,
        "vector_query",
        json!({"collection": "c", "vector": [1.0, 0.0], "top_k": 10}),
    );
    let ids: Vec<&str> = q["result"]["matches"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids.len(), 3);
    assert!(!ids.contains(&"a"), "deleted vector never returned");
    let (_, l) = h.call(&o, "collection_list", json!({}));
    assert_eq!(l["result"]["collections"][0]["shards"], 2);
}

#[test]
fn state_survives_isolate_restart() {
    let (mut h, t) = setup();
    let ed = ctx(&t, "ed", write_caps());
    h.call(
        &ed,
        "vector_upsert",
        json!({"collection": "c", "vectors": [v("a", 1.0), v("b", 3.0)]}),
    );
    let before = h
        .call(
            &ed,
            "vector_query",
            json!({"collection": "c", "vector": [1.0, 0.0], "top_k": 2}),
        )
        .1;
    h.cluster.restart();
    let after = h
        .call(
            &ed,
            "vector_query",
            json!({"collection": "c", "vector": [1.0, 0.0], "top_k": 2}),
        )
        .1;
    assert_eq!(before["result"], after["result"]);
    let (_, u) = h.call(&ed, "usage_get", json!({}));
    assert_eq!(u["result"]["usage"]["vectors"], 2);
}
