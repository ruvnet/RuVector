//! Executors end to end over the real DO cores (in-memory SQLite mock),
//! every call JSON-encoded across the DO boundary as in production.

use crate::backend::mem::MemBackend;
use crate::service::{self, Call};
use crate::testkit::*;
use ruvector_edge_store::{ErrorCode, Op, ResidentRegistry};
use ruvector_edge_tenancy::Role;
use serde_json::{json, Value as Json};

fn exec(
    b: &MemBackend,
    c: &ruvector_edge_store::CallerContext,
    op: Op,
    args: Json,
) -> Result<Json, ErrorCode> {
    block_on(service::execute(&call(b, c), op, &args.to_string()))
        .map(|(r, _)| r)
        .map_err(|e| e.code)
}

fn claim(b: &MemBackend, c: &ruvector_edge_store::CallerContext) -> Result<Json, ErrorCode> {
    block_on(service::claim(&call(b, c)))
        .map(|(r, _)| r)
        .map_err(|e| e.code)
}

fn create(
    b: &MemBackend,
    c: &ruvector_edge_store::CallerContext,
    name: &str,
    dim: u32,
    shards: u32,
) {
    let spec = json!({ "name": name, "dim": dim, "metric": "l2", "shards": shards });
    exec(b, c, Op::CollectionCreate, spec).unwrap();
}

fn upsert(
    b: &MemBackend,
    c: &ruvector_edge_store::CallerContext,
    coll: &str,
    rows: &[(String, Vec<f32>)],
) -> Result<Json, ErrorCode> {
    let vectors: Vec<Json> = rows
        .iter()
        .map(|(id, v)| json!({ "id": id, "values": v }))
        .collect();
    exec(
        b,
        c,
        Op::VectorUpsert,
        json!({ "collection": coll, "vectors": vectors }),
    )
}

#[test]
fn claim_then_owner_lifecycle_matches_brute_force_across_shards() {
    let b = MemBackend::new();
    let t = tenant("org-a");
    let owner = ctx(&t, "alice", rw());
    assert_eq!(claim(&b, &owner).unwrap(), json!({ "role": "owner" }));
    assert_eq!(claim(&b, &owner), Err(ErrorCode::Conflict));
    create(&b, &owner, "docs", 8, 3);
    let mut rng = Rng(42);
    let rows: Vec<(String, Vec<f32>)> = (0..200).map(|i| (format!("v{i}"), rng.vec(8))).collect();
    for chunk in rows.chunks(50) {
        assert_eq!(
            upsert(&b, &owner, "docs", chunk).unwrap()["upserted"],
            json!(chunk.len())
        );
    }
    let q = rng.vec(8);
    let got = exec(
        &b,
        &owner,
        Op::VectorQuery,
        json!({ "collection": "docs", "vector": q, "top_k": 10 }),
    )
    .unwrap();
    assert_eq!(got["shards_queried"], json!(3));
    let ids: Vec<&str> = got["matches"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    let mut brute: Vec<(f64, &str)> = rows
        .iter()
        .map(|(id, v)| {
            (
                v.iter()
                    .zip(&q)
                    .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
                    .sum(),
                id.as_str(),
            )
        })
        .collect();
    brute.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(b.1)));
    let expect: Vec<&str> = brute.iter().take(10).map(|(_, id)| *id).collect();
    assert_eq!(ids, expect);

    let f = exec(
        &b,
        &owner,
        Op::VectorFetch,
        json!({ "collection": "docs", "ids": ["v3", "nope", "v3"], "include_values": true }),
    )
    .unwrap();
    assert_eq!(f["vectors"].as_array().unwrap().len(), 1);
    assert_eq!(f["vectors"][0]["values"], json!(rows[3].1));

    let d = exec(
        &b,
        &owner,
        Op::VectorDelete,
        json!({ "collection": "docs", "ids": ["v1", "v2", "nope"] }),
    )
    .unwrap();
    assert_eq!(d["deleted"], json!(2));
    let u = exec(&b, &owner, Op::UsageGet, json!({})).unwrap();
    assert_eq!(u["usage"]["vectors"], json!(198));
    assert_eq!(u["usage"]["collections"], json!(1));
    assert_eq!(u["usage"]["float_budget"], json!(198 * 8));

    let g = block_on(service::collection_get(
        &call(&b, &owner),
        r#"{"collection":"docs"}"#,
    ))
    .unwrap()
    .0;
    assert_eq!(
        (g["count"].clone(), g["shards"].clone()),
        (json!(198), json!(3))
    );
    let l = exec(&b, &owner, Op::CollectionList, json!({})).unwrap();
    assert_eq!(l["collections"][0]["name"], json!("docs"));
}

#[test]
fn other_tenants_get_404_on_foreign_names() {
    let b = MemBackend::new();
    let (ta, tb) = (tenant("org-a"), tenant("org-b"));
    let a = ctx(&ta, "alice", rw());
    let bob = ctx(&tb, "bob", rw());
    claim(&b, &a).unwrap();
    claim(&b, &bob).unwrap();
    create(&b, &a, "secret", 4, 1);
    upsert(&b, &a, "secret", &[("x".into(), vec![1.0, 0.0, 0.0, 0.0])]).unwrap();
    let q = json!({ "collection": "secret", "vector": [1.0, 0.0, 0.0, 0.0], "top_k": 1 });
    assert_eq!(exec(&b, &bob, Op::VectorQuery, q), Err(ErrorCode::NotFound));
    let f = json!({ "collection": "secret", "ids": ["x"] });
    assert_eq!(
        exec(&b, &bob, Op::VectorFetch, f.clone()),
        Err(ErrorCode::NotFound)
    );
    assert_eq!(
        exec(&b, &bob, Op::VectorDelete, f),
        Err(ErrorCode::NotFound)
    );
    let r = block_on(service::collection_get(
        &call(&b, &bob),
        r#"{"collection":"secret"}"#,
    ));
    assert_eq!(r.unwrap_err().code, ErrorCode::NotFound);
    assert_eq!(
        exec(&b, &bob, Op::CollectionList, json!({})).unwrap()["collections"],
        json!([])
    );
    // A same-named collection in tenant B is a different DO: A's data stays put.
    create(&b, &bob, "secret", 4, 1);
    let got = exec(
        &b,
        &bob,
        Op::VectorFetch,
        json!({ "collection": "secret", "ids": ["x"] }),
    )
    .unwrap();
    assert_eq!(got["vectors"], json!([]));
    // A token for tenant B never reaches A's ledger, even as a member of B.
    assert_eq!(
        exec(&b, &ctx(&ta, "bob", rw()), Op::CollectionList, json!({})),
        Err(ErrorCode::RoleRequired)
    );
}

#[test]
fn viewer_cannot_write_and_scope_is_checked_before_role() {
    let b = MemBackend::new();
    let t = tenant("org-a");
    let owner = ctx(&t, "alice", rw());
    claim(&b, &owner).unwrap();
    create(&b, &owner, "docs", 2, 1);
    b.invite(&t, &sub("alice"), &sub("val"), Role::Viewer, T0)
        .unwrap();
    let viewer = ctx(&t, "val", rw());
    let row = [("a".to_string(), vec![1.0, 2.0])];
    assert_eq!(
        upsert(&b, &viewer, "docs", &row),
        Err(ErrorCode::RoleRequired)
    );
    let del = json!({ "collection": "docs", "ids": ["a"] });
    assert_eq!(
        exec(&b, &viewer, Op::VectorDelete, del),
        Err(ErrorCode::RoleRequired)
    );
    let spec = json!({ "name": "more", "dim": 2, "metric": "dot" });
    assert_eq!(
        exec(&b, &viewer, Op::CollectionCreate, spec),
        Err(ErrorCode::RoleRequired)
    );
    assert!(exec(&b, &viewer, Op::CollectionList, json!({})).is_ok());
    // The owner without `ruvector:write` is stepped up, never role-checked.
    let owner_ro = ctx(&t, "alice", ro());
    let e = block_on(service::execute(
        &call(&b, &owner_ro),
        Op::VectorUpsert,
        r#"{"collection":"docs","vectors":[]}"#,
    ))
    .unwrap_err();
    assert_eq!(
        (e.code, e.scope),
        (ErrorCode::InsufficientScope, Some("ruvector:write"))
    );
    // A read-only claim attempt is a step-up too.
    let other = ctx(&tenant("org-z"), "zed", ro());
    assert_eq!(claim(&b, &other), Err(ErrorCode::InsufficientScope));
    // Non-member of a claimed tenant / anyone in an unclaimed one.
    assert_eq!(
        exec(&b, &ctx(&t, "mallory", rw()), Op::CollectionList, json!({})),
        Err(ErrorCode::RoleRequired)
    );
    assert_eq!(
        exec(
            &b,
            &ctx(&tenant("org-z"), "zed", rw()),
            Op::CollectionList,
            json!({})
        ),
        Err(ErrorCode::NotClaimed)
    );
    // tenant_me needs no capability and reports the role.
    let me = exec(&b, &viewer, Op::TenantMe, json!({})).unwrap();
    assert_eq!(
        (me["role"].clone(), me["claimed"].clone()),
        (json!("viewer"), json!(true))
    );
}

#[test]
fn dry_runs_write_nothing_and_charge_nothing() {
    let b = MemBackend::new();
    let t = tenant("org-a");
    let owner = ctx(&t, "alice", rw());
    claim(&b, &owner).unwrap();
    let dry = |op, args: Json| {
        let c = Call {
            b: &b,
            ctx: &owner,
            dry_run: true,
            now: T0,
        };
        block_on(service::execute(&c, op, &args.to_string()))
            .unwrap()
            .0
    };
    let r = dry(
        Op::CollectionCreate,
        json!({ "name": "d", "dim": 2, "metric": "cosine" }),
    );
    assert_eq!(r["dry_run"], json!(true));
    assert_eq!(
        exec(&b, &owner, Op::CollectionList, json!({})).unwrap()["collections"],
        json!([])
    );
    create(&b, &owner, "d", 2, 1);
    let r = dry(
        Op::VectorUpsert,
        json!({ "collection": "d", "vectors": [{ "id": "a", "values": [1.0, 0.0] }] }),
    );
    assert_eq!(
        (r["upserted"].clone(), r["delta"]["vectors"].clone()),
        (json!(1), json!(1))
    );
    let u = exec(&b, &owner, Op::UsageGet, json!({})).unwrap();
    assert_eq!(u["usage"]["vectors"], json!(0));
}

#[test]
fn shard_outage_is_503_and_leaves_usage_uncharged() {
    let b = MemBackend::new();
    let t = tenant("org-a");
    let owner = ctx(&t, "alice", rw());
    claim(&b, &owner).unwrap();
    create(&b, &owner, "d", 2, 1);
    b.shard_down.set(true);
    let row = [("a".to_string(), vec![1.0, 0.0])];
    assert_eq!(
        upsert(&b, &owner, "d", &row),
        Err(ErrorCode::ShardUnavailable)
    );
    b.shard_down.set(false);
    let u = exec(&b, &owner, Op::UsageGet, json!({})).unwrap();
    assert_eq!(
        (u["usage"]["vectors"].clone(), u["usage"]["bytes"].clone()),
        (json!(0), json!(0))
    );
    upsert(&b, &owner, "d", &row).unwrap();
    assert_eq!(
        exec(&b, &owner, Op::UsageGet, json!({})).unwrap()["usage"]["vectors"],
        json!(1)
    );
}

#[test]
fn cold_shards_are_evicted_and_reload_from_storage() {
    // A cap of one byte: every touch evicts every other shard.
    let b = MemBackend::with(crate::ledger_core::FREE_PLAN, ResidentRegistry::new(1));
    let t = tenant("org-a");
    let owner = ctx(&t, "alice", rw());
    claim(&b, &owner).unwrap();
    create(&b, &owner, "one", 2, 1);
    create(&b, &owner, "two", 2, 1);
    upsert(&b, &owner, "one", &[("a".into(), vec![1.0, 0.0])]).unwrap();
    upsert(&b, &owner, "two", &[("b".into(), vec![0.0, 1.0])]).unwrap();
    assert_eq!(b.host.borrow().resident_count(), 1);
    let q = |c: &str| {
        exec(
            &b,
            &owner,
            Op::VectorQuery,
            json!({ "collection": c, "vector": [1.0, 0.0], "top_k": 1 }),
        )
        .unwrap()
    };
    assert_eq!(q("one")["matches"][0]["id"], json!("a"));
    assert_eq!(q("two")["matches"][0]["id"], json!("b"));
    b.restart();
    assert_eq!(q("one")["matches"][0]["id"], json!("a"));
}
