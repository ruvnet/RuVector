//! Regression tests for resource bounds: real resident accounting and the
//! per-shard cap, registry tracking of every loaded shard, fetch / filter /
//! body limits, bounded and metered idempotency, ledger recovery, and
//! finite distances.

mod common;

use common::*;
use ruvector_edge_store::shard::{Actor, SHARD_RESIDENT_CAP_BYTES};
use ruvector_edge_store::{
    shard_meta_for, ErrorCode, LocalCluster, MemSqlStore, Metric, ResidentRegistry, ShardConfig,
    UpsertRow, VectorShard,
};
use ruvector_edge_tenancy::{ledger_do_name, CollectionUid, Role, ShardIndex, TenantKey};
use serde_json::{json, Value as Json};

type H = Harness<MemSqlStore>;

fn setup(shards: u32) -> (H, TenantKey) {
    let mut h = H::new();
    let t = tenant("org-l");
    h.claim(&t, "owner", &[("vi", Role::Viewer)]);
    let (s, r) = h.call(
        &ctx(&t, "owner", write_caps()),
        "collection_create",
        json!({"name": "c", "dim": 2, "metric": "l2", "shards": shards, "filterable_keys": ["k", "j", "m"]}),
    );
    assert_eq!(s, 200, "{r}");
    (h, t)
}

fn vecs(n: usize) -> Json {
    let v: Vec<Json> = (0..n)
        .map(
            |i| json!({"id": format!("v{i}"), "values": [i as f32, 1.0], "metadata": {"k": i % 3}}),
        )
        .collect();
    json!({"collection": "c", "vectors": v})
}

fn idem_rows(h: &H, t: &TenantKey) -> usize {
    h.cluster
        .store(ledger_do_name(t).as_str())
        .map_or(0, |s| s.row_count("idempotency"))
}

#[test]
fn resident_bytes_count_ids_and_metadata_and_cap_the_shard() {
    let st = MemSqlStore::new();
    let mut s = VectorShard::open(&st).unwrap();
    let dm = shard_meta_for(
        &tenant("org-m"),
        CollectionUid::from_bytes([2; 16]),
        ShardIndex::ZERO,
    )
    .unwrap();
    let cfg = ShardConfig {
        dim: 1,
        metric: Metric::L2,
        filterable_keys: vec!["blob".into()],
        float_cap: 3_000_000,
    };
    let actor = Actor {
        sub: "es1_m",
        jti: "j",
        family_id: "f",
    };
    let blob = "x".repeat(4000);
    let mut err = None;
    for b in 0..20 {
        let rows = (0..500)
            .map(|i| UpsertRow {
                id: format!("{:0>256}", b * 500 + i),
                values: vec![1.0],
                metadata: Some(json!({ "blob": blob })),
            })
            .collect();
        match s.plan_upsert(&dm, &cfg, rows) {
            Ok(p) => {
                s.apply_upsert(&st, p, actor, T0).unwrap();
            }
            Err(e) => {
                err = Some(e.code);
                break;
            }
        }
    }
    // dim = 1 admits 3M rows by floats alone; real memory stops it early.
    assert_eq!(err, Some(ErrorCode::BudgetExceeded));
    assert!(s.len() < 4_000, "{} rows admitted", s.len());
    assert!(s.resident_bytes() <= SHARD_RESIDENT_CAP_BYTES);
    assert!(s.resident_bytes() >= s.len() as u64 * (4000 + 2 * 256));
    assert_eq!(
        VectorShard::open(&st).unwrap().resident_bytes(),
        s.resident_bytes()
    );
    let ids: Vec<String> = (0..10).map(|i| format!("{:0>256}", i)).collect();
    let before = s.resident_bytes();
    s.delete(&st, &dm, &ids, actor, false, T0).unwrap();
    assert!(before - s.resident_bytes() >= 10 * (4000 + 2 * 256));
    assert_eq!(
        VectorShard::open(&st).unwrap().resident_bytes(),
        s.resident_bytes()
    );
}

#[test]
fn shards_loaded_by_failing_requests_are_registered_and_evictable() {
    let mut h = H::new();
    h.cluster = LocalCluster::with_registry(limits(), ResidentRegistry::new(1));
    let t = tenant("org-g");
    h.claim(&t, "owner", &[]);
    let o = ctx(&t, "owner", write_caps());
    for c in 0..5 {
        let name = format!("c{c}");
        let spec = json!({"name": name, "dim": 2, "metric": "l2", "shards": 2});
        assert_eq!(h.call(&o, "collection_create", spec).0, 200);
        let body = json!({"collection": name, "vectors": [{"id": "a", "values": [1.0, 2.0]}, {"id": "b", "values": [2.0, 1.0]}]});
        assert_eq!(h.call(&o, "vector_upsert", body).0, 200);
    }
    h.cluster.restart();
    // A bad `include` is rejected before any shard is loaded.
    let q = json!({"collection": "c0", "vector": [1.0, 0.0], "top_k": 1, "include": ["bogus"]});
    assert_eq!(h.call(&o, "vector_query", q).0, 400);
    assert_eq!(h.cluster.resident_shards(), 0);
    for c in 0..5 {
        // Loads shards, then is a dry run or fails in planning (wrong
        // dimension; "a" lives in a non-empty shard).
        let ok =
            json!({"collection": format!("c{c}"), "vectors": [{"id": "q", "values": [1.0, 1.0]}]});
        let id = op_id(500 + c);
        assert_eq!(h.call_with(&o, "vector_upsert", &id, ok, true).0, 200);
        let bad = json!({"collection": format!("c{c}"), "vectors": [{"id": "a", "values": [1.0]}]});
        assert_eq!(h.call(&o, "vector_upsert", bad).0, 400);
    }
    // Every loaded shard was registered, so the 1-byte cap kept one resident.
    assert_eq!(h.cluster.resident_shards(), 1);
    assert!(h.cluster.resident_total() > 0);
}

#[test]
fn fetch_enforces_the_total_id_limit_and_dedupes() {
    let (mut h, t) = setup(6);
    let o = ctx(&t, "owner", write_caps());
    assert_eq!(h.call(&o, "vector_upsert", vecs(200)).0, 200);
    let ids = |n: usize| -> Vec<String> { (0..n).map(|i| format!("v{i}")).collect() };
    for n in [101, 300, 600] {
        let (s, r) = h.call(
            &o,
            "vector_fetch",
            json!({"collection": "c", "ids": ids(n)}),
        );
        assert_eq!((s, code(&r)), (413, "payload_too_large"), "{n} ids");
    }
    let (s, _) = h.call(&o, "vector_fetch", json!({"collection": "c", "ids": []}));
    assert_eq!(s, 400);
    let dup = json!({"collection": "c", "ids": ["v1", "v1", "v2", "v1"], "include_values": true});
    let (s, r) = h.call(&o, "vector_fetch", dup);
    assert_eq!(s, 200);
    let got: Vec<&str> = r["result"]["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v["id"].as_str().unwrap())
        .collect();
    assert_eq!(got, ["v1", "v2"]);
}

#[test]
fn filter_values_are_bounded() {
    let (mut h, t) = setup(1);
    let o = ctx(&t, "owner", write_caps());
    assert_eq!(h.call(&o, "vector_upsert", vecs(10)).0, 200);
    let q = |f: Json| json!({"collection": "c", "vector": [0.0, 0.0], "top_k": 5, "filter": f});
    let long = "x".repeat(257);
    assert_eq!(h.call(&o, "vector_query", q(json!({ "k": long }))).0, 400);
    let in32: Vec<u32> = (0..32).collect();
    let three = json!({"k": {"$in": in32}, "j": {"$in": in32}, "m": 1});
    assert_eq!(h.call(&o, "vector_query", q(three)).0, 400);
    let (s, r) = h.call(&o, "vector_query", q(json!({"k": {"$in": [0, 2]}})));
    assert_eq!(s, 200);
    assert_eq!(r["result"]["matches"].as_array().unwrap().len(), 5);
    // Work units scale with rows × (1 + filter values).
    assert_eq!(r["usage"]["rows"], 10);
}

#[test]
fn oversized_body_is_rejected_before_parsing() {
    let (mut h, t) = setup(1);
    let o = ctx(&t, "owner", write_caps());
    let body = vec![b' '; ruvector_edge_store::ops::dispatch::MAX_OPS_BODY_BYTES + 1];
    let r = h
        .disp
        .dispatch(&mut h.cluster, &o, &body, None, &h.clock, &h.entropy);
    let r: Json = serde_json::from_str(&r.body).unwrap();
    assert_eq!(code(&r), "payload_too_large");
    let bad = envelope(&t, "collection_list", &op_id(1), json!([1]), false);
    let r = h
        .disp
        .dispatch(&mut h.cluster, &o, &bad, None, &h.clock, &h.entropy);
    assert_eq!(r.status, 400);
}

#[test]
fn only_successful_mutations_are_remembered() {
    let (mut h, t) = setup(1);
    let base = idem_rows(&h, &t);
    // Non-member tenant_me, reads and failures never grow the store.
    let stranger = ctx(&t, "stranger", read_caps());
    let vi = ctx(&t, "vi", read_caps());
    let o = ctx(&t, "owner", write_caps());
    for _ in 0..10 {
        assert_eq!(h.call(&stranger, "tenant_me", json!({})).0, 200);
        let q = json!({"collection": "c", "vector": [0.0, 0.0], "top_k": 1, "include": ["values"]});
        assert_eq!(h.call(&vi, "vector_query", q).0, 200);
        assert_eq!(
            h.call(
                &o,
                "vector_upsert",
                json!({"collection": "c", "vectors": [{"id": "a", "values": [1.0]}]})
            )
            .0,
            400
        );
    }
    assert_eq!(idem_rows(&h, &t), base);
    // A dry run does not claim the op_id: the real call under it executes.
    let id = op_id(77);
    assert_eq!(h.call_with(&o, "vector_upsert", &id, vecs(3), true).0, 200);
    let (s, r) = h.call_with(&o, "vector_upsert", &id, vecs(3), false);
    assert_eq!(s, 200, "{r}");
    assert_eq!(r["result"]["dry_run"], false);
    assert_eq!(idem_rows(&h, &t), base + 1);
    // The stored row is charged to bytes and released when purged.
    let usage = |h: &mut H| {
        h.call(&o, "usage_get", json!({})).1["result"]["usage"]["bytes"]
            .as_u64()
            .unwrap()
    };
    let with_row = usage(&mut h);
    h.clock.advance(86_401);
    let id2 = op_id(78);
    let del = json!({"collection": "c", "ids": ["v0"]});
    assert_eq!(h.call_with(&o, "vector_delete", &id2, del, false).0, 200);
    assert_eq!(idem_rows(&h, &t), 1, "expired rows purged");
    let vec_bytes = (2 + 8 + 7) as u64; // "v0" + 2 f32 + {"k":0}
    let now = usage(&mut h);
    assert!(
        now < with_row - vec_bytes,
        "purge released the old rows: {with_row} -> {now}"
    );
}

#[test]
fn quota_errors_are_not_replayed_after_capacity_frees() {
    let mut h = H::new();
    let mut lim = limits();
    lim.max_vectors = 2;
    h.cluster = LocalCluster::new(lim);
    let t = tenant("org-x");
    h.claim(&t, "owner", &[]);
    let o = ctx(&t, "owner", write_caps());
    h.call(
        &o,
        "collection_create",
        json!({"name": "c", "dim": 2, "metric": "l2"}),
    );
    assert_eq!(h.call(&o, "vector_upsert", vecs(2)).0, 200);
    let id = op_id(4040);
    let extra = json!({"collection": "c", "vectors": [{"id": "new", "values": [1.0, 1.0]}]});
    let (s, r) = h.call_with(&o, "vector_upsert", &id, extra.clone(), false);
    assert_eq!((s, code(&r)), (413, "quota_exceeded"));
    assert_eq!(
        h.call(
            &o,
            "vector_delete",
            json!({"collection": "c", "ids": ["v0"]})
        )
        .0,
        200
    );
    let (s, r) = h.call_with(&o, "vector_upsert", &id, extra, false);
    assert_eq!(s, 200, "{r}");
}

#[test]
fn poisoned_ledger_recovers_once_storage_does() {
    let (mut h, t) = setup(1);
    let o = ctx(&t, "owner", write_caps());
    let name = ledger_do_name(&t);
    h.cluster
        .store(name.as_str())
        .unwrap()
        .set_fail_writes(true);
    assert_eq!(h.call(&o, "usage_get", json!({})).0, 503);
    h.cluster
        .store(name.as_str())
        .unwrap()
        .set_fail_writes(false);
    let (s, r) = h.call(&o, "usage_get", json!({}));
    assert_eq!(s, 200, "{r}");
    assert_eq!(h.call(&o, "vector_upsert", vecs(2)).0, 200);
}

#[test]
fn extreme_magnitudes_give_finite_ordered_distances() {
    let (mut h, t) = setup(1);
    let o = ctx(&t, "owner", write_caps());
    let body = json!({"collection": "c", "vectors": [
        {"id": "far", "values": [-3.0e38, -3.0e38]},
        {"id": "near", "values": [3.0e38, 2.9e38]},
    ]});
    assert_eq!(h.call(&o, "vector_upsert", body).0, 200);
    let q = json!({"collection": "c", "vector": [3.0e38, 3.0e38], "top_k": 2});
    let (s, r) = h.call(&o, "vector_query", q);
    assert_eq!(s, 200);
    let m = r["result"]["matches"].as_array().unwrap();
    assert_eq!(m[0]["id"], "near");
    assert_eq!(m[1]["id"], "far");
    assert!(m.iter().all(|x| x["distance"].is_number()), "{r}");
    assert_eq!(m[1]["distance"].as_f64().unwrap() as f32, f32::MAX);
}

#[test]
fn query_steps_scale_with_filter_values() {
    use ruvector_edge_store::filter::Filter;
    let st = MemSqlStore::new();
    let mut s = VectorShard::open(&st).unwrap();
    let dm = shard_meta_for(
        &tenant("org-s"),
        CollectionUid::from_bytes([5; 16]),
        ShardIndex::ZERO,
    )
    .unwrap();
    let keys = vec!["g".to_string()];
    let cfg = ShardConfig {
        dim: 1,
        metric: Metric::L2,
        filterable_keys: keys.clone(),
        float_cap: 100,
    };
    let rows = (0..10)
        .map(|i| UpsertRow {
            id: format!("r{i}"),
            values: vec![i as f32],
            metadata: None,
        })
        .collect();
    let plan = s.plan_upsert(&dm, &cfg, rows).unwrap();
    s.apply_upsert(
        &st,
        plan,
        Actor {
            sub: "es1_s",
            jti: "j",
            family_id: "f",
        },
        T0,
    )
    .unwrap();
    assert_eq!(s.query_steps(&Filter::default()), 10);
    let f = Filter::parse(&json!({"g": {"$in": ["a", "b", "c"]}}), &keys).unwrap();
    assert_eq!(s.query_steps(&f), 40);
    // No filter: a full 6-shard collection of the smallest rows fits the budget.
    const _: () = assert!(ruvector_edge_store::shard::MAX_QUERY_STEPS >= 6 * 100_000);
}

#[test]
fn idempotency_rows_are_metered_capped_and_purged() {
    use ruvector_edge_store::ledger::{IdemKey, IdemLookup, MAX_IDEM_RESPONSE_BYTES};
    use ruvector_edge_store::{ledger_meta_for, TenantLedger};
    let st = MemSqlStore::new();
    let lm = ledger_meta_for(&tenant("org-i")).unwrap();
    let mut lim = limits();
    lim.max_bytes = 200;
    let mut l = TenantLedger::open(&st, lim).unwrap();
    let h = [7u8; 32];
    let (k1, k2, k3) = (op_id(1), op_id(2), op_id(3));
    fn key<'a>(k: &'a str, h: &'a [u8; 32]) -> IdemKey<'a> {
        IdemKey {
            sub: "es1_a",
            key: k,
            body_sha256: h,
        }
    }
    let bytes = |l: &TenantLedger| l.usage(&lm, T0).unwrap().bytes;
    assert!(l.idem_store(&st, &lm, key(&k1, &h), "{}", T0).unwrap());
    let row = key(&k1, &h).row_bytes("{}");
    assert_eq!(bytes(&l), row);
    // Over the bytes quota: not remembered, nothing charged.
    assert!(!l
        .idem_store(&st, &lm, key(&k2, &h), &"x".repeat(300), T0)
        .unwrap());
    assert_eq!(
        l.idem_lookup(&st, &lm, key(&k2, &h), T0).unwrap(),
        IdemLookup::Miss
    );
    assert_eq!(bytes(&l), row);
    // Past the TTL the next store purges the old row and releases it.
    let later = T0 + 86_401;
    assert!(l.idem_store(&st, &lm, key(&k3, &h), "{}", later).unwrap());
    assert_eq!(st.row_count("idempotency"), 1);
    assert_eq!(
        l.usage(&lm, later).unwrap().bytes,
        key(&k3, &h).row_bytes("{}")
    );
    // Oversized responses are never stored.
    let st2 = MemSqlStore::new();
    let mut big = TenantLedger::open(&st2, limits()).unwrap();
    let huge = "x".repeat(MAX_IDEM_RESPONSE_BYTES + 1);
    assert!(!big.idem_store(&st2, &lm, key(&k1, &h), &huge, T0).unwrap());
    assert_eq!(st2.row_count("idempotency"), 0);
}
