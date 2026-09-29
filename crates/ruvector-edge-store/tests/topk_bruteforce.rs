//! M1 acceptance: exact top-10 over 1k × 384 equals an independent f64
//! brute force, for every metric, directly on a shard and through the
//! dispatcher with 3 shards and a metadata filter.

mod common;

use common::*;
use ruvector_edge_store::shard::Actor;
use ruvector_edge_store::{
    shard_meta_for, MemSqlStore, Metric, QueryRequest, ShardConfig, UpsertRow, VectorShard,
};
use ruvector_edge_tenancy::{CollectionUid, Role, ShardIndex};
use serde_json::{json, Value as Json};

const N: usize = 1000;
const DIM: usize = 384;
const K: usize = 10;

fn dataset(seed: u64) -> Vec<(String, Vec<f32>)> {
    let mut r = Rng(seed);
    (0..N).map(|i| (format!("v{i:04}"), r.vec(DIM))).collect()
}

fn brute(
    metric: Metric,
    data: &[(String, Vec<f32>)],
    q: &[f32],
    keep: impl Fn(usize) -> bool,
) -> Vec<(String, f64)> {
    let dot = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum::<f64>()
    };
    let mut all: Vec<(String, f64)> = data
        .iter()
        .enumerate()
        .filter(|(i, _)| keep(*i))
        .map(|(_, (id, v))| {
            let d = match metric {
                Metric::Cosine => 1.0 - dot(q, v) / (dot(q, q).sqrt() * dot(v, v).sqrt()),
                Metric::L2 => q
                    .iter()
                    .zip(v)
                    .map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2))
                    .sum::<f64>()
                    .sqrt(),
                Metric::Dot => -dot(q, v),
            };
            (id.clone(), d)
        })
        .collect();
    all.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    all.truncate(K);
    all
}

fn assert_same(got: &[(String, f32)], want: &[(String, f64)]) {
    let got_ids: Vec<&str> = got.iter().map(|g| g.0.as_str()).collect();
    let want_ids: Vec<&str> = want.iter().map(|w| w.0.as_str()).collect();
    assert_eq!(got_ids, want_ids);
    for (g, w) in got.iter().zip(want) {
        assert!(
            (f64::from(g.1) - w.1).abs() < 1e-4 * (1.0 + w.1.abs()),
            "{} {} vs {}",
            g.0,
            g.1,
            w.1
        );
    }
}

#[test]
fn shard_top10_equals_brute_force_all_metrics() {
    let data = dataset(0x5EED);
    let t = tenant("org-a");
    let actor = Actor {
        sub: "es1_x",
        jti: "j",
        family_id: "f",
        act_sub: None,
    };
    for metric in [Metric::Cosine, Metric::L2, Metric::Dot] {
        let store = MemSqlStore::new();
        let mut shard = VectorShard::open(&store).unwrap();
        let dm = shard_meta_for(&t, CollectionUid::from_bytes([7; 16]), ShardIndex::ZERO).unwrap();
        let cfg = ShardConfig {
            dim: DIM as u32,
            metric,
            filterable_keys: vec![],
            float_cap: 3_000_000,
        };
        for chunk in data.chunks(500) {
            let rows = chunk
                .iter()
                .map(|(id, v)| UpsertRow {
                    id: id.clone(),
                    values: v.clone(),
                    metadata: None,
                })
                .collect();
            let plan = shard.plan_upsert(&dm, &cfg, rows).unwrap();
            shard.apply_upsert(&store, plan, actor, T0).unwrap();
        }
        assert_eq!(shard.len(), N);
        let mut r = Rng(99);
        for _ in 0..5 {
            let q = r.vec(DIM);
            let req = QueryRequest {
                vector: q.clone(),
                top_k: K as u32,
                filter: None,
                include: vec![],
            };
            let out = shard.query(&dm, &cfg, &req).unwrap();
            assert_eq!(out.scanned, N as u64);
            let got: Vec<(String, f32)> = out
                .matches
                .iter()
                .map(|m| (m.id.clone(), m.distance))
                .collect();
            assert_same(&got, &brute(metric, &data, &q, |_| true));
        }
        // Cold load answers identically. 1000 ops exceed the op-log tail,
        // so the log was pruned and replay-from-scratch fails closed
        // (equivalence with replay is covered by `proptest_replay`).
        let reopened = VectorShard::open(&store).unwrap();
        assert_eq!(reopened.state_digest(), shard.state_digest());
        assert!(VectorShard::rebuild_from_ops(&store).is_err());
        let q = Rng(7).vec(DIM);
        let req = QueryRequest {
            vector: q,
            top_k: K as u32,
            filter: None,
            include: vec![],
        };
        assert_eq!(
            reopened.query(&dm, &cfg, &req).unwrap(),
            shard.query(&dm, &cfg, &req).unwrap()
        );
    }
}

fn matches_of(resp: &Json) -> Vec<(String, f32)> {
    resp["result"]["matches"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| {
            (
                m["id"].as_str().unwrap().to_string(),
                m["distance"].as_f64().unwrap() as f32,
            )
        })
        .collect()
}

#[test]
fn dispatcher_three_shards_with_filter_equals_brute_force() {
    let data = dataset(0xABCD);
    let mut h: Harness<MemSqlStore> = Harness::new();
    let t = tenant("org-a");
    h.claim(&t, "owner", &[("viewer", Role::Viewer)]);
    let owner = ctx(&t, "owner", write_caps());
    let (s, r) = h.call(&owner, "collection_create", json!({
        "name": "docs", "dim": DIM, "metric": "cosine", "shards": 3, "filterable_keys": ["bucket", "tag"]
    }));
    assert_eq!(s, 200, "{r}");
    // 100 × 384 floats per call keeps each body under the 1 MiB limit.
    for (b, chunk) in data.chunks(100).enumerate() {
        let vectors: Vec<Json> = chunk
            .iter()
            .enumerate()
            .map(|(j, (id, v))| {
                let i = b * 100 + j;
                json!({"id": id, "values": v, "metadata": {"bucket": i % 5, "tag": if i % 2 == 0 { "even" } else { "odd" }}})
            })
            .collect();
        let (s, r) = h.call(
            &owner,
            "vector_upsert",
            json!({"collection": "docs", "vectors": vectors}),
        );
        assert_eq!(s, 200, "{r}");
    }
    let viewer = ctx(&t, "viewer", read_caps());
    let mut rng = Rng(1234);
    for _ in 0..3 {
        let q = rng.vec(DIM);
        let (s, r) = h.call(
            &viewer,
            "vector_query",
            json!({"collection": "docs", "vector": q, "top_k": K}),
        );
        assert_eq!(s, 200, "{r}");
        assert_eq!(r["result"]["shards_queried"], 3);
        assert_eq!(r["usage"]["rows"], N as u64);
        assert_same(&matches_of(&r), &brute(Metric::Cosine, &data, &q, |_| true));
        // Filter: bucket == 2 AND tag in [even] ⇒ i ≡ 2 (mod 10).
        let (s, r) = h.call(
            &viewer,
            "vector_query",
            json!({
                "collection": "docs", "vector": q, "top_k": K, "include": ["metadata"],
                "filter": {"bucket": 2, "tag": {"$in": ["even"]}}
            }),
        );
        assert_eq!(s, 200, "{r}");
        assert_same(
            &matches_of(&r),
            &brute(Metric::Cosine, &data, &q, |i| i % 10 == 2),
        );
        for m in r["result"]["matches"].as_array().unwrap() {
            assert_eq!(m["metadata"]["bucket"], 2);
            assert_eq!(m["metadata"]["tag"], "even");
        }
    }
    // Undeclared filter key is rejected, not silently ignored.
    let (s, r) = h.call(
        &viewer,
        "vector_query",
        json!({
            "collection": "docs", "vector": vec![0.5f32; DIM], "top_k": 3, "filter": {"secret": 1}
        }),
    );
    assert_eq!((s, code(&r)), (400, "invalid_request"));
}
