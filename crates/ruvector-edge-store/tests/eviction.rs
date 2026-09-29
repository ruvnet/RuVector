//! Isolate resident cap: LRU eviction through ops, transparent reload.

mod common;

use common::*;
use ruvector_edge_store::MemSqlStore;
use serde_json::{json, Value as Json};

type H = Harness<MemSqlStore>;

const CAP: u64 = 3_500;

fn v(id: &str, x: f32) -> Json {
    json!({"id": id, "values": [x, 0.0]})
}

#[test]
fn isolate_cap_evicts_lru_shards_and_reloads_transparently() {
    use ruvector_edge_store::{shard_meta_for, LocalCluster, ResidentRegistry};
    use ruvector_edge_tenancy::{CollectionUid, ShardCount, ShardIndex};
    let mut h = H::new();
    // Room for roughly two of the three shards' resident bytes (~10 rows
    // of ~160 B each per shard, per-row overhead included).
    h.cluster = LocalCluster::with_registry(limits(), ResidentRegistry::new(CAP));
    let t = tenant("org-e");
    h.claim(&t, "owner", &[]);
    let o = ctx(&t, "owner", write_caps());
    let (_, c) = h.call(
        &o,
        "collection_create",
        json!({"name": "c", "dim": 2, "metric": "l2", "shards": 3}),
    );
    let uid = CollectionUid::parse(c["result"]["collection_uid"].as_str().unwrap()).unwrap();
    let vecs: Vec<Json> = (0..30).map(|i| v(&format!("id{i}"), i as f32)).collect();
    assert_eq!(
        h.call(
            &o,
            "vector_upsert",
            json!({"collection": "c", "vectors": vecs})
        )
        .0,
        200
    );
    let names: Vec<_> = (0..3)
        .map(|i| {
            shard_meta_for(
                &t,
                uid,
                ShardIndex::new(i, ShardCount::new(3).unwrap()).unwrap(),
            )
            .unwrap()
            .do_name()
        })
        .collect();
    let resident = names.iter().filter(|n| h.cluster.is_resident(n)).count();
    assert!(resident < 3, "the cap forced an eviction");
    assert!(h.cluster.resident_total() <= CAP);
    // Evicted shards cold-load on the next query; nothing is lost.
    let (s, r) = h.call(
        &o,
        "vector_query",
        json!({"collection": "c", "vector": [0.0, 0.0], "top_k": 30}),
    );
    assert_eq!(s, 200);
    let ids: Vec<&str> = r["result"]["matches"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids.len(), 30);
    assert_eq!(&ids[..3], ["id0", "id1", "id2"]);
}
