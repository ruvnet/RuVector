//! Property tests: any upsert/delete sequence leaves live state equal to
//! both reconstructions, and top-k equals a brute force over the model.

mod common;

use common::*;
use proptest::prelude::*;
use ruvector_edge_store::shard::Actor;
use ruvector_edge_store::{
    shard_meta_for, MemSqlStore, Metric, QueryRequest, ShardConfig, UpsertRow, VectorShard,
};
use ruvector_edge_tenancy::{CollectionUid, ShardIndex};
use serde_json::json;
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
enum Step {
    Upsert(Vec<(u8, [i8; 3], Option<u8>)>),
    Delete(Vec<u8>),
}

fn step() -> impl Strategy<Value = Step> {
    prop_oneof![
        prop::collection::vec((0u8..24, any::<[i8; 3]>(), prop::option::of(0u8..3)), 1..6)
            .prop_map(Step::Upsert),
        prop::collection::vec(0u8..24, 1..6).prop_map(Step::Delete),
    ]
}

fn to_vals(v: [i8; 3]) -> Vec<f32> {
    // Offset keeps every vector non-zero (dot/l2 allow zero, but keep it simple).
    v.iter().map(|x| f32::from(*x) / 16.0 + 0.03125).collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn live_equals_cold_load_equals_replay(steps in prop::collection::vec(step(), 1..25), metric in 0usize..3) {
        let metric = [Metric::Cosine, Metric::L2, Metric::Dot][metric];
        let st = MemSqlStore::new();
        let mut s = VectorShard::open(&st).unwrap();
        let dm = shard_meta_for(&tenant("org-p"), CollectionUid::from_bytes([4; 16]), ShardIndex::ZERO).unwrap();
        let cfg = ShardConfig { dim: 3, metric, filterable_keys: vec!["g".into()], float_cap: 1_000 };
        let actor = Actor { sub: "es1_p", jti: "j", family_id: "f" };
        let mut model: BTreeMap<String, (Vec<f32>, Option<u8>)> = BTreeMap::new();
        for step in steps {
            match step {
                Step::Upsert(rows) => {
                    let mut dedup = BTreeMap::new();
                    for (id, v, g) in rows {
                        dedup.insert(format!("k{id}"), (to_vals(v), g));
                    }
                    let batch = dedup.iter().map(|(id, (v, g))| UpsertRow {
                        id: id.clone(), values: v.clone(), metadata: g.map(|g| json!({"g": g})),
                    }).collect();
                    let plan = s.plan_upsert(&dm, &cfg, batch).unwrap();
                    s.apply_upsert(&st, plan, actor, T0).unwrap();
                    model.extend(dedup);
                }
                Step::Delete(ids) => {
                    let ids: Vec<String> = ids.iter().map(|i| format!("k{i}")).collect();
                    let o = s.delete(&st, &dm, &ids, actor, false, T0).unwrap();
                    let before = model.len();
                    for id in &ids { model.remove(id); }
                    prop_assert_eq!(o.deleted as usize, before - model.len());
                }
            }
        }
        prop_assert_eq!(s.len(), model.len());
        let live = s.state_digest();
        prop_assert_eq!(VectorShard::open(&st).unwrap().state_digest(), live);
        prop_assert_eq!(VectorShard::rebuild_from_ops(&st).unwrap().state_digest(), live);
        // Filtered top-k equals a brute force over the model.
        if !model.is_empty() {
            let q = vec![0.5f32, -0.25, 1.0];
            let req = QueryRequest { vector: q.clone(), top_k: 5, filter: Some(json!({"g": {"$in": [0, 2]}})), include: vec![] };
            let got: Vec<String> = s.query(&dm, &cfg, &req).unwrap().matches.into_iter().map(|m| m.id).collect();
            let dist = |v: &[f32]| ruvector_edge_store::distance::distance(metric, &q, ruvector_edge_store::distance::norm(&q), v, ruvector_edge_store::distance::norm(v));
            let mut want: Vec<(f64, String)> = model.iter()
                .filter(|(_, (_, g))| matches!(g, Some(0) | Some(2)))
                .map(|(id, (v, _))| (dist(v), id.clone()))
                .collect();
            want.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            let want: Vec<String> = want.into_iter().take(5).map(|w| w.1).collect();
            prop_assert_eq!(got, want);
        }
    }
}
