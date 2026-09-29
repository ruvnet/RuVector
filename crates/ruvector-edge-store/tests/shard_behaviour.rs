//! VectorShard: validation codes, float cap, delete-then-query, replay,
//! identity assertion and poisoning.

mod common;

use common::*;
use ruvector_edge_store::shard::{Actor, UsageDelta};
use ruvector_edge_store::{
    shard_meta_for, ErrorCode, MemSqlStore, Metric, QueryRequest, ShardConfig, SqlStore, UpsertRow,
    VectorShard,
};
use ruvector_edge_tenancy::{CollectionUid, DoMeta, ShardIndex};
use serde_json::json;

const ACTOR: Actor<'static> = Actor {
    sub: "es1_actor",
    jti: "jti",
    family_id: "fam",
};

fn setup(dim: u32, metric: Metric, cap: u64) -> (MemSqlStore, VectorShard, DoMeta, ShardConfig) {
    let store = MemSqlStore::new();
    let shard = VectorShard::open(&store).unwrap();
    let dm = shard_meta_for(
        &tenant("org-a"),
        CollectionUid::from_bytes([1; 16]),
        ShardIndex::ZERO,
    )
    .unwrap();
    let cfg = ShardConfig {
        dim,
        metric,
        filterable_keys: vec!["k".into()],
        float_cap: cap,
    };
    (store, shard, dm, cfg)
}

fn row(id: &str, v: Vec<f32>) -> UpsertRow {
    UpsertRow {
        id: id.into(),
        values: v,
        metadata: None,
    }
}

fn upsert(
    s: &mut VectorShard,
    st: &MemSqlStore,
    dm: &DoMeta,
    cfg: &ShardConfig,
    rows: Vec<UpsertRow>,
) -> Result<u64, ErrorCode> {
    let plan = s.plan_upsert(dm, cfg, rows).map_err(|e| e.code)?;
    s.apply_upsert(st, plan, ACTOR, T0)
        .map(|o| o.write_seq)
        .map_err(|e| e.code)
}

fn q(v: Vec<f32>, k: u32) -> QueryRequest {
    QueryRequest {
        vector: v,
        top_k: k,
        filter: None,
        include: vec![],
    }
}

#[test]
fn validation_error_codes() {
    let (st, mut s, dm, cfg) = setup(3, Metric::Cosine, 3_000_000);
    assert_eq!(
        upsert(&mut s, &st, &dm, &cfg, vec![row("a", vec![1.0, 2.0])]),
        Err(ErrorCode::DimensionMismatch)
    );
    assert_eq!(
        upsert(
            &mut s,
            &st,
            &dm,
            &cfg,
            vec![row("a", vec![1.0, f32::NAN, 0.0])]
        ),
        Err(ErrorCode::NonFiniteValue)
    );
    assert_eq!(
        upsert(&mut s, &st, &dm, &cfg, vec![row("a", vec![0.0; 3])]),
        Err(ErrorCode::InvalidRequest)
    );
    assert_eq!(
        upsert(&mut s, &st, &dm, &cfg, vec![row("", vec![1.0; 3])]),
        Err(ErrorCode::InvalidRequest)
    );
    assert_eq!(
        upsert(&mut s, &st, &dm, &cfg, vec![row("a\u{202E}", vec![1.0; 3])]),
        Err(ErrorCode::InvalidRequest)
    );
    assert_eq!(
        upsert(
            &mut s,
            &st,
            &dm,
            &cfg,
            vec![row("a", vec![1.0; 3]), row("a", vec![2.0; 3])]
        ),
        Err(ErrorCode::InvalidRequest)
    );
    let big: Vec<UpsertRow> = (0..501)
        .map(|i| row(&format!("r{i}"), vec![1.0; 3]))
        .collect();
    assert_eq!(
        upsert(&mut s, &st, &dm, &cfg, big),
        Err(ErrorCode::PayloadTooLarge)
    );
    let meta = UpsertRow {
        id: "m".into(),
        values: vec![1.0; 3],
        metadata: Some(json!({"x": "y".repeat(5000)})),
    };
    assert_eq!(
        upsert(&mut s, &st, &dm, &cfg, vec![meta]),
        Err(ErrorCode::PayloadTooLarge)
    );
    // Nothing was written by any rejected call.
    assert_eq!(st.write_count(), 0);
    upsert(&mut s, &st, &dm, &cfg, vec![row("a", vec![1.0, 0.0, 0.0])]).unwrap();
    let e = s.query(&dm, &cfg, &q(vec![1.0; 4], 1)).unwrap_err();
    assert_eq!(e.code, ErrorCode::DimensionMismatch);
    assert_eq!(
        s.query(&dm, &cfg, &q(vec![1.0; 3], 0)).unwrap_err().code,
        ErrorCode::InvalidRequest
    );
    assert_eq!(
        s.query(&dm, &cfg, &q(vec![1.0; 3], 101)).unwrap_err().code,
        ErrorCode::InvalidRequest
    );
}

#[test]
fn float_cap_is_budget_exceeded_and_replacements_are_free() {
    let (st, mut s, dm, cfg) = setup(4, Metric::L2, 12); // 3 rows of dim 4
    upsert(
        &mut s,
        &st,
        &dm,
        &cfg,
        (0..3)
            .map(|i| row(&format!("r{i}"), vec![i as f32; 4]))
            .collect(),
    )
    .unwrap();
    assert_eq!(
        upsert(&mut s, &st, &dm, &cfg, vec![row("r9", vec![1.0; 4])]),
        Err(ErrorCode::BudgetExceeded)
    );
    // Replacing an existing id does not grow the resident set.
    let plan = s
        .plan_upsert(&dm, &cfg, vec![row("r1", vec![5.0; 4])])
        .unwrap();
    assert_eq!(
        (
            plan.inserted,
            plan.replaced,
            plan.delta.vectors,
            plan.delta.floats
        ),
        (0, 1, 0, 0)
    );
    s.apply_upsert(&st, plan, ACTOR, T0).unwrap();
    assert_eq!(s.resident_floats(), 12);
}

#[test]
fn delete_then_query_and_fetch() {
    let (st, mut s, dm, cfg) = setup(2, Metric::L2, 1000);
    let rows = vec![
        row("a", vec![0.0, 0.0]),
        row("b", vec![1.0, 0.0]),
        row("c", vec![5.0, 5.0]),
    ];
    upsert(&mut s, &st, &dm, &cfg, rows).unwrap();
    let ids = |r: &QueryRequest, s: &VectorShard| -> Vec<String> {
        s.query(&dm, &cfg, r)
            .unwrap()
            .matches
            .into_iter()
            .map(|m| m.id)
            .collect()
    };
    let near = q(vec![0.1, 0.0], 3);
    assert_eq!(ids(&near, &s), ["a", "b", "c"]);
    // dry_run deletes nothing.
    let o = s.delete(&st, &dm, &["a".into()], ACTOR, true, T0).unwrap();
    assert_eq!((o.deleted, o.dry_run), (1, true));
    assert_eq!(ids(&near, &s), ["a", "b", "c"]);
    let o = s
        .delete(
            &st,
            &dm,
            &["a".into(), "zz".into(), "a".into()],
            ACTOR,
            false,
            T0,
        )
        .unwrap();
    assert_eq!(o.deleted, 1);
    assert_eq!(
        o.delta,
        UsageDelta {
            vectors: -1,
            floats: -2,
            bytes: -(1 + 8)
        }
    );
    assert_eq!(ids(&near, &s), ["b", "c"]);
    assert!(s.fetch(&dm, &["a".into()], false).unwrap().is_empty());
    assert_eq!(
        s.fetch(&dm, &["c".into(), "b".into()], true).unwrap()[0].values,
        Some(vec![5.0, 5.0])
    );
    // Re-upsert after delete gets a fresh iid and is queryable again.
    upsert(&mut s, &st, &dm, &cfg, vec![row("a", vec![0.0, 0.0])]).unwrap();
    assert_eq!(ids(&near, &s), ["a", "b", "c"]);
    for fresh in [
        VectorShard::open(&st).unwrap(),
        VectorShard::rebuild_from_ops(&st).unwrap(),
    ] {
        assert_eq!(fresh.state_digest(), s.state_digest());
        assert_eq!(ids(&near, &fresh), ["a", "b", "c"]);
    }
}

#[test]
fn replay_is_idempotent_and_fails_closed_on_gap() {
    let (st, mut s, dm, cfg) = setup(2, Metric::Dot, 1000);
    for i in 0..5 {
        upsert(
            &mut s,
            &st,
            &dm,
            &cfg,
            vec![row(&format!("r{i}"), vec![i as f32, 1.0])],
        )
        .unwrap();
    }
    s.delete(&st, &dm, &["r2".into()], ACTOR, false, T0)
        .unwrap();
    let mut r = VectorShard::rebuild_from_ops(&st).unwrap();
    assert_eq!(r.write_seq(), 6);
    assert_eq!(
        r.catch_up(&st).unwrap(),
        0,
        "already-applied ops are skipped"
    );
    assert_eq!(r.state_digest(), s.state_digest());
    // Remove op 3 from the log: replay must refuse rather than skip it.
    let copy = st.clone();
    copy.exec("DELETE FROM ops WHERE seq = ?", &[3i64.into()])
        .unwrap();
    assert!(VectorShard::rebuild_from_ops(&copy).is_err());
}

#[test]
fn foreign_identity_is_not_found_and_uninitialised_reads_are_empty() {
    let (st, mut s, dm, cfg) = setup(2, Metric::Cosine, 1000);
    let empty = s.query(&dm, &cfg, &q(vec![1.0, 0.0], 5)).unwrap();
    assert!(empty.matches.is_empty());
    assert_eq!(st.row_count("meta"), 0, "a read never initialises a DO");
    upsert(&mut s, &st, &dm, &cfg, vec![row("a", vec![1.0, 0.0])]).unwrap();
    let other_tenant = shard_meta_for(
        &tenant("org-b"),
        CollectionUid::from_bytes([1; 16]),
        ShardIndex::ZERO,
    )
    .unwrap();
    let other_uid = shard_meta_for(
        &tenant("org-a"),
        CollectionUid::from_bytes([2; 16]),
        ShardIndex::ZERO,
    )
    .unwrap();
    for wrong in [&other_tenant, &other_uid] {
        assert_eq!(
            s.query(wrong, &cfg, &q(vec![1.0, 0.0], 5))
                .unwrap_err()
                .code,
            ErrorCode::NotFound
        );
        assert_eq!(
            s.fetch(wrong, &["a".into()], true).unwrap_err().code,
            ErrorCode::NotFound
        );
        assert_eq!(
            s.plan_upsert(wrong, &cfg, vec![row("x", vec![1.0, 0.0])])
                .unwrap_err()
                .code,
            ErrorCode::NotFound
        );
        assert_eq!(
            s.delete(&st, wrong, &["a".into()], ACTOR, false, T0)
                .unwrap_err()
                .code,
            ErrorCode::NotFound
        );
        assert_eq!(s.wipe(&st, wrong).unwrap_err().code, ErrorCode::NotFound);
    }
    assert_eq!(s.len(), 1);
}

#[test]
fn storage_failure_poisons_until_reopen() {
    let (st, mut s, dm, cfg) = setup(2, Metric::L2, 1000);
    upsert(&mut s, &st, &dm, &cfg, vec![row("a", vec![1.0, 0.0])]).unwrap();
    st.set_fail_writes(true);
    assert_eq!(
        upsert(&mut s, &st, &dm, &cfg, vec![row("b", vec![1.0, 1.0])]),
        Err(ErrorCode::ShardUnavailable)
    );
    assert!(s.is_poisoned());
    assert_eq!(
        s.query(&dm, &cfg, &q(vec![1.0, 0.0], 1)).unwrap_err().code,
        ErrorCode::ShardUnavailable
    );
    st.set_fail_writes(false);
    let s2 = VectorShard::open(&st).unwrap();
    assert_eq!(s2.len(), 1);
    assert_eq!(
        s2.state_digest(),
        VectorShard::rebuild_from_ops(&st).unwrap().state_digest()
    );
}

#[test]
fn wipe_releases_everything_and_resets() {
    let (st, mut s, dm, cfg) = setup(2, Metric::L2, 1000);
    upsert(
        &mut s,
        &st,
        &dm,
        &cfg,
        vec![row("a", vec![1.0, 0.0]), row("bb", vec![0.0, 1.0])],
    )
    .unwrap();
    let d = s.wipe(&st, &dm).unwrap();
    assert_eq!(
        d,
        UsageDelta {
            vectors: -2,
            floats: -4,
            bytes: -(1 + 8 + 2 + 8)
        }
    );
    for t in ["meta", "vectors", "ops", "filter_idx"] {
        assert_eq!(st.row_count(t), 0, "{t}");
    }
    assert!(s.identity().is_none() && s.is_empty());
}
