//! Torn writes (a DO `sql.exec` that throws mid-batch does not roll back
//! the earlier statements): after any failure point, reopening must make
//! the durable tables agree with the op log, and the ledger must match
//! what the shards really hold.

mod common;

use common::*;
use ruvector_edge_store::shard::{Actor, UsageDelta};
use ruvector_edge_store::{
    shard_meta_for, ErrorCode, MemSqlStore, Metric, ShardConfig, UpsertRow, VectorShard,
};
use ruvector_edge_tenancy::{CollectionUid, DoMeta, Role, ShardCount, ShardIndex};
use serde_json::{json, Value as Json};

const ACTOR: Actor<'static> = Actor {
    sub: "es1_t",
    jti: "j",
    family_id: "f",
    act_sub: None,
};

fn setup() -> (MemSqlStore, VectorShard, DoMeta, ShardConfig) {
    let st = MemSqlStore::new();
    let s = VectorShard::open(&st).unwrap();
    let dm = shard_meta_for(
        &tenant("org-t"),
        CollectionUid::from_bytes([9; 16]),
        ShardIndex::ZERO,
    )
    .unwrap();
    let cfg = ShardConfig {
        dim: 2,
        metric: Metric::L2,
        filterable_keys: vec!["g".into()],
        float_cap: 1_000,
    };
    (st, s, dm, cfg)
}

fn rows(ids: &[&str], g: u32) -> Vec<UpsertRow> {
    ids.iter()
        .enumerate()
        .map(|(i, id)| UpsertRow {
            id: (*id).into(),
            values: vec![g as f32, i as f32],
            metadata: Some(json!({ "g": g })),
        })
        .collect()
}

fn upsert(
    s: &mut VectorShard,
    st: &MemSqlStore,
    dm: &DoMeta,
    cfg: &ShardConfig,
    r: Vec<UpsertRow>,
) -> Result<(), ErrorCode> {
    let plan = s.plan_upsert(dm, cfg, r).map_err(|e| e.code)?;
    s.apply_upsert(st, plan, ACTOR, T0)
        .map(|_| ())
        .map_err(|e| e.code)
}

/// Reopen, and check the durable tables agree with the log and with a
/// second cold load; then one more write keeps all views equal (the old
/// bug: the next write advanced `meta.write_seq` past a replayed op whose
/// `vectors` row was never written, so a later cold load lost it).
fn assert_consistent(st: &MemSqlStore, dm: &DoMeta, cfg: &ShardConfig, n: u64) {
    let mut s = VectorShard::open(st).unwrap();
    let digest = s.state_digest();
    assert_eq!(
        VectorShard::open(st).unwrap().state_digest(),
        digest,
        "n={n}"
    );
    assert_eq!(
        VectorShard::rebuild_from_ops(st).unwrap().state_digest(),
        digest,
        "n={n}: replay"
    );
    assert_eq!(st.row_count("vectors"), s.len(), "n={n}: vectors rows");
    assert_eq!(st.row_count("filter_idx"), s.len(), "n={n}: filter rows");
    upsert(&mut s, st, dm, cfg, rows(&["z"], 7)).unwrap();
    let live = s.state_digest();
    assert_eq!(VectorShard::open(st).unwrap().state_digest(), live, "n={n}");
    assert_eq!(
        VectorShard::rebuild_from_ops(st).unwrap().state_digest(),
        live,
        "n={n}: replay after next write"
    );
}

#[test]
fn torn_upsert_at_every_statement_reopens_consistently() {
    let mut completed = false;
    for n in 0..40 {
        let (st, mut s, dm, cfg) = setup();
        upsert(&mut s, &st, &dm, &cfg, rows(&["a", "b", "c"], 1)).unwrap();
        st.set_fail_after(Some(n));
        match upsert(&mut s, &st, &dm, &cfg, rows(&["a", "b", "c", "d"], 2)) {
            Ok(()) => completed = true,
            Err(code) => {
                assert_eq!(code, ErrorCode::ShardUnavailable);
                assert!(s.is_poisoned());
            }
        }
        st.set_fail_after(None);
        assert_consistent(&st, &dm, &cfg, n);
        if completed {
            break;
        }
    }
    assert!(completed, "the batch eventually fits before the fault");
}

#[test]
fn torn_delete_at_every_statement_reopens_consistently() {
    let mut completed = false;
    for n in 0..30 {
        let (st, mut s, dm, cfg) = setup();
        upsert(&mut s, &st, &dm, &cfg, rows(&["a", "b", "c", "d"], 1)).unwrap();
        st.set_fail_after(Some(n));
        let ids: Vec<String> = ["a", "b", "c"].iter().map(|x| x.to_string()).collect();
        match s.delete(&st, &dm, &ids, ACTOR, false, T0) {
            Ok(_) => completed = true,
            Err(e) => assert_eq!(e.code, ErrorCode::ShardUnavailable),
        }
        st.set_fail_after(None);
        assert_consistent(&st, &dm, &cfg, n);
        if completed {
            break;
        }
    }
    assert!(completed);
}

type H = Harness<MemSqlStore>;

fn shard_names(
    t: &ruvector_edge_tenancy::TenantKey,
    uid: &str,
    n: u32,
) -> Vec<ruvector_edge_tenancy::DoName> {
    let uid = CollectionUid::parse(uid).unwrap();
    (0..n)
        .map(|i| {
            shard_meta_for(
                t,
                uid,
                ShardIndex::new(i, ShardCount::new(n).unwrap()).unwrap(),
            )
            .unwrap()
            .do_name()
        })
        .collect()
}

fn truth(h: &mut H, names: &[ruvector_edge_tenancy::DoName]) -> UsageDelta {
    let mut t = UsageDelta::default();
    for n in names {
        let u = h.cluster.shard(n).unwrap().1.usage_totals();
        t.vectors += u.vectors;
        t.floats += u.floats;
        t.bytes += u.bytes;
    }
    t
}

fn idem_bytes(h: &H, t: &ruvector_edge_tenancy::TenantKey) -> i64 {
    use ruvector_edge_store::SqlStore;
    let st = h
        .cluster
        .store(ruvector_edge_tenancy::ledger_do_name(t).as_str())
        .unwrap();
    st.query(
        "SELECT bytes FROM idempotency WHERE expires_at > ?",
        &[0i64.into()],
    )
    .unwrap()
    .iter()
    .map(|r| r[0].as_int().unwrap())
    .sum()
}

fn vecs(prefix: &str, n: usize, meta: &str) -> Vec<Json> {
    (0..n)
        .map(|i| json!({"id": format!("{prefix}{i}"), "values": [i as f32, 1.0], "metadata": {"m": meta}}))
        .collect()
}

/// A multi-shard upsert torn on one shard: the ledger is reconciled to the
/// shards' real contents (refund of the unapplied part, including a
/// positive-bytes refund when replacements shrank rows), and the retry
/// under the same op_id re-executes (errors are never remembered).
#[test]
fn torn_multi_shard_upsert_reconciles_usage_exactly() {
    for n in [0u64, 1, 3, 6] {
        let mut h = H::new();
        let t = tenant("org-r");
        h.claim(&t, "owner", &[("ed", Role::Editor)]);
        let o = ctx(&t, "owner", write_caps());
        let (_, c) = h.call(
            &o,
            "collection_create",
            json!({"name": "c", "dim": 2, "metric": "l2", "shards": 3, "filterable_keys": ["m"]}),
        );
        let names = shard_names(&t, c["result"]["collection_uid"].as_str().unwrap(), 3);
        let big = "x".repeat(900);
        let (s, r) = h.call(
            &o,
            "vector_upsert",
            json!({"collection": "c", "vectors": vecs("v", 30, &big)}),
        );
        assert_eq!(s, 200, "{r}");
        // Replace with small metadata: every shard's delta is negative bytes.
        let st = h.cluster.store(names[1].as_str()).unwrap();
        st.set_fail_after(Some(n));
        let id = op_id(900 + n);
        let args = json!({"collection": "c", "vectors": vecs("v", 30, "s")});
        let (s, r) = h.call_with(&o, "vector_upsert", &id, args.clone(), false);
        assert_eq!((s, code(&r)), (503, "shard_unavailable"), "n={n}");
        h.cluster
            .store(names[1].as_str())
            .unwrap()
            .set_fail_after(None);
        let want = truth(&mut h, &names);
        let (_, u) = h.call(&o, "usage_get", json!({}));
        let u = &u["result"]["usage"];
        assert_eq!(u["vectors"], want.vectors, "n={n}");
        assert_eq!(u["float_budget"], want.floats, "n={n}");
        assert_eq!(
            u["bytes"].as_i64().unwrap(),
            want.bytes + idem_bytes(&h, &t),
            "n={n}"
        );
        // The failed op was not remembered: the retry executes and settles.
        let (s, r) = h.call_with(&o, "vector_upsert", &id, args, false);
        assert_eq!(s, 200, "{r}");
        let want = truth(&mut h, &names);
        let (_, u) = h.call(&o, "usage_get", json!({}));
        let u = &u["result"]["usage"];
        assert_eq!(u["vectors"], 30);
        assert_eq!(
            u["bytes"].as_i64().unwrap(),
            want.bytes + idem_bytes(&h, &t)
        );
    }
}

#[test]
fn torn_multi_shard_delete_releases_what_was_removed() {
    for n in [0u64, 1, 2, 4] {
        let mut h = H::new();
        let t = tenant("org-d");
        h.claim(&t, "owner", &[]);
        let o = ctx(&t, "owner", write_caps());
        let (_, c) = h.call(
            &o,
            "collection_create",
            json!({"name": "c", "dim": 2, "metric": "l2", "shards": 3}),
        );
        let names = shard_names(&t, c["result"]["collection_uid"].as_str().unwrap(), 3);
        assert_eq!(
            h.call(
                &o,
                "vector_upsert",
                json!({"collection": "c", "vectors": vecs("v", 30, "m")})
            )
            .0,
            200
        );
        h.cluster
            .store(names[2].as_str())
            .unwrap()
            .set_fail_after(Some(n));
        let ids: Vec<String> = (0..30).map(|i| format!("v{i}")).collect();
        let (s, _) = h.call(&o, "vector_delete", json!({"collection": "c", "ids": ids}));
        h.cluster
            .store(names[2].as_str())
            .unwrap()
            .set_fail_after(None);
        assert_eq!(s, 503, "n={n}");
        let want = truth(&mut h, &names);
        let (_, u) = h.call(&o, "usage_get", json!({}));
        assert_eq!(u["result"]["usage"]["vectors"], want.vectors, "n={n}");
        assert_eq!(
            u["result"]["usage"]["bytes"].as_i64().unwrap(),
            want.bytes + idem_bytes(&h, &t),
            "n={n}"
        );
    }
}

/// Internal corrections never fail on limits and clamp instead of
/// underflowing (a refund used to be re-admitted against the limits and
/// silently dropped).
#[test]
fn ledger_adjust_ignores_limits_and_clamps() {
    use ruvector_edge_store::{ledger_meta_for, TenantLedger};
    use ruvector_edge_tenancy::QuotaDelta;
    let st = MemSqlStore::new();
    let t = tenant("org-q");
    let lm = ledger_meta_for(&t).unwrap();
    let mut lim = limits();
    lim.max_vectors = 10;
    let mut l = TenantLedger::open(&st, lim).unwrap();
    let d = |v: i64| QuotaDelta {
        vectors: v,
        ..Default::default()
    };
    l.admit(&st, &lm, d(10), 0, T0).unwrap();
    assert_eq!(
        l.admit(&st, &lm, d(1), 0, T0).unwrap_err().code,
        ErrorCode::QuotaExceeded
    );
    assert_eq!(l.adjust(&st, &lm, d(5), T0).unwrap().vectors, 15);
    assert_eq!(l.adjust(&st, &lm, d(-100), T0).unwrap().vectors, 0);
    assert_eq!(
        TenantLedger::open(&st, lim)
            .unwrap()
            .usage(&lm, T0)
            .unwrap()
            .vectors,
        0
    );
}
