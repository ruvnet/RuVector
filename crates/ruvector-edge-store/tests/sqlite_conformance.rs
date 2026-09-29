//! SQL conformance: the same scenarios on the in-memory mock and on real
//! SQLite (what Durable Objects run) must produce identical responses and
//! identical shard state. This is what makes the `schema` constants
//! trustworthy beyond the mock's own grammar.

mod common;

use common::*;
use rusqlite::types::{ToSqlOutput, ValueRef};
use rusqlite::{params_from_iter, Connection, ToSql};
use ruvector_edge_store::{
    schema, shard_meta_for, MemSqlStore, Row, SqlStore, StoreError, Value, VectorShard,
};
use ruvector_edge_tenancy::{CollectionUid, Role, ShardIndex};
use serde_json::{json, Value as Json};

struct SqliteStore(Connection);

impl Default for SqliteStore {
    fn default() -> Self {
        SqliteStore(Connection::open_in_memory().unwrap())
    }
}

struct P<'a>(&'a Value);
impl ToSql for P<'_> {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(match self.0 {
            Value::Null => ToSqlOutput::Borrowed(ValueRef::Null),
            Value::Int(i) => ToSqlOutput::Borrowed(ValueRef::Integer(*i)),
            Value::Real(f) => ToSqlOutput::Borrowed(ValueRef::Real(*f)),
            Value::Text(s) => ToSqlOutput::Borrowed(ValueRef::Text(s.as_bytes())),
            Value::Blob(b) => ToSqlOutput::Borrowed(ValueRef::Blob(b)),
        })
    }
}

fn map_err(e: rusqlite::Error) -> StoreError {
    match e.sqlite_error_code() {
        Some(rusqlite::ErrorCode::ConstraintViolation) => StoreError::Constraint,
        _ => StoreError::Backend(e.to_string()),
    }
}

impl SqlStore for SqliteStore {
    fn exec(&self, sql: &str, params: &[Value]) -> Result<u64, StoreError> {
        self.0
            .execute(sql, params_from_iter(params.iter().map(P)))
            .map(|n| n as u64)
            .map_err(map_err)
    }
    fn query(&self, sql: &str, params: &[Value]) -> Result<Vec<Row>, StoreError> {
        let mut st = self.0.prepare(sql).map_err(map_err)?;
        let n = st.column_count();
        let rows = st
            .query_map(params_from_iter(params.iter().map(P)), |r| {
                (0..n)
                    .map(|i| {
                        Ok(match r.get_ref(i)? {
                            ValueRef::Null => Value::Null,
                            ValueRef::Integer(i) => Value::Int(i),
                            ValueRef::Real(f) => Value::Real(f),
                            ValueRef::Text(t) => {
                                Value::Text(String::from_utf8_lossy(t).into_owned())
                            }
                            ValueRef::Blob(b) => Value::Blob(b.to_vec()),
                        })
                    })
                    .collect::<rusqlite::Result<Row>>()
            })
            .map_err(map_err)?;
        rows.collect::<rusqlite::Result<Vec<Row>>>()
            .map_err(map_err)
    }
}

/// A deterministic ops scenario; returns every response in order.
fn scenario<S: SqlStore + Default>() -> Vec<Json> {
    let mut h: Harness<S> = Harness::new();
    let t = tenant("org-conf");
    h.claim(&t, "owner", &[("vi", Role::Viewer)]);
    let o = ctx(&t, "owner", write_caps());
    let vi = ctx(&t, "vi", read_caps());
    let mut out = Vec::new();
    let mut push = |x: (u16, Json)| out.push(json!({"status": x.0, "body": x.1}));
    push(h.call(
        &o,
        "collection_create",
        json!({"name": "c", "dim": 8, "metric": "cosine", "shards": 2, "filterable_keys": ["g"]}),
    ));
    let mut rng = Rng(77);
    for b in 0..4 {
        let vecs: Vec<Json> = (0..50)
            .map(|i| json!({"id": format!("id{}", b * 50 + i), "values": rng.vec(8), "metadata": {"g": i % 3}}))
            .collect();
        push(h.call(
            &o,
            "vector_upsert",
            json!({"collection": "c", "vectors": vecs}),
        ));
    }
    push(h.call(
        &o,
        "vector_delete",
        json!({"collection": "c", "ids": ["id3", "id7", "id150"]}),
    ));
    push(h.call(
        &o,
        "vector_upsert",
        json!({"collection": "c", "vectors": [{"id": "id7", "values": rng.vec(8)}]}),
    ));
    let q = rng.vec(8);
    push(h.call(
        &vi,
        "vector_query",
        json!({"collection": "c", "vector": q, "top_k": 7, "include": ["metadata"]}),
    ));
    push(h.call(
        &vi,
        "vector_query",
        json!({"collection": "c", "vector": q, "top_k": 7, "filter": {"g": {"$ne": 1}}}),
    ));
    push(h.call(
        &vi,
        "vector_fetch",
        json!({"collection": "c", "ids": ["id7", "id3", "id199"], "include_values": true}),
    ));
    push(h.call_with(
        &o,
        "vector_delete",
        &op_id(9001),
        json!({"collection": "c", "ids": ["id9"]}),
        false,
    ));
    push(h.call_with(
        &o,
        "vector_delete",
        &op_id(9001),
        json!({"collection": "c", "ids": ["id9"]}),
        false,
    ));
    push(h.call_with(
        &o,
        "vector_delete",
        &op_id(9001),
        json!({"collection": "c", "ids": ["id10"]}),
        false,
    ));
    push(h.call(&o, "usage_get", json!({})));
    h.cluster.restart();
    push(h.call(
        &vi,
        "vector_query",
        json!({"collection": "c", "vector": q, "top_k": 7}),
    ));
    push(h.call(&o, "collection_list", json!({})));
    out
}

#[test]
fn ops_scenario_identical_on_mock_and_sqlite() {
    let mock = scenario::<MemSqlStore>();
    let real = scenario::<SqliteStore>();
    assert_eq!(mock.len(), real.len());
    for (i, (m, r)) in mock.iter().zip(&real).enumerate() {
        assert_eq!(m, r, "response {i} differs");
        assert!(m["status"] == 200 || i == 12, "step {i}: {m}");
    }
    assert_eq!(mock[12]["body"]["error"]["code"], "op_replayed");
}

#[test]
fn shard_digests_identical_on_mock_and_sqlite() {
    fn run<S: SqlStore + Default>() -> [[u8; 32]; 2] {
        let st = S::default();
        let mut s = VectorShard::open(&st).unwrap();
        let dm = shard_meta_for(
            &tenant("org-d"),
            CollectionUid::from_bytes([3; 16]),
            ShardIndex::ZERO,
        )
        .unwrap();
        let cfg = ruvector_edge_store::ShardConfig {
            dim: 4,
            metric: ruvector_edge_store::Metric::L2,
            filterable_keys: vec!["k".into()],
            float_cap: 1_000_000,
        };
        let actor = ruvector_edge_store::shard::Actor {
            sub: "es1_a",
            jti: "j",
            family_id: "f",
        };
        let mut rng = Rng(5);
        // > PAGE_ROWS rows so the paged cold load crosses a page boundary.
        for b in 0..3 {
            let rows = (0..500)
                .map(|i| ruvector_edge_store::UpsertRow {
                    id: format!("r{}", b * 500 + i),
                    values: rng.vec(4),
                    metadata: Some(json!({"k": i % 4})),
                })
                .collect();
            let plan = s.plan_upsert(&dm, &cfg, rows).unwrap();
            s.apply_upsert(&st, plan, actor, T0).unwrap();
        }
        let ids: Vec<String> = (0..1500).step_by(7).map(|i| format!("r{i}")).collect();
        for chunk in ids.chunks(100) {
            s.delete(&st, &dm, chunk, actor, false, T0).unwrap();
        }
        assert_eq!(
            st.query(schema::VEC_PAGE, &[0i64.into(), 5i64.into()])
                .unwrap()
                .len(),
            5
        );
        // The op log is pruned to a bounded tail once folded into `vectors`
        // (bounded growth); replay-from-scratch then fails closed.
        let logged = st
            .query(schema::OPS_PAGE, &[0i64.into(), 100_000i64.into()])
            .unwrap()
            .len() as u64;
        assert_eq!(logged, ruvector_edge_store::shard::OPS_TAIL);
        assert_eq!(s.snapshot_seq(), s.write_seq() - logged);
        assert!(VectorShard::rebuild_from_ops(&st).is_err());
        [
            s.state_digest(),
            VectorShard::open(&st).unwrap().state_digest(),
        ]
    }
    let m = run::<MemSqlStore>();
    let r = run::<SqliteStore>();
    assert_eq!(m, r);
    assert!(m.iter().all(|d| *d == m[0]), "live == cold load");
}

#[test]
fn constraint_errors_map_identically() {
    for st in [
        &SqliteStore::default() as &dyn SqlStore,
        &MemSqlStore::new(),
    ] {
        for ddl in schema::LEDGER_SCHEMA {
            st.exec(ddl, &[]).unwrap();
        }
        let row = [
            Value::from("es1_x"),
            "owner".into(),
            Value::Null,
            1i64.into(),
        ];
        st.exec(schema::MEMBER_INSERT, &row).unwrap();
        assert_eq!(
            st.exec(schema::MEMBER_INSERT, &row),
            Err(StoreError::Constraint)
        );
        st.exec(schema::MEMBER_PUT, &row).unwrap();
        assert_eq!(st.query(schema::MEMBER_SELECT_ALL, &[]).unwrap().len(), 1);
    }
}
