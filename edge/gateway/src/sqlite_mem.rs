//! `SqlStore` over an in-memory **real** SQLite (rusqlite), for the M4
//! tests: the quant-shard SQL runs against the engine Durable Objects use
//! (with its indexes), which the linear-scan `MemSqlStore` mock cannot do
//! at 50k rows.

use rusqlite::types::{Value as SVal, ValueRef};
use ruvector_edge_store::{Row, SqlStore, StoreError, Value};

/// One in-memory SQLite database.
pub struct SqliteStore(pub rusqlite::Connection);

impl Default for SqliteStore {
    fn default() -> Self {
        SqliteStore(rusqlite::Connection::open_in_memory().expect("sqlite"))
    }
}

fn err(e: rusqlite::Error) -> StoreError {
    let t = e.to_string();
    if t.contains("constraint") || t.contains("UNIQUE") {
        StoreError::Constraint
    } else {
        StoreError::Backend(t)
    }
}

fn to_sql(v: &Value) -> SVal {
    match v {
        Value::Null => SVal::Null,
        Value::Int(i) => SVal::Integer(*i),
        Value::Real(f) => SVal::Real(*f),
        Value::Text(s) => SVal::Text(s.clone()),
        Value::Blob(b) => SVal::Blob(b.clone()),
    }
}

impl SqlStore for SqliteStore {
    fn exec(&self, sql: &str, params: &[Value]) -> Result<u64, StoreError> {
        let mut st = self.0.prepare_cached(sql).map_err(err)?;
        let n = st
            .execute(rusqlite::params_from_iter(params.iter().map(to_sql)))
            .map_err(err)?;
        Ok(n as u64)
    }

    fn query(&self, sql: &str, params: &[Value]) -> Result<Vec<Row>, StoreError> {
        let mut st = self.0.prepare_cached(sql).map_err(err)?;
        let cols = st.column_count();
        let mut rows = st
            .query(rusqlite::params_from_iter(params.iter().map(to_sql)))
            .map_err(err)?;
        let mut out = Vec::new();
        while let Some(r) = rows.next().map_err(err)? {
            let mut row = Vec::with_capacity(cols);
            for i in 0..cols {
                row.push(match r.get_ref(i).map_err(err)? {
                    ValueRef::Null => Value::Null,
                    ValueRef::Integer(i) => Value::Int(i),
                    ValueRef::Real(f) => Value::Real(f),
                    ValueRef::Text(t) => Value::Text(String::from_utf8_lossy(t).into_owned()),
                    ValueRef::Blob(b) => Value::Blob(b.to_vec()),
                });
            }
            out.push(row);
        }
        Ok(out)
    }
}
