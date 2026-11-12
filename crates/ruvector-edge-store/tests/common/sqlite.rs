//! Real SQLite (what Durable Objects run) behind the `SqlStore` port, for
//! conformance and for the large M2 scenarios (the mock is O(n) per write).

use rusqlite::types::{ToSqlOutput, ValueRef};
use rusqlite::{params_from_iter, Connection, ToSql};
use ruvector_edge_store::{Row, SqlStore, StoreError, Value};

pub struct SqliteStore(pub Connection);

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
