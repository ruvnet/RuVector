//! The registry crate's `KvStore` port over Durable Object SQLite, plus
//! borrowed entropy for the registry services (ADR-351 §3 rv-registry, M5).

use ruvector_edge_registry::ports::KvStore;
use ruvector_edge_store::{SqlStore, StoreError, Value};
use ruvector_edge_tenancy::{EntropySource, TenancyError};

/// The registry crate's [`KvStore`] over the DO's SQLite (`rkv` table).
/// Keys are ASCII, so `prefix || 0x7f` bounds a prefix scan.
pub struct SqlKv<S>(pub S);

impl<S: SqlStore> SqlKv<S> {
    /// Ensure the table exists.
    pub fn open(sql: S) -> Result<Self, StoreError> {
        sql.exec(
            "CREATE TABLE IF NOT EXISTS rkv (k TEXT, v BLOB, PRIMARY KEY (k))",
            &[],
        )?;
        Ok(SqlKv(sql))
    }

    /// Number of keys under `prefix`, counting at most `limit` (keys only:
    /// no values are read).
    pub fn count_keys(&self, prefix: &str, limit: usize) -> Result<usize, StoreError> {
        let rows = self.0.query(
            "SELECT k FROM rkv WHERE k >= ? AND k < ? ORDER BY k LIMIT ?",
            &[
                Value::Text(prefix.into()),
                Value::Text(format!("{prefix}\u{7f}")),
                Value::Int(i64::try_from(limit).unwrap_or(i64::MAX)),
            ],
        )?;
        Ok(rows.len())
    }
}

fn blob_of(v: &Value) -> Result<Vec<u8>, StoreError> {
    match v {
        Value::Blob(b) => Ok(b.clone()),
        Value::Text(t) => Ok(t.as_bytes().to_vec()),
        _ => Err(StoreError::Corrupt("rkv value")),
    }
}

impl<S: SqlStore> KvStore for SqlKv<S> {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>, StoreError> {
        let rows = self
            .0
            .query("SELECT v FROM rkv WHERE k = ?", &[Value::Text(key.into())])?;
        rows.first()
            .map(|r| {
                r.first()
                    .ok_or(StoreError::Corrupt("rkv row"))
                    .and_then(blob_of)
            })
            .transpose()
    }
    fn put(&self, key: &str, value: &[u8]) -> Result<(), StoreError> {
        self.0.exec(
            "INSERT OR REPLACE INTO rkv (k, v) VALUES (?, ?)",
            &[Value::Text(key.into()), Value::Blob(value.to_vec())],
        )?;
        Ok(())
    }
    fn delete(&self, key: &str) -> Result<(), StoreError> {
        self.0
            .exec("DELETE FROM rkv WHERE k = ?", &[Value::Text(key.into())])?;
        Ok(())
    }
    fn list(
        &self,
        prefix: &str,
        start_after: Option<&str>,
        limit: usize,
    ) -> Result<Vec<(String, Vec<u8>)>, StoreError> {
        let upper = Value::Text(format!("{prefix}\u{7f}"));
        let n = Value::Int(i64::try_from(limit).unwrap_or(i64::MAX));
        let rows = match start_after.filter(|s| *s >= prefix) {
            Some(after) => self.0.query(
                "SELECT k, v FROM rkv WHERE k > ? AND k < ? ORDER BY k LIMIT ?",
                &[Value::Text(after.into()), upper, n],
            )?,
            None => self.0.query(
                "SELECT k, v FROM rkv WHERE k >= ? AND k < ? ORDER BY k LIMIT ?",
                &[Value::Text(prefix.into()), upper, n],
            )?,
        };
        rows.iter()
            .map(|r| match (r.first(), r.get(1)) {
                (Some(Value::Text(k)), Some(v)) => Ok((k.clone(), blob_of(v)?)),
                _ => Err(StoreError::Corrupt("rkv row")),
            })
            .collect()
    }
}

/// Borrowed entropy (the registry takes its ports by value).
pub struct EntropyRef<'a>(pub &'a dyn EntropySource);

impl EntropySource for EntropyRef<'_> {
    fn fill(&self, out: &mut [u8]) -> Result<(), TenancyError> {
        self.0.fill(out)
    }
}
