//! Ports: storage (`SqlStore`), plus the clock / entropy ports re-exported
//! from the auth and tenancy crates so callers implement one trait each.
//!
//! [`SqlStore`] mirrors the Durable Object SQLite API (`ctx.storage.sql.exec`)
//! closely enough that the Worker adapter is a thin pass-through: positional
//! `?` bindings, rows returned as vectors of [`Value`] in `SELECT` column
//! order. Methods take `&self` because the DO handle is a shared JS object;
//! the in-memory mock ([`crate::mem_store::MemSqlStore`]) uses interior
//! mutability.
//!
//! No `transaction()` is used (ADR §6.1): callers validate fully in memory
//! first and then issue every write back to back, so on a DO they coalesce
//! into one implicit commit.

pub use ruvector_edge_auth::Clock;
pub use ruvector_edge_tenancy::EntropySource;
use thiserror::Error;

/// A SQLite value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// SQL `NULL`.
    Null,
    /// `INTEGER`.
    Int(i64),
    /// `REAL`.
    Real(f64),
    /// `TEXT`.
    Text(String),
    /// `BLOB`.
    Blob(Vec<u8>),
}

impl Value {
    /// Text accessor.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Value::Text(s) => Some(s),
            _ => None,
        }
    }
    /// Integer accessor.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }
    /// Blob accessor.
    pub fn as_blob(&self) -> Option<&[u8]> {
        match self {
            Value::Blob(b) => Some(b),
            _ => None,
        }
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::Text(s.to_string())
    }
}
impl From<String> for Value {
    fn from(s: String) -> Self {
        Value::Text(s)
    }
}
impl From<i64> for Value {
    fn from(i: i64) -> Self {
        Value::Int(i)
    }
}
impl From<Vec<u8>> for Value {
    fn from(b: Vec<u8>) -> Self {
        Value::Blob(b)
    }
}

/// One result row, columns in `SELECT` order.
pub type Row = Vec<Value>;

/// Storage failure.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum StoreError {
    /// The backend rejected or failed the statement (retryable at the
    /// request level: `503 shard_unavailable`).
    #[error("storage backend error: {0}")]
    Backend(String),
    /// A primary-key / unique constraint was violated.
    #[error("constraint violation")]
    Constraint,
    /// A stored row failed decoding: corruption or a forged value; fail
    /// closed (`500`).
    #[error("corrupt stored row: {0}")]
    Corrupt(&'static str),
}

/// DO-SQLite-shaped storage port.
pub trait SqlStore {
    /// Execute a statement that returns no rows. Returns rows written.
    fn exec(&self, sql: &str, params: &[Value]) -> Result<u64, StoreError>;
    /// Execute a `SELECT`, returning every row.
    fn query(&self, sql: &str, params: &[Value]) -> Result<Vec<Row>, StoreError>;
}

impl<T: SqlStore + ?Sized> SqlStore for &T {
    fn exec(&self, sql: &str, params: &[Value]) -> Result<u64, StoreError> {
        (**self).exec(sql, params)
    }
    fn query(&self, sql: &str, params: &[Value]) -> Result<Vec<Row>, StoreError> {
        (**self).query(sql, params)
    }
}

/// Column accessor helpers that fail closed on type confusion.
pub(crate) fn col_text(row: &Row, i: usize, what: &'static str) -> Result<String, StoreError> {
    row.get(i)
        .and_then(Value::as_text)
        .map(str::to_string)
        .ok_or(StoreError::Corrupt(what))
}

pub(crate) fn col_int(row: &Row, i: usize, what: &'static str) -> Result<i64, StoreError> {
    row.get(i)
        .and_then(Value::as_int)
        .ok_or(StoreError::Corrupt(what))
}

/// Read a `meta`-style `(k, v)` table as pairs.
pub(crate) fn read_kv(
    store: &dyn SqlStore,
    sql: &str,
) -> Result<Vec<(String, String)>, StoreError> {
    store
        .query(sql, &[])?
        .iter()
        .map(|r| Ok((col_text(r, 0, "meta.k")?, col_text(r, 1, "meta.v")?)))
        .collect()
}
