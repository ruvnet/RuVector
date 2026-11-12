//! Minimal SQL port so the storage adapters run against Durable Object
//! SQLite in production and against an in-memory SQLite (rusqlite) in native
//! tests. Only text and integer parameters are used: hashes are hex text,
//! records are JSON text.

use ruvector_edge_authz::StoreError;

/// Bound parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SqlArg {
    /// TEXT.
    Text(String),
    /// INTEGER.
    Int(i64),
}

impl From<&str> for SqlArg {
    fn from(s: &str) -> Self {
        SqlArg::Text(s.to_string())
    }
}

impl From<String> for SqlArg {
    fn from(s: String) -> Self {
        SqlArg::Text(s)
    }
}

impl From<u64> for SqlArg {
    /// Saturates at `i64::MAX` (unix seconds never get there).
    fn from(v: u64) -> Self {
        SqlArg::Int(i64::try_from(v).unwrap_or(i64::MAX))
    }
}

/// Result cell.
#[derive(Debug, Clone, PartialEq)]
pub enum SqlVal {
    /// NULL.
    Null,
    /// INTEGER (also booleans).
    Int(i64),
    /// REAL.
    Real(f64),
    /// TEXT.
    Text(String),
    /// BLOB.
    Blob(Vec<u8>),
}

impl SqlVal {
    /// Text content, if TEXT.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            SqlVal::Text(s) => Some(s),
            _ => None,
        }
    }

    /// Integer content (REAL with an integral value is accepted, since JS
    /// numbers cross the Workers boundary as `f64`).
    pub fn as_int(&self) -> Option<i64> {
        match self {
            SqlVal::Int(i) => Some(*i),
            SqlVal::Real(f) if f.fract() == 0.0 && f.abs() < 9.0e15 => Some(*f as i64),
            _ => None,
        }
    }
}

/// Synchronous SQL execution (Durable Object `sql.exec` is synchronous).
pub trait SqlExec {
    /// Run a multi-statement script without parameters (schema).
    fn exec_script(&self, sql: &str) -> Result<(), StoreError>;
    /// Run one statement and return every result row (fully consumed, so
    /// writes have happened when this returns).
    fn query(&self, sql: &str, args: Vec<SqlArg>) -> Result<Vec<Vec<SqlVal>>, StoreError>;
}

/// Lowercase hex of a 32-byte digest (hash columns are TEXT).
pub fn hex32(bytes: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(64);
    for b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

mod worker_impl {
    use super::{SqlArg, SqlExec, SqlVal};
    use ruvector_edge_authz::StoreError;
    use worker::{SqlStorage, SqlStorageValue};

    fn err(e: worker::Error) -> StoreError {
        StoreError(e.to_string())
    }

    impl From<SqlArg> for SqlStorageValue {
        fn from(a: SqlArg) -> Self {
            match a {
                SqlArg::Text(s) => SqlStorageValue::String(s),
                SqlArg::Int(i) => SqlStorageValue::Integer(i),
            }
        }
    }

    fn val(v: SqlStorageValue) -> SqlVal {
        match v {
            SqlStorageValue::Null => SqlVal::Null,
            SqlStorageValue::Boolean(b) => SqlVal::Int(i64::from(b)),
            SqlStorageValue::Integer(i) => SqlVal::Int(i),
            SqlStorageValue::Float(f) => SqlVal::Real(f),
            SqlStorageValue::String(s) => SqlVal::Text(s),
            SqlStorageValue::Blob(b) => SqlVal::Blob(b),
        }
    }

    impl SqlExec for SqlStorage {
        fn exec_script(&self, sql: &str) -> Result<(), StoreError> {
            let cursor = self.exec(sql, None).map_err(err)?;
            for row in cursor.raw() {
                row.map_err(err)?;
            }
            Ok(())
        }

        fn query(&self, sql: &str, args: Vec<SqlArg>) -> Result<Vec<Vec<SqlVal>>, StoreError> {
            let args: Vec<SqlStorageValue> = args.into_iter().map(Into::into).collect();
            let cursor = self.exec(sql, args).map_err(err)?;
            cursor
                .raw()
                .map(|row| {
                    row.map(|cells| cells.into_iter().map(val).collect())
                        .map_err(err)
                })
                .collect()
        }
    }
}

/// In-memory SQLite for native tests.
#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) mod memory {
    use super::{SqlArg, SqlExec, SqlVal};
    use rusqlite::types::ValueRef;
    use ruvector_edge_authz::StoreError;

    pub(crate) struct MemoryDb(pub rusqlite::Connection);

    impl MemoryDb {
        pub(crate) fn new() -> Self {
            MemoryDb(rusqlite::Connection::open_in_memory().unwrap())
        }
    }

    fn err(e: rusqlite::Error) -> StoreError {
        StoreError(e.to_string())
    }

    impl SqlExec for MemoryDb {
        fn exec_script(&self, sql: &str) -> Result<(), StoreError> {
            self.0.execute_batch(sql).map_err(err)
        }

        fn query(&self, sql: &str, args: Vec<SqlArg>) -> Result<Vec<Vec<SqlVal>>, StoreError> {
            let mut stmt = self.0.prepare(sql).map_err(err)?;
            let n = stmt.column_count();
            let params = rusqlite::params_from_iter(args.into_iter().map(|a| match a {
                SqlArg::Text(s) => rusqlite::types::Value::Text(s),
                SqlArg::Int(i) => rusqlite::types::Value::Integer(i),
            }));
            let mut rows = stmt.query(params).map_err(err)?;
            let mut out = Vec::new();
            while let Some(row) = rows.next().map_err(err)? {
                let mut cells = Vec::with_capacity(n);
                for i in 0..n {
                    cells.push(match row.get_ref(i).map_err(err)? {
                        ValueRef::Null => SqlVal::Null,
                        ValueRef::Integer(v) => SqlVal::Int(v),
                        ValueRef::Real(v) => SqlVal::Real(v),
                        ValueRef::Text(t) => SqlVal::Text(String::from_utf8_lossy(t).into()),
                        ValueRef::Blob(b) => SqlVal::Blob(b.to_vec()),
                    });
                }
                out.push(cells);
            }
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex32_is_lowercase_and_64_chars() {
        let mut b = [0u8; 32];
        b[0] = 0xab;
        b[31] = 0x0f;
        let h = hex32(&b);
        assert_eq!(h.len(), 64);
        assert!(h.starts_with("ab00"));
        assert!(h.ends_with("0f"));
    }

    #[test]
    fn integral_reals_read_as_ints() {
        assert_eq!(SqlVal::Real(42.0).as_int(), Some(42));
        assert_eq!(SqlVal::Real(1.5).as_int(), None);
        assert_eq!(SqlVal::Text("1".into()).as_int(), None);
        assert_eq!(SqlArg::from(u64::MAX), SqlArg::Int(i64::MAX));
    }
}
