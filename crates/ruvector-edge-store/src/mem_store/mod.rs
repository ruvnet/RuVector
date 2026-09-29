//! In-memory [`SqlStore`] for tests and native tooling.
//!
//! Interprets the restricted grammar in `parse` over row vectors, with
//! SQLite-like semantics for everything the crate uses: primary keys
//! (`INSERT` conflicts, `INSERT OR REPLACE` replaces), `NULL` never compares
//! equal, `ORDER BY` / `LIMIT`. `UNIQUE` column constraints other than the
//! primary key are not enforced (the crate never relies on them).
//! `tests/sqlite_conformance.rs` runs the same scenarios on real SQLite.

mod parse;

use crate::ports::{Row, SqlStore, StoreError, Value};
use core::cell::{Cell, RefCell};
use core::cmp::Ordering;
use parse::{wh_params, CmpOp, Cond, Stmt};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default)]
struct Table {
    cols: Vec<String>,
    pk: Vec<usize>,
    rows: Vec<Row>,
}

impl Table {
    fn col(&self, name: &str) -> Result<usize, StoreError> {
        self.cols
            .iter()
            .position(|c| c == name)
            .ok_or_else(|| StoreError::Backend(format!("mem_store: no such column {name}")))
    }

    fn matches(&self, row: &Row, wh: &[(usize, CmpOp)], params: &[Value]) -> bool {
        let mut at = 0;
        wh.iter().all(|(ci, op)| {
            if let CmpOp::In(n) = op {
                let set = params.get(at..at + n).unwrap_or(&[]);
                at += n;
                return set
                    .iter()
                    .any(|p| cmp(&row[*ci], p) == Some(Ordering::Equal));
            }
            let Some(p) = params.get(at) else {
                return false;
            };
            at += 1;
            let Some(ord) = cmp(&row[*ci], p) else {
                return false;
            };
            match op {
                CmpOp::In(_) => false,
                CmpOp::Eq => ord == Ordering::Equal,
                CmpOp::Ne => ord != Ordering::Equal,
                CmpOp::Lt => ord == Ordering::Less,
                CmpOp::Le => ord != Ordering::Greater,
                CmpOp::Gt => ord == Ordering::Greater,
                CmpOp::Ge => ord != Ordering::Less,
            }
        })
    }

    fn resolve(&self, wh: &[Cond]) -> Result<Vec<(usize, CmpOp)>, StoreError> {
        wh.iter().map(|c| Ok((self.col(&c.col)?, c.op))).collect()
    }
}

/// SQLite-style comparison; `None` when either side is `NULL` or the types
/// are incomparable.
fn cmp(a: &Value, b: &Value) -> Option<Ordering> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Some(x.cmp(y)),
        (Value::Real(x), Value::Real(y)) => x.partial_cmp(y),
        (Value::Int(x), Value::Real(y)) => (*x as f64).partial_cmp(y),
        (Value::Real(x), Value::Int(y)) => x.partial_cmp(&(*y as f64)),
        (Value::Text(x), Value::Text(y)) => Some(x.cmp(y)),
        (Value::Blob(x), Value::Blob(y)) => Some(x.cmp(y)),
        _ => None,
    }
}

/// Sort key for `ORDER BY` (`NULL` first, as SQLite does).
fn order_cmp(a: &Value, b: &Value) -> Ordering {
    match (a, b) {
        (Value::Null, Value::Null) => Ordering::Equal,
        (Value::Null, _) => Ordering::Less,
        (_, Value::Null) => Ordering::Greater,
        _ => cmp(a, b).unwrap_or(Ordering::Equal),
    }
}

/// In-memory SQL store. Cheap to clone (deep copy) for crash/restore tests.
#[derive(Debug, Clone, Default)]
pub struct MemSqlStore {
    tables: RefCell<BTreeMap<String, Table>>,
    fail_writes: Cell<bool>,
    fail_after: Cell<Option<u64>>,
    writes: Cell<u64>,
}

/// Largest TEXT/BLOB payload one write may bind, as on Durable Object
/// SQLite ("maximum string, BLOB or row size: 2 MB").
pub const MAX_ROW_BYTES: usize = 2_000_000;

impl MemSqlStore {
    /// Empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fault injection: while set, every write statement fails with
    /// [`StoreError::Backend`].
    pub fn set_fail_writes(&self, fail: bool) {
        self.fail_writes.set(fail);
    }

    /// Transient fault injection (torn writes): the next `n` write
    /// statements succeed, the one after fails once, then writes succeed
    /// again. `None` disarms.
    pub fn set_fail_after(&self, n: Option<u64>) {
        self.fail_after.set(n);
    }

    /// Number of successful write statements so far.
    pub fn write_count(&self) -> u64 {
        self.writes.get()
    }

    /// Number of rows currently in `table` (0 if absent).
    pub fn row_count(&self, table: &str) -> usize {
        self.tables.borrow().get(table).map_or(0, |t| t.rows.len())
    }

    fn run(&self, stmt: Stmt, params: &[Value]) -> Result<(u64, Vec<Row>), StoreError> {
        if stmt.param_count() != params.len() {
            return Err(StoreError::Backend(
                "mem_store: wrong parameter count".into(),
            ));
        }
        let is_write = !matches!(
            stmt,
            Stmt::Select { .. } | Stmt::Create { .. } | Stmt::Index { .. }
        );
        if is_write && self.fail_writes.get() {
            return Err(StoreError::Backend(
                "mem_store: injected write failure".into(),
            ));
        }
        if is_write {
            match self.fail_after.get() {
                Some(0) => {
                    self.fail_after.set(None);
                    return Err(StoreError::Backend("mem_store: injected torn write".into()));
                }
                Some(n) => self.fail_after.set(Some(n - 1)),
                None => {}
            }
            let payload: usize = params
                .iter()
                .map(|v| match v {
                    Value::Text(s) => s.len(),
                    Value::Blob(b) => b.len(),
                    _ => 8,
                })
                .sum();
            if payload > MAX_ROW_BYTES {
                return Err(StoreError::Backend("mem_store: row too large".into()));
            }
        }
        let mut tables = self.tables.borrow_mut();
        let out = match stmt {
            Stmt::Create { table, cols, pk } => {
                tables.entry(table).or_insert_with(|| {
                    let pk = pk
                        .iter()
                        .filter_map(|k| cols.iter().position(|c| c == k))
                        .collect();
                    Table {
                        cols,
                        pk,
                        rows: Vec::new(),
                    }
                });
                (0, Vec::new())
            }
            Stmt::Index { table, cols } => {
                let t = get(&mut tables, &table)?;
                for c in &cols {
                    t.col(c)?;
                }
                (0, Vec::new())
            }
            Stmt::Insert {
                table,
                replace,
                cols,
            } => {
                let t = get(&mut tables, &table)?;
                let mut row = vec![Value::Null; t.cols.len()];
                for (c, v) in cols.iter().zip(params) {
                    row[t.col(c)?] = v.clone();
                }
                if !t.pk.is_empty() {
                    let clash = t.rows.iter().position(|r| {
                        t.pk.iter()
                            .all(|&k| cmp(&r[k], &row[k]) == Some(Ordering::Equal))
                    });
                    match clash {
                        Some(i) if replace => {
                            t.rows.remove(i);
                        }
                        Some(_) => return Err(StoreError::Constraint),
                        None => {}
                    }
                }
                t.rows.push(row);
                (1, Vec::new())
            }
            Stmt::Update { table, sets, wh } => {
                let t = get(&mut tables, &table)?;
                let set_idx: Vec<usize> =
                    sets.iter().map(|c| t.col(c)).collect::<Result<_, _>>()?;
                let wh = t.resolve(&wh)?;
                let (set_vals, wh_vals) = params.split_at(set_idx.len());
                let hits: Vec<usize> = (0..t.rows.len())
                    .filter(|&i| t.matches(&t.rows[i], &wh, wh_vals))
                    .collect();
                for &i in &hits {
                    for (ci, v) in set_idx.iter().zip(set_vals) {
                        t.rows[i][*ci] = v.clone();
                    }
                }
                (hits.len() as u64, Vec::new())
            }
            Stmt::Delete { table, wh } => {
                let t = get(&mut tables, &table)?;
                let wh = t.resolve(&wh)?;
                let before = t.rows.len();
                let rows = core::mem::take(&mut t.rows);
                t.rows = rows
                    .into_iter()
                    .filter(|r| !t.matches(r, &wh, params))
                    .collect();
                ((before - t.rows.len()) as u64, Vec::new())
            }
            Stmt::Select {
                table,
                cols,
                wh,
                order,
                limit,
            } => {
                let t = get(&mut tables, &table)?;
                let sel: Vec<usize> = cols.iter().map(|c| t.col(c)).collect::<Result<_, _>>()?;
                let wh_res = t.resolve(&wh)?;
                let mut hits: Vec<&Row> = t
                    .rows
                    .iter()
                    .filter(|r| t.matches(r, &wh_res, params))
                    .collect();
                if let Some((col, desc)) = order {
                    let ci = t.col(&col)?;
                    hits.sort_by(|a, b| {
                        let o = order_cmp(&a[ci], &b[ci]);
                        if desc {
                            o.reverse()
                        } else {
                            o
                        }
                    });
                }
                if limit {
                    let n = params
                        .get(wh_params(&wh))
                        .and_then(Value::as_int)
                        .ok_or_else(|| {
                            StoreError::Backend("mem_store: LIMIT needs INTEGER".into())
                        })?;
                    hits.truncate(usize::try_from(n.max(0)).unwrap_or(usize::MAX));
                }
                let rows = hits
                    .into_iter()
                    .map(|r| sel.iter().map(|&i| r[i].clone()).collect())
                    .collect();
                (0, rows)
            }
        };
        if is_write {
            self.writes.set(self.writes.get() + 1);
        }
        Ok(out)
    }
}

fn get<'a>(
    tables: &'a mut BTreeMap<String, Table>,
    name: &str,
) -> Result<&'a mut Table, StoreError> {
    tables
        .get_mut(name)
        .ok_or_else(|| StoreError::Backend(format!("mem_store: no such table {name}")))
}

impl SqlStore for MemSqlStore {
    fn exec(&self, sql: &str, params: &[Value]) -> Result<u64, StoreError> {
        let stmt = parse::parse(sql)?;
        if matches!(stmt, Stmt::Select { .. }) {
            return Err(StoreError::Backend("mem_store: exec of SELECT".into()));
        }
        self.run(stmt, params).map(|(n, _)| n)
    }

    fn query(&self, sql: &str, params: &[Value]) -> Result<Vec<Row>, StoreError> {
        let stmt = parse::parse(sql)?;
        if !matches!(stmt, Stmt::Select { .. }) {
            return Err(StoreError::Backend("mem_store: query of non-SELECT".into()));
        }
        self.run(stmt, params).map(|(_, rows)| rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> MemSqlStore {
        let s = MemSqlStore::new();
        s.exec(
            "CREATE TABLE IF NOT EXISTS t (a TEXT PRIMARY KEY, b INTEGER)",
            &[],
        )
        .unwrap();
        s
    }

    #[test]
    fn insert_conflict_and_replace() {
        let s = store();
        let ins = "INSERT INTO t (a, b) VALUES (?, ?)";
        s.exec(ins, &["x".into(), 1.into()]).unwrap();
        assert_eq!(
            s.exec(ins, &["x".into(), 2.into()]),
            Err(StoreError::Constraint)
        );
        s.exec(
            "INSERT OR REPLACE INTO t (a, b) VALUES (?, ?)",
            &["x".into(), 3.into()],
        )
        .unwrap();
        let rows = s
            .query("SELECT b FROM t WHERE a = ?", &["x".into()])
            .unwrap();
        assert_eq!(rows, vec![vec![Value::Int(3)]]);
    }

    #[test]
    fn where_order_limit_update_delete() {
        let s = store();
        for (a, b) in [("p", 5), ("q", 1), ("r", 3), ("n", 9)] {
            s.exec("INSERT INTO t (a, b) VALUES (?, ?)", &[a.into(), b.into()])
                .unwrap();
        }
        let rows = s
            .query(
                "SELECT a FROM t WHERE b > ? ORDER BY b DESC LIMIT ?",
                &[1.into(), 2.into()],
            )
            .unwrap();
        assert_eq!(rows, vec![vec![Value::from("n")], vec![Value::from("p")]]);
        assert_eq!(
            s.exec("UPDATE t SET b = ? WHERE a = ?", &[0.into(), "n".into()])
                .unwrap(),
            1
        );
        assert_eq!(
            s.exec("DELETE FROM t WHERE b <= ?", &[1.into()]).unwrap(),
            2
        );
        assert_eq!(s.row_count("t"), 2);
        // NULL never compares equal.
        s.exec(
            "INSERT INTO t (a, b) VALUES (?, ?)",
            &["z".into(), Value::Null],
        )
        .unwrap();
        assert!(s
            .query("SELECT a FROM t WHERE b = ?", &[Value::Null])
            .unwrap()
            .is_empty());
    }

    #[test]
    fn param_count_and_fault_injection() {
        let s = store();
        assert!(s
            .exec("INSERT INTO t (a, b) VALUES (?, ?)", &["x".into()])
            .is_err());
        s.set_fail_writes(true);
        assert!(s.exec("DELETE FROM t", &[]).is_err());
        assert!(s.query("SELECT a FROM t", &[]).is_ok());
    }

    #[test]
    fn torn_write_injection_and_row_size_limit() {
        let s = store();
        let ins = "INSERT OR REPLACE INTO t (a, b) VALUES (?, ?)";
        s.set_fail_after(Some(1));
        s.exec(ins, &["x".into(), 1.into()]).unwrap();
        assert!(s.exec(ins, &["y".into(), 1.into()]).is_err());
        s.exec(ins, &["z".into(), 1.into()]).unwrap();
        assert_eq!(s.row_count("t"), 2);
        let big = "x".repeat(MAX_ROW_BYTES + 1);
        assert!(s.exec(ins, &[big.into(), 1.into()]).is_err());
        s.exec("CREATE INDEX IF NOT EXISTS t_b ON t (b)", &[])
            .unwrap();
        assert!(s
            .exec("CREATE INDEX IF NOT EXISTS t_q ON t (q)", &[])
            .is_err());
    }
}
