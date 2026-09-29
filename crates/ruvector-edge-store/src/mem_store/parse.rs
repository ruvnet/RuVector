//! Parser for the restricted SQL grammar the mock understands:
//!
//! ```text
//! CREATE TABLE IF NOT EXISTS t (col TYPE ..., [PRIMARY KEY (a, b)])
//! CREATE INDEX IF NOT EXISTS i ON t (c, ...)
//! INSERT [OR REPLACE] INTO t (c, ...) VALUES (?, ...)
//! UPDATE t SET c = ?, ... [WHERE cond]
//! DELETE FROM t [WHERE cond]
//! SELECT c, ... FROM t [WHERE cond] [ORDER BY c [ASC|DESC]] [LIMIT ?]
//! cond := c (= | != | < | <= | > | >=) ? [AND cond]
//! ```
//!
//! Anything else is rejected, so a statement the mock would silently
//! misinterpret fails loudly instead.

use crate::ports::StoreError;

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Word(String),
    Num,
    Param,
    Punct(&'static str),
}

fn tokenize(sql: &str) -> Result<Vec<Tok>, StoreError> {
    let b = sql.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c.is_ascii_whitespace() {
            i += 1;
        } else if c.is_ascii_alphabetic() || c == b'_' {
            let s = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            out.push(Tok::Word(sql[s..i].to_ascii_lowercase()));
        } else if c.is_ascii_digit() {
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            out.push(Tok::Num);
        } else {
            let two = sql.get(i..i + 2).unwrap_or("");
            let p = match two {
                "<=" => Some("<="),
                ">=" => Some(">="),
                "!=" => Some("!="),
                _ => None,
            };
            if let Some(p) = p {
                out.push(Tok::Punct(p));
                i += 2;
                continue;
            }
            let p = match c {
                b'?' => {
                    out.push(Tok::Param);
                    i += 1;
                    continue;
                }
                b'(' => "(",
                b')' => ")",
                b',' => ",",
                b'=' => "=",
                b'<' => "<",
                b'>' => ">",
                b'*' => "*",
                _ => return Err(bad("unsupported character")),
            };
            out.push(Tok::Punct(p));
            i += 1;
        }
    }
    Ok(out)
}

fn bad(what: &str) -> StoreError {
    StoreError::Backend(format!("mem_store: {what}"))
}

/// Comparison operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// `column op ?`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Cond {
    pub col: String,
    pub op: CmpOp,
}

/// Parsed statement.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Stmt {
    Create {
        table: String,
        cols: Vec<String>,
        pk: Vec<String>,
    },
    /// Secondary index: a no-op for the mock beyond checking the columns.
    Index {
        table: String,
        cols: Vec<String>,
    },
    Insert {
        table: String,
        replace: bool,
        cols: Vec<String>,
    },
    Update {
        table: String,
        sets: Vec<String>,
        wh: Vec<Cond>,
    },
    Delete {
        table: String,
        wh: Vec<Cond>,
    },
    Select {
        table: String,
        cols: Vec<String>,
        wh: Vec<Cond>,
        order: Option<(String, bool)>,
        limit: bool,
    },
}

impl Stmt {
    /// Number of `?` bindings the statement expects.
    pub fn param_count(&self) -> usize {
        match self {
            Stmt::Create { .. } | Stmt::Index { .. } => 0,
            Stmt::Insert { cols, .. } => cols.len(),
            Stmt::Update { sets, wh, .. } => sets.len() + wh.len(),
            Stmt::Delete { wh, .. } => wh.len(),
            Stmt::Select { wh, limit, .. } => wh.len() + usize::from(*limit),
        }
    }
}

struct P {
    t: Vec<Tok>,
    i: usize,
}

impl P {
    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.i)
    }
    fn next(&mut self) -> Option<Tok> {
        let t = self.t.get(self.i).cloned();
        self.i += 1;
        t
    }
    fn is_kw(&self, kw: &str) -> bool {
        matches!(self.peek(), Some(Tok::Word(w)) if w == kw)
    }
    fn kw(&mut self, kw: &str) -> Result<(), StoreError> {
        if self.is_kw(kw) {
            self.i += 1;
            Ok(())
        } else {
            Err(bad("expected keyword"))
        }
    }
    fn punct(&mut self, p: &str) -> Result<(), StoreError> {
        match self.next() {
            Some(Tok::Punct(q)) if q == p => Ok(()),
            _ => Err(bad("expected punctuation")),
        }
    }
    fn ident(&mut self) -> Result<String, StoreError> {
        match self.next() {
            Some(Tok::Word(w)) => Ok(w),
            _ => Err(bad("expected identifier")),
        }
    }
    fn param(&mut self) -> Result<(), StoreError> {
        match self.next() {
            Some(Tok::Param) => Ok(()),
            _ => Err(bad("expected ?")),
        }
    }
    fn ident_list(&mut self) -> Result<Vec<String>, StoreError> {
        let mut v = vec![self.ident()?];
        while matches!(self.peek(), Some(Tok::Punct(","))) {
            self.i += 1;
            v.push(self.ident()?);
        }
        Ok(v)
    }
    fn done(&self) -> Result<(), StoreError> {
        if self.i == self.t.len() {
            Ok(())
        } else {
            Err(bad("trailing tokens"))
        }
    }
    fn where_clause(&mut self) -> Result<Vec<Cond>, StoreError> {
        let mut out = Vec::new();
        if !self.is_kw("where") {
            return Ok(out);
        }
        self.i += 1;
        loop {
            let col = self.ident()?;
            let op = match self.next() {
                Some(Tok::Punct("=")) => CmpOp::Eq,
                Some(Tok::Punct("!=")) => CmpOp::Ne,
                Some(Tok::Punct("<")) => CmpOp::Lt,
                Some(Tok::Punct("<=")) => CmpOp::Le,
                Some(Tok::Punct(">")) => CmpOp::Gt,
                Some(Tok::Punct(">=")) => CmpOp::Ge,
                _ => return Err(bad("expected comparison")),
            };
            self.param()?;
            out.push(Cond { col, op });
            if !self.is_kw("and") {
                return Ok(out);
            }
            self.i += 1;
        }
    }
}

/// Parse one statement.
pub(crate) fn parse(sql: &str) -> Result<Stmt, StoreError> {
    let mut p = P {
        t: tokenize(sql)?,
        i: 0,
    };
    let stmt = match p.next() {
        Some(Tok::Word(w)) if w == "create" => parse_create(&mut p)?,
        Some(Tok::Word(w)) if w == "insert" => {
            let replace = p.is_kw("or");
            if replace {
                p.i += 1;
                p.kw("replace")?;
            }
            p.kw("into")?;
            let table = p.ident()?;
            p.punct("(")?;
            let cols = p.ident_list()?;
            p.punct(")")?;
            p.kw("values")?;
            p.punct("(")?;
            for k in 0..cols.len() {
                if k > 0 {
                    p.punct(",")?;
                }
                p.param()?;
            }
            p.punct(")")?;
            Stmt::Insert {
                table,
                replace,
                cols,
            }
        }
        Some(Tok::Word(w)) if w == "update" => {
            let table = p.ident()?;
            p.kw("set")?;
            let mut sets = Vec::new();
            loop {
                sets.push(p.ident()?);
                p.punct("=")?;
                p.param()?;
                if !matches!(p.peek(), Some(Tok::Punct(","))) {
                    break;
                }
                p.i += 1;
            }
            let wh = p.where_clause()?;
            Stmt::Update { table, sets, wh }
        }
        Some(Tok::Word(w)) if w == "delete" => {
            p.kw("from")?;
            let table = p.ident()?;
            let wh = p.where_clause()?;
            Stmt::Delete { table, wh }
        }
        Some(Tok::Word(w)) if w == "select" => {
            let cols = p.ident_list()?;
            p.kw("from")?;
            let table = p.ident()?;
            let wh = p.where_clause()?;
            let mut order = None;
            if p.is_kw("order") {
                p.i += 1;
                p.kw("by")?;
                let col = p.ident()?;
                let desc = if p.is_kw("desc") {
                    p.i += 1;
                    true
                } else {
                    if p.is_kw("asc") {
                        p.i += 1;
                    }
                    false
                };
                order = Some((col, desc));
            }
            let limit = p.is_kw("limit");
            if limit {
                p.i += 1;
                p.param()?;
            }
            Stmt::Select {
                table,
                cols,
                wh,
                order,
                limit,
            }
        }
        _ => return Err(bad("unsupported statement")),
    };
    p.done()?;
    Ok(stmt)
}

fn parse_create(p: &mut P) -> Result<Stmt, StoreError> {
    if p.is_kw("index") {
        p.i += 1;
        p.kw("if")?;
        p.kw("not")?;
        p.kw("exists")?;
        p.ident()?;
        p.kw("on")?;
        let table = p.ident()?;
        p.punct("(")?;
        let cols = p.ident_list()?;
        p.punct(")")?;
        return Ok(Stmt::Index { table, cols });
    }
    p.kw("table")?;
    p.kw("if")?;
    p.kw("not")?;
    p.kw("exists")?;
    let table = p.ident()?;
    p.punct("(")?;
    let mut cols = Vec::new();
    let mut pk = Vec::new();
    loop {
        if p.is_kw("primary") {
            p.i += 1;
            p.kw("key")?;
            p.punct("(")?;
            pk = p.ident_list()?;
            p.punct(")")?;
        } else {
            let col = p.ident()?;
            // Skip type and constraints up to the next top-level `,` / `)`.
            while let Some(t) = p.peek() {
                match t {
                    Tok::Punct(",") | Tok::Punct(")") => break,
                    Tok::Word(w) if w == "primary" => {
                        p.i += 1;
                        p.kw("key")?;
                        pk = vec![col.clone()];
                    }
                    Tok::Word(_) | Tok::Num => p.i += 1,
                    _ => return Err(bad("unsupported column constraint")),
                }
            }
            cols.push(col);
        }
        match p.next() {
            Some(Tok::Punct(",")) => continue,
            Some(Tok::Punct(")")) => break,
            _ => return Err(bad("expected , or )")),
        }
    }
    if pk.iter().any(|k| !cols.contains(k)) {
        return Err(bad("primary key names an unknown column"));
    }
    Ok(Stmt::Create { table, cols, pk })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema;

    #[test]
    fn every_schema_constant_parses() {
        for s in schema::SHARD_SCHEMA.iter().chain(schema::LEDGER_SCHEMA) {
            assert!(
                matches!(parse(s).unwrap(), Stmt::Create { .. } | Stmt::Index { .. }),
                "{s}"
            );
        }
        for s in [
            schema::META_SELECT_ALL,
            schema::META_PUT,
            schema::META_DELETE_ALL,
            schema::VEC_PUT,
            schema::VEC_DELETE,
            schema::VEC_PAGE,
            schema::VEC_DELETE_ALL,
            schema::OPS_APPEND,
            schema::OPS_PAGE,
            schema::OPS_DELETE_ALL,
            schema::FILTER_PUT,
            schema::FILTER_DELETE_ID,
            schema::FILTER_DELETE_ALL,
            schema::LMETA_SELECT_ALL,
            schema::LMETA_PUT,
            schema::MEMBER_SELECT_ALL,
            schema::MEMBER_INSERT,
            schema::MEMBER_PUT,
            schema::MEMBER_DELETE,
            schema::CATALOG_SELECT_ALL,
            schema::CATALOG_INSERT,
            schema::CATALOG_SET_STATE,
            schema::IDEM_SELECT,
            schema::IDEM_PUT,
            schema::IDEM_EXPIRED,
            schema::IDEM_DELETE,
            schema::OPS_PRUNE,
        ] {
            parse(s).unwrap_or_else(|e| panic!("{s}: {e}"));
        }
    }

    #[test]
    fn pk_detection() {
        match parse(schema::SHARD_SCHEMA[3]).unwrap() {
            Stmt::Create { pk, .. } => assert_eq!(pk, vec!["key", "value", "id"]),
            _ => unreachable!(),
        }
        match parse(schema::SHARD_SCHEMA[1]).unwrap() {
            Stmt::Create { pk, cols, .. } => {
                assert_eq!(pk, vec!["id"]);
                assert_eq!(cols.len(), 8);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn rejects_unsupported_sql() {
        for s in [
            "SELECT COUNT(*) FROM t",
            "SELECT a FROM t WHERE a = 1",
            "DROP TABLE t",
            "SELECT a FROM t; DELETE FROM t",
            "INSERT INTO t (a) SELECT a FROM u",
            "SELECT a FROM t WHERE a = ? OR b = ?",
        ] {
            assert!(parse(s).is_err(), "{s}");
        }
    }
}
