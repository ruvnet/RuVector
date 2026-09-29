//! Cypher over a tenant graph (ADR-351 §3 rv-graph, §10 budgets): parse
//! with rvlite's pure-Rust engine, classify (`MATCH`/`RETURN`/`WITH` only
//! → read, anything else → write), estimate the work *before* executing,
//! and bound the result.
//!
//! Budgets (§10 "Cypher ≤ 10k steps, ≤ 1k rows"): the executor has no step
//! counter to interrupt and Workers' clock does not advance during CPU
//! work, so cost is estimated from the AST and the graph's indexes: each
//! `MATCH` pattern enumerates its candidate nodes (a label's index, or
//! every node) and, for a relationship pattern, their incident edges. Over
//! [`MAX_STEPS`] → `413 budget_exceeded` before anything runs; more than
//! [`MAX_ROWS`] result rows → `413` (a mutating query's effect is then
//! discarded by the caller).
//!
//! Every refusal says what failed ([`Refusal`], `graph_cypher_dx`).

use crate::graph_cypher_dx::{self as dx, Refusal};
use ruvector_edge_store::{ErrorCode, OpError};
use rvlite::cypher::ast::{Direction, NodePattern, Pattern, Query, Statement};
use rvlite::cypher::executor::{ContextValue, ExecutionError};
use rvlite::cypher::graph_store::Value as CValue;
use rvlite::cypher::{parse_cypher, Executor, PropertyGraph};
use serde_json::{json, Map, Value as Json};

/// Longest query text.
pub const MAX_QUERY_BYTES: usize = 4096;
/// Estimated enumeration steps per query.
pub const MAX_STEPS: u64 = 10_000;
/// Result rows per query.
pub const MAX_ROWS: usize = 1_000;

/// A parsed, classified query.
pub struct Parsed {
    /// The AST.
    pub ast: Query,
    /// Only `MATCH` / `RETURN` / `WITH`.
    pub read_only: bool,
}

/// Deepest `(` / `[` / `{` nesting a query may use. rvlite's recursive-
/// descent parser spends about ten stack frames per level and has no depth
/// limit of its own: at 1 MB of stack (the wasm32 default) a ~500-level
/// query overflows it.
pub const MAX_NESTING: usize = 32;

/// Lexical pre-check before rvlite sees the text (it also runs pre-auth,
/// in `graph_routes::extra_class`). Mirroring rvlite's lexer (`'` / `"`
/// strings with backslash escapes, backtick names, `//` line comments), it
/// refuses (`400`) a `$` outside them and nesting deeper than
/// [`MAX_NESTING`].
///
/// `$` is refused because rvlite@c6ece78 routes it to `scan_identifier`,
/// which does not consume it: `tokenize` then pushes empty tokens forever
/// until the isolate runs out of memory. Parameters are unsupported anyway.
pub fn precheck(query: &str) -> Result<(), Refusal> {
    let b = query.as_bytes();
    let (mut i, mut depth) = (0usize, 0usize);
    while i < b.len() {
        match b[i] {
            q @ (b'"' | b'\'') => {
                i += 1;
                while i < b.len() && b[i] != q {
                    i += if b[i] == b'\\' { 2 } else { 1 };
                }
            }
            b'`' => {
                i += 1;
                while i < b.len() && b[i] != b'`' {
                    i += 1;
                }
            }
            b'/' if b.get(i + 1) == Some(&b'/') => {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            }
            b'$' => {
                return Err(Refusal::invalid(format!(
                    "unsupported: query parameters ($ at byte {i}); inline the value"
                )))
            }
            b'(' | b'[' | b'{' => {
                depth += 1;
                if depth > MAX_NESTING {
                    return Err(Refusal::invalid(format!(
                        "query nests ( [ {{ deeper than {MAX_NESTING} levels (at byte {i})"
                    )));
                }
            }
            b')' | b']' | b'}' => depth = depth.saturating_sub(1),
            _ => {}
        }
        i += 1;
    }
    Ok(())
}

/// Parse and classify (`400` on syntax errors, parameters or deep nesting;
/// `413` on oversized text); unaliased `RETURN` items are named by their
/// expression text ([`dx::name_columns`]).
pub fn parse(query: &str) -> Result<Parsed, Refusal> {
    if query.len() > MAX_QUERY_BYTES {
        return Err(Refusal::new(
            OpError::new(ErrorCode::PayloadTooLarge, "cypher query too long"),
            format!(
                "query is {} bytes, over the {MAX_QUERY_BYTES}-byte limit",
                query.len()
            ),
        ));
    }
    precheck(query)?;
    let mut ast = parse_cypher(query).map_err(|e| Refusal::invalid(dx::parse_detail(&e)))?;
    if ast.statements.is_empty() {
        return Err(Refusal::invalid("empty cypher query"));
    }
    dx::name_columns(&mut ast);
    let read_only = ast.is_read_only();
    Ok(Parsed { ast, read_only })
}

fn budget() -> Refusal {
    Refusal::new(
        OpError::new(ErrorCode::BudgetExceeded, "cypher step budget"),
        format!("query exceeds the {MAX_STEPS}-step budget (estimated from the MATCH patterns and graph size)"),
    )
}

fn node_candidates(g: &PropertyGraph, p: &NodePattern) -> u64 {
    if p.labels.is_empty() {
        g.stats().node_count as u64
    } else {
        p.labels
            .iter()
            .map(|l| g.find_nodes_by_label(l).len() as u64)
            .sum()
    }
}

/// Estimated steps of `ast` over `g` (saturating; stops counting past the
/// budget). Unsupported patterns are `400` here, before execution.
pub fn estimate(g: &PropertyGraph, ast: &Query) -> Result<u64, Refusal> {
    let mut steps = 0u64;
    for st in &ast.statements {
        let pats = match st {
            Statement::Match(m) => &m.patterns,
            Statement::Create(c) => {
                steps = steps.saturating_add(c.patterns.len() as u64);
                continue;
            }
            _ => continue,
        };
        for p in pats {
            steps = steps.saturating_add(match p {
                Pattern::Node(n) => node_candidates(g, n),
                Pattern::Relationship(r) => {
                    if r.range.is_some() {
                        return Err(Refusal::invalid(
                            "unsupported: variable-length relationships ([*a..b])",
                        ));
                    }
                    let from = node_candidates(g, &r.from);
                    let edges = g.stats().edge_count as u64;
                    let fan = match r.direction {
                        Direction::Undirected => edges.saturating_mul(2),
                        _ => edges,
                    };
                    from.saturating_add(fan)
                }
                _ => {
                    return Err(Refusal::invalid(
                        "unsupported: this MATCH pattern (e.g. named paths or hyperedges)",
                    ))
                }
            });
            if steps > MAX_STEPS {
                return Err(budget());
            }
        }
    }
    Ok(steps)
}

/// The `RETURN` columns of `ast` (alias, variable, else the expression
/// text — as [`dx::name_columns`] names them).
fn return_columns(ast: &Query) -> Vec<String> {
    let mut cols = Vec::new();
    for st in &ast.statements {
        if let Statement::Return(r) = st {
            for (i, item) in r.items.iter().enumerate() {
                let name = item
                    .alias
                    .clone()
                    .unwrap_or_else(|| match &item.expression {
                        rvlite::cypher::Expression::Variable(v) => v.clone(),
                        e => dx::column_name(e, i),
                    });
                if !cols.contains(&name) {
                    cols.push(name);
                }
            }
        }
    }
    cols
}

/// Run `parsed` on `g` within the budgets; the result as public JSON.
///
/// rvlite evaluates `RETURN` against an empty binding set when `MATCH`
/// found nothing and fails with `VariableNotFound` for the pattern's own
/// variable; a read-only query failing that way on a variable its `MATCH`
/// binds has matched nothing, so it answers zero rows.
pub fn run(g: &mut PropertyGraph, parsed: &Parsed) -> Result<(Json, u64), Refusal> {
    let steps = estimate(g, &parsed.ast)?;
    let res = match Executor::new(g).execute(&parsed.ast) {
        Ok(res) => res,
        Err(ExecutionError::VariableNotFound(v))
            if parsed.read_only && dx::bound_vars(&parsed.ast, false).contains(&v.as_str()) =>
        {
            let columns = return_columns(&parsed.ast);
            return Ok((json!({ "columns": columns, "rows": [] }), steps));
        }
        Err(e) => {
            return Err(Refusal::new(
                OpError::invalid("cypher execution error"),
                dx::exec_detail(&e, &parsed.ast),
            ))
        }
    };
    if res.rows.len() > MAX_ROWS {
        return Err(Refusal::new(
            OpError::new(ErrorCode::BudgetExceeded, "cypher row budget"),
            format!(
                "query returns more than {MAX_ROWS} rows; narrow the MATCH (LIMIT is not applied)"
            ),
        ));
    }
    let rows: Vec<Json> = res
        .rows
        .iter()
        .map(|r| {
            let mut m = Map::new();
            for c in &res.columns {
                m.insert(c.clone(), r.get(c).map_or(Json::Null, ctx_json));
            }
            Json::Object(m)
        })
        .collect();
    Ok((json!({ "columns": res.columns, "rows": rows }), steps))
}

/// Plain JSON of a Cypher value.
pub fn value_json(v: &CValue) -> Json {
    match v {
        CValue::Null => Json::Null,
        CValue::Boolean(b) => json!(b),
        CValue::Integer(i) => json!(i),
        CValue::Float(f) => json!(f),
        CValue::String(s) => json!(s),
        CValue::List(l) => Json::Array(l.iter().map(value_json).collect()),
        CValue::Map(m) => Json::Object(m.iter().map(|(k, v)| (k.clone(), value_json(v))).collect()),
    }
}

fn props(p: &std::collections::HashMap<String, CValue>) -> Json {
    let mut keys: Vec<&String> = p.keys().collect();
    keys.sort();
    Json::Object(
        keys.into_iter()
            .map(|k| (k.clone(), value_json(&p[k])))
            .collect(),
    )
}

fn ctx_json(v: &ContextValue) -> Json {
    match v {
        ContextValue::Node(n) => {
            json!({ "id": n.id, "labels": n.labels, "properties": props(&n.properties) })
        }
        ContextValue::Edge(e) => json!({
            "id": e.id, "from": e.from, "to": e.to, "type": e.edge_type,
            "properties": props(&e.properties),
        }),
        ContextValue::Value(v) => value_json(v),
        ContextValue::List(l) => Json::Array(l.iter().map(ctx_json).collect()),
        ContextValue::Map(m) => {
            Json::Object(m.iter().map(|(k, v)| (k.clone(), ctx_json(v))).collect())
        }
    }
}
