//! Cypher developer experience (ADR-351 §3 rv-graph): say *why* a query
//! was refused, and name unaliased `RETURN` columns the way openCypher
//! does.
//!
//! - [`Refusal`] is an [`OpError`] (whose `code` and status clients match
//!   on, unchanged) plus an optional explanation. The explanation crosses
//!   the `GraphStore` wire in `WireErr::detail` and replaces the problem /
//!   tool-error `detail`; it is built here from fixed phrases, the caller's
//!   own query (a variable name, a token) and positions — never from
//!   rvlite's `Debug` AST dumps, other tenants' data or internal paths —
//!   and [`clamp`]ed to [`MAX_DETAIL`] chars at every render.
//! - rvlite names every unaliased non-variable `RETURN` item `?column?`
//!   and keeps one column per name, so `RETURN n.id, n.name` lost
//!   `n.name`. [`name_columns`] aliases such items with their expression
//!   text (`n.name`, `count(n)`) before execution.

use ruvector_edge_store::OpError;
use rvlite::cypher::ast::{
    AggregationFunction as Agg, BinaryOperator as Bin, Pattern, Query, Statement,
    UnaryOperator as Un,
};
use rvlite::cypher::{ExecutionError, Expression as E, ParseError};

/// Longest explanation a client sees.
pub const MAX_DETAIL: usize = 200;
/// Longest generated column name.
pub const MAX_COLUMN: usize = 64;
/// Deepest expression rendered into a column name.
const MAX_RENDER_DEPTH: usize = 12;

/// A refused graph call: the stable error and, for Cypher, what failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    /// Code, status and static detail.
    pub op: OpError,
    /// The explanation replacing `op.detail`, when there is one.
    pub detail: Option<String>,
}

impl Refusal {
    /// `op` explained by `detail`.
    pub fn new(op: OpError, detail: impl Into<String>) -> Self {
        Refusal {
            op,
            detail: Some(detail.into()),
        }
    }

    /// `invalid_request` explained by `detail`.
    pub fn invalid(detail: impl Into<String>) -> Self {
        Refusal::new(OpError::invalid("invalid cypher"), detail)
    }

    /// The detail to show, bounded.
    pub fn shown(&self) -> String {
        clamp(self.detail.as_deref().unwrap_or(self.op.detail))
    }
}

impl From<OpError> for Refusal {
    fn from(op: OpError) -> Self {
        Refusal { op, detail: None }
    }
}

impl From<Refusal> for OpError {
    fn from(r: Refusal) -> Self {
        r.op
    }
}

/// Control characters dropped, at most [`MAX_DETAIL`] chars (`…` marks a
/// cut).
pub fn clamp(s: &str) -> String {
    cut(s, MAX_DETAIL)
}

fn cut(s: &str, max: usize) -> String {
    let clean: Vec<char> = s.chars().filter(|c| !c.is_control()).collect();
    if clean.len() <= max {
        return clean.into_iter().collect();
    }
    let mut out: String = clean[..max - 1].iter().collect();
    out.push('…');
    out
}

/// A parse failure, with its position.
pub fn parse_detail(e: &ParseError) -> String {
    match e {
        ParseError::UnexpectedToken {
            expected,
            found,
            line,
            column,
        } => format!(
            "parse error at line {line}, column {column}: expected {}, found {}",
            cut(expected, 40),
            cut(found, 40)
        ),
        ParseError::UnexpectedEof => "parse error: unexpected end of query".into(),
        other => format!("parse error: {other}"),
    }
}

/// Variables bound by the patterns of `create` (true) or `MATCH` (false)
/// statements.
pub fn bound_vars(ast: &Query, create: bool) -> Vec<&str> {
    fn walk<'a>(p: &'a Pattern, out: &mut Vec<&'a str>) {
        match p {
            Pattern::Node(n) => out.extend(n.variable.as_deref()),
            Pattern::Relationship(r) => {
                out.extend(r.from.variable.as_deref());
                out.extend(r.variable.as_deref());
                walk(&r.to, out);
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    for st in &ast.statements {
        let pats = match (st, create) {
            (Statement::Create(c), true) => &c.patterns,
            (Statement::Match(m), false) => &m.patterns,
            _ => continue,
        };
        pats.iter().for_each(|p| walk(p, &mut out));
    }
    out
}

/// The first word of `s` after `prefix` (a variant name in rvlite's
/// message), if any.
fn word_after<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    let rest = s.split_once(prefix)?.1;
    let end = rest
        .find(|c: char| !c.is_ascii_alphanumeric())
        .unwrap_or(rest.len());
    (end > 0).then(|| &rest[..end])
}

fn unsupported(msg: &str) -> String {
    if let Some(f) = word_after(msg, "Aggregation { function: ") {
        return format!(
            "unsupported: aggregation {}() (aggregate functions are not implemented)",
            f.to_ascii_lowercase()
        );
    }
    if let Some(kind) = word_after(msg, "Expression ") {
        let what = match kind {
            "BinaryOp" => "operator expressions (e.g. n.x + 1) outside WHERE",
            "UnaryOp" => "unary operators outside WHERE",
            "FunctionCall" => "function calls",
            "Case" => "CASE expressions",
            "PatternPredicate" => "pattern predicates",
            "List" | "Map" => "list / map expressions here",
            _ => "this expression",
        };
        return format!("unsupported: {what}");
    }
    if let Some(kind) = word_after(msg, "Statement ") {
        return format!("unsupported: {} clause", kind.to_ascii_uppercase());
    }
    if msg.contains("Pattern type not yet supported") {
        return "unsupported: this MATCH pattern (e.g. named paths or hyperedges)".into();
    }
    "unsupported cypher operation".into()
}

/// An execution failure of `ast`.
pub fn exec_detail(e: &ExecutionError, ast: &Query) -> String {
    match e {
        ExecutionError::VariableNotFound(v) => {
            let v = cut(v, 40);
            if bound_vars(ast, true).contains(&v.as_str()) {
                format!(
                    "unsupported: RETURN of variables bound by CREATE (`{v}`); \
                     run the CREATE, then MATCH it in a second query"
                )
            } else {
                format!("unknown variable `{v}`")
            }
        }
        ExecutionError::UnsupportedOperation(m) => unsupported(m),
        ExecutionError::TypeError(m) => format!("type error: {}", cut(m, 120)),
        ExecutionError::ExecutionError(m) => format!("execution error: {}", cut(m, 120)),
        ExecutionError::GraphError(g) => format!("graph error: {}", cut(&g.to_string(), 120)),
    }
}

/// Alias every unaliased, non-variable `RETURN` item with its expression
/// text (openCypher's column naming); explicit aliases are unchanged.
pub fn name_columns(ast: &mut Query) {
    for st in &mut ast.statements {
        let Statement::Return(r) = st else { continue };
        for (i, item) in r.items.iter_mut().enumerate() {
            if item.alias.is_none() && !matches!(item.expression, E::Variable(_)) {
                item.alias = Some(column_name(&item.expression, i));
            }
        }
    }
}

/// The column name of `e` (item `index`): its text, at most
/// [`MAX_COLUMN`] chars.
pub fn column_name(e: &E, index: usize) -> String {
    let mut s = String::new();
    render(e, &mut s, 0);
    if s.is_empty() {
        return format!("column_{index}");
    }
    cut(&s, MAX_COLUMN)
}

fn bin(op: Bin) -> &'static str {
    match op {
        Bin::Add => "+",
        Bin::Subtract => "-",
        Bin::Multiply => "*",
        Bin::Divide => "/",
        Bin::Modulo => "%",
        Bin::Power => "^",
        Bin::Equal => "=",
        Bin::NotEqual => "<>",
        Bin::LessThan => "<",
        Bin::LessThanOrEqual => "<=",
        Bin::GreaterThan => ">",
        Bin::GreaterThanOrEqual => ">=",
        Bin::And => "AND",
        Bin::Or => "OR",
        Bin::Xor => "XOR",
        Bin::Contains => "CONTAINS",
        Bin::StartsWith => "STARTS WITH",
        Bin::EndsWith => "ENDS WITH",
        Bin::Matches => "=~",
        Bin::In => "IN",
        Bin::Is => "IS",
        Bin::IsNot => "IS NOT",
    }
}

fn agg(f: Agg) -> &'static str {
    match f {
        Agg::Count => "count",
        Agg::Sum => "sum",
        Agg::Avg => "avg",
        Agg::Min => "min",
        Agg::Max => "max",
        Agg::Collect => "collect",
        Agg::StdDev => "stDev",
        Agg::StdDevP => "stDevP",
        Agg::Percentile => "percentileDisc",
    }
}

fn list(items: &[&E], out: &mut String, depth: usize) {
    for (i, e) in items.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        render(e, out, depth);
    }
}

/// Canonical text of `e` (there are no source spans). Stops at
/// [`MAX_RENDER_DEPTH`] or once past [`MAX_COLUMN`]; `CASE` and pattern
/// predicates render as nothing (the caller then uses `column_<i>`).
fn render(e: &E, out: &mut String, depth: usize) {
    if depth > MAX_RENDER_DEPTH || out.len() > MAX_COLUMN {
        out.push('…');
        return;
    }
    let d = depth + 1;
    match e {
        E::Integer(i) => out.push_str(&i.to_string()),
        E::Float(f) => out.push_str(&f.to_string()),
        E::String(s) => {
            out.push('\'');
            out.push_str(&cut(s, MAX_COLUMN));
            out.push('\'');
        }
        E::Boolean(b) => out.push_str(if *b { "true" } else { "false" }),
        E::Null => out.push_str("null"),
        E::Variable(v) => out.push_str(v),
        E::Property { object, property } => {
            render(object, out, d);
            out.push('.');
            out.push_str(property);
        }
        E::List(items) => {
            out.push('[');
            list(&items.iter().collect::<Vec<_>>(), out, d);
            out.push(']');
        }
        E::Map(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(k);
                out.push_str(": ");
                render(&m[k], out, d);
            }
            out.push('}');
        }
        E::BinaryOp { left, op, right } => {
            render(left, out, d);
            out.push(' ');
            out.push_str(bin(*op));
            out.push(' ');
            render(right, out, d);
        }
        E::UnaryOp { op, operand } => match op {
            Un::Not => {
                out.push_str("NOT ");
                render(operand, out, d);
            }
            Un::Minus | Un::Plus => {
                out.push(if *op == Un::Minus { '-' } else { '+' });
                render(operand, out, d);
            }
            Un::IsNull | Un::IsNotNull => {
                render(operand, out, d);
                out.push_str(if *op == Un::IsNull {
                    " IS NULL"
                } else {
                    " IS NOT NULL"
                });
            }
        },
        E::FunctionCall { name, args } => {
            out.push_str(name);
            out.push('(');
            list(&args.iter().collect::<Vec<_>>(), out, d);
            out.push(')');
        }
        E::Aggregation {
            function,
            expression,
            distinct,
        } => {
            out.push_str(agg(*function));
            out.push('(');
            if *distinct {
                out.push_str("DISTINCT ");
            }
            render(expression, out, d);
            out.push(')');
        }
        E::PatternPredicate(_) | E::Case { .. } => {}
    }
}
