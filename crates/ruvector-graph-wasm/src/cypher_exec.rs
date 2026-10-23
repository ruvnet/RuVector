//! Cypher evaluation over graph-wasm's own node and edge maps.
//!
//! Parsing uses `ruvector_graph::cypher::parse_cypher`. The `executor` module of
//! `ruvector-graph` is a physical-operator skeleton (its scan operators return
//! `Ok(None)`) and nothing lowers a Cypher AST into its plans, so this module walks
//! the parsed AST directly over the same storage that `createNode` / `createEdge`
//! write to. There is one copy of the data and nothing to keep in sync.
//!
//! Supported: MATCH (comma-separated patterns, chained hops, all three directions,
//! label and property filters, OPTIONAL MATCH), WHERE, CREATE (nodes and
//! relationships, including between variables bound by an earlier MATCH), RETURN
//! with DISTINCT, aggregation (count, sum, avg, min, max, collect), ORDER BY, SKIP
//! and LIMIT.
//!
//! Not supported (returns an error instead of a wrong answer): WITH, MERGE, SET,
//! REMOVE, DELETE, variable-length relationships, named paths, hyperedge patterns,
//! pattern predicates, regular expressions.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};

use parking_lot::MutexGuard;
use ruvector_core::advanced::hypergraph::HypergraphIndex;
use ruvector_graph::cypher::ast::{
    AggregationFunction, BinaryOperator, CreateClause, Direction, Expression, MatchClause,
    NodePattern, Pattern, PropertyMap, ReturnClause, Statement, UnaryOperator,
};
use ruvector_graph::cypher::parse_cypher;
use serde_json::{json, Number, Value as Json};
use uuid::Uuid;

use crate::types::{Edge, EdgeId, Node, NodeId};

/// Every lock the graph needs, held together for the duration of one query.
///
/// Built by `GraphDB::lock_store`, which acquires them in one fixed order. Code
/// that holds a `Store` must not call a method that locks the graph again.
pub(crate) struct Store<'a> {
    pub nodes: MutexGuard<'a, HashMap<NodeId, Node>>,
    pub edges: MutexGuard<'a, HashMap<EdgeId, Edge>>,
    pub labels_index: MutexGuard<'a, HashMap<String, Vec<NodeId>>>,
    pub edge_types_index: MutexGuard<'a, HashMap<String, Vec<EdgeId>>>,
    pub node_edges_out: MutexGuard<'a, HashMap<NodeId, Vec<EdgeId>>>,
    pub node_edges_in: MutexGuard<'a, HashMap<NodeId, Vec<EdgeId>>>,
    pub hypergraph: MutexGuard<'a, HypergraphIndex>,
}

impl Store<'_> {
    /// Inserts a node. This is the single insertion path for `createNode` and for
    /// Cypher CREATE. An `embedding` property also goes into the hypergraph index.
    pub(crate) fn insert_node(
        &mut self,
        labels: Vec<String>,
        properties: HashMap<String, Json>,
    ) -> NodeId {
        let id = Uuid::new_v4().to_string();
        let embedding = properties
            .get("embedding")
            .and_then(|v| serde_json::from_value::<Vec<f32>>(v.clone()).ok());

        for label in &labels {
            self.labels_index
                .entry(label.clone())
                .or_default()
                .push(id.clone());
        }
        if let Some(emb) = &embedding {
            self.hypergraph.add_entity(id.clone(), emb.clone());
        }
        self.nodes.insert(
            id.clone(),
            Node {
                id: id.clone(),
                labels,
                properties,
                embedding,
            },
        );
        self.node_edges_out.insert(id.clone(), Vec::new());
        self.node_edges_in.insert(id.clone(), Vec::new());
        id
    }

    /// Inserts a directed edge. Both endpoints must already exist.
    pub(crate) fn insert_edge(
        &mut self,
        from: NodeId,
        to: NodeId,
        edge_type: String,
        properties: HashMap<String, Json>,
    ) -> Result<EdgeId, String> {
        if !self.nodes.contains_key(&from) {
            return Err(format!("Node {} not found", from));
        }
        if !self.nodes.contains_key(&to) {
            return Err(format!("Node {} not found", to));
        }
        let id = Uuid::new_v4().to_string();
        self.edge_types_index
            .entry(edge_type.clone())
            .or_default()
            .push(id.clone());
        self.node_edges_out
            .entry(from.clone())
            .or_default()
            .push(id.clone());
        self.node_edges_in
            .entry(to.clone())
            .or_default()
            .push(id.clone());
        self.edges.insert(
            id.clone(),
            Edge {
                id: id.clone(),
                from,
                to,
                edge_type,
                properties,
            },
        );
        Ok(id)
    }
}

/// Runtime value during evaluation. `Node` and `Edge` refer into the store.
#[derive(Debug, Clone)]
enum Val {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<Val>),
    Map(BTreeMap<String, Val>),
    Node(NodeId),
    Edge(EdgeId),
}

/// Variable bindings for one row.
type Env = HashMap<String, Val>;

/// Rows returned by RETURN, already projected and ordered.
#[derive(Default)]
pub(crate) struct Output {
    /// Each row is a list of (column name, value) pairs in RETURN order.
    pub rows: Vec<Vec<(String, Json)>>,
    /// Distinct node IDs that appear as bare values in the rows.
    pub node_ids: Vec<NodeId>,
    /// Distinct edge IDs that appear as bare values in the rows.
    pub edge_ids: Vec<EdgeId>,
}

/// Parses and runs a Cypher query against the store.
pub(crate) fn execute(st: &mut Store, cypher: &str) -> Result<Output, String> {
    let query = parse_cypher(cypher).map_err(|e| format!("Cypher parse error: {}", e))?;

    let mut rows: Vec<Env> = vec![Env::new()];
    let mut output: Option<Output> = None;

    for statement in &query.statements {
        if output.is_some() {
            return Err("Clauses after RETURN are not supported".to_string());
        }
        rows = match statement {
            Statement::Match(clause) => match_clause(st, clause, rows)?,
            Statement::Create(clause) => {
                let mut next = Vec::with_capacity(rows.len());
                for row in rows {
                    next.push(create_clause(st, clause, row)?);
                }
                next
            }
            Statement::Return(clause) => {
                output = Some(project(st, rows, clause)?);
                Vec::new()
            }
            Statement::With(_) => return Err("WITH is not supported yet".to_string()),
            Statement::Merge(_) => return Err("MERGE is not supported yet".to_string()),
            Statement::Set(_) | Statement::Remove(_) | Statement::Delete(_) => {
                return Err("SET, REMOVE and DELETE are not supported yet".to_string())
            }
        };
    }

    // A query without RETURN still applies its writes and returns no rows.
    Ok(output.unwrap_or_default())
}

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

fn from_json(j: &Json) -> Val {
    match j {
        Json::Null => Val::Null,
        Json::Bool(b) => Val::Bool(*b),
        Json::Number(n) => match n.as_i64() {
            Some(i) => Val::Int(i),
            None => Val::Float(n.as_f64().unwrap_or(f64::NAN)),
        },
        Json::String(s) => Val::Str(s.clone()),
        Json::Array(items) => Val::List(items.iter().map(from_json).collect()),
        Json::Object(map) => Val::Map(map.iter().map(|(k, v)| (k.clone(), from_json(v))).collect()),
    }
}

/// Converts a value for return to JavaScript. Nodes and edges become objects.
fn to_json(st: &Store, v: &Val) -> Json {
    match v {
        Val::Null => Json::Null,
        Val::Bool(b) => Json::Bool(*b),
        Val::Int(i) => Json::from(*i),
        Val::Float(f) => Number::from_f64(*f).map_or(Json::Null, Json::Number),
        Val::Str(s) => Json::String(s.clone()),
        Val::List(items) => Json::Array(items.iter().map(|x| to_json(st, x)).collect()),
        Val::Map(map) => Json::Object(
            map.iter()
                .map(|(k, x)| (k.clone(), to_json(st, x)))
                .collect(),
        ),
        Val::Node(id) => match st.nodes.get(id) {
            Some(n) => json!({ "id": n.id, "labels": n.labels, "properties": n.properties }),
            None => Json::Null,
        },
        Val::Edge(id) => match st.edges.get(id) {
            Some(e) => json!({
                "id": e.id,
                "type": e.edge_type,
                "from": e.from,
                "to": e.to,
                "properties": e.properties,
            }),
            None => Json::Null,
        },
    }
}

/// Converts a value for storage in a property. Graph elements are rejected.
fn to_property(v: &Val) -> Result<Json, String> {
    Ok(match v {
        Val::Null => Json::Null,
        Val::Bool(b) => Json::Bool(*b),
        Val::Int(i) => Json::from(*i),
        Val::Float(f) => Number::from_f64(*f).map_or(Json::Null, Json::Number),
        Val::Str(s) => Json::String(s.clone()),
        Val::List(items) => Json::Array(
            items
                .iter()
                .map(to_property)
                .collect::<Result<Vec<_>, _>>()?,
        ),
        Val::Map(map) => {
            let mut out = serde_json::Map::new();
            for (k, x) in map {
                out.insert(k.clone(), to_property(x)?);
            }
            Json::Object(out)
        }
        Val::Node(_) | Val::Edge(_) => {
            return Err("Nodes and relationships cannot be stored as property values".to_string())
        }
    })
}

fn as_f64(v: &Val) -> Option<f64> {
    match v {
        Val::Int(i) => Some(*i as f64),
        Val::Float(f) => Some(*f),
        _ => None,
    }
}

/// Cypher equality for non-null operands. Null never equals anything here;
/// callers that need three-valued logic check for Null first.
fn values_equal(a: &Val, b: &Val) -> bool {
    match (a, b) {
        (Val::Null, _) | (_, Val::Null) => false,
        (Val::Int(x), Val::Int(y)) => x == y,
        (Val::Bool(x), Val::Bool(y)) => x == y,
        (Val::Str(x), Val::Str(y)) => x == y,
        (Val::Node(x), Val::Node(y)) => x == y,
        (Val::Edge(x), Val::Edge(y)) => x == y,
        (Val::List(x), Val::List(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| values_equal(p, q))
        }
        (Val::Map(x), Val::Map(y)) => {
            x.len() == y.len()
                && x.iter()
                    .all(|(k, v)| y.get(k).is_some_and(|w| values_equal(v, w)))
        }
        _ => match (as_f64(a), as_f64(b)) {
            (Some(x), Some(y)) => x == y,
            _ => false,
        },
    }
}

/// Ordering for `<`, `>` and friends. None means the values are not comparable.
fn compare(a: &Val, b: &Val) -> Option<Ordering> {
    match (a, b) {
        (Val::Int(x), Val::Int(y)) => Some(x.cmp(y)),
        (Val::Str(x), Val::Str(y)) => Some(x.cmp(y)),
        (Val::Bool(x), Val::Bool(y)) => Some(x.cmp(y)),
        _ => match (as_f64(a), as_f64(b)) {
            (Some(x), Some(y)) => x.partial_cmp(&y),
            _ => None,
        },
    }
}

fn type_rank(v: &Val) -> u8 {
    match v {
        Val::Bool(_) => 0,
        Val::Int(_) | Val::Float(_) => 1,
        Val::Str(_) => 2,
        _ => 3,
    }
}

/// Total order for ORDER BY. Nulls sort last ascending, as in Cypher.
fn sort_cmp(a: &Val, b: &Val) -> Ordering {
    match (matches!(a, Val::Null), matches!(b, Val::Null)) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        _ => compare(a, b).unwrap_or_else(|| type_rank(a).cmp(&type_rank(b))),
    }
}

/// Key used for DISTINCT and grouping. Nodes and edges key on their IDs.
fn group_key(v: &Val) -> String {
    format!("{:?}", v)
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

fn eval(st: &Store, env: &Env, e: &Expression) -> Result<Val, String> {
    Ok(match e {
        Expression::Integer(i) => Val::Int(*i),
        Expression::Float(f) => Val::Float(*f),
        Expression::String(s) => Val::Str(s.clone()),
        Expression::Boolean(b) => Val::Bool(*b),
        Expression::Null => Val::Null,
        Expression::Variable(name) => env
            .get(name)
            .cloned()
            .ok_or_else(|| format!("Unknown variable `{}`", name))?,
        Expression::Property { object, property } => {
            property_of(st, &eval(st, env, object)?, property)?
        }
        Expression::List(items) => Val::List(
            items
                .iter()
                .map(|x| eval(st, env, x))
                .collect::<Result<_, _>>()?,
        ),
        Expression::Map(map) => {
            let mut out = BTreeMap::new();
            for (k, x) in map {
                out.insert(k.clone(), eval(st, env, x)?);
            }
            Val::Map(out)
        }
        Expression::UnaryOp { op, operand } => unary(*op, eval(st, env, operand)?)?,
        Expression::BinaryOp { left, op, right } => match op {
            BinaryOperator::And | BinaryOperator::Or | BinaryOperator::Xor => {
                logic(*op, &eval(st, env, left)?, &eval(st, env, right)?)?
            }
            _ => binary(*op, eval(st, env, left)?, eval(st, env, right)?)?,
        },
        Expression::FunctionCall { name, args } => call_function(st, env, name, args)?,
        Expression::Aggregation { .. } => {
            return Err(
                "Aggregation functions are only supported as top-level RETURN items".to_string(),
            )
        }
        Expression::PatternPredicate(_) => {
            return Err("Pattern predicates are not supported yet".to_string())
        }
        Expression::Case {
            expression,
            alternatives,
            default,
        } => {
            let subject = match expression {
                Some(x) => Some(eval(st, env, x)?),
                None => None,
            };
            for (cond, result) in alternatives {
                let hit = match &subject {
                    Some(s) => values_equal(s, &eval(st, env, cond)?),
                    None => matches!(eval(st, env, cond)?, Val::Bool(true)),
                };
                if hit {
                    return eval(st, env, result);
                }
            }
            match default {
                Some(d) => eval(st, env, d)?,
                None => Val::Null,
            }
        }
    })
}

fn property_of(st: &Store, target: &Val, prop: &str) -> Result<Val, String> {
    match target {
        Val::Node(id) => Ok(st
            .nodes
            .get(id)
            .and_then(|n| n.properties.get(prop))
            .map_or(Val::Null, from_json)),
        Val::Edge(id) => Ok(st
            .edges
            .get(id)
            .and_then(|e| e.properties.get(prop))
            .map_or(Val::Null, from_json)),
        Val::Map(map) => Ok(map.get(prop).cloned().unwrap_or(Val::Null)),
        Val::Null => Ok(Val::Null),
        _ => Err(format!(
            "Cannot read property `{}` of a non-node value",
            prop
        )),
    }
}

fn unary(op: UnaryOperator, v: Val) -> Result<Val, String> {
    Ok(match op {
        UnaryOperator::IsNull => Val::Bool(matches!(v, Val::Null)),
        UnaryOperator::IsNotNull => Val::Bool(!matches!(v, Val::Null)),
        UnaryOperator::Not => match v {
            Val::Null => Val::Null,
            Val::Bool(b) => Val::Bool(!b),
            _ => return Err("NOT applied to a non-boolean value".to_string()),
        },
        UnaryOperator::Minus => match v {
            Val::Null => Val::Null,
            Val::Int(i) => i.checked_neg().map_or(Val::Float(-(i as f64)), Val::Int),
            Val::Float(f) => Val::Float(-f),
            _ => return Err("Unary minus applied to a non-numeric value".to_string()),
        },
        UnaryOperator::Plus => match v {
            Val::Null | Val::Int(_) | Val::Float(_) => v,
            _ => return Err("Unary plus applied to a non-numeric value".to_string()),
        },
    })
}

/// Three-valued AND, OR and XOR.
fn logic(op: BinaryOperator, l: &Val, r: &Val) -> Result<Val, String> {
    let tri = |v: &Val| -> Result<Option<bool>, String> {
        match v {
            Val::Null => Ok(None),
            Val::Bool(b) => Ok(Some(*b)),
            _ => Err("Boolean operator applied to a non-boolean value".to_string()),
        }
    };
    let (a, b) = (tri(l)?, tri(r)?);
    Ok(match op {
        BinaryOperator::And => match (a, b) {
            (Some(false), _) | (_, Some(false)) => Val::Bool(false),
            (Some(true), Some(true)) => Val::Bool(true),
            _ => Val::Null,
        },
        BinaryOperator::Or => match (a, b) {
            (Some(true), _) | (_, Some(true)) => Val::Bool(true),
            (Some(false), Some(false)) => Val::Bool(false),
            _ => Val::Null,
        },
        _ => match (a, b) {
            (Some(x), Some(y)) => Val::Bool(x != y),
            _ => Val::Null,
        },
    })
}

fn arith(
    l: &Val,
    r: &Val,
    int_op: fn(i64, i64) -> Option<i64>,
    float_op: fn(f64, f64) -> f64,
) -> Result<Val, String> {
    match (l, r) {
        (Val::Int(a), Val::Int(b)) => Ok(match int_op(*a, *b) {
            Some(v) => Val::Int(v),
            None => Val::Float(float_op(*a as f64, *b as f64)),
        }),
        _ => match (as_f64(l), as_f64(r)) {
            (Some(a), Some(b)) => Ok(Val::Float(float_op(a, b))),
            _ => Err("Arithmetic applied to a non-numeric value".to_string()),
        },
    }
}

fn binary(op: BinaryOperator, l: Val, r: Val) -> Result<Val, String> {
    // IS and IS NOT compare nullness, so they are the only operators that see nulls.
    match op {
        BinaryOperator::Is => {
            return Ok(Val::Bool(matches!(l, Val::Null) == matches!(r, Val::Null)))
        }
        BinaryOperator::IsNot => {
            return Ok(Val::Bool(matches!(l, Val::Null) != matches!(r, Val::Null)))
        }
        _ => {}
    }
    if matches!(l, Val::Null) || matches!(r, Val::Null) {
        return Ok(Val::Null);
    }
    Ok(match op {
        BinaryOperator::Equal => Val::Bool(values_equal(&l, &r)),
        BinaryOperator::NotEqual => Val::Bool(!values_equal(&l, &r)),
        BinaryOperator::LessThan => ordering(&l, &r, |o| o == Ordering::Less)?,
        BinaryOperator::LessThanOrEqual => ordering(&l, &r, |o| o != Ordering::Greater)?,
        BinaryOperator::GreaterThan => ordering(&l, &r, |o| o == Ordering::Greater)?,
        BinaryOperator::GreaterThanOrEqual => ordering(&l, &r, |o| o != Ordering::Less)?,
        BinaryOperator::Add => match (&l, &r) {
            (Val::Str(a), Val::Str(b)) => Val::Str(format!("{}{}", a, b)),
            _ => arith(&l, &r, i64::checked_add, |a, b| a + b)?,
        },
        BinaryOperator::Subtract => arith(&l, &r, i64::checked_sub, |a, b| a - b)?,
        BinaryOperator::Multiply => arith(&l, &r, i64::checked_mul, |a, b| a * b)?,
        BinaryOperator::Divide => match (&l, &r) {
            (Val::Int(_), Val::Int(0)) => return Err("Division by zero".to_string()),
            (Val::Int(a), Val::Int(b)) => Val::Int(a.wrapping_div(*b)),
            _ => arith(&l, &r, |_, _| None, |a, b| a / b)?,
        },
        BinaryOperator::Modulo => match (&l, &r) {
            (Val::Int(_), Val::Int(0)) => return Err("Modulo by zero".to_string()),
            (Val::Int(a), Val::Int(b)) => Val::Int(a.wrapping_rem(*b)),
            _ => arith(&l, &r, |_, _| None, |a, b| a % b)?,
        },
        BinaryOperator::Power => match (as_f64(&l), as_f64(&r)) {
            (Some(a), Some(b)) => Val::Float(a.powf(b)),
            _ => return Err("Exponentiation applied to a non-numeric value".to_string()),
        },
        BinaryOperator::Contains | BinaryOperator::StartsWith | BinaryOperator::EndsWith => {
            match (&l, &r) {
                (Val::Str(a), Val::Str(b)) => Val::Bool(match op {
                    BinaryOperator::Contains => a.contains(b.as_str()),
                    BinaryOperator::StartsWith => a.starts_with(b.as_str()),
                    _ => a.ends_with(b.as_str()),
                }),
                _ => Val::Null,
            }
        }
        BinaryOperator::In => match &r {
            Val::List(items) => Val::Bool(items.iter().any(|x| values_equal(&l, x))),
            _ => return Err("IN expects a list on the right-hand side".to_string()),
        },
        BinaryOperator::Matches => {
            return Err("Regular expression operators are not supported yet".to_string())
        }
        BinaryOperator::And | BinaryOperator::Or | BinaryOperator::Xor => return logic(op, &l, &r),
        BinaryOperator::Is | BinaryOperator::IsNot => unreachable!("handled above"),
    })
}

fn ordering(l: &Val, r: &Val, pred: fn(Ordering) -> bool) -> Result<Val, String> {
    match compare(l, r) {
        Some(o) => Ok(Val::Bool(pred(o))),
        None => Ok(Val::Null),
    }
}

fn call_function(st: &Store, env: &Env, name: &str, args: &[Expression]) -> Result<Val, String> {
    let lname = name.to_ascii_lowercase();
    match (lname.as_str(), args) {
        ("id", [arg]) => match eval(st, env, arg)? {
            Val::Node(id) | Val::Edge(id) => Ok(Val::Str(id)),
            Val::Null => Ok(Val::Null),
            _ => Err("id() expects a node or relationship".to_string()),
        },
        ("labels", [arg]) => match eval(st, env, arg)? {
            Val::Node(id) => Ok(Val::List(
                st.nodes
                    .get(&id)
                    .map(|n| n.labels.iter().cloned().map(Val::Str).collect())
                    .unwrap_or_default(),
            )),
            Val::Null => Ok(Val::Null),
            _ => Err("labels() expects a node".to_string()),
        },
        ("type", [arg]) => match eval(st, env, arg)? {
            Val::Edge(id) => Ok(st
                .edges
                .get(&id)
                .map_or(Val::Null, |e| Val::Str(e.edge_type.clone()))),
            Val::Null => Ok(Val::Null),
            _ => Err("type() expects a relationship".to_string()),
        },
        ("size", [arg]) => match eval(st, env, arg)? {
            Val::Str(s) => Ok(Val::Int(s.chars().count() as i64)),
            Val::List(items) => Ok(Val::Int(items.len() as i64)),
            Val::Null => Ok(Val::Null),
            _ => Err("size() expects a string or list".to_string()),
        },
        ("coalesce", _) => {
            for arg in args {
                let v = eval(st, env, arg)?;
                if !matches!(v, Val::Null) {
                    return Ok(v);
                }
            }
            Ok(Val::Null)
        }
        _ => Err(format!("Unsupported function `{}`", name)),
    }
}

// ---------------------------------------------------------------------------
// MATCH
// ---------------------------------------------------------------------------

/// One hop of a flattened pattern. `nodes[i] -hops[i]- nodes[i + 1]`.
struct Hop<'p> {
    variable: Option<&'p str>,
    rel_type: Option<&'p str>,
    properties: Option<&'p PropertyMap>,
    direction: Direction,
}

#[derive(Default)]
struct Path<'p> {
    nodes: Vec<&'p NodePattern>,
    hops: Vec<Hop<'p>>,
}

/// Flattens a parsed pattern. `(a)-[r]->(b)-[s]->(c)` parses as a relationship
/// whose `to` is another relationship starting at `b`. This unwraps that nesting.
fn flatten<'p>(pattern: &'p Pattern, path: &mut Path<'p>) -> Result<(), String> {
    match pattern {
        Pattern::Node(node) => path.nodes.push(node),
        Pattern::Relationship(rel) => {
            if rel.range.is_some() {
                return Err("Variable-length relationships are not supported yet".to_string());
            }
            // The first node of a chained hop is the node the previous hop ended on.
            if path.nodes.len() == path.hops.len() {
                path.nodes.push(&rel.from);
            }
            path.hops.push(Hop {
                variable: rel.variable.as_deref(),
                rel_type: rel.rel_type.as_deref(),
                properties: rel.properties.as_ref(),
                direction: rel.direction,
            });
            flatten(&rel.to, path)?;
        }
        Pattern::Path(_) => return Err("Named paths are not supported yet".to_string()),
        Pattern::Hyperedge(_) => return Err("Hyperedge patterns are not supported yet".to_string()),
    }
    Ok(())
}

fn flatten_all<'p>(patterns: &'p [Pattern]) -> Result<Vec<Path<'p>>, String> {
    patterns
        .iter()
        .map(|p| {
            let mut path = Path::default();
            flatten(p, &mut path)?;
            Ok(path)
        })
        .collect()
}

fn with_binding(env: &Env, var: Option<&str>, val: Val) -> Env {
    let mut out = env.clone();
    if let Some(v) = var {
        out.insert(v.to_string(), val);
    }
    out
}

/// Candidate start nodes. Uses the label index when the pattern has a label.
fn candidate_nodes(st: &Store, pat: &NodePattern) -> Vec<NodeId> {
    match pat.labels.first() {
        Some(label) => {
            let mut seen = HashSet::new();
            st.labels_index
                .get(label)
                .map(|ids| {
                    ids.iter()
                        .filter(|id| seen.insert((*id).clone()))
                        .cloned()
                        .collect()
                })
                .unwrap_or_default()
        }
        None => st.nodes.keys().cloned().collect(),
    }
}

fn props_match(
    st: &Store,
    env: &Env,
    stored: &HashMap<String, Json>,
    wanted: &PropertyMap,
) -> Result<bool, String> {
    for (key, expr) in wanted {
        let want = eval(st, env, expr)?;
        let Some(have) = stored.get(key) else {
            return Ok(false);
        };
        if matches!(want, Val::Null) || !values_equal(&from_json(have), &want) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether node `id` satisfies a node pattern under the current bindings.
fn node_matches(st: &Store, id: &str, pat: &NodePattern, env: &Env) -> Result<bool, String> {
    let Some(node) = st.nodes.get(id) else {
        return Ok(false);
    };
    if let Some(var) = &pat.variable {
        if let Some(bound) = env.get(var) {
            if !matches!(bound, Val::Node(b) if b.as_str() == id) {
                return Ok(false);
            }
        }
    }
    if !pat.labels.iter().all(|l| node.labels.contains(l)) {
        return Ok(false);
    }
    match &pat.properties {
        Some(props) => props_match(st, env, &node.properties, props),
        None => Ok(true),
    }
}

/// Edges leaving `cur` in the given direction, paired with the node at the other end.
fn incident(st: &Store, cur: &str, dir: Direction) -> Vec<(EdgeId, NodeId)> {
    let mut hits = Vec::new();
    if matches!(dir, Direction::Outgoing | Direction::Undirected) {
        for eid in st.node_edges_out.get(cur).into_iter().flatten() {
            if let Some(e) = st.edges.get(eid) {
                hits.push((eid.clone(), e.to.clone()));
            }
        }
    }
    if matches!(dir, Direction::Incoming | Direction::Undirected) {
        for eid in st.node_edges_in.get(cur).into_iter().flatten() {
            if let Some(e) = st.edges.get(eid) {
                // A self-loop is already in the outgoing list.
                if dir == Direction::Undirected && e.from == cur {
                    continue;
                }
                hits.push((eid.clone(), e.from.clone()));
            }
        }
    }
    hits
}

type Partial = (Env, Vec<EdgeId>);

/// Every extension of `env` that matches `path`. `used` lists relationships already
/// bound in this MATCH clause. A relationship may not be matched twice.
fn match_path(st: &Store, path: &Path, env: &Env, used: &[EdgeId]) -> Result<Vec<Partial>, String> {
    let mut out = Vec::new();
    let first = path.nodes[0];
    let starts: Vec<NodeId> = match first.variable.as_ref().and_then(|v| env.get(v)) {
        Some(Val::Node(id)) => vec![id.clone()],
        Some(_) => return Ok(out),
        None => candidate_nodes(st, first),
    };
    for start in starts {
        if node_matches(st, &start, first, env)? {
            let env0 = with_binding(env, first.variable.as_deref(), Val::Node(start.clone()));
            extend(st, path, 0, &start, env0, used.to_vec(), &mut out)?;
        }
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn extend(
    st: &Store,
    path: &Path,
    i: usize,
    cur: &str,
    env: Env,
    used: Vec<EdgeId>,
    out: &mut Vec<Partial>,
) -> Result<(), String> {
    if i == path.hops.len() {
        out.push((env, used));
        return Ok(());
    }
    let hop = &path.hops[i];
    let next = path.nodes[i + 1];

    for (eid, other) in incident(st, cur, hop.direction) {
        if used.contains(&eid) {
            continue;
        }
        let Some(edge) = st.edges.get(&eid) else {
            continue;
        };
        if hop.rel_type.is_some_and(|t| edge.edge_type != t) {
            continue;
        }
        if let Some(wanted) = hop.properties {
            if !props_match(st, &env, &edge.properties, wanted)? {
                continue;
            }
        }
        if let Some(var) = hop.variable {
            if let Some(bound) = env.get(var) {
                if !matches!(bound, Val::Edge(b) if *b == eid) {
                    continue;
                }
            }
        }
        if !node_matches(st, &other, next, &env)? {
            continue;
        }
        let env1 = with_binding(&env, hop.variable, Val::Edge(eid.clone()));
        let env2 = with_binding(&env1, next.variable.as_deref(), Val::Node(other.clone()));
        let mut used2 = used.clone();
        used2.push(eid);
        extend(st, path, i + 1, &other, env2, used2, out)?;
    }
    Ok(())
}

fn match_clause(st: &Store, clause: &MatchClause, rows: Vec<Env>) -> Result<Vec<Env>, String> {
    let paths = flatten_all(&clause.patterns)?;

    // Variables this clause introduces. OPTIONAL MATCH binds them to null on no match.
    let mut new_vars: Vec<String> = Vec::new();
    for path in &paths {
        for node in &path.nodes {
            new_vars.extend(node.variable.clone());
        }
        for hop in &path.hops {
            new_vars.extend(hop.variable.map(str::to_string));
        }
    }

    let mut out = Vec::new();
    for row in rows {
        let mut partial: Vec<Partial> = vec![(row.clone(), Vec::new())];
        for path in &paths {
            let mut next = Vec::new();
            for (env, used) in &partial {
                next.extend(match_path(st, path, env, used)?);
            }
            partial = next;
            if partial.is_empty() {
                break;
            }
        }

        let mut matched = Vec::new();
        for (env, _) in partial {
            if let Some(w) = &clause.where_clause {
                if !matches!(eval(st, &env, &w.condition)?, Val::Bool(true)) {
                    continue;
                }
            }
            matched.push(env);
        }

        if matched.is_empty() && clause.optional {
            let mut env = row;
            for v in &new_vars {
                env.entry(v.clone()).or_insert(Val::Null);
            }
            out.push(env);
        } else {
            out.extend(matched);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// CREATE
// ---------------------------------------------------------------------------

fn create_props(
    st: &Store,
    env: &Env,
    props: Option<&PropertyMap>,
) -> Result<HashMap<String, Json>, String> {
    let mut out = HashMap::new();
    if let Some(map) = props {
        for (key, expr) in map {
            let v = eval(st, env, expr)?;
            // Cypher does not store null properties.
            if !matches!(v, Val::Null) {
                out.insert(key.clone(), to_property(&v)?);
            }
        }
    }
    Ok(out)
}

/// Runs one CREATE clause for one row. Variables already bound by MATCH are reused,
/// so `MATCH (a), (b) CREATE (a)-[:T]->(b)` links existing nodes.
fn create_clause(st: &mut Store, clause: &CreateClause, mut env: Env) -> Result<Env, String> {
    for pattern in &clause.patterns {
        let mut path = Path::default();
        flatten(pattern, &mut path)?;

        let mut ids: Vec<NodeId> = Vec::with_capacity(path.nodes.len());
        for node in &path.nodes {
            let bound = node.variable.as_ref().and_then(|v| env.get(v)).cloned();
            let id = match bound {
                Some(Val::Node(id)) => {
                    if !node.labels.is_empty() || node.properties.is_some() {
                        return Err(format!(
                            "Variable `{}` is already bound; CREATE cannot add labels or properties to it",
                            node.variable.as_deref().unwrap_or("")
                        ));
                    }
                    id
                }
                Some(_) => {
                    return Err(format!(
                        "Variable `{}` is bound to a non-node value",
                        node.variable.as_deref().unwrap_or("")
                    ))
                }
                None => {
                    let props = create_props(st, &env, node.properties.as_ref())?;
                    let id = st.insert_node(node.labels.clone(), props);
                    if let Some(var) = &node.variable {
                        env.insert(var.clone(), Val::Node(id.clone()));
                    }
                    id
                }
            };
            ids.push(id);
        }

        for (i, hop) in path.hops.iter().enumerate() {
            let rel_type = hop
                .rel_type
                .ok_or_else(|| "CREATE requires a relationship type".to_string())?;
            let (from, to) = match hop.direction {
                Direction::Outgoing => (ids[i].clone(), ids[i + 1].clone()),
                Direction::Incoming => (ids[i + 1].clone(), ids[i].clone()),
                Direction::Undirected => {
                    return Err("CREATE requires a relationship direction".to_string())
                }
            };
            if let Some(var) = hop.variable {
                if env.contains_key(var) {
                    return Err(format!("Variable `{}` is already bound", var));
                }
            }
            let props = create_props(st, &env, hop.properties)?;
            let eid = st.insert_edge(from, to, rel_type.to_string(), props)?;
            if let Some(var) = hop.variable {
                env.insert(var.to_string(), Val::Edge(eid));
            }
        }
    }
    Ok(env)
}

// ---------------------------------------------------------------------------
// RETURN
// ---------------------------------------------------------------------------

fn agg_name(f: AggregationFunction) -> &'static str {
    match f {
        AggregationFunction::Count => "count",
        AggregationFunction::Sum => "sum",
        AggregationFunction::Avg => "avg",
        AggregationFunction::Min => "min",
        AggregationFunction::Max => "max",
        AggregationFunction::Collect => "collect",
        AggregationFunction::StdDev => "stdev",
        AggregationFunction::StdDevP => "stdevp",
        AggregationFunction::Percentile => "percentile",
    }
}

/// Column name for an expression without an alias. Mirrors the source text.
fn expr_text(e: &Expression) -> String {
    match e {
        Expression::Variable(v) => v.clone(),
        Expression::Property { object, property } => {
            format!("{}.{}", expr_text(object), property)
        }
        Expression::Integer(i) => i.to_string(),
        Expression::Float(f) => f.to_string(),
        Expression::String(s) => format!("'{}'", s),
        Expression::Boolean(b) => b.to_string(),
        Expression::Null => "null".to_string(),
        Expression::Aggregation {
            function,
            expression,
            distinct,
        } => format!(
            "{}({}{})",
            agg_name(*function),
            if *distinct { "DISTINCT " } else { "" },
            expr_text(expression)
        ),
        Expression::FunctionCall { name, args } => format!(
            "{}({})",
            name,
            args.iter().map(expr_text).collect::<Vec<_>>().join(", ")
        ),
        Expression::BinaryOp { left, op, right } => format!(
            "{} {} {}",
            expr_text(left),
            binary_symbol(*op),
            expr_text(right)
        ),
        Expression::UnaryOp { op, operand } => match op {
            UnaryOperator::Not => format!("NOT {}", expr_text(operand)),
            UnaryOperator::Minus => format!("-{}", expr_text(operand)),
            UnaryOperator::Plus => format!("+{}", expr_text(operand)),
            UnaryOperator::IsNull => format!("{} IS NULL", expr_text(operand)),
            UnaryOperator::IsNotNull => format!("{} IS NOT NULL", expr_text(operand)),
        },
        Expression::List(items) => format!(
            "[{}]",
            items.iter().map(expr_text).collect::<Vec<_>>().join(", ")
        ),
        Expression::Case { .. } => "CASE".to_string(),
        other => format!("{:?}", other),
    }
}

fn binary_symbol(op: BinaryOperator) -> &'static str {
    match op {
        BinaryOperator::Add => "+",
        BinaryOperator::Subtract => "-",
        BinaryOperator::Multiply => "*",
        BinaryOperator::Divide => "/",
        BinaryOperator::Modulo => "%",
        BinaryOperator::Power => "^",
        BinaryOperator::Equal => "=",
        BinaryOperator::NotEqual => "<>",
        BinaryOperator::LessThan => "<",
        BinaryOperator::LessThanOrEqual => "<=",
        BinaryOperator::GreaterThan => ">",
        BinaryOperator::GreaterThanOrEqual => ">=",
        BinaryOperator::And => "AND",
        BinaryOperator::Or => "OR",
        BinaryOperator::Xor => "XOR",
        BinaryOperator::Contains => "CONTAINS",
        BinaryOperator::StartsWith => "STARTS WITH",
        BinaryOperator::EndsWith => "ENDS WITH",
        BinaryOperator::Matches => "=~",
        BinaryOperator::In => "IN",
        BinaryOperator::Is => "IS",
        BinaryOperator::IsNot => "IS NOT",
    }
}

fn is_aggregation(e: &Expression) -> bool {
    matches!(e, Expression::Aggregation { .. })
}

fn aggregate(
    st: &Store,
    group: &[Env],
    function: AggregationFunction,
    expression: &Expression,
    distinct: bool,
) -> Result<Val, String> {
    // count(*) is parsed as count over the marker `*` and counts rows.
    if matches!(expression, Expression::Variable(v) if v == "*") {
        return Ok(Val::Int(group.len() as i64));
    }

    let mut values = Vec::new();
    for env in group {
        let v = eval(st, env, expression)?;
        if !matches!(v, Val::Null) {
            values.push(v);
        }
    }
    if distinct {
        let mut seen = HashSet::new();
        values.retain(|v| seen.insert(group_key(v)));
    }

    Ok(match function {
        AggregationFunction::Count => Val::Int(values.len() as i64),
        AggregationFunction::Collect => Val::List(values),
        AggregationFunction::Sum => {
            if values.iter().all(|v| matches!(v, Val::Int(_))) {
                let mut total: i64 = 0;
                for v in &values {
                    if let Val::Int(i) = v {
                        total = total
                            .checked_add(*i)
                            .ok_or_else(|| "Integer overflow in sum()".to_string())?;
                    }
                }
                Val::Int(total)
            } else {
                let mut total = 0.0;
                for v in &values {
                    total +=
                        as_f64(v).ok_or_else(|| "sum() requires numeric values".to_string())?;
                }
                Val::Float(total)
            }
        }
        AggregationFunction::Avg => {
            if values.is_empty() {
                Val::Null
            } else {
                let mut total = 0.0;
                for v in &values {
                    total +=
                        as_f64(v).ok_or_else(|| "avg() requires numeric values".to_string())?;
                }
                Val::Float(total / values.len() as f64)
            }
        }
        AggregationFunction::Min | AggregationFunction::Max => {
            let want_max = function == AggregationFunction::Max;
            let mut best: Option<Val> = None;
            for v in values {
                best = Some(match best {
                    None => v,
                    Some(b) => {
                        let ord = compare(&v, &b).ok_or_else(|| {
                            "min()/max() requires values of one comparable type".to_string()
                        })?;
                        let take_new = if want_max {
                            ord == Ordering::Greater
                        } else {
                            ord == Ordering::Less
                        };
                        if take_new {
                            v
                        } else {
                            b
                        }
                    }
                });
            }
            best.unwrap_or(Val::Null)
        }
        AggregationFunction::StdDev
        | AggregationFunction::StdDevP
        | AggregationFunction::Percentile => {
            return Err(format!("{}() is not supported yet", agg_name(function)))
        }
    })
}

/// Evaluates an ORDER BY key. A key that names a returned column reads that column.
/// Otherwise it is evaluated against the source row, which only works without aggregation.
fn order_key(
    st: &Store,
    expr: &Expression,
    columns: &[String],
    values: &[Val],
    env: Option<&Env>,
) -> Result<Val, String> {
    let text = expr_text(expr);
    if let Some(idx) = columns.iter().position(|c| *c == text) {
        return Ok(values[idx].clone());
    }
    match env {
        Some(env) => eval(st, env, expr),
        None => Err(format!(
            "ORDER BY `{}` must name a returned column in an aggregating query",
            text
        )),
    }
}

fn eval_limit(st: &Store, expr: &Expression, what: &str) -> Result<usize, String> {
    match eval(st, &Env::new(), expr)? {
        Val::Int(n) if n >= 0 => Ok(n as usize),
        _ => Err(format!("{} must be a non-negative integer", what)),
    }
}

fn project(st: &Store, rows: Vec<Env>, ret: &ReturnClause) -> Result<Output, String> {
    let items = &ret.items;
    let columns: Vec<String> = items
        .iter()
        .map(|it| {
            it.alias
                .clone()
                .unwrap_or_else(|| expr_text(&it.expression))
        })
        .collect();
    let order_items = ret
        .order_by
        .as_ref()
        .map(|o| o.items.as_slice())
        .unwrap_or(&[]);

    // Each projected row carries its values and its ORDER BY keys.
    let mut projected: Vec<(Vec<Val>, Vec<Val>)> = Vec::new();
    let aggregating = items.iter().any(|it| is_aggregation(&it.expression));

    if aggregating {
        // Group by the non-aggregate items. With none, all rows form one group,
        // and that group exists even when there are no rows.
        let key_items: Vec<usize> = (0..items.len())
            .filter(|&i| !is_aggregation(&items[i].expression))
            .collect();
        let mut groups: Vec<Vec<Env>> = Vec::new();
        if key_items.is_empty() {
            groups.push(Vec::new());
        }
        let mut index: HashMap<String, usize> = HashMap::new();
        for env in rows {
            if key_items.is_empty() {
                groups[0].push(env);
                continue;
            }
            let mut key = String::new();
            for &i in &key_items {
                key.push_str(&group_key(&eval(st, &env, &items[i].expression)?));
                key.push('|');
            }
            match index.get(&key) {
                Some(&g) => groups[g].push(env),
                None => {
                    index.insert(key, groups.len());
                    groups.push(vec![env]);
                }
            }
        }

        for group in &groups {
            let mut values = Vec::with_capacity(items.len());
            for item in items {
                values.push(match &item.expression {
                    Expression::Aggregation {
                        function,
                        expression,
                        distinct,
                    } => aggregate(st, group, *function, expression, *distinct)?,
                    other => match group.first() {
                        Some(env) => eval(st, env, other)?,
                        None => Val::Null,
                    },
                });
            }
            let keys = order_items
                .iter()
                .map(|o| order_key(st, &o.expression, &columns, &values, None))
                .collect::<Result<Vec<_>, _>>()?;
            projected.push((values, keys));
        }
    } else {
        for env in &rows {
            let mut values = Vec::with_capacity(items.len());
            for item in items {
                values.push(eval(st, env, &item.expression)?);
            }
            let keys = order_items
                .iter()
                .map(|o| order_key(st, &o.expression, &columns, &values, Some(env)))
                .collect::<Result<Vec<_>, _>>()?;
            projected.push((values, keys));
        }
    }

    if ret.distinct {
        let mut seen = HashSet::new();
        projected.retain(|(values, _)| {
            seen.insert(values.iter().map(group_key).collect::<Vec<_>>().join("|"))
        });
    }

    if !order_items.is_empty() {
        projected.sort_by(|a, b| {
            for (i, item) in order_items.iter().enumerate() {
                let mut ord = sort_cmp(&a.1[i], &b.1[i]);
                if !item.ascending {
                    ord = ord.reverse();
                }
                if ord != Ordering::Equal {
                    return ord;
                }
            }
            Ordering::Equal
        });
    }

    let skip = match &ret.skip {
        Some(e) => eval_limit(st, e, "SKIP")?,
        None => 0,
    };
    let limit = match &ret.limit {
        Some(e) => eval_limit(st, e, "LIMIT")?,
        None => usize::MAX,
    };

    let mut out = Output::default();
    let mut seen_nodes = HashSet::new();
    let mut seen_edges = HashSet::new();
    for (values, _) in projected.into_iter().skip(skip).take(limit) {
        for v in &values {
            match v {
                Val::Node(id) if seen_nodes.insert(id.clone()) => out.node_ids.push(id.clone()),
                Val::Edge(id) if seen_edges.insert(id.clone()) => out.edge_ids.push(id.clone()),
                _ => {}
            }
        }
        let row: Vec<(String, Json)> = columns
            .iter()
            .cloned()
            .zip(values.iter().map(|v| to_json(st, v)))
            .collect();
        out.rows.push(row);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GraphDB;
    use ruvector_graph::cypher::ast::Statement;

    /// The shipment graph from the acceptance test: 2 ports, 5 containers, 8 POs,
    /// 3 customers and 2 DCs.
    ///
    /// Containers C1, C2 and C4 arrive at LA. C3 and C5 arrive at SEA.
    /// PO n ships in C(((n-1) % 5) + 1) with value n * 10000, so POs 1, 2, 4, 6 and 7
    /// are at LA. Customer A orders PO1 and PO2, B orders PO3, PO4 and PO5, and C
    /// orders PO6, PO7 and PO8.
    fn fixture() -> GraphDB {
        let db = GraphDB::new(None).expect("db");
        {
            let mut st = db.lock_store();
            let p = |pairs: Vec<(&str, Json)>| -> HashMap<String, Json> {
                pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
            };
            let la = st.insert_node(vec!["Port".into()], p(vec![("name", json!("LA"))]));
            let sea = st.insert_node(vec!["Port".into()], p(vec![("name", json!("SEA"))]));
            let mut containers = Vec::new();
            for (i, port) in [(1, &la), (2, &la), (3, &sea), (4, &la), (5, &sea)] {
                let c = st.insert_node(
                    vec!["Container".into()],
                    p(vec![("cid", json!(format!("C{}", i)))]),
                );
                st.insert_edge(c.clone(), port.clone(), "ARRIVES_AT".into(), HashMap::new())
                    .unwrap();
                containers.push(c);
            }
            let mut pos = Vec::new();
            for n in 1..=8usize {
                let po = st.insert_node(
                    vec!["PO".into()],
                    p(vec![
                        ("po", json!(format!("PO{}", n))),
                        ("value", json!(n * 10000)),
                    ]),
                );
                st.insert_edge(
                    po.clone(),
                    containers[(n - 1) % 5].clone(),
                    "SHIPS_IN".into(),
                    HashMap::new(),
                )
                .unwrap();
                pos.push(po);
            }
            let customers: Vec<_> = ["A", "B", "C"]
                .iter()
                .map(|name| st.insert_node(vec!["Customer".into()], p(vec![("name", json!(name))])))
                .collect();
            let order_of = [0usize, 0, 1, 1, 1, 2, 2, 2];
            for (n, po) in pos.iter().enumerate() {
                st.insert_edge(
                    customers[order_of[n]].clone(),
                    po.clone(),
                    "ORDERED".into(),
                    HashMap::new(),
                )
                .unwrap();
            }
            for name in ["DC1", "DC2"] {
                let dc = st.insert_node(vec!["DC".into()], p(vec![("name", json!(name))]));
                st.insert_edge(pos[0].clone(), dc, "DELIVERS_TO".into(), HashMap::new())
                    .unwrap();
            }
        }
        db
    }

    fn run(db: &GraphDB, cypher: &str) -> Vec<HashMap<String, Json>> {
        match db.execute_cypher(cypher) {
            Ok(result) => result.data,
            Err(e) => panic!("query failed: {}\n  {}", e, cypher),
        }
    }

    fn col(rows: &[HashMap<String, Json>], name: &str) -> Vec<Json> {
        rows.iter().map(|r| r[name].clone()).collect()
    }

    #[test]
    fn parses_acceptance_queries_into_expected_statements() {
        let cases = [
            ("MATCH (n) RETURN n", vec!["Match", "Return"]),
            (
                "MATCH (c:Container)-[:ARRIVES_AT]->(p:Port {name: 'LA'}) RETURN c",
                vec!["Match", "Return"],
            ),
            (
                "MATCH (o:PO)-[:SHIPS_IN]->(c:Container)-[:ARRIVES_AT]->(p:Port {name: 'LA'}) RETURN o.po, o.value, c.cid",
                vec!["Match", "Return"],
            ),
            (
                "MATCH (cu:Customer)-[:ORDERED]->(o:PO) WHERE o.value > 50000 RETURN cu.name",
                vec!["Match", "Return"],
            ),
            (
                "MATCH (a:Port {name: 'LA'}), (b:Container {cid: 'C1'}) CREATE (b)-[:ARRIVES_AT]->(a)",
                vec!["Match", "Create"],
            ),
        ];
        for (q, kinds) in cases {
            let parsed = parse_cypher(q).unwrap_or_else(|e| panic!("{}: {}", q, e));
            let got: Vec<&str> = parsed
                .statements
                .iter()
                .map(|s| match s {
                    Statement::Match(_) => "Match",
                    Statement::Return(_) => "Return",
                    Statement::Create(_) => "Create",
                    _ => "Other",
                })
                .collect();
            assert_eq!(got, kinds, "{}", q);
        }
        // Keyword-prefixed relationship type must not split into OR + DERED.
        let parsed = parse_cypher("MATCH (a)-[:ORDERED]->(b) RETURN a").unwrap();
        match &parsed.statements[0] {
            Statement::Match(m) => match &m.patterns[0] {
                Pattern::Relationship(r) => assert_eq!(r.rel_type.as_deref(), Some("ORDERED")),
                other => panic!("unexpected pattern {:?}", other),
            },
            other => panic!("unexpected statement {:?}", other),
        }
    }

    #[test]
    fn match_all_nodes() {
        let db = fixture();
        assert_eq!(run(&db, "MATCH (n) RETURN n").len(), 20);
    }

    #[test]
    fn containers_arriving_at_la() {
        let db = fixture();
        let rows = run(
            &db,
            "MATCH (c:Container)-[:ARRIVES_AT]->(p:Port {name: 'LA'}) RETURN c",
        );
        let mut cids: Vec<String> = rows
            .iter()
            .map(|r| r["c"]["properties"]["cid"].as_str().unwrap().to_string())
            .collect();
        cids.sort();
        assert_eq!(cids, vec!["C1", "C2", "C4"]);
    }

    #[test]
    fn three_hop_po_rows_at_la() {
        let db = fixture();
        let rows = run(
            &db,
            "MATCH (o:PO)-[:SHIPS_IN]->(c:Container)-[:ARRIVES_AT]->(p:Port {name: 'LA'}) RETURN o.po, o.value, c.cid",
        );
        let mut got: Vec<(String, i64, String)> = rows
            .iter()
            .map(|r| {
                (
                    r["o.po"].as_str().unwrap().to_string(),
                    r["o.value"].as_i64().unwrap(),
                    r["c.cid"].as_str().unwrap().to_string(),
                )
            })
            .collect();
        got.sort();
        assert_eq!(
            got,
            vec![
                ("PO1".into(), 10000, "C1".into()),
                ("PO2".into(), 20000, "C2".into()),
                ("PO4".into(), 40000, "C4".into()),
                ("PO6".into(), 60000, "C1".into()),
                ("PO7".into(), 70000, "C2".into()),
            ]
        );
    }

    #[test]
    fn customer_hop_and_where_filter() {
        let db = fixture();
        let rows = run(
            &db,
            "MATCH (cu:Customer)-[:ORDERED]->(o:PO)-[:SHIPS_IN]->(c:Container)-[:ARRIVES_AT]->(p:Port {name: 'LA'}) RETURN cu.name, o.po",
        );
        let mut got: Vec<(String, String)> = rows
            .iter()
            .map(|r| {
                (
                    r["cu.name"].as_str().unwrap().to_string(),
                    r["o.po"].as_str().unwrap().to_string(),
                )
            })
            .collect();
        got.sort();
        assert_eq!(
            got,
            vec![
                ("A".into(), "PO1".into()),
                ("A".into(), "PO2".into()),
                ("B".into(), "PO4".into()),
                ("C".into(), "PO6".into()),
                ("C".into(), "PO7".into()),
            ]
        );

        // PO6, PO7 and PO8 are above 50000.
        let rows = run(&db, "MATCH (o:PO) WHERE o.value > 50000 RETURN o.po");
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn aggregation_and_order() {
        let db = fixture();
        let rows = run(
            &db,
            "MATCH (o:PO)-[:SHIPS_IN]->(c:Container)-[:ARRIVES_AT]->(p:Port {name: 'LA'}) RETURN count(o), sum(o.value)",
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["count(o)"], json!(5));
        assert_eq!(rows[0]["sum(o.value)"], json!(200000));

        let rows = run(
            &db,
            "MATCH (o:PO) RETURN o.po ORDER BY o.value DESC LIMIT 2",
        );
        assert_eq!(col(&rows, "o.po"), vec![json!("PO8"), json!("PO7")]);

        let rows = run(
            &db,
            "MATCH (p:Port)<-[:ARRIVES_AT]-(c:Container) RETURN p.name, count(c)",
        );
        let mut counts: Vec<(String, i64)> = rows
            .iter()
            .map(|r| {
                (
                    r["p.name"].as_str().unwrap().to_string(),
                    r["count(c)"].as_i64().unwrap(),
                )
            })
            .collect();
        counts.sort();
        assert_eq!(counts, vec![("LA".into(), 3), ("SEA".into(), 2)]);
    }

    #[test]
    fn optional_match_and_empty_aggregate() {
        let db = fixture();
        let rows = run(&db, "MATCH (o:PO {po: 'NOPE'}) RETURN count(o)");
        assert_eq!(rows[0]["count(o)"], json!(0));
        let rows = run(
            &db,
            "MATCH (c:Container {cid: 'C3'}) OPTIONAL MATCH (c)-[:SHIPS_IN]->(x) RETURN c.cid, x",
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["x"], Json::Null);
    }

    #[test]
    fn create_relationship_between_matched_nodes() {
        let db = fixture();
        let before = run(
            &db,
            "MATCH (c:Container)-[:ARRIVES_AT]->(p:Port {name: 'SEA'}) RETURN c",
        )
        .len();
        assert_eq!(before, 2);
        let out = run(
            &db,
            "MATCH (c:Container {cid: 'C1'}), (p:Port {name: 'SEA'}) CREATE (c)-[:ARRIVES_AT]->(p) RETURN c.cid",
        );
        assert_eq!(out.len(), 1);
        let after = run(
            &db,
            "MATCH (c:Container)-[:ARRIVES_AT]->(p:Port {name: 'SEA'}) RETURN c",
        )
        .len();
        assert_eq!(after, 3);
        // No node was created by the MATCH + CREATE.
        assert_eq!(run(&db, "MATCH (n) RETURN n").len(), 20);
    }

    #[test]
    fn create_nodes_and_edges_in_one_statement() {
        let db = GraphDB::new(None).expect("db");
        let out = run(
            &db,
            "CREATE (a:Person {name: 'Ann'})-[:KNOWS {since: 2020}]->(b:Person {name: 'Bo'}) RETURN a.name",
        );
        assert_eq!(out[0]["a.name"], json!("Ann"));
        let rows = run(
            &db,
            "MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a.name, r.since, b.name",
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["r.since"], json!(2020));
    }

    #[test]
    fn operators_functions_and_column_names() {
        let db = fixture();
        let po_values = |q: &str, col: &str| -> Vec<Json> {
            let mut v: Vec<Json> = run(&db, q).iter().map(|r| r[col].clone()).collect();
            v.sort_by_key(|j| j.to_string());
            v
        };

        // IS NULL / IS NOT NULL. Every PO has a value, so nothing is null.
        assert_eq!(
            run(&db, "MATCH (o:PO) WHERE o.value IS NULL RETURN o").len(),
            0
        );
        assert_eq!(
            run(&db, "MATCH (o:PO) WHERE o.nope IS NOT NULL RETURN o").len(),
            0
        );
        assert_eq!(
            run(&db, "MATCH (o:PO) WHERE o.nope IS NULL RETURN o").len(),
            8
        );

        // IN, CONTAINS, STARTS WITH, ENDS WITH. Native fixture POs are PO1..PO8.
        assert_eq!(
            po_values(
                "MATCH (o:PO) WHERE o.po IN ['PO1', 'PO2'] RETURN o.po AS po",
                "po"
            ),
            vec![json!("PO1"), json!("PO2")]
        );
        assert_eq!(
            po_values(
                "MATCH (o:PO) WHERE o.po CONTAINS '1' RETURN o.po AS po",
                "po"
            ),
            vec![json!("PO1")]
        );
        assert_eq!(
            po_values(
                "MATCH (o:PO) WHERE o.po STARTS WITH 'PO' RETURN o.po AS po",
                "po"
            )
            .len(),
            8
        );
        assert_eq!(
            po_values(
                "MATCH (o:PO) WHERE o.po ENDS WITH '8' RETURN o.po AS po",
                "po"
            ),
            vec![json!("PO8")]
        );

        // CASE, searched and simple. PO7 has value 70000.
        let rows = run(
            &db,
            "MATCH (o:PO {po: 'PO7'}) RETURN CASE WHEN o.value > 50000 THEN 'big' ELSE 'small' END AS size",
        );
        assert_eq!(rows[0]["size"], json!("big"));
        let rows = run(
            &db,
            "MATCH (c:Container {cid: 'C1'}) RETURN CASE c.cid WHEN 'C2' THEN 'two' WHEN 'C1' THEN 'one' END AS word",
        );
        assert_eq!(rows[0]["word"], json!("one"));

        // count(*) counts rows, including when there is no MATCH.
        assert_eq!(
            run(&db, "MATCH (o:PO) RETURN count(*)")[0]["count(*)"],
            json!(8)
        );
        assert_eq!(run(&db, "RETURN count(*)")[0]["count(*)"], json!(1));

        // Default column names mirror the source text.
        let rows = run(&db, "MATCH (o:PO) RETURN o.value * 2 LIMIT 1");
        assert!(rows[0].contains_key("o.value * 2"), "{:?}", rows[0].keys());
        let rows = run(&db, "MATCH (o:PO) RETURN COUNT(o)");
        assert!(rows[0].contains_key("count(o)"));

        // avg, min, max, collect, SKIP, undirected and arrow shorthand
        // Native fixture values are n * 10000 for n in 1..=8.
        let rows = run(
            &db,
            "MATCH (o:PO) RETURN min(o.value) AS lo, max(o.value) AS hi, avg(o.value) AS a",
        );
        assert_eq!(rows[0]["lo"], json!(10000));
        assert_eq!(rows[0]["hi"], json!(80000));
        assert_eq!(rows[0]["a"], json!(45000.0));
        let rows = run(&db, "MATCH (o:PO) RETURN collect(o.po) AS pos");
        assert_eq!(rows[0]["pos"].as_array().unwrap().len(), 8);
        assert_eq!(
            run(&db, "MATCH (o:PO) RETURN o.po ORDER BY o.po SKIP 6").len(),
            2
        );
        assert_eq!(
            run(&db, "MATCH (c:Container)-[:ARRIVES_AT]-(p:Port) RETURN p").len(),
            5
        );
        assert_eq!(run(&db, "MATCH (c:Container)-->(p:Port) RETURN p").len(), 5);
        assert_eq!(run(&db, "MATCH (p:Port)<--(c:Container) RETURN c").len(), 5);

        // Incoming CREATE: (p)<-[:T]-(c) creates an edge from c to p.
        let before = db.lock_store().edges.len();
        run(
            &db,
            "MATCH (p:Port {name: 'SEA'}), (c:Container {cid: 'C1'}) CREATE (p)<-[:ARRIVES_AT]-(c)",
        );
        assert_eq!(db.lock_store().edges.len(), before + 1);
        let rows = run(
            &db,
            "MATCH (c:Container)-[:ARRIVES_AT]->(p:Port {name: 'SEA'}) RETURN c.cid",
        );
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn unsupported_clauses_are_reported() {
        let db = fixture();
        let err = db.execute_cypher("MATCH (n) SET n.x = 1").err().unwrap();
        assert!(err.contains("not supported"), "{}", err);
        let err = db
            .execute_cypher("MATCH (a)-[*1..3]->(b) RETURN a")
            .err()
            .unwrap();
        assert!(err.contains("Variable-length"), "{}", err);
    }
}
