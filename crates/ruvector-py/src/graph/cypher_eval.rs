//! Cypher expression evaluator.
//!
//! Ported from `crates/ruvector-graph-node/src/cypher_exec.rs` @
//! `cebb38bc4` (logic unchanged). That module is a NAPI binding crate's
//! source file, but this half of it has no NAPI-specific types — it only
//! imports from `ruvector_graph::{cypher::ast, GraphDB}` and `std`, which
//! is exactly what made the port possible. The one deliberate rename:
//! the original's `pub enum Bound` is `Binding` here, purely so this file
//! can never be confused with `pyo3::Bound` — nothing in this module
//! touches pyo3 at all, by design, so `graph.rs` (the only sibling that
//! does) stays the sole pyo3-aware file in this slice.
//!
//! Lowers both Cypher literals and stored `PropertyValue`s into one
//! `EvalValue` domain so comparison has exactly one set of rules to
//! follow, then evaluates `WHERE`/inline-property expressions against a
//! row's variable bindings.

use std::collections::HashMap;

use ruvector_graph::cypher::ast::{BinaryOperator, Expression, UnaryOperator};
use ruvector_graph::{Edge, Node, PropertyValue};

/// A value in the expression evaluator's own domain.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum EvalValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<EvalValue>),
}

impl EvalValue {
    pub(super) fn truthy(&self) -> bool {
        matches!(self, EvalValue::Bool(true))
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            EvalValue::Int(i) => Some(*i as f64),
            EvalValue::Float(f) => Some(*f),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            EvalValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

impl From<&PropertyValue> for EvalValue {
    fn from(value: &PropertyValue) -> Self {
        match value {
            PropertyValue::Null => EvalValue::Null,
            PropertyValue::Boolean(b) => EvalValue::Bool(*b),
            PropertyValue::Integer(i) => EvalValue::Int(*i),
            PropertyValue::Float(f) => EvalValue::Float(*f),
            PropertyValue::String(s) => EvalValue::Str(s.clone()),
            PropertyValue::Array(items) | PropertyValue::List(items) => {
                EvalValue::List(items.iter().map(EvalValue::from).collect())
            }
            PropertyValue::FloatArray(items) => {
                EvalValue::List(items.iter().map(|f| EvalValue::Float(*f as f64)).collect())
            }
            // A map has no ordering or equality semantics we can honour here;
            // treating it as Null makes every comparison against it false
            // rather than accidentally true.
            PropertyValue::Map(_) => EvalValue::Null,
        }
    }
}

/// What a pattern variable is bound to for the duration of one candidate row.
#[derive(Debug, Clone)]
pub(super) enum Binding {
    Node(Node),
    Edge(Edge),
}

pub(super) type Bindings = HashMap<String, Binding>;

/// Resolve `<variable>.<property>` against the bound entity.
///
/// `id` is resolved from the entity's identity field when no stored property
/// shadows it. That is the whole point of the exercise: `MATCH (n) WHERE
/// n.id = '...'` is the point-lookup shape the original port (ruvnet/
/// ruvector#879) calls out, and in this data model the id lives beside the
/// property bag rather than inside it.
pub(super) fn lookup_property(bound: &Binding, property: &str) -> EvalValue {
    match bound {
        Binding::Node(node) => {
            if let Some(value) = node.properties.get(property) {
                return EvalValue::from(value);
            }
            match property {
                "id" => EvalValue::Str(node.id.clone()),
                "labels" => EvalValue::List(
                    node.labels
                        .iter()
                        .map(|l| EvalValue::Str(l.name.clone()))
                        .collect(),
                ),
                _ => EvalValue::Null,
            }
        }
        Binding::Edge(edge) => {
            if let Some(value) = edge.properties.get(property) {
                return EvalValue::from(value);
            }
            match property {
                "id" => EvalValue::Str(edge.id.clone()),
                "from" | "source" => EvalValue::Str(edge.from.clone()),
                "to" | "target" => EvalValue::Str(edge.to.clone()),
                "type" => EvalValue::Str(edge.edge_type.clone()),
                _ => EvalValue::Null,
            }
        }
    }
}

/// Evaluate an expression to a value. Unresolvable references yield `Null`,
/// which makes every downstream comparison false — Cypher's own rule.
pub(super) fn eval(expr: &Expression, bindings: &Bindings) -> EvalValue {
    match expr {
        Expression::Integer(i) => EvalValue::Int(*i),
        Expression::Float(f) => EvalValue::Float(*f),
        Expression::String(s) => EvalValue::Str(s.clone()),
        Expression::Boolean(b) => EvalValue::Bool(*b),
        Expression::Null => EvalValue::Null,
        Expression::List(items) => {
            EvalValue::List(items.iter().map(|i| eval(i, bindings)).collect())
        }
        Expression::Variable(name) => match bindings.get(name) {
            // A bare variable in a predicate position is only meaningful as an
            // existence check; comparing it directly is not supported.
            Some(_) => EvalValue::Bool(true),
            None => EvalValue::Null,
        },
        Expression::Property { object, property } => {
            let Expression::Variable(name) = object.as_ref() else {
                return EvalValue::Null;
            };
            match bindings.get(name) {
                Some(bound) => lookup_property(bound, property),
                None => EvalValue::Null,
            }
        }
        Expression::UnaryOp { op, operand } => {
            let value = eval(operand, bindings);
            match op {
                UnaryOperator::Not => EvalValue::Bool(!value.truthy()),
                UnaryOperator::Minus => match value {
                    EvalValue::Int(i) => EvalValue::Int(-i),
                    EvalValue::Float(f) => EvalValue::Float(-f),
                    _ => EvalValue::Null,
                },
                UnaryOperator::Plus => value,
                UnaryOperator::IsNull => EvalValue::Bool(matches!(value, EvalValue::Null)),
                UnaryOperator::IsNotNull => EvalValue::Bool(!matches!(value, EvalValue::Null)),
            }
        }
        Expression::BinaryOp { left, op, right } => eval_binary(left, *op, right, bindings),
        // Functions, aggregations, CASE and pattern predicates are out of
        // scope for this executor.
        _ => EvalValue::Null,
    }
}

fn eval_binary(
    left: &Expression,
    op: BinaryOperator,
    right: &Expression,
    bindings: &Bindings,
) -> EvalValue {
    // Short-circuit the logical operators before evaluating both sides.
    match op {
        BinaryOperator::And => {
            return EvalValue::Bool(eval(left, bindings).truthy() && eval(right, bindings).truthy())
        }
        BinaryOperator::Or => {
            return EvalValue::Bool(eval(left, bindings).truthy() || eval(right, bindings).truthy())
        }
        BinaryOperator::Xor => {
            return EvalValue::Bool(eval(left, bindings).truthy() != eval(right, bindings).truthy())
        }
        _ => {}
    }

    let l = eval(left, bindings);
    let r = eval(right, bindings);

    match op {
        BinaryOperator::Equal => EvalValue::Bool(values_equal(&l, &r)),
        BinaryOperator::NotEqual => EvalValue::Bool(!values_equal(&l, &r)),
        BinaryOperator::LessThan
        | BinaryOperator::LessThanOrEqual
        | BinaryOperator::GreaterThan
        | BinaryOperator::GreaterThanOrEqual => EvalValue::Bool(compare(&l, &r, op)),
        BinaryOperator::Contains => match (l.as_str(), r.as_str()) {
            (Some(hay), Some(needle)) => EvalValue::Bool(hay.contains(needle)),
            _ => EvalValue::Bool(false),
        },
        BinaryOperator::StartsWith => match (l.as_str(), r.as_str()) {
            (Some(hay), Some(needle)) => EvalValue::Bool(hay.starts_with(needle)),
            _ => EvalValue::Bool(false),
        },
        BinaryOperator::EndsWith => match (l.as_str(), r.as_str()) {
            (Some(hay), Some(needle)) => EvalValue::Bool(hay.ends_with(needle)),
            _ => EvalValue::Bool(false),
        },
        BinaryOperator::In => match r {
            EvalValue::List(items) => {
                EvalValue::Bool(items.iter().any(|item| values_equal(&l, item)))
            }
            _ => EvalValue::Bool(false),
        },
        // `IS NULL` / `IS NOT NULL`: the parser puts NULL on the right.
        BinaryOperator::Is => EvalValue::Bool(matches!(l, EvalValue::Null)),
        BinaryOperator::IsNot => EvalValue::Bool(!matches!(l, EvalValue::Null)),
        BinaryOperator::Add => arith(&l, &r, op),
        BinaryOperator::Subtract => arith(&l, &r, op),
        BinaryOperator::Multiply => arith(&l, &r, op),
        BinaryOperator::Divide => arith(&l, &r, op),
        BinaryOperator::Modulo => arith(&l, &r, op),
        BinaryOperator::Power => arith(&l, &r, op),
        // No regex engine is linked into this crate; `=~` is reported as
        // unsupported by the caller rather than quietly matching nothing.
        BinaryOperator::Matches => EvalValue::Null,
        BinaryOperator::And | BinaryOperator::Or | BinaryOperator::Xor => unreachable!(),
    }
}

pub(super) fn values_equal(l: &EvalValue, r: &EvalValue) -> bool {
    match (l, r) {
        (EvalValue::Null, _) | (_, EvalValue::Null) => false,
        (EvalValue::Str(a), EvalValue::Str(b)) => a == b,
        (EvalValue::Bool(a), EvalValue::Bool(b)) => a == b,
        (EvalValue::List(a), EvalValue::List(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| values_equal(x, y))
        }
        // Numbers compare across Int/Float rather than by representation, so
        // `WHERE n.age = 30` matches a stored 30.0.
        _ => match (l.as_f64(), r.as_f64()) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        },
    }
}

fn compare(l: &EvalValue, r: &EvalValue, op: BinaryOperator) -> bool {
    let ordering = match (l, r) {
        (EvalValue::Str(a), EvalValue::Str(b)) => a.as_str().partial_cmp(b.as_str()),
        _ => match (l.as_f64(), r.as_f64()) {
            // `partial_cmp` on a NaN operand yields None, which falls through
            // to `false` below — never a panic.
            (Some(a), Some(b)) => a.partial_cmp(&b),
            _ => None,
        },
    };
    let Some(ordering) = ordering else {
        return false;
    };
    match op {
        BinaryOperator::LessThan => ordering.is_lt(),
        BinaryOperator::LessThanOrEqual => ordering.is_le(),
        BinaryOperator::GreaterThan => ordering.is_gt(),
        BinaryOperator::GreaterThanOrEqual => ordering.is_ge(),
        _ => false,
    }
}

fn arith(l: &EvalValue, r: &EvalValue, op: BinaryOperator) -> EvalValue {
    // String concatenation is the one non-numeric `+`.
    if let (BinaryOperator::Add, Some(a), Some(b)) = (op, l.as_str(), r.as_str()) {
        return EvalValue::Str(format!("{a}{b}"));
    }
    let (Some(a), Some(b)) = (l.as_f64(), r.as_f64()) else {
        return EvalValue::Null;
    };
    let both_int = matches!(l, EvalValue::Int(_)) && matches!(r, EvalValue::Int(_));
    let result = match op {
        BinaryOperator::Add => a + b,
        BinaryOperator::Subtract => a - b,
        BinaryOperator::Multiply => a * b,
        BinaryOperator::Divide => {
            if b == 0.0 {
                return EvalValue::Null;
            }
            a / b
        }
        BinaryOperator::Modulo => {
            if b == 0.0 {
                return EvalValue::Null;
            }
            a % b
        }
        BinaryOperator::Power => a.powf(b),
        _ => return EvalValue::Null,
    };
    if both_int && op != BinaryOperator::Divide && result.fract() == 0.0 {
        EvalValue::Int(result as i64)
    } else {
        EvalValue::Float(result)
    }
}
