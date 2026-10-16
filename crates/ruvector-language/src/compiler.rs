//! Static type and resource validation independent of parsing.
use crate::{
    error::{Error, Result},
    types::{Effect, Op, Program},
};
use std::collections::BTreeMap;

pub const MAX_STEPS: usize = 1024;
pub const MAX_VECTOR_DIM: usize = 4096;
pub const MAX_DEPTH: usize = 64;
pub const MAX_LIST_ITEMS: usize = 10_000;
pub const MAX_BUDGET: u64 = 1_000_000_000;
pub const MAX_NAME_BYTES: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Type {
    Nodes,
    Observations,
    Derived,
    Predictions,
    Decisions,
}

/// Validate a program, including callers constructing the Rust structs directly.
pub fn check(program: &Program) -> Result<()> {
    if program.version != 1 {
        return Err(Error::new("version", "supported program version is 1"));
    }
    identifier(&program.name, "program name")?;
    if program.budget == 0 || program.budget > MAX_BUDGET {
        return Err(Error::new(
            "budget",
            format!("budget must be between 1 and {MAX_BUDGET}"),
        ));
    }
    if program.steps.is_empty() || program.steps.len() > MAX_STEPS {
        return Err(Error::new(
            "step_limit",
            "program requires between 1 and 1024 steps",
        ));
    }
    // Rust callers bypass the parser; enforce the same total bound without
    // allocating an oversized JSON string or traversing unbounded input lists.
    serde_json::to_writer(SizeLimit(0), program).map_err(|_| {
        Error::new(
            "source_limit",
            "program exceeds 1 MiB JSON representation limit",
        )
    })?;
    require_effect(program, Effect::DecisionEmit)?;
    let mut bindings = BTreeMap::new();
    for step in &program.steps {
        identifier(&step.bind, "binding")?;
        if bindings.contains_key(step.bind.as_str()) {
            return Err(Error::new(
                "duplicate_binding",
                format!("duplicate binding {}", step.bind),
            ));
        }
        if !matches!(step.op, Op::Assess { .. } | Op::Infer { .. }) {
            require_effect(program, Effect::GraphRead)?;
        }
        let kind = match &step.op {
            Op::Nodes { label } => {
                name(label, "node label")?;
                Type::Nodes
            }
            Op::Select { ids } => {
                names(ids, "selected IDs")?;
                Type::Nodes
            }
            Op::Traverse {
                input,
                relation,
                depth,
            } => {
                expect(&bindings, input, &[Type::Nodes])?;
                name(relation, "relation")?;
                bounded(*depth, MAX_DEPTH, "traversal depth")?;
                Type::Nodes
            }
            Op::Similar { input, vector, k } => {
                expect(&bindings, input, &[Type::Nodes])?;
                bounded(vector.len(), MAX_VECTOR_DIM, "vector dimensions")?;
                if vector.iter().any(|x| !x.is_finite()) {
                    return Err(Error::new(
                        "number",
                        "vectors require finite f32 components",
                    ));
                }
                bounded(*k, MAX_LIST_ITEMS, "top k")?;
                Type::Nodes
            }
            Op::Observe {
                input,
                predicate,
                within,
            } => {
                expect(&bindings, input, &[Type::Nodes])?;
                name(predicate, "predicate")?;
                if *within == 0 {
                    return Err(Error::new("range", "observation window must be positive"));
                }
                Type::Observations
            }
            Op::Infer { inputs, rules } => {
                names(inputs, "inference inputs")?;
                names(rules, "inference rules")?;
                for input in inputs {
                    expect(&bindings, input, &[Type::Observations, Type::Derived])?;
                }
                Type::Derived
            }
            Op::Predict {
                input,
                relation,
                model,
                k,
            } => {
                expect(&bindings, input, &[Type::Nodes])?;
                require_effect(program, Effect::ModelPredict)?;
                name(relation, "prediction relation")?;
                name(model, "model")?;
                bounded(*k, MAX_LIST_ITEMS, "top k")?;
                Type::Predictions
            }
            Op::Assess { input } => {
                expect(&bindings, input, &[Type::Observations, Type::Derived])?;
                Type::Decisions
            }
        };
        bindings.insert(step.bind.as_str(), kind);
    }
    expect(
        &bindings,
        &program.emit,
        &[Type::Decisions, Type::Predictions],
    )?;
    Ok(())
}

/// Canonical compact JSON for this version, with deterministic object/effect order.
/// Array order remains significant; no algebraic equivalence is claimed.
pub fn canonical_json(program: &Program) -> Result<String> {
    check(program)?;
    // Struct order is fixed even if a host enables serde_json/preserve_order.
    // Program contains no dynamically ordered object maps.
    serde_json::to_string(program).map_err(|e| Error::new("json_encode", e.to_string()))
}

fn require_effect(program: &Program, effect: Effect) -> Result<()> {
    if !program.effects.contains(&effect) {
        return Err(Error::new(
            "effect",
            format!("missing declared effect {effect:?}"),
        ));
    }
    Ok(())
}
fn expect(bindings: &BTreeMap<&str, Type>, input: &str, allowed: &[Type]) -> Result<()> {
    identifier(input, "input binding")?;
    let kind = bindings
        .get(input)
        .ok_or_else(|| Error::new("reference", format!("unknown or forward reference {input}")))?;
    if !allowed.contains(kind) {
        return Err(Error::new(
            "type",
            format!("binding {input} has type {kind:?}; expected {allowed:?}"),
        ));
    }
    Ok(())
}
fn identifier(value: &str, description: &str) -> Result<()> {
    let mut chars = value.chars();
    let first = chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
    if !first || value.len() > 128 || chars.any(|c| !c.is_ascii_alphanumeric() && c != '_') {
        return Err(Error::new(
            "name",
            format!("{description} must be an ASCII identifier of at most 128 bytes"),
        ));
    }
    Ok(())
}
fn name(value: &str, description: &str) -> Result<()> {
    if value.trim().is_empty() || value.len() > MAX_NAME_BYTES {
        return Err(Error::new(
            "name",
            format!("{description} must be nonempty and at most {MAX_NAME_BYTES} bytes"),
        ));
    }
    Ok(())
}
fn names(values: &[String], description: &str) -> Result<()> {
    bounded(values.len(), MAX_LIST_ITEMS, description)?;
    for value in values {
        name(value, description)?;
    }
    Ok(())
}
fn bounded(value: usize, max: usize, description: &str) -> Result<()> {
    if value == 0 || value > max {
        return Err(Error::new(
            "range",
            format!("{description} must be between 1 and {max}"),
        ));
    }
    Ok(())
}

struct SizeLimit(usize);
impl std::io::Write for SizeLimit {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let remaining = crate::parser::MAX_SOURCE_BYTES.saturating_sub(self.0);
        if bytes.len() > remaining {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "program size limit",
            ));
        }
        self.0 += bytes.len();
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
