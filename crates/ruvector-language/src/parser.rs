//! Bounded readable and JSON frontends for the same versioned program contract.
mod json;
mod text;

use crate::{
    compiler,
    error::{Error, Result},
    types::{Program, RunRequest},
};

pub const MAX_SOURCE_BYTES: usize = 1_048_576;
pub const MAX_JSON_BYTES: usize = 16 * 1_048_576;
pub const MAX_JSON_DEPTH: usize = 64;
pub use json::strict_json;

/// Parse either the readable language or its strict JSON representation.
pub fn parse(source: &str) -> Result<Program> {
    source_size(source)?;
    if source.trim_start().starts_with('{') {
        parse_json(source)
    } else {
        let program = text::parse(source)?;
        compiler::check(&program)?;
        Ok(program)
    }
}

/// Parse JSON without accepting duplicate keys, unknown fields, or trailing data.
pub fn parse_json(source: &str) -> Result<Program> {
    source_size(source)?;
    let value = strict_json(source)?;
    validate_program_value(&value)?;
    let program: Program =
        serde_json::from_value(value).map_err(|e| Error::new("json_schema", e.to_string()))?;
    compiler::check(&program)?;
    Ok(program)
}

/// Parse the combined CLI request with the same program checks as standalone JSON.
pub fn parse_request_json(source: &str) -> Result<RunRequest> {
    let value = strict_json(source)?;
    let program = value
        .get("program")
        .ok_or_else(|| Error::new("json_schema", "request requires program"))?;
    validate_program_value(program)?;
    let request: RunRequest =
        serde_json::from_value(value).map_err(|e| Error::new("json_schema", e.to_string()))?;
    compiler::check(&request.program)?;
    Ok(request)
}

fn validate_program_value(value: &serde_json::Value) -> Result<()> {
    if value.to_string().len() > MAX_SOURCE_BYTES {
        return Err(Error::new(
            "source_limit",
            "program exceeds 1 MiB JSON representation limit",
        ));
    }
    if let Some(effects) = value.get("effects").and_then(|v| v.as_array()) {
        let mut seen = std::collections::BTreeSet::new();
        for effect in effects {
            if let Some(name) = effect.as_str() {
                if !seen.insert(name) {
                    return Err(Error::new(
                        "duplicate_effect",
                        format!("duplicate effect {name}"),
                    ));
                }
            }
        }
    }
    Ok(())
}

fn source_size(source: &str) -> Result<()> {
    if source.len() > MAX_SOURCE_BYTES {
        return Err(Error::new(
            "source_limit",
            "program exceeds 1 MiB source limit",
        ));
    }
    Ok(())
}
