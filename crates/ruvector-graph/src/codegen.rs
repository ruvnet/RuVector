//! Schema-driven typed client codegen (HelixDB-inspired, ADR-252 P6).
//!
//! HelixDB compiles a schema into typed API endpoints + multi-language SDKs so
//! callers get compile-time-checked node/edge/vector types. This module does the
//! same from a [`GraphSchema`]: it emits TypeScript, Python, and Rust type
//! definitions plus a vector-type manifest. Output is deterministic (schema
//! elements are sorted) so it can be checked in and diffed.
//!
//! These are *type* stubs — the single source of truth is the schema. The
//! generated code carries the labels, property names/types, edge `from`/`to`
//! constraints, and vector dimensions/metrics across the language boundary.

use crate::schema::{DistanceMetric, GraphSchema, PropertySchema, PropertyType};

fn metric_name(m: DistanceMetric) -> &'static str {
    match m {
        DistanceMetric::Cosine => "Cosine",
        DistanceMetric::DotProduct => "DotProduct",
        DistanceMetric::Euclidean => "Euclidean",
    }
}

// ---- TypeScript ------------------------------------------------------------

fn ts_type(t: PropertyType) -> &'static str {
    match t {
        PropertyType::Boolean => "boolean",
        PropertyType::Integer | PropertyType::Float => "number",
        PropertyType::String => "string",
        PropertyType::Vector => "number[]",
        PropertyType::Array => "unknown[]",
        PropertyType::Map => "Record<string, unknown>",
        PropertyType::Any => "unknown",
    }
}

fn ts_property(p: &PropertySchema) -> String {
    let opt = if p.required { "" } else { "?" };
    let indexed = if p.indexed { "  /** @indexed */\n" } else { "" };
    format!("{indexed}  {}{}: {};\n", p.name, opt, ts_type(p.ptype))
}

/// Generate TypeScript interfaces + a vector-type manifest from the schema.
pub fn generate_typescript(schema: &GraphSchema) -> String {
    let mut out = String::new();
    out.push_str(
        "// Auto-generated from RuVector GraphSchema (ADR-252 P6). Do not edit by hand.\n\n",
    );

    for n in schema.node_schemas_sorted() {
        out.push_str(&format!("export interface {} {{\n", n.label));
        for p in &n.properties {
            out.push_str(&ts_property(p));
        }
        out.push_str("}\n\n");
    }

    for e in schema.edge_schemas_sorted() {
        out.push_str(&format!(
            "/** Edge {0}: {1} -> {2} */\nexport interface {0} {{\n  from: string;\n  to: string;\n",
            e.edge_type, e.from_label, e.to_label
        ));
        for p in &e.properties {
            out.push_str(&ts_property(p));
        }
        out.push_str("}\n\n");
    }

    out.push_str("export const VectorTypes = {\n");
    for v in schema.vector_schemas_sorted() {
        out.push_str(&format!(
            "  {}: {{ label: \"{}\", property: \"{}\", dimensions: {}, metric: \"{}\" }},\n",
            v.name,
            v.label,
            v.property,
            v.dimensions,
            metric_name(v.metric)
        ));
    }
    out.push_str("} as const;\n\nexport type VectorTypeName = keyof typeof VectorTypes;\n");
    out
}

// ---- Python ----------------------------------------------------------------

fn py_type(t: PropertyType) -> &'static str {
    match t {
        PropertyType::Boolean => "bool",
        PropertyType::Integer => "int",
        PropertyType::Float => "float",
        PropertyType::String => "str",
        PropertyType::Vector => "list[float]",
        PropertyType::Array => "list",
        PropertyType::Map => "dict",
        PropertyType::Any => "Any",
    }
}

fn py_property(p: &PropertySchema) -> String {
    let ty = py_type(p.ptype);
    if p.required {
        format!("    {}: {}\n", p.name, ty)
    } else {
        format!("    {}: NotRequired[{}]\n", p.name, ty)
    }
}

/// Generate Python `TypedDict` classes + a vector-type manifest from the schema.
pub fn generate_python(schema: &GraphSchema) -> String {
    let mut out = String::new();
    out.push_str("# Auto-generated from RuVector GraphSchema (ADR-252 P6). Do not edit by hand.\n");
    out.push_str("from __future__ import annotations\n");
    out.push_str("from typing import Any, NotRequired, TypedDict\n\n");

    for n in schema.node_schemas_sorted() {
        out.push_str(&format!("class {}(TypedDict):\n", n.label));
        if n.properties.is_empty() {
            out.push_str("    pass\n\n");
            continue;
        }
        for p in &n.properties {
            out.push_str(&py_property(p));
        }
        out.push('\n');
    }

    for e in schema.edge_schemas_sorted() {
        out.push_str(&format!("class {}(TypedDict):\n", e.edge_type));
        out.push_str(&format!("    # {} -> {}\n", e.from_label, e.to_label));
        out.push_str("    from_: str\n    to: str\n");
        for p in &e.properties {
            out.push_str(&py_property(p));
        }
        out.push('\n');
    }

    out.push_str("VECTOR_TYPES = {\n");
    for v in schema.vector_schemas_sorted() {
        out.push_str(&format!(
            "    \"{}\": {{\"label\": \"{}\", \"property\": \"{}\", \"dimensions\": {}, \"metric\": \"{}\"}},\n",
            v.name, v.label, v.property, v.dimensions, metric_name(v.metric)
        ));
    }
    out.push_str("}\n");
    out
}

// ---- Rust ------------------------------------------------------------------

fn rust_type(t: PropertyType) -> &'static str {
    match t {
        PropertyType::Boolean => "bool",
        PropertyType::Integer => "i64",
        PropertyType::Float => "f64",
        PropertyType::String => "String",
        PropertyType::Vector => "Vec<f32>",
        PropertyType::Array => "Vec<serde_json::Value>",
        PropertyType::Map => "std::collections::HashMap<String, serde_json::Value>",
        PropertyType::Any => "serde_json::Value",
    }
}

fn rust_field(p: &PropertySchema) -> String {
    let ty = rust_type(p.ptype);
    if p.required {
        format!("    pub {}: {},\n", p.name, ty)
    } else {
        format!("    pub {}: Option<{}>,\n", p.name, ty)
    }
}

/// Generate Rust structs from the schema (serde-ready).
pub fn generate_rust(schema: &GraphSchema) -> String {
    let mut out = String::new();
    out.push_str(
        "// Auto-generated from RuVector GraphSchema (ADR-252 P6). Do not edit by hand.\n",
    );
    out.push_str("use serde::{Deserialize, Serialize};\n\n");

    for n in schema.node_schemas_sorted() {
        out.push_str("#[derive(Debug, Clone, Serialize, Deserialize)]\n");
        out.push_str(&format!("pub struct {} {{\n", n.label));
        for p in &n.properties {
            out.push_str(&rust_field(p));
        }
        out.push_str("}\n\n");
    }

    for e in schema.edge_schemas_sorted() {
        out.push_str(&format!(
            "/// Edge {}: {} -> {}\n",
            e.edge_type, e.from_label, e.to_label
        ));
        out.push_str("#[derive(Debug, Clone, Serialize, Deserialize)]\n");
        out.push_str(&format!(
            "pub struct {} {{\n    pub from: String,\n    pub to: String,\n",
            e.edge_type
        ));
        for p in &e.properties {
            out.push_str(&rust_field(p));
        }
        out.push_str("}\n\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{EdgeSchema, NodeSchema, VectorSchema};

    fn schema() -> GraphSchema {
        let mut s = GraphSchema::new();
        s.add_node(
            NodeSchema::new("Person")
                .property(
                    PropertySchema::new("name", PropertyType::String)
                        .required()
                        .indexed(),
                )
                .property(PropertySchema::new("age", PropertyType::Integer))
                .property(PropertySchema::new("embedding", PropertyType::Vector)),
        );
        s.add_node(NodeSchema::new("Company"));
        s.add_edge(EdgeSchema::new("WORKS_AT", "Person", "Company"));
        s.add_vector(VectorSchema::new(
            "PersonEmb",
            "Person",
            "embedding",
            384,
            DistanceMetric::Cosine,
        ));
        s
    }

    #[test]
    fn typescript_has_typed_interfaces_and_manifest() {
        let ts = generate_typescript(&schema());
        assert!(ts.contains("export interface Person {"));
        assert!(ts.contains("name: string;")); // required
        assert!(ts.contains("age?: number;")); // optional
        assert!(ts.contains("embedding?: number[];")); // vector
        assert!(ts.contains("@indexed"));
        assert!(ts.contains("export interface WORKS_AT {"));
        assert!(ts.contains("Person -> Company"));
        assert!(ts.contains("PersonEmb: { label: \"Person\""));
        assert!(ts.contains("dimensions: 384"));
        assert!(ts.contains("export type VectorTypeName"));
    }

    #[test]
    fn python_has_typeddicts_and_manifest() {
        let py = generate_python(&schema());
        assert!(py.contains("class Person(TypedDict):"));
        assert!(py.contains("    name: str"));
        assert!(py.contains("    age: NotRequired[int]"));
        assert!(py.contains("class Company(TypedDict):"));
        assert!(py.contains("    pass")); // empty node
        assert!(py.contains("\"PersonEmb\": {\"label\": \"Person\""));
    }

    #[test]
    fn rust_has_structs() {
        let rs = generate_rust(&schema());
        assert!(rs.contains("pub struct Person {"));
        assert!(rs.contains("pub name: String,"));
        assert!(rs.contains("pub age: Option<i64>,"));
        assert!(rs.contains("pub embedding: Option<Vec<f32>>,"));
        assert!(rs.contains("pub struct WORKS_AT {"));
    }

    #[test]
    fn output_is_deterministic() {
        let s = schema();
        assert_eq!(generate_typescript(&s), generate_typescript(&s));
        assert_eq!(generate_python(&s), generate_python(&s));
        assert_eq!(generate_rust(&s), generate_rust(&s));
    }
}
