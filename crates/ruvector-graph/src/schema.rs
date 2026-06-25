//! Optional, schema-first type layer for the graph (HelixDB-inspired, ADR-252 P1/P2).
//!
//! RuVector's graph is schemaless by default and its Cypher engine is interpreted
//! at runtime. This module adds an **opt-in** schema that catches type errors
//! *before* execution — declared node labels, typed edges with `from`/`to` label
//! constraints, indexed properties, and **vector types bound to a node label +
//! property** (so a vector hit can be traversed back into the graph as a
//! first-class, validated relationship rather than a runtime string + property
//! name).
//!
//! The module is pure-Rust with no storage/HNSW dependency, so it compiles for
//! WASM. It coexists with schemaless mode: only declared labels/edges are checked,
//! and undeclared ones pass through untouched.

use crate::edge::Edge;
use crate::error::{GraphError, Result};
use crate::node::Node;
use crate::types::PropertyValue;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Declared type of a node/edge property.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropertyType {
    Boolean,
    Integer,
    /// Accepts `Float` and (widening) `Integer`.
    Float,
    String,
    /// Dense embedding (`FloatArray`, or a homogeneous numeric `Array`/`List`).
    Vector,
    /// Heterogeneous list.
    Array,
    Map,
    /// Accepts any value (escape hatch).
    Any,
}

impl PropertyType {
    /// Does `value` satisfy this declared type?
    pub fn accepts(&self, value: &PropertyValue) -> bool {
        match self {
            PropertyType::Any => true,
            PropertyType::Boolean => matches!(value, PropertyValue::Boolean(_)),
            PropertyType::Integer => matches!(value, PropertyValue::Integer(_)),
            // Float is permissive: an integer literal is a valid float.
            PropertyType::Float => {
                matches!(value, PropertyValue::Float(_) | PropertyValue::Integer(_))
            }
            PropertyType::String => matches!(value, PropertyValue::String(_)),
            PropertyType::Vector => extract_vector(value).is_some(),
            PropertyType::Array => {
                matches!(value, PropertyValue::Array(_) | PropertyValue::List(_))
            }
            PropertyType::Map => matches!(value, PropertyValue::Map(_)),
        }
    }
}

/// Distance metric for a vector type. Search always ranks by a *higher-is-better*
/// score, so `Euclidean` is surfaced as the negated distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    Cosine,
    DotProduct,
    Euclidean,
}

impl DistanceMetric {
    /// Higher score == more similar, for any metric. Convenience wrapper that
    /// computes the query's norm inline; prefer [`DistanceMetric::query_norm`] +
    /// [`DistanceMetric::score_pre`] in a scan loop to amortize the query norm.
    pub fn score(&self, a: &[f32], b: &[f32]) -> f32 {
        self.score_pre(a, b, self.query_norm(a))
    }

    /// Precompute the query-side norm once per query. Only `Cosine` needs it;
    /// the others return `1.0`.
    #[inline]
    pub fn query_norm(&self, q: &[f32]) -> f32 {
        match self {
            DistanceMetric::Cosine => dot(q, q).sqrt(),
            _ => 1.0,
        }
    }

    /// Score `candidate` against `query`, reusing a precomputed `query_norm`.
    /// Hoists the query norm out of the per-candidate hot loop.
    #[inline]
    pub fn score_pre(&self, query: &[f32], candidate: &[f32], query_norm: f32) -> f32 {
        match self {
            DistanceMetric::DotProduct => dot(query, candidate),
            DistanceMetric::Cosine => {
                // Single fused pass: accumulate q·c and c·c together so the
                // candidate slice is read once (half the memory traffic of two
                // separate `dot` calls).
                let n = query.len().min(candidate.len());
                let mut qc = 0.0f32;
                let mut cc = 0.0f32;
                for i in 0..n {
                    let c = candidate[i];
                    qc += query[i] * c;
                    cc += c * c;
                }
                let cn = cc.sqrt();
                if query_norm == 0.0 || cn == 0.0 {
                    0.0
                } else {
                    qc / (query_norm * cn)
                }
            }
            DistanceMetric::Euclidean => {
                let n = query.len().min(candidate.len());
                let mut sum = 0.0f32;
                for i in 0..n {
                    let d = query[i] - candidate[i];
                    sum += d * d;
                }
                -sum.sqrt()
            }
        }
    }
}

/// Score a vector-shaped property against a query without allocating in the
/// common `FloatArray` case (zero-copy slice scoring). Returns `None` if the
/// property is not vector-shaped or its dimension does not match the query.
#[inline]
pub fn score_property(
    metric: DistanceMetric,
    query: &[f32],
    query_norm: f32,
    value: &PropertyValue,
) -> Option<f32> {
    match value {
        // Fast path: borrow the stored slice directly, no clone.
        PropertyValue::FloatArray(v) => {
            if v.len() == query.len() {
                Some(metric.score_pre(query, v, query_norm))
            } else {
                None
            }
        }
        // Slow path: heterogeneous numeric list must be materialized.
        PropertyValue::Array(_) | PropertyValue::List(_) => {
            let v = extract_vector(value)?;
            if v.len() == query.len() {
                Some(metric.score_pre(query, &v, query_norm))
            } else {
                None
            }
        }
        _ => None,
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    // Iterator form so LLVM auto-vectorizes (SSE/AVX/NEON) without bounds checks.
    // SIMD via `simsimd`/ruvector-core is a follow-up (ADR-252 P5) but is
    // deliberately not a hard dependency here so the schema layer stays WASM- and
    // no-feature-build-safe.
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Coerce a property value into a dense `Vec<f32>` if it is vector-shaped.
pub fn extract_vector(value: &PropertyValue) -> Option<Vec<f32>> {
    match value {
        PropertyValue::FloatArray(v) => Some(v.clone()),
        PropertyValue::Array(items) | PropertyValue::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                match it {
                    PropertyValue::Float(f) => out.push(*f as f32),
                    PropertyValue::Integer(i) => out.push(*i as f32),
                    _ => return None,
                }
            }
            if out.is_empty() {
                None
            } else {
                Some(out)
            }
        }
        _ => None,
    }
}

/// Declaration for a single property.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertySchema {
    pub name: String,
    pub ptype: PropertyType,
    /// Must be present on every instance.
    pub required: bool,
    /// Hint that this property is secondary-indexed (HelixQL `INDEX`).
    pub indexed: bool,
}

impl PropertySchema {
    pub fn new(name: impl Into<String>, ptype: PropertyType) -> Self {
        Self {
            name: name.into(),
            ptype,
            required: false,
            indexed: false,
        }
    }
    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }
    pub fn indexed(mut self) -> Self {
        self.indexed = true;
        self
    }
}

/// Schema for a node label (`N::` in HelixQL).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSchema {
    pub label: String,
    pub properties: Vec<PropertySchema>,
    /// If true, properties not declared here are rejected.
    pub strict: bool,
}

impl NodeSchema {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            properties: Vec::new(),
            strict: false,
        }
    }
    pub fn property(mut self, p: PropertySchema) -> Self {
        self.properties.push(p);
        self
    }
    pub fn strict(mut self) -> Self {
        self.strict = true;
        self
    }
}

/// Schema for an edge type (`E::` in HelixQL) with `from`/`to` label constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeSchema {
    pub edge_type: String,
    pub from_label: String,
    pub to_label: String,
    pub properties: Vec<PropertySchema>,
}

impl EdgeSchema {
    pub fn new(
        edge_type: impl Into<String>,
        from_label: impl Into<String>,
        to_label: impl Into<String>,
    ) -> Self {
        Self {
            edge_type: edge_type.into(),
            from_label: from_label.into(),
            to_label: to_label.into(),
            properties: Vec::new(),
        }
    }
    pub fn property(mut self, p: PropertySchema) -> Self {
        self.properties.push(p);
        self
    }
}

/// Schema for a vector type (`V::` in HelixQL), bound to a node label + property.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSchema {
    /// Vector type name (referenced by `search_then_traverse`).
    pub name: String,
    /// Node label whose instances carry this embedding.
    pub label: String,
    /// Property holding the embedding.
    pub property: String,
    pub dimensions: usize,
    pub metric: DistanceMetric,
}

impl VectorSchema {
    pub fn new(
        name: impl Into<String>,
        label: impl Into<String>,
        property: impl Into<String>,
        dimensions: usize,
        metric: DistanceMetric,
    ) -> Self {
        Self {
            name: name.into(),
            label: label.into(),
            property: property.into(),
            dimensions,
            metric,
        }
    }
}

/// A complete, optional graph schema.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphSchema {
    nodes: HashMap<String, NodeSchema>,
    edges: HashMap<String, EdgeSchema>,
    vectors: HashMap<String, VectorSchema>,
}

impl GraphSchema {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, schema: NodeSchema) -> &mut Self {
        self.nodes.insert(schema.label.clone(), schema);
        self
    }
    pub fn add_edge(&mut self, schema: EdgeSchema) -> &mut Self {
        self.edges.insert(schema.edge_type.clone(), schema);
        self
    }
    pub fn add_vector(&mut self, schema: VectorSchema) -> &mut Self {
        self.vectors.insert(schema.name.clone(), schema);
        self
    }

    pub fn node(&self, label: &str) -> Option<&NodeSchema> {
        self.nodes.get(label)
    }
    pub fn edge(&self, edge_type: &str) -> Option<&EdgeSchema> {
        self.edges.get(edge_type)
    }
    pub fn vector(&self, name: &str) -> Option<&VectorSchema> {
        self.vectors.get(name)
    }

    /// Node schemas sorted by label (deterministic — for codegen).
    pub fn node_schemas_sorted(&self) -> Vec<&NodeSchema> {
        let mut v: Vec<&NodeSchema> = self.nodes.values().collect();
        v.sort_by(|a, b| a.label.cmp(&b.label));
        v
    }
    /// Edge schemas sorted by edge type (deterministic — for codegen).
    pub fn edge_schemas_sorted(&self) -> Vec<&EdgeSchema> {
        let mut v: Vec<&EdgeSchema> = self.edges.values().collect();
        v.sort_by(|a, b| a.edge_type.cmp(&b.edge_type));
        v
    }
    /// Vector schemas sorted by name (deterministic — for codegen).
    pub fn vector_schemas_sorted(&self) -> Vec<&VectorSchema> {
        let mut v: Vec<&VectorSchema> = self.vectors.values().collect();
        v.sort_by(|a, b| a.name.cmp(&b.name));
        v
    }

    /// Validate the schema's own internal consistency: every edge's `from`/`to`
    /// label and every vector's bound label must reference a declared node. Run
    /// this once after building the schema (HelixQL's compile-time check).
    pub fn validate_self(&self) -> Result<()> {
        for e in self.edges.values() {
            if !self.nodes.contains_key(&e.from_label) {
                return Err(GraphError::SchemaViolation(format!(
                    "edge '{}' references undeclared from-label '{}'",
                    e.edge_type, e.from_label
                )));
            }
            if !self.nodes.contains_key(&e.to_label) {
                return Err(GraphError::SchemaViolation(format!(
                    "edge '{}' references undeclared to-label '{}'",
                    e.edge_type, e.to_label
                )));
            }
        }
        for v in self.vectors.values() {
            if !self.nodes.contains_key(&v.label) {
                return Err(GraphError::SchemaViolation(format!(
                    "vector '{}' bound to undeclared label '{}'",
                    v.name, v.label
                )));
            }
        }
        Ok(())
    }

    /// Validate a node against any declared schema for its labels. Labels with no
    /// schema pass through (schemaless coexistence).
    pub fn validate_node(&self, node: &Node) -> Result<()> {
        // Collect every property allowed by any matching (declared) label.
        let mut allowed: Vec<&str> = Vec::new();
        let mut any_strict = false;
        let mut matched_any = false;

        for label in &node.labels {
            let Some(ns) = self.nodes.get(&label.name) else {
                continue;
            };
            matched_any = true;
            any_strict |= ns.strict;
            for p in &ns.properties {
                allowed.push(p.name.as_str());
                match node.properties.get(&p.name) {
                    None if p.required => {
                        return Err(GraphError::SchemaViolation(format!(
                            "node '{}' (:{}) missing required property '{}'",
                            node.id, label.name, p.name
                        )));
                    }
                    Some(v) if !p.ptype.accepts(v) => {
                        return Err(GraphError::SchemaViolation(format!(
                            "node '{}' (:{}) property '{}' has wrong type (expected {:?})",
                            node.id, label.name, p.name, p.ptype
                        )));
                    }
                    _ => {}
                }
            }
        }

        if matched_any && any_strict {
            for key in node.properties.keys() {
                if !allowed.iter().any(|a| a == key) {
                    return Err(GraphError::SchemaViolation(format!(
                        "node '{}' has undeclared property '{}' (strict schema)",
                        node.id, key
                    )));
                }
            }
        }
        Ok(())
    }

    /// Validate an edge given the labels of its endpoints. Undeclared edge types
    /// pass through. Pass the actual from/to node labels so direction + endpoint
    /// types are checked.
    pub fn validate_edge(
        &self,
        edge: &Edge,
        from_labels: &[String],
        to_labels: &[String],
    ) -> Result<()> {
        let Some(es) = self.edges.get(&edge.edge_type) else {
            return Ok(());
        };
        if !from_labels.iter().any(|l| l == &es.from_label) {
            return Err(GraphError::SchemaViolation(format!(
                "edge '{}' requires from-label '{}', got {:?}",
                edge.edge_type, es.from_label, from_labels
            )));
        }
        if !to_labels.iter().any(|l| l == &es.to_label) {
            return Err(GraphError::SchemaViolation(format!(
                "edge '{}' requires to-label '{}', got {:?}",
                edge.edge_type, es.to_label, to_labels
            )));
        }
        for p in &es.properties {
            match edge.properties.get(&p.name) {
                None if p.required => {
                    return Err(GraphError::SchemaViolation(format!(
                        "edge '{}' missing required property '{}'",
                        edge.edge_type, p.name
                    )));
                }
                Some(v) if !p.ptype.accepts(v) => {
                    return Err(GraphError::SchemaViolation(format!(
                        "edge '{}' property '{}' has wrong type (expected {:?})",
                        edge.edge_type, p.name, p.ptype
                    )));
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Validate that a query vector matches a declared vector type's dimension.
    pub fn validate_vector_dims(&self, vector_type: &str, query: &[f32]) -> Result<&VectorSchema> {
        let vs = self.vectors.get(vector_type).ok_or_else(|| {
            GraphError::SchemaViolation(format!("unknown vector type '{}'", vector_type))
        })?;
        if query.len() != vs.dimensions {
            return Err(GraphError::SchemaViolation(format!(
                "vector type '{}' expects dimension {}, got {}",
                vector_type,
                vs.dimensions,
                query.len()
            )));
        }
        Ok(vs)
    }
}

/// Reciprocal Rank Fusion over several ranked id lists (ADR-252 P4 core).
///
/// `score(id) = Σ 1 / (k_const + rank)` with `rank` 1-based per list. The common
/// default for `k_const` is 60. Returns ids sorted by fused score, descending.
pub fn reciprocal_rank_fusion(rankings: &[Vec<String>], k_const: f32) -> Vec<(String, f32)> {
    let mut scores: HashMap<String, f32> = HashMap::new();
    for ranking in rankings {
        for (rank, id) in ranking.iter().enumerate() {
            let contribution = 1.0 / (k_const + (rank as f32 + 1.0));
            *scores.entry(id.clone()).or_insert(0.0) += contribution;
        }
    }
    let mut fused: Vec<(String, f32)> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fused
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::NodeBuilder;
    use crate::types::Label;

    fn person_schema() -> GraphSchema {
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
            3,
            DistanceMetric::Cosine,
        ));
        s
    }

    #[test]
    fn self_validation_catches_dangling_refs() {
        let mut s = GraphSchema::new();
        s.add_edge(EdgeSchema::new("KNOWS", "Person", "Person"));
        assert!(s.validate_self().is_err());
        s.add_node(NodeSchema::new("Person"));
        assert!(s.validate_self().is_ok());
    }

    #[test]
    fn node_validation_required_and_types() {
        let s = person_schema();
        // Valid.
        let ok = NodeBuilder::new()
            .label("Person")
            .property("name", "Alice")
            .property("age", 30i64)
            .build();
        assert!(s.validate_node(&ok).is_ok());
        // Missing required `name`.
        let missing = NodeBuilder::new()
            .label("Person")
            .property("age", 30i64)
            .build();
        assert!(s.validate_node(&missing).is_err());
        // Wrong type for `age` (string where integer expected).
        let wrong = NodeBuilder::new()
            .label("Person")
            .property("name", "Bob")
            .property("age", "old")
            .build();
        assert!(s.validate_node(&wrong).is_err());
        // Undeclared label passes through (schemaless coexistence).
        let other = NodeBuilder::new()
            .label("Alien")
            .property("planet", "Mars")
            .build();
        assert!(s.validate_node(&other).is_ok());
    }

    #[test]
    fn strict_node_rejects_undeclared_props() {
        let mut s = GraphSchema::new();
        s.add_node(
            NodeSchema::new("Tag")
                .property(PropertySchema::new("name", PropertyType::String))
                .strict(),
        );
        let bad = NodeBuilder::new()
            .label("Tag")
            .property("name", "x")
            .property("extra", 1i64)
            .build();
        assert!(s.validate_node(&bad).is_err());
    }

    #[test]
    fn edge_validation_checks_endpoint_labels() {
        let s = person_schema();
        let e = Edge::create("p1".into(), "c1".into(), "WORKS_AT");
        assert!(s
            .validate_edge(&e, &["Person".into()], &["Company".into()])
            .is_ok());
        // Wrong from-label.
        assert!(s
            .validate_edge(&e, &["Company".into()], &["Company".into()])
            .is_err());
        // Undeclared edge type passes through.
        let e2 = Edge::create("p1".into(), "p2".into(), "LIKES");
        assert!(s
            .validate_edge(&e2, &["Person".into()], &["Person".into()])
            .is_ok());
    }

    #[test]
    fn vector_dim_validation() {
        let s = person_schema();
        assert!(s
            .validate_vector_dims("PersonEmb", &[1.0, 2.0, 3.0])
            .is_ok());
        assert!(s.validate_vector_dims("PersonEmb", &[1.0, 2.0]).is_err());
        assert!(s.validate_vector_dims("Missing", &[1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn distance_metrics_rank_higher_is_better() {
        let q = [1.0f32, 0.0, 0.0];
        let near = [0.9f32, 0.1, 0.0];
        let far = [0.0f32, 1.0, 0.0];
        for m in [
            DistanceMetric::Cosine,
            DistanceMetric::DotProduct,
            DistanceMetric::Euclidean,
        ] {
            assert!(m.score(&q, &near) > m.score(&q, &far), "{:?}", m);
        }
    }

    #[test]
    fn extract_vector_handles_shapes() {
        assert_eq!(
            extract_vector(&PropertyValue::FloatArray(vec![1.0, 2.0])),
            Some(vec![1.0, 2.0])
        );
        assert_eq!(
            extract_vector(&PropertyValue::Array(vec![
                PropertyValue::Integer(1),
                PropertyValue::Float(2.0)
            ])),
            Some(vec![1.0, 2.0])
        );
        assert_eq!(extract_vector(&PropertyValue::String("x".into())), None);
    }

    #[test]
    fn rrf_fuses_and_ranks() {
        let a = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        let b = vec!["y".to_string(), "x".to_string()];
        let fused = reciprocal_rank_fusion(&[a, b], 60.0);
        // `y`: 1/62 + 1/61; `x`: 1/61 + 1/62 — tie; `z`: 1/63. x & y lead z.
        assert_eq!(fused.len(), 3);
        assert_eq!(fused[2].0, "z");
    }

    #[test]
    fn multi_label_node_validation() {
        let mut s = GraphSchema::new();
        s.add_node(
            NodeSchema::new("A")
                .property(PropertySchema::new("a", PropertyType::Integer).required()),
        );
        s.add_node(
            NodeSchema::new("B")
                .property(PropertySchema::new("b", PropertyType::String).required()),
        );
        let n = Node::new(
            "n1".into(),
            vec![Label::new("A"), Label::new("B")],
            [
                ("a".to_string(), PropertyValue::Integer(1)),
                ("b".to_string(), PropertyValue::String("x".into())),
            ]
            .into_iter()
            .collect(),
        );
        assert!(s.validate_node(&n).is_ok());
    }
}
