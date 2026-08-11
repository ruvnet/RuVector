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

/// Test-only witness for which cosine backend `score_pre` actually took. Lets a
/// test assert *selection*, not just numerical agreement: a scalar reference
/// can match the kernel's output by construction while the kernel call itself
/// has been silently reverted to the fallback arm.
///
/// Per-thread (rather than a shared global) so that `cargo test`'s default
/// one-thread-per-test execution can't let one test's cosine calls satisfy
/// another test's assertion. Holds the actual value the lattice call
/// returned (not just a flag), set only after that call returns, so the
/// witness proves what came back rather than merely that a guarded branch
/// was entered.
#[cfg(all(test, feature = "lattice-simd"))]
thread_local! {
    static COSINE_LATTICE_ROUTE_HIT: std::cell::Cell<Option<f32>> = const { std::cell::Cell::new(None) };
}

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
                // Takes the precomputed `query_norm` rather than recomputing
                // `‖query‖` per candidate, so the hoist this signature exists for
                // survives, and a caller-supplied norm that differs from `‖query‖`
                // rescales the result exactly as the scalar arm below does.
                //
                // Equal-length guard for the same reason as `dot`: the fused loop
                // truncates to the shorter slice, the kernel returns 0.0.
                #[cfg(feature = "lattice-simd")]
                {
                    if query.len() == candidate.len() {
                        let result = lattice_embed::simd::cosine_similarity_pre_normalized(
                            query, candidate, query_norm,
                        );
                        #[cfg(test)]
                        COSINE_LATTICE_ROUTE_HIT.with(|hit| hit.set(Some(result)));
                        return result;
                    }
                }

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
                // Same equal-length guard as `dot`: the loop below truncates to
                // the shorter slice, the kernel does not.
                #[cfg(feature = "lattice-simd")]
                {
                    if query.len() == candidate.len() {
                        return -lattice_embed::simd::euclidean_distance(query, candidate);
                    }
                }

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
    // With `lattice-simd`, use explicit kernels rather than relying on
    // autovectorization. This stays WASM-safe and no-feature-build-safe, the two
    // properties that kept `simsimd` out of this layer: the dependency is
    // optional, and its kernels compile to `simd128` on wasm32 — where the
    // SimSIMD-backed paths are cfg'd out and fall back to scalar.
    //
    // The equal-length guard is required, not defensive. The iterator form below
    // truncates to the shorter slice, while the kernel returns 0.0 on a length
    // mismatch, so unequal inputs must keep taking the scalar path to preserve
    // this function's existing behaviour.
    #[cfg(feature = "lattice-simd")]
    {
        if a.len() == b.len() {
            return lattice_embed::simd::dot_product(a, b);
        }
    }

    // Iterator form so LLVM auto-vectorizes (SSE/AVX/NEON) without bounds checks.
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

    /// Deterministic pseudo-random vectors, no dev-dependency needed.
    fn vecs(dim: usize, seed: u32) -> (Vec<f32>, Vec<f32>) {
        let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        (
            (0..dim).map(|_| next()).collect(),
            (0..dim).map(|_| next()).collect(),
        )
    }

    /// Whichever `dot` backend is compiled in must agree with a naive scalar sum.
    ///
    /// This is what catches a wrong kernel or a wrong adapter: swapping
    /// `dot_product` for a different kernel, or dropping the negation on the
    /// Euclidean arm, still compiles and fails here.
    #[test]
    fn test_score_pre_matches_scalar_reference() {
        // Dimensions straddling 4/8/16-lane widths and their remainders.
        for dim in [1usize, 3, 4, 7, 8, 15, 16, 17, 31, 64, 127, 384, 768] {
            for seed in 0..4u32 {
                let (q, c) = vecs(dim, seed);

                let want_dot: f32 = q.iter().zip(&c).map(|(x, y)| x * y).sum();
                let got_dot = DistanceMetric::DotProduct.score_pre(&q, &c, 0.0);
                assert!(
                    (got_dot - want_dot).abs() <= 1e-3 * want_dot.abs().max(1.0),
                    "dot mismatch dim={dim} seed={seed}: got {got_dot}, want {want_dot}"
                );

                let want_euc = -q
                    .iter()
                    .zip(&c)
                    .map(|(x, y)| (x - y) * (x - y))
                    .sum::<f32>()
                    .sqrt();
                let got_euc = DistanceMetric::Euclidean.score_pre(&q, &c, 0.0);
                assert!(
                    (got_euc - want_euc).abs() <= 1e-3 * want_euc.abs().max(1.0),
                    "euclidean mismatch dim={dim} seed={seed}: got {got_euc}, want {want_euc}"
                );

                let qn = DistanceMetric::Cosine.query_norm(&q);
                let want_qn = q.iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!(
                    (qn - want_qn).abs() <= 1e-3 * want_qn.abs().max(1.0),
                    "query_norm mismatch dim={dim} seed={seed}: got {qn}, want {want_qn}"
                );

                // The cosine score itself, not just its hoisted norm. Reference is
                // built from the naive sums rather than from `query_norm` above, so
                // a norm that is wrong in the same direction on both sides cannot
                // cancel out and pass.
                let qc: f32 = q.iter().zip(&c).map(|(x, y)| x * y).sum();
                let cn = c.iter().map(|y| y * y).sum::<f32>().sqrt();
                let want_cos = if want_qn == 0.0 || cn == 0.0 {
                    0.0
                } else {
                    qc / (want_qn * cn)
                };
                let got_cos = DistanceMetric::Cosine.score_pre(&q, &c, qn);
                assert!(
                    (got_cos - want_cos).abs() <= 1e-3 * want_cos.abs().max(1.0),
                    "cosine mismatch dim={dim} seed={seed}: got {got_cos}, want {want_cos}"
                );

                // A caller-supplied norm that is not `‖query‖` must rescale the
                // result, not be ignored. This is the property that made a plain
                // two-argument kernel unusable for this signature, so it is
                // asserted rather than assumed.
                let scaled = DistanceMetric::Cosine.score_pre(&q, &c, qn * 2.0);
                assert!(
                    (scaled - want_cos / 2.0).abs() <= 1e-3 * (want_cos / 2.0).abs().max(1.0),
                    "cosine ignored the supplied query_norm dim={dim} seed={seed}: \
                     got {scaled}, want {}",
                    want_cos / 2.0
                );
            }
        }
    }

    /// Unequal lengths must keep the truncating scalar behaviour. The kernels
    /// return 0.0 on a length mismatch, so a missing guard would show up here as
    /// a zero instead of the truncated dot product.
    #[test]
    fn test_unequal_lengths_truncate_not_zero() {
        let q = vec![1.0f32, 2.0, 3.0, 4.0];
        let c = vec![1.0f32, 1.0, 1.0];

        let got = DistanceMetric::DotProduct.score_pre(&q, &c, 0.0);
        assert!(
            (got - 6.0).abs() < 1e-5,
            "expected truncated dot 6.0, got {got}"
        );

        let got = DistanceMetric::Euclidean.score_pre(&q, &c, 0.0);
        let want = -(0.0f32 + 1.0 + 4.0).sqrt();
        assert!(
            (got - want).abs() < 1e-5,
            "expected truncated euclidean {want}, got {got}"
        );

        // Cosine truncates too: q·c and ‖c‖ both over the first 3 lanes.
        let qn = (1.0f32 + 4.0 + 9.0 + 16.0).sqrt();
        let got = DistanceMetric::Cosine.score_pre(&q, &c, qn);
        let want = 6.0f32 / (qn * 3.0f32.sqrt());
        assert!(
            (got - want).abs() < 1e-5,
            "expected truncated cosine {want}, got {got}"
        );
    }

    /// Guards *selection*, not just numerical agreement: reverting the
    /// `lattice-simd` cosine arm to the scalar fallback still produces a
    /// correct score (that's the point of the fallback), so
    /// `test_score_pre_matches_scalar_reference` alone would keep passing.
    ///
    /// The witness is written only after
    /// `lattice_embed::simd::cosine_similarity_pre_normalized` returns, and
    /// this test cross-checks the recorded value bit-for-bit against a
    /// second, independent direct call to that same kernel function. A
    /// reversion that swaps the call for an inline scalar computation (even
    /// one bound to the same local and stored the same way) still shows up
    /// here: the scalar sum's rounding practically never matches the
    /// kernel's, so the two values diverge. Reverting the whole arm instead
    /// leaves the witness unset, which the `.expect` below catches.
    #[cfg(feature = "lattice-simd")]
    #[test]
    fn test_cosine_equal_length_routes_through_lattice_backend() {
        COSINE_LATTICE_ROUTE_HIT.with(|hit| hit.set(None));

        // dim=17 straddles the 16-lane width with a one-element remainder,
        // so the kernel's reduction order can't coincidentally match a
        // sequential scalar sum.
        let (q, c) = vecs(17, 5);
        let qn = DistanceMetric::Cosine.query_norm(&q);
        let got = DistanceMetric::Cosine.score_pre(&q, &c, qn);

        let recorded = COSINE_LATTICE_ROUTE_HIT.with(|hit| hit.get()).expect(
            "expected the equal-length cosine path to record a post-call \
                 witness; the scalar fallback ran instead (or the lattice \
                 arm never returned through the witnessed path)",
        );
        assert_eq!(
            recorded.to_bits(),
            got.to_bits(),
            "witness value diverged from score_pre's own return value"
        );

        let direct = lattice_embed::simd::cosine_similarity_pre_normalized(&q, &c, qn);
        assert_eq!(
            recorded.to_bits(),
            direct.to_bits(),
            "expected the equal-length cosine path's witnessed value to \
             bit-match a fresh, independent call into \
             lattice_embed::simd::cosine_similarity_pre_normalized; got a \
             different value, so the scan path did not actually return the \
             kernel's result"
        );
    }
}
