//! Cypher query language parser and execution engine
//!
//! This module provides a pure-Rust Cypher implementation including:
//! - Lexical analysis (tokenization)
//! - Syntax parsing (AST generation)
//! - In-memory property graph storage
//! - Query execution engine
//!
//! With the default `browser` feature, [`CypherEngine`] is also exported to
//! JavaScript through wasm-bindgen. Without it, use the Rust API
//! ([`CypherEngine::run`], [`CypherEngine::graph_stats`],
//! [`CypherEngine::export_state`] / [`CypherEngine::load_state`]).
//!
//! Supported operations:
//! - CREATE: Create nodes and relationships
//! - MATCH: Pattern matching
//! - WHERE: Filtering
//! - RETURN: Projection
//! - SET: Update properties
//! - DELETE/DETACH DELETE: Remove nodes and edges

pub mod ast;
pub mod executor;
pub mod graph_store;
pub mod lexer;
pub mod parser;

pub use ast::{Expression, Pattern, Query, Statement};
pub use executor::{ContextValue, ExecutionError, ExecutionResult, Executor};
pub use graph_store::{Edge, EdgeId, GraphStats, Node, NodeId, PropertyGraph, Value};
pub use lexer::{tokenize, Token, TokenKind};
pub use parser::{parse_cypher, ParseError};

use crate::storage::state::{EdgeState, GraphState, NodeState, PropertyValue};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
#[cfg(feature = "browser")]
use wasm_bindgen::prelude::*;

/// Error returned by the pure-Rust [`CypherEngine`] API.
///
/// The `Display` text matches the error strings the JavaScript bindings have
/// always thrown, so the browser wrappers stay byte-for-byte compatible.
#[derive(Debug, Clone, PartialEq)]
pub enum CypherError {
    /// The query could not be parsed.
    Parse(String),
    /// The query parsed but failed while executing against the graph.
    Execution(String),
    /// A persisted edge could not be restored by [`CypherEngine::load_state`].
    Import(String),
}

impl std::fmt::Display for CypherError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "Parse error: {}", e),
            Self::Execution(e) => write!(f, "Execution error: {}", e),
            Self::Import(e) => write!(f, "Failed to import edge: {}", e),
        }
    }
}

impl std::error::Error for CypherError {}

/// WASM-compatible Cypher engine
// (Exported to JS only with the default `browser` feature; the pure-Rust API
// below is always available.)
#[cfg_attr(feature = "browser", wasm_bindgen)]
pub struct CypherEngine {
    graph: PropertyGraph,
}

#[cfg_attr(feature = "browser", wasm_bindgen)]
impl CypherEngine {
    /// Create a new Cypher engine with empty graph
    #[cfg_attr(feature = "browser", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        Self {
            graph: PropertyGraph::new(),
        }
    }

    /// Clear the graph
    pub fn clear(&mut self) {
        self.graph = PropertyGraph::new();
    }
}

/// JavaScript bindings (default `browser` feature)
#[cfg(feature = "browser")]
#[wasm_bindgen]
impl CypherEngine {
    /// Execute a Cypher query and return JSON results
    pub fn execute(&mut self, query: &str) -> Result<JsValue, JsValue> {
        let result = self
            .run(query)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Convert to JS value
        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    /// Get graph statistics
    pub fn stats(&self) -> Result<JsValue, JsValue> {
        let stats = self.graph_stats();
        serde_wasm_bindgen::to_value(&stats)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }
}

#[cfg(feature = "browser")]
impl CypherEngine {
    /// Import state to restore the graph (JavaScript-error flavour of
    /// [`CypherEngine::load_state`])
    pub fn import_state(&mut self, state: &GraphState) -> Result<(), JsValue> {
        self.load_state(state)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

/// Pure-Rust API, available with or without the `browser` feature
impl CypherEngine {
    /// Parse and execute a Cypher query against the graph
    pub fn run(&mut self, query: &str) -> Result<ExecutionResult, CypherError> {
        let ast = parse_cypher(query).map_err(|e| CypherError::Parse(e.to_string()))?;

        let mut executor = Executor::new(&mut self.graph);
        executor
            .execute(&ast)
            .map_err(|e| CypherError::Execution(e.to_string()))
    }

    /// Get graph statistics
    pub fn graph_stats(&self) -> GraphStats {
        self.graph.stats()
    }

    /// Borrow the underlying property graph
    pub fn graph(&self) -> &PropertyGraph {
        &self.graph
    }

    /// Export graph state for persistence
    pub fn export_state(&self) -> GraphState {
        let nodes: Vec<NodeState> = self
            .graph
            .all_nodes()
            .into_iter()
            .map(|n| NodeState {
                id: n.id.clone(),
                labels: n.labels.clone(),
                properties: n
                    .properties
                    .iter()
                    .map(|(k, v)| (k.clone(), value_to_property(v)))
                    .collect(),
            })
            .collect();

        let edges: Vec<EdgeState> = self
            .graph
            .all_edges()
            .into_iter()
            .map(|e| EdgeState {
                id: e.id.clone(),
                from: e.from.clone(),
                to: e.to.clone(),
                edge_type: e.edge_type.clone(),
                properties: e
                    .properties
                    .iter()
                    .map(|(k, v)| (k.clone(), value_to_property(v)))
                    .collect(),
            })
            .collect();

        let stats = self.graph.stats();

        GraphState {
            nodes,
            edges,
            next_node_id: stats.node_count,
            next_edge_id: stats.edge_count,
        }
    }

    /// Restore the graph from persisted state, replacing the current graph
    pub fn load_state(&mut self, state: &GraphState) -> Result<(), CypherError> {
        self.graph = PropertyGraph::new();

        // Import nodes first
        for node_state in &state.nodes {
            let mut node = Node::new(node_state.id.clone());
            for label in &node_state.labels {
                node = node.with_label(label.clone());
            }
            for (key, value) in &node_state.properties {
                node = node.with_property(key.clone(), property_to_value(value));
            }
            self.graph.add_node(node);
        }

        // Then import edges
        for edge_state in &state.edges {
            let mut edge = Edge::new(
                edge_state.id.clone(),
                edge_state.from.clone(),
                edge_state.to.clone(),
                edge_state.edge_type.clone(),
            );
            for (key, value) in &edge_state.properties {
                edge = edge.with_property(key.clone(), property_to_value(value));
            }
            if let Err(e) = self.graph.add_edge(edge) {
                return Err(CypherError::Import(e.to_string()));
            }
        }

        Ok(())
    }
}

impl Default for CypherEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert graph Value to serializable PropertyValue
fn value_to_property(v: &Value) -> PropertyValue {
    match v {
        Value::Null => PropertyValue::Null,
        Value::Boolean(b) => PropertyValue::Boolean(*b),
        Value::Integer(i) => PropertyValue::Integer(*i),
        Value::Float(f) => PropertyValue::Float(*f),
        Value::String(s) => PropertyValue::String(s.clone()),
        Value::List(list) => PropertyValue::List(list.iter().map(value_to_property).collect()),
        Value::Map(map) => PropertyValue::Map(
            map.iter()
                .map(|(k, v)| (k.clone(), value_to_property(v)))
                .collect(),
        ),
    }
}

/// Convert PropertyValue back to graph Value
fn property_to_value(p: &PropertyValue) -> Value {
    match p {
        PropertyValue::Null => Value::Null,
        PropertyValue::Boolean(b) => Value::Boolean(*b),
        PropertyValue::Integer(i) => Value::Integer(*i),
        PropertyValue::Float(f) => Value::Float(*f),
        PropertyValue::String(s) => Value::String(s.clone()),
        PropertyValue::List(list) => Value::List(list.iter().map(property_to_value).collect()),
        PropertyValue::Map(map) => Value::Map(
            map.iter()
                .map(|(k, v)| (k.clone(), property_to_value(v)))
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_node() {
        let mut engine = CypherEngine::new();
        let query = "CREATE (n:Person {name: 'Alice', age: 30})";

        let ast = parse_cypher(query).unwrap();
        let mut executor = Executor::new(&mut engine.graph);
        let result = executor.execute(&ast);

        assert!(result.is_ok());
        assert_eq!(engine.graph.stats().node_count, 1);
    }

    #[test]
    fn test_create_relationship() {
        let mut engine = CypherEngine::new();
        let query = "CREATE (a:Person {name: 'Alice'})-[r:KNOWS]->(b:Person {name: 'Bob'})";

        let ast = parse_cypher(query).unwrap();
        let mut executor = Executor::new(&mut engine.graph);
        let result = executor.execute(&ast);

        assert!(result.is_ok());
        let stats = engine.graph.stats();
        assert_eq!(stats.node_count, 2);
        assert_eq!(stats.edge_count, 1);
    }

    #[test]
    fn test_match_nodes() {
        let mut engine = CypherEngine::new();

        // Create data
        let create = "CREATE (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})";
        let ast = parse_cypher(create).unwrap();
        let mut executor = Executor::new(&mut engine.graph);
        executor.execute(&ast).unwrap();

        // Match nodes
        let match_query = "MATCH (n:Person) RETURN n";
        let ast = parse_cypher(match_query).unwrap();
        let mut executor = Executor::new(&mut engine.graph);
        let result = executor.execute(&ast);

        assert!(result.is_ok());
    }

    #[test]
    fn test_parser() {
        let queries = vec![
            "MATCH (n:Person) RETURN n",
            "CREATE (n:Person {name: 'Alice'})",
            "MATCH (a)-[r:KNOWS]->(b) RETURN a, r, b",
            "CREATE (a:Person)-[r:KNOWS]->(b:Person)",
        ];

        for query in queries {
            let result = parse_cypher(query);
            assert!(result.is_ok(), "Failed to parse: {}", query);
        }
    }
}
